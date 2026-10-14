#!/usr/bin/env python3
"""Apply explicit Codex adjudications to blinded domain-model judge traces."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.judge_astronomy_domain_answers import gold_payload, validate_judgment


MODEL_KEYS = ("qwen3_8b", "astrosage_8b")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--adjudications", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = {
        row["native_qa_id"]: row
        for row in read_jsonl(args.dataset)
        if row["question_type"] != "mcq"
    }
    policy = json.loads(args.adjudications.read_text(encoding="utf-8"))
    overrides = policy["adjudications"]
    rubric_resolutions = policy.get("rubric_resolutions", {})
    traces = read_jsonl(args.traces)
    if {row["item_id"] for row in traces} != set(records):
        raise ValueError("judge traces do not cover the frozen non-MCQ set exactly")

    rows: list[dict[str, Any]] = []
    for trace in traces:
        item_id = trace["item_id"]
        result = copy.deepcopy(trace.get("result"))
        if not isinstance(result, dict):
            raise ValueError(f"{item_id} has no recoverable judge result")
        stripped_fields: list[str] = []
        if set(result) - {"item_id", "rubric_status", "rubric_issue", "candidates"} == {"type"}:
            result.pop("type")
            stripped_fields.append("type")
        rubric_resolution = rubric_resolutions.get(item_id)
        if rubric_resolution:
            result["rubric_status"] = rubric_resolution["status"]
            result["rubric_issue"] = None
        criterion_ids = {
            row["criterion_id"] for row in gold_payload(records[item_id])["rubric"]
        }
        validate_judgment(result, item_id, criterion_ids)
        by_label = {row["label"]: row for row in result["candidates"]}
        for label, model_key in trace["blind_map"].items():
            candidate = copy.deepcopy(by_label[label])
            override = overrides.get(item_id, {}).get(model_key, {})
            by_criterion = {
                row["criterion_id"]: row for row in candidate["criterion_results"]
            }
            for criterion_id, edit in override.get("coverage", {}).items():
                if criterion_id not in by_criterion:
                    raise ValueError(f"unknown criterion override: {(item_id, model_key, criterion_id)}")
                by_criterion[criterion_id]["covered"] = edit["covered"]
                by_criterion[criterion_id]["evidence"] = edit["evidence"]
            if "scientific_errors" in override:
                candidate["scientific_errors"] = override["scientific_errors"]
            rows.append(
                {
                    "schema_version": "domain_model_judgment.v1",
                    "item_id": item_id,
                    "model_key": model_key,
                    "rubric_status": result["rubric_status"],
                    "rubric_issue": result["rubric_issue"],
                    "criterion_results": candidate["criterion_results"],
                    "scientific_errors": candidate["scientific_errors"],
                    "overall_notes": candidate["overall_notes"],
                    "confidence": candidate["confidence"],
                    "review_status": "codex_audited",
                    "audit": {
                        "policy_id": policy["id"],
                        "disposition": (
                            "edited"
                            if override or stripped_fields or rubric_resolution
                            else "accepted_first_pass"
                        ),
                        "note": override.get(
                            "audit_note", "First-pass judgment accepted after paired-answer audit."
                        ),
                        "stripped_extra_fields": stripped_fields,
                        "rubric_resolution": rubric_resolution,
                    },
                }
            )

    expected = {(item_id, model) for item_id in records for model in MODEL_KEYS}
    actual = {(row["item_id"], row["model_key"]) for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError("audited judgments do not cover every item/model pair exactly once")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"audited_rows={len(rows)} edited={sum(r['audit']['disposition'] == 'edited' for r in rows)} output={args.output}")


if __name__ == "__main__":
    main()
