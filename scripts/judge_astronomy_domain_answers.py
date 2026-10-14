#!/usr/bin/env python3
"""Blindly grade non-MCQ astronomy answers against a frozen audited rubric.

This is an optional DeepSeek first pass, not the final authority. It randomizes the
model-to-A/B mapping deterministically, stores the mapping only in the local trace,
and never sends model names. The resulting review packet must be audited before it is
used as the final judgments file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.baseline_runner import system_with_schema
from src.generation.deepseek_api import DeepSeekJSONClient, DeepSeekResponseParseError


MODEL_KEYS = ("qwen3_8b", "astrosage_8b")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def unique_traces(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        item_id = row["item_id"]
        if item_id in rows:
            raise ValueError(f"duplicate trace in {path}: {item_id}")
        rows[item_id] = row
    return rows


def parse_trace_specs(specs: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for spec in specs:
        key, separator, raw_path = spec.partition("=")
        if not separator or key not in MODEL_KEYS or key in result:
            raise ValueError(f"invalid or duplicate trace specification: {spec}")
        result[key] = Path(raw_path)
    if set(result) != set(MODEL_KEYS):
        raise ValueError(f"expected traces for {MODEL_KEYS}")
    return result


def gold_payload(record: dict[str, Any]) -> dict[str, Any]:
    gold = record.get("gold_enrichment")
    if not isinstance(gold, dict) or gold.get("accepted") is not True:
        raise ValueError(f"{record['native_qa_id']} lacks accepted audited Gold")
    minimum_criteria = 1 if record["question_type"] == "short_answer" else 3
    if not minimum_criteria <= len(gold["scoring_rubric"]) <= 5:
        raise ValueError(
            f"{record['native_qa_id']} {record['question_type']} Gold must have "
            f"{minimum_criteria}-5 audited criteria"
        )
    rubric = [
        {
            "criterion_id": f"C{index}",
            "criterion": row["criterion"],
            "points": row["points"],
        }
        for index, row in enumerate(gold["scoring_rubric"], 1)
    ]
    return {
        "gold_answer": gold["gold_answer"]["answer_text"],
        "standard_solution": gold["standard_solution"],
        "rubric": rubric,
    }


def validate_judgment(
    result: dict[str, Any], item_id: str, criterion_ids: set[str]
) -> None:
    if set(result) != {"item_id", "rubric_status", "rubric_issue", "candidates"}:
        raise ValueError("judge returned unexpected top-level fields")
    if result["item_id"] != item_id or result["rubric_status"] not in {"usable", "problematic"}:
        raise ValueError("judge returned invalid item id or rubric status")
    if result["rubric_issue"] is not None and not isinstance(result["rubric_issue"], str):
        raise ValueError("rubric_issue must be a string or null")
    candidates = result["candidates"]
    if not isinstance(candidates, list) or {row.get("label") for row in candidates} != {"A", "B"}:
        raise ValueError("judge must return exactly candidates A and B")
    for candidate in candidates:
        if set(candidate) != {
            "label", "criterion_results", "scientific_errors", "overall_notes", "confidence"
        }:
            raise ValueError("candidate judgment has unexpected fields")
        criteria = candidate["criterion_results"]
        if len(criteria) != len(criterion_ids) or {
            row.get("criterion_id") for row in criteria
        } != criterion_ids:
            raise ValueError("candidate judgment does not cover every rubric criterion exactly once")
        for row in criteria:
            if set(row) != {"criterion_id", "covered", "evidence"} or not isinstance(
                row["covered"], bool
            ):
                raise ValueError("invalid criterion result")
        for error in candidate["scientific_errors"]:
            if set(error) != {"claim", "correction", "severity"} or error["severity"] not in {
                "material", "minor"
            }:
                raise ValueError("invalid scientific error")
        confidence = candidate["confidence"]
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0 <= confidence <= 1
        ):
            raise ValueError("invalid confidence")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--trace", action="append", required=True)
    parser.add_argument(
        "--prompt",
        type=Path,
        default=PROJECT_ROOT / "src/prompts/domain_model_answer_judge_v1.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    records = [
        row for row in read_jsonl(args.dataset) if row["question_type"] != "mcq"
    ]
    if args.limit is not None:
        records = records[: args.limit]
    trace_paths = parse_trace_specs(args.trace)
    traces = {key: unique_traces(path) for key, path in trace_paths.items()}
    prompt = json.loads(args.prompt.read_text(encoding="utf-8"))
    client = DeepSeekJSONClient(model=args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "blind_judge_traces.jsonl"
    existing = read_jsonl(raw_path) if raw_path.exists() else []
    completed = {row["item_id"] for row in existing}

    with raw_path.open("a", encoding="utf-8") as handle:
        for index, record in enumerate(records, 1):
            item_id = record["native_qa_id"]
            if item_id in completed:
                continue
            missing = [key for key in MODEL_KEYS if item_id not in traces[key]]
            if missing:
                raise ValueError(f"{item_id} missing model traces: {missing}")
            swap = hashlib.sha256(item_id.encode("utf-8")).digest()[0] % 2 == 1
            ordered = MODEL_KEYS[::-1] if swap else MODEL_KEYS
            blind_map = {"A": ordered[0], "B": ordered[1]}
            gold = gold_payload(record)
            candidates = {
                label: traces[key][item_id]["final_prediction"]
                for label, key in blind_map.items()
            }
            request = {
                "item_id": item_id,
                "question": record["question"],
                "choices": record.get("choices") or None,
                **gold,
                "candidates": candidates,
            }
            result = None
            metadata = None
            raw_response_content = None
            try:
                result, metadata = client.generate_json(
                    [
                        {"role": "system", "content": system_with_schema(prompt)},
                        {"role": "user", "content": json.dumps(request, ensure_ascii=False, indent=2)},
                    ],
                    max_tokens=args.max_tokens,
                    thinking_enabled=False,
                    retries=0,
                )
            except DeepSeekResponseParseError as exc:
                raw_response_content = exc.raw_content
                metadata = exc.metadata
                error = f"{type(exc).__name__}: {exc}"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            else:
                try:
                    criterion_ids = {row["criterion_id"] for row in gold["rubric"]}
                    validate_judgment(result, item_id, criterion_ids)
                    error = None
                except Exception as exc:
                    # Preserve the paid response and usage even when its semantic
                    # contract fails deterministic validation.
                    error = f"{type(exc).__name__}: {exc}"
            trace = {
                "schema_version": "domain_model_blind_judge_trace.v2",
                "item_id": item_id,
                "prompt_id": prompt["id"],
                "blind_map": blind_map,
                "request": request,
                "result": result,
                "raw_response_content": raw_response_content,
                "api_metadata": metadata,
                "error": error,
            }
            handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index}/{len(records)}] {item_id} error={error is not None}", flush=True)

    review_rows: list[dict[str, Any]] = []
    for trace in read_jsonl(raw_path):
        if trace.get("error") or not trace.get("result"):
            continue
        by_label = {row["label"]: row for row in trace["result"]["candidates"]}
        for label, model_key in trace["blind_map"].items():
            candidate = by_label[label]
            review_rows.append(
                {
                    "schema_version": "domain_model_judgment.v1",
                    "item_id": trace["item_id"],
                    "model_key": model_key,
                    "rubric_status": trace["result"]["rubric_status"],
                    "rubric_issue": trace["result"]["rubric_issue"],
                    "criterion_results": candidate["criterion_results"],
                    "scientific_errors": candidate["scientific_errors"],
                    "overall_notes": candidate["overall_notes"],
                    "confidence": candidate["confidence"],
                    "review_status": "deepseek_first_pass_requires_audit",
                }
            )
    with (args.output_dir / "review_packet.jsonl").open("w", encoding="utf-8") as handle:
        for row in review_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"review_rows={len(review_rows)} output={args.output_dir / 'review_packet.jsonl'}")


if __name__ == "__main__":
    main()
