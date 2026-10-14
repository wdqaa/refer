#!/usr/bin/env python3
"""Build a lossless baseline-compatible view of curated native-QA evaluation Gold.

Input: the frozen native-QA JSONL containing ``gold_enrichment``. Output: deterministic
JSONL in the schema consumed by the existing Qwen baseline and blinded-review scripts.
Questions, choices, required domains, solutions, and rubric text are not rewritten.

Example:
    python scripts/build_native_qa_baseline_view.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def convert(record: dict[str, Any]) -> dict[str, Any]:
    gold = record["gold_enrichment"]
    if gold.get("accepted") is not True:
        raise ValueError(f"Gold is not accepted: {record['native_qa_id']}")
    rubric = gold["scoring_rubric"]
    if sum(item["points"] for item in rubric) != 10:
        raise ValueError(f"Rubric does not sum to 10: {record['native_qa_id']}")
    origin_type = (
        "native_cross_domain"
        if record["domain_type"] == "native_cross_domain"
        else "native_single"
    )
    choices = record.get("choices") or None
    return {
        "schema_version": "native_qa_baseline_view.v1",
        "id": record["native_qa_id"],
        "source": {
            "source_file": record["source_file"],
            "source_id": record["source_id"],
            "source_index": record["source_index"],
        },
        "origin_type": origin_type,
        "question_type": record["question_type"],
        "question": record["question"],
        "choices": choices,
        "gold": {
            "answer": gold["gold_answer"]["answer_text"],
            "correct_choice": gold["gold_answer"]["correct_choice"],
            "normalized_answer": {
                "canonical": gold["gold_answer"]["answer_text"],
                "value": None,
                "unit": None,
                "tolerance": None,
                "aliases": [],
                "canonical_expression": None,
            },
            "standard_solution": gold["standard_solution"],
            "rubric": {
                "scoring_items": rubric,
                "total_points": 10,
            },
        },
        "required_domains": record["required_domains"],
        "verification": {
            "accepted": True,
            "gold_human_review": record["gold_human_review"]["decision"],
            "evaluation_only": True,
        },
        "provenance": {
            "native_qa_id": record["native_qa_id"],
            "gold_prompt_version": record["gold_prompt_version"],
            "gold_review_policy_version": record["gold_human_review"]["review_policy_version"],
        },
    }


def main() -> None:
    base = PROJECT_ROOT / "datasets/processed/native_qa_mining_v1/pilot_500/eval_split_v1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=base / "eval_50_gold_curated.jsonl")
    parser.add_argument("--output", type=Path, default=base / "eval_50_baseline_view.jsonl")
    args = parser.parse_args()

    source = read_jsonl(args.input)
    converted = [convert(record) for record in source]
    if len(converted) != 50 or len({record["id"] for record in converted}) != len(converted):
        raise ValueError("frozen baseline view must contain exactly 50 unique records")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in converted:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"records={len(converted)} output={args.output}")


if __name__ == "__main__":
    main()
