#!/usr/bin/env python3
"""Apply the semantic-review decisions to machine-accepted QA records.

Input: machine-accepted QA JSONL plus a complete review JSON.
Output: curated QA JSONL and a small acceptance summary.
Example: ``python scripts/apply_qa_semantic_review.py``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = PROJECT_ROOT / "datasets/processed/deepseek_qa_pilot_v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=BASE / "verified_qa.jsonl")
    parser.add_argument("--review", type=Path, default=BASE / "codex_semantic_review.json")
    parser.add_argument("--output", type=Path, default=BASE / "verified_qa_curated.jsonl")
    parser.add_argument("--summary", type=Path, default=BASE / "curated_summary.json")
    args = parser.parse_args()

    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    review = json.loads(args.review.read_text(encoding="utf-8"))["items"]
    record_ids = {record["id"] for record in records}
    if record_ids != set(review):
        raise ValueError(
            f"Review must cover machine-accepted records exactly; "
            f"missing={sorted(record_ids - set(review))}, extra={sorted(set(review) - record_ids)}"
        )
    curated = []
    for record in records:
        decision = review[record["id"]]
        if not decision["accepted"]:
            continue
        record = deepcopy(record)
        if "required_domains_override" in decision:
            record["required_domains"] = decision["required_domains_override"]
        if "helpful_domains_override" in decision:
            record["helpful_domains"] = decision["helpful_domains_override"]
        curated.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in curated:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    rejected = [record_id for record_id in sorted(record_ids) if not review[record_id]["accepted"]]
    summary = {
        "machine_accepted": len(records),
        "semantic_review_accepted": len(curated),
        "semantic_review_rejected": len(rejected),
        "rejected_ids": rejected,
        "origin_type": dict(Counter(record["origin_type"] for record in curated)),
        "question_type": dict(Counter(record["question_type"] for record in curated)),
        "difficulty": dict(Counter(record["difficulty"] for record in curated)),
        "required_domains": dict(
            Counter("+".join(sorted(record["required_domains"])) for record in curated)
        ),
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"accepted={len(curated)}/{len(records)} output={args.output}")


if __name__ == "__main__":
    main()
