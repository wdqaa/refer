#!/usr/bin/env python3
"""Merge curated QA JSONL files while rejecting duplicate IDs.

Input: two or more repeated ``--input`` curated QA files.
Output: one deterministic JSONL file plus a compact source-count summary.
Example: ``python scripts/merge_curated_qa.py --input a.jsonl --input b.jsonl --output merged.jsonl --summary summary.json``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    records = []
    source_counts = {}
    seen = set()
    for path in args.input:
        items = read_jsonl(path)
        source_counts[str(path)] = len(items)
        for item in items:
            if item["id"] in seen:
                raise ValueError(f"Duplicate QA ID: {item['id']}")
            seen.add(item["id"])
            records.append(item)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "total": len(records),
        "inputs": source_counts,
        "origin_type": dict(Counter(item["origin_type"] for item in records)),
        "question_type": dict(Counter(item["question_type"] for item in records)),
        "difficulty": dict(Counter(item["difficulty"] for item in records)),
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"merged={len(records)} output={args.output}")


if __name__ == "__main__":
    main()
