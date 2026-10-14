#!/usr/bin/env python3
"""Select a model-blind 72-item Gold-audit pool from astronomy candidates."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MCQ_PER_SOURCE = 12
OPEN_TARGET = 36


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def stable_key(row: dict[str, Any], seed: int) -> str:
    return hashlib.sha256(f"{seed}:{row['native_qa_id']}".encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    base = PROJECT_ROOT / "datasets/processed/domain_model_benchmark_v2/astronomy_eval60_candidates"
    parser.add_argument("--input", type=Path, default=base / "classification_v1/all_labeled.jsonl")
    parser.add_argument("--output-dir", type=Path, default=base / "gold_pool_v1")
    parser.add_argument("--seed", type=int, default=211)
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    eligible = [
        row
        for row in rows
        if row["valid"] is True
        and "astronomy" in row["required_domains"]
        and row["answer_verifiability"] != "unreliable"
    ]
    mcq = [row for row in eligible if row.get("choices")]
    open_rows = [row for row in eligible if not row.get("choices")]

    selected_mcq: list[dict[str, Any]] = []
    for source_file in sorted({row["source_file"] for row in mcq}):
        source_rows = [row for row in mcq if row["source_file"] == source_file]
        source_rows.sort(
            key=lambda row: (
                0 if row["keep_priority"] == "high" else 1,
                stable_key(row, args.seed),
            )
        )
        if len(source_rows) < MCQ_PER_SOURCE:
            raise ValueError(f"not enough eligible MCQ records in {source_file}")
        selected_mcq.extend(source_rows[:MCQ_PER_SOURCE])

    open_rows.sort(
        key=lambda row: (
            0 if row["keep_priority"] == "high" else 1,
            0 if row["answer_verifiability"] == "rule_verifiable" else 1,
            stable_key(row, args.seed),
        )
    )
    if len(open_rows) < OPEN_TARGET:
        raise ValueError("not enough eligible open records")
    selected_open = open_rows[:OPEN_TARGET]
    selected = selected_mcq + selected_open
    ids = [row["native_qa_id"] for row in selected]
    if len(selected) != 72 or len(set(ids)) != 72:
        raise ValueError("Gold pool must contain 72 unique records")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "gold_candidates_72.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for position, row in enumerate(selected):
            frozen = {
                **row,
                "benchmark_candidate": {
                    "version": "astronomy_eval60_expansion_gold_pool.v1",
                    "position": position,
                    "model_outputs_seen_during_selection": False,
                    "evaluation_only": True,
                    "excluded_from_training": True,
                },
            }
            handle.write(json.dumps(frozen, ensure_ascii=False) + "\n")
    payload = output.read_bytes()
    summary = {
        "schema_version": "astronomy_eval60_expansion_gold_pool_summary.v1",
        "records": len(selected),
        "seed": args.seed,
        "surface": dict(sorted(Counter("mcq" if row.get("choices") else "open" for row in selected).items())),
        "source_file": dict(sorted(Counter(row["source_file"] for row in selected).items())),
        "question_type": dict(sorted(Counter(row["question_type"] for row in selected).items())),
        "keep_priority": dict(sorted(Counter(row["keep_priority"] for row in selected).items())),
        "answer_verifiability": dict(sorted(Counter(row["answer_verifiability"] for row in selected).items())),
        "required_domain_sets": {
            "+".join(domains): count
            for domains, count in sorted(Counter(tuple(row["required_domains"]) for row in selected).items())
        },
        "model_outputs_seen_during_selection": False,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    (args.output_dir / "gold_pool_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
