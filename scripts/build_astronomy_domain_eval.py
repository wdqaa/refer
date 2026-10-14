#!/usr/bin/env python3
"""Freeze the untouched astronomy model-comparison source set.

The split contains every reviewed, valid pilot-500 item whose required domains
include astronomy, excluding all source identities already used by native eval50.
Questions and answers are copied without modification.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def source_key(row: dict[str, Any]) -> tuple[str, int]:
    return row["source_file"], int(row["source_index"])


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    base = PROJECT_ROOT / "datasets/processed/native_qa_mining_v1/pilot_500"
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewed", type=Path, default=base / "reviewed_native_qa.jsonl")
    parser.add_argument("--exclude", type=Path, default=base / "eval_split_v1/eval_50.jsonl")
    parser.add_argument(
        "--train-candidates",
        type=Path,
        default=base / "eval_split_v1/train_candidates_164.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets/processed/domain_model_benchmark_v1/astronomy_eval30",
    )
    args = parser.parse_args()

    reviewed = read_jsonl(args.reviewed)
    excluded = read_jsonl(args.exclude)
    train_candidates = read_jsonl(args.train_candidates)
    excluded_ids = {row["native_qa_id"] for row in excluded}
    excluded_sources = {source_key(row) for row in excluded}
    selected = [
        row
        for row in reviewed
        if row.get("valid") is True
        and "astronomy" in row.get("required_domains", [])
        and row["native_qa_id"] not in excluded_ids
        and source_key(row) not in excluded_sources
    ]
    if len(selected) != 30:
        raise ValueError(f"expected the frozen 30-item astronomy split, found {len(selected)}")
    if len({row["native_qa_id"] for row in selected}) != len(selected):
        raise ValueError("astronomy split contains duplicate native_qa_id values")
    if len({source_key(row) for row in selected}) != len(selected):
        raise ValueError("astronomy split contains duplicate source identities")
    train_ids = {row["native_qa_id"] for row in train_candidates}
    selected_ids = {row["native_qa_id"] for row in selected}
    if not selected_ids <= train_ids:
        raise ValueError("astronomy evaluation records are not all in the prior training pool")

    frozen = []
    for position, row in enumerate(selected):
        frozen.append(
            {
                **row,
                "benchmark_freeze": {
                    "version": "astronomy_domain_eval30.v1",
                    "position": position,
                    "evaluation_only": True,
                    "excluded_from_training": True,
                    "selection_rule": (
                        "reviewed valid pilot-500 item requiring astronomy; excludes native eval50"
                    ),
                },
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "source_items.jsonl"
    write_jsonl(output, frozen)
    write_jsonl(
        args.output_dir / "training_exclusions.jsonl",
        [
            {
                "native_qa_id": row["native_qa_id"],
                "reason": "frozen in astronomy_domain_eval30.v1",
                "source_file": row["source_file"],
                "source_index": row["source_index"],
            }
            for row in frozen
        ],
    )
    payload = output.read_bytes()
    summary = {
        "schema_version": "astronomy_domain_eval30_summary.v1",
        "records": len(frozen),
        "question_types": dict(sorted(Counter(row["question_type"] for row in frozen).items())),
        "domain_types": dict(sorted(Counter(row["domain_type"] for row in frozen).items())),
        "required_domain_sets": {
            "+".join(domains): count
            for domains, count in sorted(Counter(tuple(row["required_domains"]) for row in frozen).items())
        },
        "keep_priority": dict(sorted(Counter(row["keep_priority"] for row in frozen).items())),
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "gold_status": {
            "mcq": "key available; scientific audit still required before final reporting",
            "qualitative_and_short_answer": "standard solution and fixed rubric pending",
        },
        "training_exclusion": "all 30 records are evaluation-only",
        "historical_train_candidates": len(train_candidates),
        "excluded_from_historical_train_candidates": len(selected_ids & train_ids),
        "remaining_training_eligible": len(train_ids - selected_ids),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
