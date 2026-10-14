#!/usr/bin/env python3
"""Freeze a deterministic 50-item native-QA evaluation split.

Input: reviewed native single- and cross-domain JSONL from the fixed 500-item pilot.
Output: 25 cross-domain plus 25 matched single-domain evaluation records and the
remaining reviewed records as training candidates. Questions and answers are unchanged.

Example:
    python scripts/build_native_qa_eval_split.py
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build(records: list[dict[str, Any]], seed: int = 131) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    cross = [item for item in records if item["domain_type"] == "native_cross_domain"]
    single = [item for item in records if item["domain_type"] == "single"]
    rng.shuffle(cross)
    # Preserve broad question-type coverage rather than selecting only MCQ or qualitative items.
    cross.sort(key=lambda item: Counter(x["question_type"] for x in cross[:25])[item["question_type"]])
    eval_cross = cross[:25]

    available = list(single)
    rng.shuffle(available)
    eval_single = []
    for target in eval_cross:
        exact = [item for item in available if item["question_type"] == target["question_type"] and item["required_domains"][0] in target["required_domains"]]
        same_type = [item for item in available if item["question_type"] == target["question_type"]]
        same_domain = [item for item in available if item["required_domains"][0] in target["required_domains"]]
        chosen = (exact or same_type or same_domain or available)[0]
        eval_single.append(chosen)
        available.remove(chosen)

    evaluation = eval_cross + eval_single
    rng.shuffle(evaluation)
    eval_ids = {item["native_qa_id"] for item in evaluation}
    remaining = [item for item in records if item["native_qa_id"] not in eval_ids]
    if len(evaluation) != 50 or len(eval_ids) != 50 or sum(item["domain_type"] == "native_cross_domain" for item in evaluation) != 25:
        raise RuntimeError("Evaluation split invariant failed")
    return evaluation, remaining


def main() -> None:
    parser = argparse.ArgumentParser()
    base = PROJECT_ROOT / "datasets/processed/native_qa_mining_v1/pilot_500"
    parser.add_argument("--input", type=Path, default=base / "reviewed_native_qa.jsonl")
    parser.add_argument("--output-dir", type=Path, default=base / "eval_split_v1")
    parser.add_argument("--seed", type=int, default=131)
    args = parser.parse_args()
    records = read_jsonl(args.input)
    evaluation, remaining = build(records, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, items in (("eval_50.jsonl", evaluation), ("train_candidates_164.jsonl", remaining)):
        with (args.output_dir / name).open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary = {
        "seed": args.seed,
        "eval_total": len(evaluation),
        "eval_domain_type": dict(Counter(item["domain_type"] for item in evaluation)),
        "eval_question_type": dict(Counter(item["question_type"] for item in evaluation)),
        "eval_required_domains": dict(Counter(domain for item in evaluation for domain in item["required_domains"])),
        "remaining_total": len(remaining),
        "overlap": 0,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
