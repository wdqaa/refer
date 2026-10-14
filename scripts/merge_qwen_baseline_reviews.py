#!/usr/bin/env python3
"""Merge deterministic and arm-blinded manual Qwen baseline scores.

Input: verified QA, preliminary scores, private arm mapping, and manual review JSONL.
Output: one final score per item/arm and aggregate paired-comparison JSON.
Example: ``python scripts/merge_qwen_baseline_reviews.py --manual outputs/baselines/qwen3_8b_full49_v1/review_v1/manual_root.jsonl --review-dir outputs/baselines/qwen3_8b_full49_v1/review_v1``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import statistics
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARMS = ("direct", "routed_multi_agent", "oracle_multi_agent")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    index = min(len(values) - 1, max(0, round(fraction * (len(values) - 1))))
    return values[index]


def paired_bootstrap(
    left: dict[str, bool], right: dict[str, bool], *, seed: int, iterations: int
) -> dict[str, float]:
    ids = sorted(left)
    rng = random.Random(seed)
    diffs = []
    for _ in range(iterations):
        sample = [rng.choice(ids) for _ in ids]
        diffs.append(
            statistics.mean(right[item] for item in sample)
            - statistics.mean(left[item] for item in sample)
        )
    observed = statistics.mean(right.values()) - statistics.mean(left.values())
    return {
        "difference": observed,
        "bootstrap_95_low": percentile(diffs, 0.025),
        "bootstrap_95_high": percentile(diffs, 0.975),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--manual", type=Path, action="append", required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "datasets/processed/deepseek_qa_curated_v2/verified_qa_curated_49.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    args = parser.parse_args()

    records = {item["id"]: item for item in read_jsonl(args.dataset)}
    preliminary = {
        (row["item_id"], row["arm"]): row
        for row in read_jsonl(args.review_dir / "preliminary_scores.jsonl")
    }
    mapping = {
        (row["item_id"], entry["candidate_code"]): entry["arm"]
        for row in read_jsonl(args.review_dir / "arm_mapping.jsonl")
        for entry in row["mapping"]
    }
    manual_by_key = {}
    for path in args.manual:
        for row in read_jsonl(path):
            for judgment in row["judgments"]:
                mapped = (row["item_id"], judgment["candidate_code"])
                if mapped not in mapping:
                    raise ValueError(f"Manual judgment has no blinded mapping: {mapped}")
                key = (row["item_id"], mapping[mapped])
                if key in manual_by_key:
                    raise ValueError(f"Duplicate manual judgment: {key}")
                manual_by_key[key] = judgment

    final = []
    for key, score in sorted(preliminary.items()):
        item_id, arm = key
        if score["auto_correct"] is None:
            if key not in manual_by_key:
                raise ValueError(f"Missing manual judgment: {key}")
            judgment = manual_by_key[key]
            correct = judgment["correct"]
            method = "blinded_rubric_review"
            reason = judgment["rubric_notes"]
        else:
            correct = score["auto_correct"]
            method = score["method"]
            reason = score["reason"]
        final.append(
            {
                "item_id": item_id,
                "arm": arm,
                "correct": bool(correct),
                "method": method,
                "reason": reason,
                "question_type": records[item_id]["question_type"],
                "origin_type": records[item_id]["origin_type"],
            }
        )

    expected = len(records) * len(ARMS)
    if len(final) != expected:
        raise ValueError(f"Expected {expected} final scores, found {len(final)}")
    with (args.review_dir / "final_scores.jsonl").open("w", encoding="utf-8") as handle:
        for row in final:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_arm: dict[str, dict[str, bool]] = defaultdict(dict)
    summaries = {}
    for arm in ARMS:
        rows = [row for row in final if row["arm"] == arm]
        by_arm[arm] = {row["item_id"]: row["correct"] for row in rows}
        summaries[arm] = {
            "correct": sum(row["correct"] for row in rows),
            "total": len(rows),
            "accuracy": statistics.mean(row["correct"] for row in rows),
            "by_question_type": {
                name: dict(Counter(str(row["correct"]).lower() for row in rows if row["question_type"] == name))
                for name in sorted({row["question_type"] for row in rows})
            },
            "by_origin_type": {
                name: dict(Counter(str(row["correct"]).lower() for row in rows if row["origin_type"] == name))
                for name in sorted({row["origin_type"] for row in rows})
            },
        }
    comparison_specs = {
        "routed_minus_direct": ("direct", "routed_multi_agent", args.seed),
        "oracle_minus_direct": ("direct", "oracle_multi_agent", args.seed + 1),
        "oracle_minus_routed": ("routed_multi_agent", "oracle_multi_agent", args.seed + 2),
    }
    comparisons = {}
    for name, (baseline_arm, target_arm, comparison_seed) in comparison_specs.items():
        baseline = by_arm[baseline_arm]
        target = by_arm[target_arm]
        comparison = paired_bootstrap(
            baseline, target, seed=comparison_seed, iterations=args.bootstrap_iterations
        )
        comparison.update(
            baseline_arm=baseline_arm,
            target_arm=target_arm,
            target_only_correct=sum(target[item] and not baseline[item] for item in baseline),
            baseline_only_correct=sum(baseline[item] and not target[item] for item in baseline),
        )
        comparisons[name] = comparison

    summary = {
        "manual_judgments": len(manual_by_key),
        "arms": summaries,
        "paired_comparisons": comparisons,
        "interpretation": "Oracle-domain Multi-Agent is a go only if its paired gain over Direct is positive and practically meaningful; intervals include finite-set uncertainty.",
    }
    (args.review_dir / "final_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
