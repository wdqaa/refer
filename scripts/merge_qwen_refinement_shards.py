#!/usr/bin/env python3
"""Merge completed refinement trace shards in frozen eval order.

Input: the audited baseline-view dataset, exclusion audit, one arm name, and
explicit shard trace paths. Output: one validated 48-record ``traces.jsonl``.
Duplicate item IDs are allowed only when their final prediction is identical.

Example::

    python scripts/merge_qwen_refinement_shards.py \
      --dataset datasets/processed/native_qa_mining_v1/pilot_500/eval_split_v1/eval_50_baseline_view.jsonl \
      --exclusion-audit outputs/baselines/qwen3_8b_native_eval50_v2/review_v1/posthoc_item_audit.jsonl \
      --arm expert_critique --trace shard0/traces.jsonl --trace shard1/traces.jsonl \
      --output outputs/baselines/qwen3_8b_native_eval48_refinement_v1/expert_critique/traces.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--exclusion-audit", type=Path, required=True)
    parser.add_argument("--arm", choices=("expert_critique", "self_refine"), required=True)
    parser.add_argument("--trace", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    excluded = {
        row["item_id"]
        for row in read_jsonl(args.exclusion_audit)
        if row["disposition"] == "exclude_from_audited_summary"
    }
    order = [row["id"] for row in read_jsonl(args.dataset) if row["id"] not in excluded]
    if len(order) != 48:
        parser.error(f"expected audited eval48, found {len(order)}")

    merged: dict[str, dict[str, Any]] = {}
    for path in args.trace:
        for row in read_jsonl(path):
            if row.get("arm") != args.arm or row.get("status") != "ok":
                parser.error(f"invalid row for {args.arm} in {path}: {row.get('item_id')}")
            item_id = row["item_id"]
            if item_id in merged and merged[item_id]["final_prediction"] != row["final_prediction"]:
                parser.error(f"non-identical duplicate prediction for {item_id}")
            merged[item_id] = row
    missing = [item_id for item_id in order if item_id not in merged]
    extra = sorted(set(merged) - set(order))
    if missing or extra:
        parser.error(f"incomplete merge: missing={missing}, extra={extra}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item_id in order:
            handle.write(json.dumps(merged[item_id], ensure_ascii=False) + "\n")
    print(f"merged {len(order)} {args.arm} traces -> {args.output}")


if __name__ == "__main__":
    main()
