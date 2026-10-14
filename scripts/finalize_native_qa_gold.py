#!/usr/bin/env python3
"""Apply an explicit human-review policy to enriched native-QA Gold records.

Input: enriched evaluation JSONL plus a versioned human-review policy. Output: one
review record per input, a curated accepted evaluation JSONL, rejected records, and a
summary. The raw DeepSeek enrichment and source fields are never overwritten.

Example:
    python scripts/finalize_native_qa_gold.py
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from enrich_native_qa_gold import read_jsonl, validate


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_review(record: dict[str, Any], policy: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    item_id = record["native_qa_id"]
    override = policy.get("overrides", {}).get(item_id, {})
    decision = override.get("decision", policy["default_decision"])
    reason = override.get("reason", policy["default_reason"])
    if decision not in {"accept", "accept_with_edit", "reject"}:
        raise ValueError(f"invalid review decision for {item_id}: {decision}")

    original_gold = record["gold_enrichment"]
    curated_gold = json.loads(json.dumps(original_gold, ensure_ascii=False))
    edits = override.get("gold_enrichment", {})
    for field, value in edits.items():
        if field not in {"gold_answer", "standard_solution", "scoring_rubric"}:
            raise ValueError(f"unsupported Gold edit for {item_id}: {field}")
        curated_gold[field] = value

    review = {
        "native_qa_id": item_id,
        "decision": decision,
        "reason": reason,
        "edited_fields": sorted(edits),
        "review_policy_version": policy["version"],
    }
    curated = {
        **record,
        "gold_enrichment": curated_gold,
        "gold_human_review": review,
        "gold_source_trace_preserved": True,
    }
    if decision != "reject":
        errors = validate(curated, curated_gold)
        if errors:
            raise ValueError(f"curated Gold failed validation for {item_id}: {errors}")
    return review, curated


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    base = PROJECT_ROOT / "datasets/processed/native_qa_mining_v1/pilot_500/eval_split_v1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=base / "eval_50_enriched.jsonl")
    parser.add_argument(
        "--policy",
        type=Path,
        default=PROJECT_ROOT / "configs/data/native_qa_gold_human_review_v1.json",
    )
    parser.add_argument("--output-dir", type=Path, default=base)
    parser.add_argument(
        "--artifact-prefix",
        default="eval_50",
        help="Output filename prefix; the default preserves the historical eval50 names.",
    )
    args = parser.parse_args()

    records = read_jsonl(args.input)
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    if len(records) != policy["expected_input_records"]:
        raise ValueError("input record count does not match frozen review policy")
    if sha256(args.input) != policy["expected_input_sha256"]:
        raise ValueError("input SHA-256 does not match frozen review policy")
    unknown = set(policy.get("overrides", {})) - {x["native_qa_id"] for x in records}
    if unknown:
        raise ValueError(f"review overrides contain unknown IDs: {sorted(unknown)}")

    reviews: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for record in records:
        review, curated = apply_review(record, policy)
        reviews.append(review)
        (rejected if review["decision"] == "reject" else accepted).append(curated)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    review_name = (
        "gold_human_review.jsonl"
        if args.artifact_prefix == "eval_50"
        else f"{args.artifact_prefix}_gold_human_review.jsonl"
    )
    rejected_name = (
        "gold_review_rejected.jsonl"
        if args.artifact_prefix == "eval_50"
        else f"{args.artifact_prefix}_gold_review_rejected.jsonl"
    )
    write_jsonl(args.output_dir / review_name, reviews)
    write_jsonl(
        args.output_dir / f"{args.artifact_prefix}_gold_curated.jsonl", accepted
    )
    write_jsonl(args.output_dir / rejected_name, rejected)
    counts = Counter(x["decision"] for x in reviews)
    summary = {
        "input": len(records),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "review_decisions": dict(sorted(counts.items())),
        "edited_records": sum(bool(x["edited_fields"]) for x in reviews),
        "input_sha256": sha256(args.input),
        "policy_version": policy["version"],
    }
    summary_name = (
        "gold_review_summary.json"
        if args.artifact_prefix == "eval_50"
        else f"{args.artifact_prefix}_gold_review_summary.json"
    )
    (args.output_dir / summary_name).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
