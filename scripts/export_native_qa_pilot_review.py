#!/usr/bin/env python3
"""Validate human pilot reviews and export only verified high-priority cross-domain QA.

Input: classifier labels plus high-cross and sampled single/invalid human-review JSONL.
Output: reviewed cross-domain JSONL and a compact pilot assessment JSON. The source
question and answer are copied unchanged; only reviewed required-domain labels may be
corrected in the verified export.

Example:
    python scripts/export_native_qa_pilot_review.py --pilot-dir \
        datasets/processed/native_qa_mining_v1/pilot_500
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOMAINS = {"physics", "chemistry", "biology", "astronomy", "geography", "mathematics"}
DISPOSITIONS = {"accept_cross", "not_cross_domain", "reject_qa"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def export(pilot_dir: Path) -> dict[str, Any]:
    labeled = {item["native_qa_id"]: item for item in read_jsonl(pilot_dir / "all_labeled.jsonl")}
    high_cross = read_jsonl(pilot_dir / "high_priority_cross_domain.jsonl")
    reviews = read_jsonl(pilot_dir / "high_cross_human_review.jsonl")
    medium_cross = [
        item for item in labeled.values()
        if item["valid"] and item["domain_type"] == "native_cross_domain" and item["keep_priority"] == "medium"
    ]
    medium_reviews = read_jsonl(pilot_dir / "medium_cross_human_review.jsonl")
    sampled = read_jsonl(pilot_dir / "sampled_single_invalid_human_review.jsonl")
    single_policy = json.loads((pilot_dir / "high_single_review_policy.json").read_text(encoding="utf-8"))
    candidate_ids = {item["native_qa_id"] for item in high_cross}
    review_ids = [item["native_qa_id"] for item in reviews]
    if len(review_ids) != len(set(review_ids)) or set(review_ids) != candidate_ids:
        raise ValueError("Human review must cover every high-cross candidate exactly once")
    if {item["native_qa_id"] for item in medium_reviews} != {item["native_qa_id"] for item in medium_cross}:
        raise ValueError("Human review must cover every medium-cross candidate exactly once")

    verified = []
    reclassified_single = []
    for review in reviews + medium_reviews:
        if review.get("disposition") not in DISPOSITIONS:
            raise ValueError(f"Invalid disposition for {review['native_qa_id']}")
        domains = review.get("verified_required_domains")
        if not isinstance(domains, list) or len(domains) != len(set(domains)):
            raise ValueError(f"Invalid reviewed domains for {review['native_qa_id']}")
        if any(domain not in DOMAINS for domain in domains):
            raise ValueError(f"Unknown reviewed domain for {review['native_qa_id']}")
        if review["disposition"] == "accept_cross" and len(domains) < 2:
            raise ValueError(f"Accepted cross item needs at least two domains: {review['native_qa_id']}")
        if review["disposition"] == "not_cross_domain" and len(domains) != 1:
            raise ValueError(f"Non-cross item needs one domain: {review['native_qa_id']}")
        if review["disposition"] == "reject_qa" and domains:
            raise ValueError(f"Rejected QA cannot have verified domains: {review['native_qa_id']}")
        if review["disposition"] == "not_cross_domain":
            source = labeled[review["native_qa_id"]]
            reclassified_single.append(
                {
                    **source,
                    "machine_required_domains": source["required_domains"],
                    "required_domains": domains,
                    "domain_type": "single",
                    "human_review": {
                        "disposition": review["disposition"],
                        "reason": review["reason"],
                    },
                }
            )
            continue
        if review["disposition"] != "accept_cross":
            continue
        source = labeled[review["native_qa_id"]]
        verified.append(
            {
                **source,
                "machine_required_domains": source["required_domains"],
                "required_domains": domains,
                "human_review": {
                    "disposition": review["disposition"],
                    "reason": review["reason"],
                },
            }
        )

    output = pilot_dir / "verified_high_priority_cross_domain.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for item in verified:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    single_output = pilot_dir / "reviewed_reclassified_single.jsonl"
    with single_output.open("w", encoding="utf-8") as handle:
        for item in reclassified_single:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    high_single = read_jsonl(pilot_dir / "high_priority_single.jsonl")
    high_single_ids = {item["native_qa_id"] for item in high_single}
    downgrade = single_policy.get("downgrade_medium") or {}
    rejected_single = single_policy.get("reject") or {}
    override_ids = set(downgrade) | set(rejected_single)
    if single_policy.get("reviewed_all") is not True or single_policy.get("default_disposition") != "keep_high":
        raise ValueError("High-single review policy must explicitly cover all records")
    if set(downgrade) & set(rejected_single) or not override_ids <= high_single_ids:
        raise ValueError("High-single review overrides are inconsistent with candidates")
    reviewed_high = []
    reviewed_medium = []
    for source in high_single:
        item_id = source["native_qa_id"]
        if item_id in rejected_single:
            continue
        if item_id in downgrade:
            reviewed_medium.append(
                {
                    **source,
                    "keep_priority": "medium",
                    "machine_keep_priority": source["keep_priority"],
                    "human_review": {"disposition": "downgrade_medium", "reason": downgrade[item_id]},
                }
            )
        else:
            reviewed_high.append(
                {
                    **source,
                    "human_review": {"disposition": "keep_high", "reason": "Reviewed as valid, grounded, and useful single-domain QA."},
                }
            )
    for name, records in (
        ("verified_high_priority_single.jsonl", reviewed_high),
        ("verified_medium_priority_single.jsonl", reviewed_medium),
    ):
        with (pilot_dir / name).open("w", encoding="utf-8") as handle:
            for item in records:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    reviewed_single = reviewed_high + reviewed_medium + reclassified_single
    reviewed_all = reviewed_single + verified
    for name, records in (
        ("reviewed_native_single.jsonl", reviewed_single),
        ("reviewed_native_qa.jsonl", reviewed_all),
    ):
        if len({item["native_qa_id"] for item in records}) != len(records):
            raise ValueError(f"Duplicate native_qa_id in {name}")
        with (pilot_dir / name).open("w", encoding="utf-8") as handle:
            for item in records:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    dispositions = Counter(item["disposition"] for item in reviews)
    medium_dispositions = Counter(item["disposition"] for item in medium_reviews)
    sampled_groups: dict[str, dict[str, int | float]] = {}
    for group in sorted({item["audit_group"] for item in sampled}):
        group_items = [item for item in sampled if item["audit_group"] == group]
        agreements = sum(bool(item["human_agree"]) for item in group_items)
        sampled_groups[group] = {
            "sampled": len(group_items),
            "human_agree": agreements,
            "agreement_rate": round(agreements / len(group_items), 4),
        }
    summary = json.loads((pilot_dir / "summary.json").read_text(encoding="utf-8"))
    assessment = {
        "experiment_id": "EXP032",
        "prompt_version": "native-qa-classifier-v1.3",
        "total": summary["total_questions"],
        "schema_valid": summary["schema_valid_labeled"],
        "technical_schema_failures": summary["total_questions"] - summary["schema_valid_labeled"],
        "machine_valid": summary["valid"],
        "machine_high_single": summary["high_priority_single"],
        "machine_high_cross": summary["high_priority_cross_domain"],
        "high_cross_human_review": {
            **dict(dispositions),
            "acceptance_rate": round(dispositions["accept_cross"] / len(reviews), 4),
        },
        "medium_cross_human_review": {
            **dict(medium_dispositions),
            "acceptance_rate": round(medium_dispositions["accept_cross"] / len(medium_reviews), 4),
        },
        "sampled_audits": sampled_groups,
        "verified_high_cross": len(verified),
        "reviewed_reclassified_single": len(reclassified_single),
        "high_single_human_review": {
            "total": len(high_single),
            "keep_high": len(reviewed_high),
            "downgrade_medium": len(reviewed_medium),
            "reject": len(rejected_single),
        },
        "reviewed_native_single_total": len(reviewed_single),
        "reviewed_native_qa_total": len(reviewed_all),
        "reviewed_distribution": {
            "domain_type": dict(Counter(item["domain_type"] for item in reviewed_all)),
            "keep_priority": dict(Counter(item["keep_priority"] for item in reviewed_all)),
            "question_type": dict(Counter(item["question_type"] for item in reviewed_all)),
            "required_domains": dict(
                Counter(domain for item in reviewed_all for domain in item["required_domains"])
            ),
        },
        "api_usage": summary["api_usage"],
        "actual_cost_usd": summary["actual_cost_usd"],
        "decision": "stop_before_full_scan",
    }
    (pilot_dir / "pilot_assessment.json").write_text(
        json.dumps(assessment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return assessment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets/processed/native_qa_mining_v1/pilot_500",
    )
    args = parser.parse_args()
    print(json.dumps(export(args.pilot_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
