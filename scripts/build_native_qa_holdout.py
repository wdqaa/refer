#!/usr/bin/env python3
"""Build the fixed 30-item held-out native-QA classifier evaluation.

Input: the unified verified QA, audited DeepSeek source analyses, raw-source pointers,
and the prior calibration manifest. Output: an unchanged-source manifest containing
15 verified native-single, 9 verified native-cross-domain, and 6 invalid controls.

Example:
    python scripts/build_native_qa_holdout.py
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

VERIFIED_SINGLE_IDS = [
    "qa-pilot-v1-biology-11457",
    "qa-pilot-v1-astronomy-3654",
    "qa-pilot-v1-chemistry-9315",
    "qa-pilot-v1-biology-4324",
    "qa-pilot-v1-astronomy-456",
    "qa-pilot-v1-physics-5114",
    "qa-pilot-v1-geography-481",
    "qa-pilot-v1-physics-9885",
    "qa-batch-v2-astronomy-383",
    "qa-batch-v2-biology-12824",
    "qa-batch-v2-chemistry-10656",
    "qa-batch-v2-chemistry-7904",
    "qa-batch-v2-geography-15776",
    "qa-batch-v2-geography-28410",
    "qa-batch-v2-physics-14715",
]

VERIFIED_CROSS_IDS = [
    "qa-pilot-v1-physics-4416",
    "qa-pilot-v1-chemistry-15521",
    "qa-pilot-v1-physics-13303",
    "qa-pilot-v1-geography-1614",
    "qa-batch-v2-astronomy-3016",
    "qa-batch-v2-astronomy-3033",
    "qa-batch-v2-chemistry-16377",
    "qa-batch-v2-chemistry-19509",
    "qa-batch-v2-physics-7460",
]

INVALID_PROTOCOL_IDS = {
    "protocol-v1-astronomy-2690": ["astronomy"],
    "protocol-v1-biology-9502": ["biology", "mathematics"],
    "protocol-v1-chemistry-8748": ["chemistry"],
    "protocol-v1-physics-11360": ["physics"],
    "protocol-v1-geography-27555": ["geography"],
    "protocol-v1-chemistry-3909": ["chemistry"],
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def source_index(source_id: str) -> int:
    if not source_id.startswith("index:"):
        raise ValueError(f"Expected index source id, got {source_id}")
    return int(source_id.split(":", 1)[1])


def native_id(source_file: str, index: int) -> str:
    digest = hashlib.sha1(f"{source_file}:{index}".encode("utf-8")).hexdigest()[:14]
    return f"native-{digest}"


def gold_domains(domains: list[str]) -> list[dict[str, str]]:
    return [{"domain": domain} for domain in domains]


def from_verified(item: dict[str, Any]) -> dict[str, Any]:
    source = item["source"]
    index = source_index(source["source_id"])
    domains = list(item["required_domains"])
    group = "verified_native_cross_domain" if item["origin_type"] == "native_cross_domain" else "verified_native_single"
    return {
        "native_qa_id": native_id(source["source_file"], index),
        "calibration_group": group,
        "source_file": source["source_file"],
        "source_index": index,
        "source_id": source["source_id"],
        "directory_domain": source["source_file"].split("/", 1)[0].replace("math", "mathematics"),
        "question": source["original_question"],
        "answer": source["original_answer"],
        "choices": item.get("choices") or {},
        "calibration_gold": {
            "id": item["id"],
            "category": group,
            "scientifically_valid": True,
            "required_domains": gold_domains(domains),
            "question_type": item["question_type"],
        },
    }


def from_invalid(source: dict[str, Any], analysis: dict[str, Any], domains: list[str]) -> dict[str, Any]:
    return {
        "native_qa_id": native_id(source["source_file"], int(source["source_index"])),
        "calibration_group": "audited_invalid",
        "source_file": source["source_file"],
        "source_index": int(source["source_index"]),
        "source_id": source["source_id"],
        "directory_domain": source["source_domain"].replace("math", "mathematics"),
        "question": source["question"],
        "answer": source["answer"],
        "choices": source.get("choices") or {},
        "calibration_gold": {
            "id": source["protocol_id"],
            "category": "audited_invalid",
            "scientifically_valid": False,
            "required_domains": gold_domains(domains),
            "question_type": analysis["question_type"],
            "validity_issue": analysis["validity_issue"],
        },
    }


def build(
    verified_path: Path,
    sources_path: Path,
    analyses_path: Path,
    prior_manifest_path: Path,
) -> list[dict[str, Any]]:
    verified = {item["id"]: item for item in read_jsonl(verified_path)}
    sources = {item["protocol_id"]: item for item in read_jsonl(sources_path)}
    analyses = {item["protocol_id"]: item for item in read_jsonl(analyses_path)}
    selected = [from_verified(verified[item_id]) for item_id in VERIFIED_SINGLE_IDS + VERIFIED_CROSS_IDS]
    for protocol_id, domains in INVALID_PROTOCOL_IDS.items():
        trace = analyses[protocol_id]
        analysis = trace["analysis"]
        if not trace.get("schema_valid") or analysis.get("scientifically_valid") is not False:
            raise RuntimeError(f"Invalid control is not an audited invalid source: {protocol_id}")
        selected.append(from_invalid(sources[protocol_id], analysis, domains))

    prior = read_jsonl(prior_manifest_path)
    prior_identities = {(item["source_file"], int(item["source_index"])) for item in prior}
    identities = [(item["source_file"], int(item["source_index"])) for item in selected]
    overlap = sorted(set(identities) & prior_identities)
    if overlap:
        raise RuntimeError(f"Held-out sources overlap prior calibration: {overlap}")
    if len(selected) != 30 or len(set(identities)) != 30:
        raise RuntimeError("Held-out manifest must contain 30 unique sources")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    base = PROJECT_ROOT / "datasets/processed/native_qa_mining_v1"
    parser.add_argument("--verified", type=Path, default=base.parent / "deepseek_qa_curated_v2/verified_qa_curated_49.jsonl")
    parser.add_argument("--sources", type=Path, default=base.parent / "deepseek_protocol_v1/protocol_sources_150.jsonl")
    parser.add_argument("--analyses", type=Path, default=base.parent / "deepseek_protocol_v1/source_analysis_v2_1.jsonl")
    parser.add_argument("--prior-manifest", type=Path, default=base / "calibration_60/manifest.jsonl")
    parser.add_argument("--output", type=Path, default=base / "holdout_30/manifest.jsonl")
    parser.add_argument("--summary", type=Path, default=base / "holdout_30/manifest_summary.json")
    args = parser.parse_args()
    records = build(args.verified, args.sources, args.analyses, args.prior_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary = {
        "total": len(records),
        "calibration_group": dict(Counter(item["calibration_group"] for item in records)),
        "gold_domain_type": dict(Counter("native_cross_domain" if len(item["calibration_gold"]["required_domains"]) > 1 else "single" for item in records if item["calibration_gold"]["scientifically_valid"])),
        "valid": sum(item["calibration_gold"]["scientifically_valid"] for item in records),
        "invalid": sum(not item["calibration_gold"]["scientifically_valid"] for item in records),
        "overlap_with_calibration_60": 0,
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
