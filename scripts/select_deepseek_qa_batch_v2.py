#!/usr/bin/env python3
"""Select the bounded second DeepSeek QA-production batch.

Input: the 150-source protocol batch, Source Analyzer v2.1, and prior manifests.
Output: a deterministic 78-record QA manifest plus a compact selection summary.
Example: ``python scripts/select_deepseek_qa_batch_v2.py``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = PROJECT_ROOT / "datasets/processed/deepseek_protocol_v1"
PILOT = PROJECT_ROOT / "datasets/processed/deepseek_qa_pilot_v1/manifest_30.jsonl"
VALIDATION = PROJECT_ROOT / "datasets/processed/deepseek_qa_validation_v2/manifest_fresh6.jsonl"

# Precision-first exclusions made from question text and the v2 failure taxonomy.
EXCLUDE_NATIVE_CROSS = {
    "protocol-v1-astronomy-1126",  # black-hole physics split as astronomy
    "protocol-v1-physics-2342",  # underspecified numerical calculation
    "protocol-v1-geography-13821",  # study-specific MESS result
    "protocol-v1-geography-16345",  # study-specific niche-map result
    "protocol-v1-geography-7638",  # study-specific fire/GIS result
}
EXCLUDE_TRANSFORM = {
    "protocol-v1-biology-15434",  # cultural-art extension is not a clean science dependency
    "protocol-v1-chemistry-15968",  # materials/solid-state boundary split
    "protocol-v1-physics-16690",  # atomic quantum-chemistry boundary split
    "protocol-v1-physics-17081",  # routine phase-change thermodynamics boundary split
    "protocol-v1-geography-31104",  # study-specific year comparison
    "protocol-v1-geography-5773",  # specialized regional reference dependency
    "protocol-v1-geography-17094",  # study-specific fishery finding
    "protocol-v1-geography-947",  # study-specific decision-support result
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    base = PROJECT_ROOT / "datasets/processed/deepseek_qa_batch_v2"
    parser.add_argument("--sources", type=Path, default=PROTOCOL / "protocol_sources_150.jsonl")
    parser.add_argument("--analysis", type=Path, default=PROTOCOL / "source_analysis_v2_1.jsonl")
    parser.add_argument("--pilot-manifest", type=Path, default=PILOT)
    parser.add_argument("--validation-manifest", type=Path, default=VALIDATION)
    parser.add_argument("--output", type=Path, default=base / "manifest_78.jsonl")
    parser.add_argument("--summary", type=Path, default=base / "selection_summary.json")
    args = parser.parse_args()

    sources = {item["protocol_id"]: item for item in read_jsonl(args.sources)}
    analyses = {item["protocol_id"]: item for item in read_jsonl(args.analysis)}
    used = {
        item["source"]["protocol_id"]
        for path in (args.pilot_manifest, args.validation_manifest)
        for item in read_jsonl(path)
    }
    output = []
    for protocol_id in sorted(analyses):
        if protocol_id in used:
            continue
        trace = analyses[protocol_id]
        analysis = trace["analysis"]
        route = analysis["decision"]
        if not trace["schema_valid"] or not analysis["scientifically_valid"]:
            continue
        if analysis["reference_answer_role"] == "unreliable_or_insufficient":
            continue
        if route == "keep_cross_domain" and protocol_id in EXCLUDE_NATIVE_CROSS:
            continue
        if route == "transform_candidate" and protocol_id in EXCLUDE_TRANSFORM:
            continue
        if route not in {"keep_control", "keep_cross_domain", "transform_candidate"}:
            continue
        output.append(
            {
                "qa_id": protocol_id.replace("protocol-v1", "qa-batch-v2"),
                "route": route,
                "source": sources[protocol_id],
                "source_analysis": analysis,
            }
        )

    route_counts = Counter(item["route"] for item in output)
    expected = {"keep_control": 47, "keep_cross_domain": 9, "transform_candidate": 22}
    if len(output) != 78 or dict(route_counts) != expected:
        raise RuntimeError(f"Unexpected selection: total={len(output)} routes={dict(route_counts)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in output:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary = {
        "total": len(output),
        "route": dict(route_counts),
        "question_type": dict(Counter(item["source_analysis"]["question_type"] for item in output)),
        "primary_domain": dict(Counter(item["source_analysis"]["primary_domain"] for item in output)),
        "planned_first_pass_api_calls": 200,
        "estimated_retry_allowance": 10,
        "top_up_candidates_remaining": 8,
        "estimated_top_up_max_api_calls": 32,
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote={len(output)} routes={dict(route_counts)} output={args.output}")


if __name__ == "__main__":
    main()
