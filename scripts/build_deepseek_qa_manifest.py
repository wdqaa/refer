#!/usr/bin/env python3
"""Build a validated QA-production manifest from explicit protocol IDs.

Input: protocol sources, Source Analyzer v2.1 traces, and repeated ``--protocol-id`` values.
Output: a self-contained JSONL manifest accepted by ``generate_deepseek_qa_pilot.py``.
Example: ``python scripts/build_deepseek_qa_manifest.py --protocol-id protocol-v1-geography-414 --output /tmp/manifest.jsonl``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = PROJECT_ROOT / "datasets/processed/deepseek_protocol_v1"
ALLOWED_ROUTES = {"keep_control", "keep_cross_domain", "transform_candidate"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build_manifest(sources: list[dict], analyses: list[dict], protocol_ids: list[str]) -> list[dict]:
    source_by_id = {item["protocol_id"]: item for item in sources}
    analysis_by_id = {item["protocol_id"]: item for item in analyses}
    if len(protocol_ids) != len(set(protocol_ids)):
        raise ValueError("Protocol IDs must be unique")
    output = []
    for protocol_id in protocol_ids:
        if protocol_id not in source_by_id or protocol_id not in analysis_by_id:
            raise ValueError(f"Unknown protocol ID: {protocol_id}")
        trace = analysis_by_id[protocol_id]
        analysis = trace["analysis"]
        route = analysis["decision"]
        if not trace["schema_valid"] or not analysis["scientifically_valid"]:
            raise ValueError(f"{protocol_id} does not have a valid source analysis")
        if route not in ALLOWED_ROUTES:
            raise ValueError(f"{protocol_id} has non-production decision: {route}")
        if analysis["reference_answer_role"] == "unreliable_or_insufficient":
            raise ValueError(f"{protocol_id} has an unreliable source answer")
        output.append(
            {
                "qa_id": protocol_id.replace("protocol-v1", "qa-validation-v2"),
                "route": route,
                "source": source_by_id[protocol_id],
                "source_analysis": analysis,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, default=BASE / "protocol_sources_150.jsonl")
    parser.add_argument("--analysis", type=Path, default=BASE / "source_analysis_v2_1.jsonl")
    parser.add_argument("--protocol-id", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = build_manifest(read_jsonl(args.sources), read_jsonl(args.analysis), args.protocol_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote={len(records)} output={args.output}")


if __name__ == "__main__":
    main()
