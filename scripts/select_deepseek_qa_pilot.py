#!/usr/bin/env python3
"""Build the curated 30-item QA-production pilot manifest.

Input: the 150-source protocol batch and its complete Source Analyzer v2.1 output.
Output: a self-contained 30-record JSONL manifest for DeepSeek QA production.
Example: ``python scripts/select_deepseek_qa_pilot.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = PROJECT_ROOT / "datasets/processed/deepseek_protocol_v1"

SELECTION = {
    "keep_control": [
        "protocol-v1-astronomy-456",
        "protocol-v1-astronomy-3654",
        "protocol-v1-biology-4324",
        "protocol-v1-biology-11457",
        "protocol-v1-chemistry-9315",
        "protocol-v1-chemistry-2522",
        "protocol-v1-geography-21470",
        "protocol-v1-geography-481",
        "protocol-v1-physics-5114",
        "protocol-v1-physics-9885",
    ],
    "keep_cross_domain": [
        "protocol-v1-physics-4416",
        "protocol-v1-chemistry-19224",
        "protocol-v1-geography-25048",
        "protocol-v1-chemistry-15521",
        "protocol-v1-geography-7743",
        "protocol-v1-geography-1614",
        "protocol-v1-biology-546",
        "protocol-v1-physics-13303",
    ],
    "transform_candidate": [
        "protocol-v1-astronomy-122",
        "protocol-v1-astronomy-2233",
        "protocol-v1-biology-22378",
        "protocol-v1-biology-11832",
        "protocol-v1-chemistry-13483",
        "protocol-v1-chemistry-11280",
        "protocol-v1-geography-20701",
        "protocol-v1-geography-17103",
        "protocol-v1-geography-23365",
        "protocol-v1-chemistry-15183",
        "protocol-v1-physics-13938",
        "protocol-v1-physics-1935",
    ],
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, default=BASE / "protocol_sources_150.jsonl")
    parser.add_argument("--analysis", type=Path, default=BASE / "source_analysis_v2_1.jsonl")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "datasets/processed/deepseek_qa_pilot_v1/manifest_30.jsonl",
    )
    args = parser.parse_args()

    sources = {item["protocol_id"]: item for item in read_jsonl(args.sources)}
    analyses = {item["protocol_id"]: item for item in read_jsonl(args.analysis)}
    output = []
    for expected_decision, protocol_ids in SELECTION.items():
        for protocol_id in protocol_ids:
            source = sources[protocol_id]
            trace = analyses[protocol_id]
            analysis = trace["analysis"]
            if not trace["schema_valid"] or analysis["decision"] != expected_decision:
                raise ValueError(f"{protocol_id} no longer matches {expected_decision}")
            if not analysis["scientifically_valid"]:
                raise ValueError(f"{protocol_id} is not scientifically valid")
            if analysis["reference_answer_role"] == "unreliable_or_insufficient":
                raise ValueError(f"{protocol_id} has an unreliable source answer")
            output.append(
                {
                    "qa_id": protocol_id.replace("protocol-v1", "qa-pilot-v1"),
                    "route": expected_decision,
                    "source": source,
                    "source_analysis": analysis,
                }
            )

    if len(output) != 30 or len({item["qa_id"] for item in output}) != 30:
        raise RuntimeError("QA pilot must contain 30 unique records")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in output:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"wrote={len(output)} output={args.output}")


if __name__ == "__main__":
    main()

