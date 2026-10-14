#!/usr/bin/env python3
"""Build the fixed 25-source bridge-guided calibration manifest.

Input: a small manual ID/domain config and the existing unused 150-source pool.
Output: source records with audited primary domains; no model calls or matching scores.
Example: ``python scripts/build_bridge_calibration_manifest.py``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOMAINS = {"physics", "chemistry", "biology", "astronomy", "geography"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def question_type(source: dict[str, Any]) -> str:
    if source.get("choices"):
        return "mcq"
    text = source.get("question", "").lower()
    if any(word in text for word in ("calculate", "time difference", "magnetic moment")):
        return "numeric"
    return "qualitative"


def build_manifest(
    sources: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    source_by_id = {item["selector_source_id"]: item for item in sources}
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for selection in config["items"]:
        source_id = selection["selector_source_id"]
        domain = selection["primary_domain"]
        if source_id in seen:
            raise ValueError(f"Duplicate source ID: {source_id}")
        if source_id not in source_by_id:
            raise ValueError(f"Unknown source ID: {source_id}")
        if domain not in DOMAINS:
            raise ValueError(f"Invalid primary domain: {domain}")
        seen.add(source_id)
        source = source_by_id[source_id]
        output.append(
            {
                "bridge_source_id": source_id.replace("selector-v1-", "bridge-v1-", 1),
                "calibration_id": config["id"],
                "primary_domain": domain,
                "source": source,
                "source_analysis": {
                    "primary_domain": domain,
                    "required_domains": [
                        {
                            "domain": domain,
                            "evidence": "Audited semantic primary domain for bridge calibration.",
                            "necessity_reason": "The original question is treated as a single-domain source during bridge selection."
                        }
                    ],
                    "question_type": question_type(source),
                    "answerability": "objective" if source.get("choices") else "qualitative",
                    "reference_answer_role": "untrusted_reference"
                }
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    base = PROJECT_ROOT / "datasets/processed/bridge_guided_calibration_v1"
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/data/bridge_guided_calibration_v1.json",
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=PROJECT_ROOT
        / "datasets/processed/transformability_selector_calibration_v1/candidate_sources_150.jsonl",
    )
    parser.add_argument("--output", type=Path, default=base / "sources_25.jsonl")
    parser.add_argument("--summary", type=Path, default=base / "source_summary.json")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    records = build_manifest(read_jsonl(args.sources), config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "calibration_id": config["id"],
        "total": len(records),
        "primary_domain": dict(Counter(item["primary_domain"] for item in records)),
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
