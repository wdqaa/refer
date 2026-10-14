#!/usr/bin/env python3
"""Select the bounded Stage 01.5 QA calibration manifest after source analysis.

Input: the 50 fresh sources and matching Source Analyzer v2.1 traces.
Output: at most 20 objective non-transform items and 12 balanced non-mechanism transforms.
Example: ``python scripts/select_stage015_calibration_qa.py``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OBJECTIVE_TYPES = {"mcq", "numeric", "short_answer", "symbolic"}
NON_MECHANISM_RELATIONS = (
    "quantitative_modeling",
    "constraint_coupling",
    "measurement_observation",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def eligible(trace: dict[str, Any]) -> bool:
    analysis = trace.get("analysis") or {}
    return bool(
        trace.get("schema_valid")
        and analysis.get("scientifically_valid")
        and analysis.get("reference_answer_role") != "unreliable_or_insufficient"
    )


def manifest_record(source: dict[str, Any], analysis: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "qa_id": source["protocol_id"].replace("protocol-", "qa-", 1),
        "route": analysis["decision"],
        "source": source,
        "source_analysis": analysis,
        **extra,
    }


def select_manifest(
    sources: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    *,
    objective_limit: int = 20,
    per_relation_limit: int = 4,
) -> list[dict[str, Any]]:
    source_by_id = {item["protocol_id"]: item for item in sources}
    valid = [trace for trace in traces if eligible(trace) and trace["protocol_id"] in source_by_id]
    objective = []
    for trace in sorted(valid, key=lambda item: item["protocol_id"]):
        source = source_by_id[trace["protocol_id"]]
        analysis = trace["analysis"]
        if (
            source.get("sampling_stratum") in {"mcq", "likely_objective_open"}
            and analysis.get("decision") in {"keep_control", "keep_cross_domain"}
            and analysis.get("question_type") in OBJECTIVE_TYPES
            and analysis.get("answerability") == "objective"
        ):
            objective.append(manifest_record(source, analysis))
    # Retain astronomy emphasis without allowing it to consume the entire objective cap.
    astronomy = [item for item in objective if item["source"]["source_domain"] == "astronomy"]
    other = [item for item in objective if item["source"]["source_domain"] != "astronomy"]
    objective_selected = astronomy[:8] + other[: max(0, objective_limit - min(8, len(astronomy)))]
    if len(objective_selected) < objective_limit:
        already = {item["qa_id"] for item in objective_selected}
        objective_selected.extend(
            item for item in objective if item["qa_id"] not in already
        )
        objective_selected = objective_selected[:objective_limit]

    transforms = []
    for relation in NON_MECHANISM_RELATIONS:
        candidates = []
        for trace in sorted(valid, key=lambda item: item["protocol_id"]):
            source = source_by_id[trace["protocol_id"]]
            analysis = trace["analysis"]
            if (
                source.get("sampling_stratum") == "non_mechanism_affordance"
                and source.get("affordance_hint") == relation
                and analysis.get("decision") == "transform_candidate"
            ):
                candidates.append(
                    manifest_record(source, analysis, preferred_fusion_relation=relation)
                )
        transforms.extend(candidates[:per_relation_limit])
    return objective_selected + transforms


def main() -> None:
    parser = argparse.ArgumentParser()
    base = PROJECT_ROOT / "datasets/processed/stage015_calibration_v1"
    parser.add_argument("--sources", type=Path, default=base / "sources_50.jsonl")
    parser.add_argument("--analysis", type=Path, default=base / "source_analysis_v2_1.jsonl")
    parser.add_argument("--output", type=Path, default=base / "qa_manifest.jsonl")
    parser.add_argument("--summary", type=Path, default=base / "qa_selection_summary.json")
    parser.add_argument("--objective-limit", type=int, default=20)
    parser.add_argument("--per-relation-limit", type=int, default=4)
    args = parser.parse_args()

    output = select_manifest(
        read_jsonl(args.sources),
        read_jsonl(args.analysis),
        objective_limit=args.objective_limit,
        per_relation_limit=args.per_relation_limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in output:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    route_counts = Counter(item["route"] for item in output)
    relation_counts = Counter(
        item.get("preferred_fusion_relation") for item in output
        if item.get("preferred_fusion_relation")
    )
    summary = {
        "total": len(output),
        "route": dict(route_counts),
        "preferred_fusion_relation": dict(relation_counts),
        "planned_qa_api_calls_max": sum(
            4 if item["route"] == "transform_candidate" else 2 for item in output
        ),
        "calibration_gate": "at_least_2_non_mechanism_transforms_survive_semantic_review",
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
