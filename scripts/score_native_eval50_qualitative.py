#!/usr/bin/env python3
"""Score frozen native-eval50 qualitative outputs with an audited core rubric.

Input: a versioned rubric JSON, the frozen curated dataset, three existing trace
JSONL files, and optional versioned adjudications. Output: criterion-level JSONL,
an aggregate JSON summary, and a compact Markdown report. This script never calls
a model and never changes the source data or trace files.

Example::

    python scripts/score_native_eval50_qualitative.py \
      --rubric configs/evaluation/native_eval50_qualitative_rubric_v2.json \
      --adjudications configs/evaluation/native_eval50_qualitative_adjudications_v2.json \
      --dataset datasets/processed/native_qa_mining_v1/pilot_500/eval_split_v1/eval_50_gold_curated.jsonl \
      --trace direct=outputs/baselines/qwen3_8b_native_eval50_v2/direct/traces.jsonl \
      --trace routed_multi_agent=outputs/baselines/qwen3_8b_native_eval50_v2/routed/traces.jsonl \
      --trace oracle_multi_agent=outputs/baselines/qwen3_8b_native_eval50_v2/oracle/traces.jsonl \
      --historical-scores outputs/baselines/qwen3_8b_native_eval50_v2/review_v1/final_scores.jsonl \
      --output-dir outputs/baselines/qwen3_8b_native_eval50_v2/qualitative_rubric_v2
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import statistics
from typing import Any


ARMS = ("direct", "routed_multi_agent", "oracle_multi_agent")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def prediction_text(prediction: dict[str, Any]) -> str:
    fields = [prediction.get("answer") or ""]
    fields.extend(prediction.get("reasoning") or [])
    if prediction.get("canonical_expression"):
        fields.append(str(prediction["canonical_expression"]))
    return "\n".join(str(value) for value in fields).lower()


def match_criterion(text: str, criterion: dict[str, Any]) -> dict[str, Any]:
    group_hits: list[list[str]] = []
    for alternatives in criterion["anchor_groups"]:
        hits = [pattern for pattern in alternatives if re.search(pattern, text, re.IGNORECASE)]
        group_hits.append(hits)
    matched_groups = sum(bool(hits) for hits in group_hits)
    required_groups = criterion.get("min_groups", len(group_hits))
    covered = matched_groups >= required_groups
    review_required = (0 < matched_groups < required_groups) or any(
        re.search(pattern, text, re.IGNORECASE)
        for pattern in criterion.get("review_patterns", [])
    )
    return {
        "auto_covered": covered,
        "covered": covered,
        "matched_groups": matched_groups,
        "required_groups": required_groups,
        "group_hits": group_hits,
        "review_required": review_required,
        "adjudication": None,
    }


def validate_rubric(rubric: dict[str, Any], qualitative_ids: set[str]) -> None:
    items = rubric.get("items")
    if not isinstance(items, list):
        raise ValueError("rubric.items must be a list")
    ids = [item.get("item_id") for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("rubric contains duplicate item_id values")
    if set(ids) != qualitative_ids:
        missing = sorted(qualitative_ids - set(ids))
        extra = sorted(set(ids) - qualitative_ids)
        raise ValueError(f"rubric IDs do not match qualitative set; missing={missing}, extra={extra}")
    for item in items:
        criteria = item.get("criteria")
        if not isinstance(criteria, list) or not 3 <= len(criteria) <= 5:
            raise ValueError(f"{item['item_id']} must have 3-5 criteria")
        criterion_ids = [criterion.get("criterion_id") for criterion in criteria]
        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError(f"duplicate criterion IDs for {item['item_id']}")
        for criterion in criteria:
            groups = criterion.get("anchor_groups")
            if not isinstance(groups, list) or not groups or not all(
                isinstance(group, list) and group and all(isinstance(pattern, str) for pattern in group)
                for group in groups
            ):
                raise ValueError(
                    f"invalid anchor_groups for {item['item_id']}:{criterion.get('criterion_id')}"
                )
            minimum = criterion.get("min_groups", len(groups))
            if not isinstance(minimum, int) or not 1 <= minimum <= len(groups):
                raise ValueError(
                    f"invalid min_groups for {item['item_id']}:{criterion.get('criterion_id')}"
                )


def parse_trace_specs(specs: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"trace must be ARM=PATH, got {spec!r}")
        arm, path = spec.split("=", 1)
        if arm not in ARMS:
            raise ValueError(f"unsupported arm {arm!r}")
        if arm in result:
            raise ValueError(f"duplicate trace arm {arm!r}")
        result[arm] = Path(path)
    if set(result) != set(ARMS):
        raise ValueError(f"expected traces for {ARMS}, got {sorted(result)}")
    return result


def load_adjudications(path: Path | None) -> tuple[dict[tuple[str, str, str], dict[str, Any]], list[dict[str, Any]]]:
    if path is None:
        return {}, []
    data = json.loads(path.read_text(encoding="utf-8"))
    overrides: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in data.get("coverage_overrides", []):
        key = (row["item_id"], row["arm"], row["criterion_id"])
        if key in overrides:
            raise ValueError(f"duplicate coverage override {key}")
        overrides[key] = row
    errors = data.get("scientific_errors", [])
    seen_errors: set[tuple[str, str, str]] = set()
    for row in errors:
        key = (row["item_id"], row["arm"], row["error_id"])
        if key in seen_errors:
            raise ValueError(f"duplicate scientific error {key}")
        seen_errors.add(key)
    return overrides, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rubric", type=Path, required=True)
    parser.add_argument("--adjudications", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--trace", action="append", required=True)
    parser.add_argument("--historical-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dataset = read_jsonl(args.dataset)
    qualitative = {
        row["native_qa_id"]: row for row in dataset if row["question_type"] == "qualitative"
    }
    rubric = json.loads(args.rubric.read_text(encoding="utf-8"))
    validate_rubric(rubric, set(qualitative))
    rubric_by_id = {row["item_id"]: row for row in rubric["items"]}
    overrides, scientific_errors = load_adjudications(args.adjudications)

    trace_specs = parse_trace_specs(args.trace)
    traces: dict[str, dict[str, dict[str, Any]]] = {}
    for arm, path in trace_specs.items():
        rows = {row["item_id"]: row for row in read_jsonl(path)}
        missing = sorted(set(qualitative) - set(rows))
        if missing:
            raise ValueError(f"{arm} trace is missing qualitative IDs: {missing}")
        traces[arm] = rows

    criterion_rows: list[dict[str, Any]] = []
    item_coverage: dict[str, dict[str, float]] = defaultdict(dict)
    unresolved_review_flags: list[dict[str, str]] = []
    for item_id in sorted(qualitative):
        criteria = rubric_by_id[item_id]["criteria"]
        for arm in ARMS:
            prediction = traces[arm][item_id]["final_prediction"]
            text = prediction_text(prediction)
            covered_count = 0
            for criterion in criteria:
                match = match_criterion(text, criterion)
                key = (item_id, arm, criterion["criterion_id"])
                if key in overrides:
                    override = overrides[key]
                    match["covered"] = bool(override["covered"])
                    match["adjudication"] = {
                        "reason": override["reason"],
                        "evidence": override.get("evidence"),
                    }
                elif match["review_required"]:
                    unresolved_review_flags.append(
                        {"item_id": item_id, "arm": arm, "criterion_id": criterion["criterion_id"]}
                    )
                covered_count += int(match["covered"])
                criterion_rows.append(
                    {
                        "item_id": item_id,
                        "arm": arm,
                        "criterion_id": criterion["criterion_id"],
                        "criterion": criterion["criterion"],
                        **match,
                    }
                )
            item_coverage[arm][item_id] = covered_count / len(criteria)

    historical = read_jsonl(args.historical_scores)
    historical_strict = {
        arm: {
            "correct": sum(
                bool(row["correct"])
                for row in historical
                if row["arm"] == arm and row["question_type"] == "qualitative"
            ),
            "total": sum(
                1 for row in historical if row["arm"] == arm and row["question_type"] == "qualitative"
            ),
        }
        for arm in ARMS
    }

    errors_by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for error in scientific_errors:
        if error["item_id"] not in qualitative:
            raise ValueError(f"scientific error has non-qualitative item {error['item_id']}")
        if error["arm"] not in ARMS:
            raise ValueError(f"scientific error has unsupported arm {error['arm']}")
        errors_by_arm[error["arm"]].append(error)

    arms_summary: dict[str, Any] = {}
    for arm in ARMS:
        values = list(item_coverage[arm].values())
        arm_criterion_rows = [row for row in criterion_rows if row["arm"] == arm]
        erroneous_items = {row["item_id"] for row in errors_by_arm[arm]}
        arms_summary[arm] = {
            "items": len(values),
            "mean_item_coverage": statistics.mean(values),
            "covered_criteria": sum(bool(row["covered"]) for row in arm_criterion_rows),
            "total_criteria": len(arm_criterion_rows),
            "micro_criterion_coverage": (
                sum(bool(row["covered"]) for row in arm_criterion_rows) / len(arm_criterion_rows)
            ),
            "responses_with_scientific_error": len(erroneous_items),
            "scientific_error_claims": len(errors_by_arm[arm]),
            "historical_strict_complete": historical_strict[arm],
        }

    comparisons: dict[str, Any] = {}
    for target in ("routed_multi_agent", "oracle_multi_agent"):
        deltas = {
            item_id: item_coverage[target][item_id] - item_coverage["direct"][item_id]
            for item_id in sorted(qualitative)
        }
        comparisons[f"{target}_minus_direct"] = {
            "mean_item_coverage_delta": statistics.mean(deltas.values()),
            "improved_items": [
                {"item_id": item_id, "delta": delta} for item_id, delta in deltas.items() if delta > 0
            ],
            "degraded_items": [
                {"item_id": item_id, "delta": delta} for item_id, delta in deltas.items() if delta < 0
            ],
            "tied_items": sum(delta == 0 for delta in deltas.values()),
        }

    summary = {
        "schema_version": "native_eval50_qualitative_coverage.v2",
        "rubric_version": rubric["version"],
        "dataset": str(args.dataset),
        "source_traces": {arm: str(path) for arm, path in trace_specs.items()},
        "qualitative_items": len(qualitative),
        "criteria_per_item": {
            "minimum": min(len(item["criteria"]) for item in rubric["items"]),
            "maximum": max(len(item["criteria"]) for item in rubric["items"]),
            "total": sum(len(item["criteria"]) for item in rubric["items"]),
        },
        "unresolved_review_flags": unresolved_review_flags,
        "arms": arms_summary,
        "paired_comparisons": comparisons,
        "scientific_errors": scientific_errors,
        "interpretation_boundary": (
            "Coverage is evaluator-side partial credit under the audited fixed rubric; it does not "
            "replace the preserved historical strict complete-rubric result."
        ),
    }

    item_rows = []
    error_count_by_item_arm = defaultdict(int)
    for row in scientific_errors:
        error_count_by_item_arm[(row["item_id"], row["arm"])] += 1
    for item_id in sorted(qualitative):
        item_rows.append(
            {
                "item_id": item_id,
                "question": qualitative[item_id]["question"],
                "criterion_count": len(rubric_by_id[item_id]["criteria"]),
                "coverage": {arm: item_coverage[arm][item_id] for arm in ARMS},
                "delta_vs_direct": {
                    arm: item_coverage[arm][item_id] - item_coverage["direct"][item_id]
                    for arm in ("routed_multi_agent", "oracle_multi_agent")
                },
                "scientific_error_claims": {
                    arm: error_count_by_item_arm[(item_id, arm)] for arm in ARMS
                },
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "criterion_scores.jsonl").open("w", encoding="utf-8") as handle:
        for row in criterion_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.output_dir / "item_scores.jsonl").open("w", encoding="utf-8") as handle:
        for row in item_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    report = [
        "# Native eval50 qualitative coverage v2",
        "",
        f"Rubric: `{rubric['version']}`; items: {len(qualitative)}; criteria: {summary['criteria_per_item']['total']}.",
        "",
        "| Arm | Mean item coverage | Micro criterion coverage | Error responses | Error claims | Historical strict |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = arms_summary[arm]
        strict = row["historical_strict_complete"]
        report.append(
            f"| {arm} | {row['mean_item_coverage']:.2%} | {row['micro_criterion_coverage']:.2%} | "
            f"{row['responses_with_scientific_error']} | {row['scientific_error_claims']} | "
            f"{strict['correct']}/{strict['total']} |"
        )
    report.extend(["", "## Paired coverage changes", ""])
    for name, comparison in comparisons.items():
        report.append(
            f"- `{name}`: mean delta {comparison['mean_item_coverage_delta']:+.2%}; "
            f"improved {len(comparison['improved_items'])}; degraded {len(comparison['degraded_items'])}; "
            f"tied {comparison['tied_items']}."
        )
    report.extend(
        [
            "",
            f"Unresolved review flags: {len(unresolved_review_flags)}.",
            "",
            summary["interpretation_boundary"],
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
