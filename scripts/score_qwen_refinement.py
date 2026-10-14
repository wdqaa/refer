#!/usr/bin/env python3
"""Score Direct, Expert-Critique, and Self-Refine on audited native eval48.

Input: frozen baseline-view data, the fixed qualitative rubric, cached/new trace
files, the two-item exclusion audit, and versioned human adjudications. Output:
criterion- and item-level JSONL plus a compact aggregate JSON report. No model is
called and no source data or inference trace is changed.

Example::

    python scripts/score_qwen_refinement.py \
      --dataset datasets/processed/native_qa_mining_v1/pilot_500/eval_split_v1/eval_50_baseline_view.jsonl \
      --exclusion-audit outputs/baselines/qwen3_8b_native_eval50_v2/review_v1/posthoc_item_audit.jsonl \
      --rubric configs/evaluation/native_eval50_qualitative_rubric_v2.json \
      --base-adjudications configs/evaluation/native_eval50_qualitative_adjudications_v2.json \
      --refinement-adjudications configs/evaluation/native_eval48_refinement_adjudications_v1.json \
      --trace direct=outputs/baselines/qwen3_8b_native_eval50_v2/direct/traces.jsonl \
      --trace expert_critique=outputs/baselines/qwen3_8b_native_eval48_refinement_v1/expert_critique/traces.jsonl \
      --trace self_refine=outputs/baselines/qwen3_8b_native_eval48_refinement_v1/self_refine/traces.jsonl \
      --output-dir outputs/baselines/qwen3_8b_native_eval48_refinement_v1/evaluation
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.score_native_eval50_qualitative import (
    load_adjudications,
    match_criterion,
    prediction_text,
    validate_rubric,
)
from src.evaluation.answer_scoring import score_prediction


ARMS = ("direct", "expert_critique", "self_refine")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def parse_traces(specs: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for spec in specs:
        arm, separator, raw_path = spec.partition("=")
        if not separator or arm not in ARMS or arm in result:
            raise ValueError(f"invalid or duplicate trace specification: {spec}")
        result[arm] = Path(raw_path)
    if set(result) != set(ARMS):
        raise ValueError(f"expected exactly these arms: {ARMS}")
    return result


def efficiency(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    calls = [call for row in rows.values() for call in row.get("calls", [])]
    item_latency = [sum(call.get("latency_seconds") or 0 for call in row.get("calls", [])) for row in rows.values()]
    return {
        "items": len(rows),
        "calls": len(calls),
        "parse_failed_calls": sum(call.get("parsed") is None for call in calls),
        "input_tokens": sum(call.get("input_tokens") or 0 for call in calls),
        "output_tokens": sum(call.get("output_tokens") or 0 for call in calls),
        "summed_model_latency_seconds": sum(item_latency),
        "mean_calls_per_item": len(calls) / len(rows),
        "mean_input_tokens_per_item": sum(call.get("input_tokens") or 0 for call in calls) / len(rows),
        "mean_output_tokens_per_item": sum(call.get("output_tokens") or 0 for call in calls) / len(rows),
        "mean_model_latency_seconds_per_item": statistics.mean(item_latency),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--exclusion-audit", type=Path, required=True)
    parser.add_argument("--rubric", type=Path, required=True)
    parser.add_argument("--base-adjudications", type=Path, required=True)
    parser.add_argument("--refinement-adjudications", type=Path, required=True)
    parser.add_argument("--trace", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    excluded = {
        row["item_id"]
        for row in read_jsonl(args.exclusion_audit)
        if row["disposition"] == "exclude_from_audited_summary"
    }
    records = {row["id"]: row for row in read_jsonl(args.dataset) if row["id"] not in excluded}
    if len(records) != 48:
        raise ValueError(f"expected audited eval48, found {len(records)} items")
    qualitative = {item_id: row for item_id, row in records.items() if row["question_type"] == "qualitative"}
    mcq = {item_id: row for item_id, row in records.items() if row["question_type"] == "mcq"}
    if (len(qualitative), len(mcq)) != (32, 16):
        raise ValueError("expected 32 qualitative and 16 MCQ items")

    rubric = json.loads(args.rubric.read_text(encoding="utf-8"))
    validate_rubric(rubric, set(qualitative))
    rubric_by_id = {row["item_id"]: row for row in rubric["items"]}

    direct_overrides, direct_errors = load_adjudications(args.base_adjudications)
    refinement_overrides, refinement_errors = load_adjudications(args.refinement_adjudications)
    overrides = {
        key: value
        for key, value in direct_overrides.items()
        if key[1] == "direct"
    }
    for key, value in refinement_overrides.items():
        if key[1] not in {"expert_critique", "self_refine"}:
            raise ValueError(f"invalid refinement adjudication arm: {key[1]}")
        if key in overrides:
            raise ValueError(f"duplicate adjudication: {key}")
        overrides[key] = value
    scientific_errors = [row for row in direct_errors if row["arm"] == "direct"] + refinement_errors

    trace_paths = parse_traces(args.trace)
    traces: dict[str, dict[str, dict[str, Any]]] = {}
    for arm, path in trace_paths.items():
        rows = {row["item_id"]: row for row in read_jsonl(path)}
        missing = sorted(set(records) - set(rows))
        if missing:
            raise ValueError(f"{arm} is missing {len(missing)} items: {missing}")
        traces[arm] = {item_id: rows[item_id] for item_id in records}

    criterion_rows: list[dict[str, Any]] = []
    coverage: dict[str, dict[str, float]] = defaultdict(dict)
    unresolved: list[dict[str, str]] = []
    for item_id in sorted(qualitative):
        criteria = rubric_by_id[item_id]["criteria"]
        for arm in ARMS:
            text = prediction_text(traces[arm][item_id]["final_prediction"])
            covered = 0
            for criterion in criteria:
                result = match_criterion(text, criterion)
                key = (item_id, arm, criterion["criterion_id"])
                if key in overrides:
                    override = overrides[key]
                    result["covered"] = bool(override["covered"])
                    result["adjudication"] = {
                        "reason": override["reason"],
                        "evidence": override.get("evidence"),
                    }
                elif result["review_required"]:
                    unresolved.append({"item_id": item_id, "arm": arm, "criterion_id": criterion["criterion_id"]})
                covered += int(result["covered"])
                criterion_rows.append({
                    "item_id": item_id,
                    "arm": arm,
                    "criterion_id": criterion["criterion_id"],
                    "criterion": criterion["criterion"],
                    **result,
                })
            coverage[arm][item_id] = covered / len(criteria)

    errors_by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for error in scientific_errors:
        if error["arm"] not in ARMS or error["item_id"] not in qualitative:
            raise ValueError(f"invalid scientific error record: {error}")
        errors_by_arm[error["arm"]].append(error)

    mcq_scores: dict[str, dict[str, bool]] = defaultdict(dict)
    for arm in ARMS:
        for item_id, record in mcq.items():
            result = score_prediction(record, traces[arm][item_id]["final_prediction"])
            mcq_scores[arm][item_id] = bool(result["auto_correct"])

    arms_summary: dict[str, Any] = {}
    for arm in ARMS:
        values = list(coverage[arm].values())
        criterion_arm = [row for row in criterion_rows if row["arm"] == arm]
        arms_summary[arm] = {
            "mcq_correct": sum(mcq_scores[arm].values()),
            "mcq_total": len(mcq),
            "mcq_accuracy": sum(mcq_scores[arm].values()) / len(mcq),
            "qualitative_mean_item_coverage": statistics.mean(values),
            "qualitative_covered_criteria": sum(bool(row["covered"]) for row in criterion_arm),
            "qualitative_total_criteria": len(criterion_arm),
            "responses_with_scientific_error": len({row["item_id"] for row in errors_by_arm[arm]}),
            "scientific_error_claims": len(errors_by_arm[arm]),
            "efficiency": efficiency(traces[arm]),
        }

    comparisons: dict[str, Any] = {}
    direct_error_items = {row["item_id"] for row in errors_by_arm["direct"]}
    for arm in ("expert_critique", "self_refine"):
        qual_deltas = {item_id: coverage[arm][item_id] - coverage["direct"][item_id] for item_id in qualitative}
        rescued = [item_id for item_id in mcq if mcq_scores[arm][item_id] and not mcq_scores["direct"][item_id]]
        damaged = [item_id for item_id in mcq if not mcq_scores[arm][item_id] and mcq_scores["direct"][item_id]]
        error_items = {row["item_id"] for row in errors_by_arm[arm]}
        comparisons[f"{arm}_minus_direct"] = {
            "mcq_rescued": sorted(rescued),
            "mcq_damaged": sorted(damaged),
            "qualitative_mean_coverage_delta": statistics.mean(qual_deltas.values()),
            "qualitative_improved": sorted(
                ({"item_id": item_id, "delta": delta} for item_id, delta in qual_deltas.items() if delta > 0),
                key=lambda row: row["item_id"],
            ),
            "qualitative_degraded": sorted(
                ({"item_id": item_id, "delta": delta} for item_id, delta in qual_deltas.items() if delta < 0),
                key=lambda row: row["item_id"],
            ),
            "qualitative_tied": sum(delta == 0 for delta in qual_deltas.values()),
            "direct_error_items_corrected": sorted(direct_error_items - error_items),
            "new_error_items_introduced": sorted(error_items - direct_error_items),
        }

    item_rows = []
    for item_id, record in sorted(records.items()):
        row: dict[str, Any] = {"item_id": item_id, "question_type": record["question_type"]}
        if item_id in mcq:
            row["correct"] = {arm: mcq_scores[arm][item_id] for arm in ARMS}
        else:
            row["coverage"] = {arm: coverage[arm][item_id] for arm in ARMS}
            row["delta_vs_direct"] = {
                arm: coverage[arm][item_id] - coverage["direct"][item_id]
                for arm in ("expert_critique", "self_refine")
            }
        item_rows.append(row)

    summary = {
        "schema_version": "native_eval48_refinement_evaluation.v1",
        "items": len(records),
        "mcq_items": len(mcq),
        "qualitative_items": len(qualitative),
        "qualitative_criteria": sum(len(item["criteria"]) for item in rubric["items"]),
        "rubric_version": rubric["version"],
        "unresolved_review_flags": unresolved,
        "arms": arms_summary,
        "paired_comparisons": comparisons,
        "scientific_errors": scientific_errors,
        "interpretation_boundary": "The same audited rubric is used for all arms. This eval48 was used to design the fixed refinement prompts, so results are diagnostic rather than an independent generalization estimate.",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("criterion_scores.jsonl", criterion_rows), ("item_scores.jsonl", item_rows)):
        with (args.output_dir / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# Native eval48 Expert-Critique vs Self-Refine",
        "",
        "| Arm | MCQ | Qualitative mean coverage | Error responses | Error claims | Calls | Output tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = arms_summary[arm]
        eff = row["efficiency"]
        report.append(
            f"| {arm} | {row['mcq_correct']}/{row['mcq_total']} | "
            f"{row['qualitative_mean_item_coverage']:.2%} | "
            f"{row['responses_with_scientific_error']} | {row['scientific_error_claims']} | "
            f"{eff['calls']} | {eff['output_tokens']} |"
        )
    report.extend(["", "## Paired changes", ""])
    for name, row in comparisons.items():
        report.append(
            f"- `{name}`: MCQ rescued {len(row['mcq_rescued'])}, damaged {len(row['mcq_damaged'])}; "
            f"qualitative coverage {row['qualitative_mean_coverage_delta']:+.2%}, "
            f"improved {len(row['qualitative_improved'])}, degraded {len(row['qualitative_degraded'])}, "
            f"tied {row['qualitative_tied']}."
        )
    report.extend(["", f"Unresolved review flags: {len(unresolved)}.", "", summary["interpretation_boundary"]])
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
