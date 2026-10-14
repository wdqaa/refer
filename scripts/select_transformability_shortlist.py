#!/usr/bin/env python3
"""Build the strict human-review shortlist from Transformability Selector traces.

Input: candidate sources plus matching Selector traces.
Output: up to 30 high/self-contained/grounded/no-risk records for human review.
Example: ``python scripts/select_transformability_shortlist.py --limit 30``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def eligible(trace: dict[str, Any]) -> bool:
    analysis = trace.get("analysis") or {}
    return bool(
        trace.get("schema_valid")
        and trace.get("error") is None
        and analysis.get("self_contained") is True
        and analysis.get("answer_grounded") is True
        and analysis.get("transformability") == "high"
        and analysis.get("risk_flags") == []
        and analysis.get("candidate_relations")
        and analysis.get("possible_new_domains")
    )


def select_shortlist(
    sources: list[dict[str, Any]], traces: list[dict[str, Any]], *, limit: int = 30
) -> list[dict[str, Any]]:
    """Preserve deterministic candidate-pool order and never fill from lower tiers."""

    trace_by_id = {trace["selector_source_id"]: trace for trace in traces}
    selected: list[dict[str, Any]] = []
    for source in sources:
        trace = trace_by_id.get(source["selector_source_id"])
        if trace is None or not eligible(trace):
            continue
        selected.append(
            {
                "selector_source_id": source["selector_source_id"],
                "source": source,
                "selector": trace["analysis"],
                "selector_prompt_version": trace["prompt_version"],
            }
        )
        if len(selected) == limit:
            break
    return selected


def summarize(
    sources: list[dict[str, Any]], traces: list[dict[str, Any]], shortlist: list[dict[str, Any]]
) -> dict[str, Any]:
    valid = [trace for trace in traces if trace.get("schema_valid")]
    analyses = [trace["analysis"] for trace in valid]
    strict = [trace for trace in traces if eligible(trace)]
    return {
        "initial_candidate_sources": len(sources),
        "selector_traces": len(traces),
        "schema_valid": len(valid),
        "failed_or_invalid": len(traces) - len(valid),
        "transformability": dict(Counter(a["transformability"] for a in analyses)),
        "candidate_relations": dict(
            Counter(relation for a in analyses for relation in a["candidate_relations"])
        ),
        "risk_flags": dict(Counter(flag for a in analyses for flag in a["risk_flags"])),
        "strict_high_eligible": len(strict),
        "human_shortlist": len(shortlist),
        "shortlist_relations": dict(
            Counter(
                relation
                for item in shortlist
                for relation in item["selector"]["candidate_relations"]
            )
        ),
        "shortlist_source_domains": dict(
            Counter(item["source"]["source_domain"] for item in shortlist)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    base = PROJECT_ROOT / "datasets/processed/transformability_selector_calibration_v1"
    parser.add_argument("--sources", type=Path, default=base / "candidate_sources_150.jsonl")
    parser.add_argument("--traces", type=Path, default=base / "selector_traces.jsonl")
    parser.add_argument("--output", type=Path, default=base / "human_shortlist.jsonl")
    parser.add_argument("--summary", type=Path, default=base / "shortlist_summary.json")
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")

    sources = read_jsonl(args.sources)
    traces = read_jsonl(args.traces)
    shortlist = select_shortlist(sources, traces, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in shortlist:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = summarize(sources, traces, shortlist)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
