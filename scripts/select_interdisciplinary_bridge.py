#!/usr/bin/env python3
"""Select one allowed interdisciplinary bridge or NONE for each source.

Input: fixed bridge-calibration sources, the small YAML library, and an environment key.
Output: one preserved Selector trace per source and a compact aggregate summary.
Example: ``python scripts/select_interdisciplinary_bridge.py --workers 4``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.deepseek_api import DeepSeekJSONClient


DOMAINS = {"physics", "chemistry", "biology", "astronomy", "geography", "mathematics"}
HARD_RISKS = {
    "false_boundary_split",
    "external_context_required",
    "source_answer_unreliable",
    "no_natural_extension",
}
SOFT_RISKS = {"too_broad", "underspecified_for_transformation"}
RISKS = HARD_RISKS | SOFT_RISKS
REQUIRED_FIELDS = {
    "decision",
    "target_domain",
    "bridge",
    "reason",
    "expected_new_dependency",
    "confidence",
    "risk_flags",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_library(path: Path) -> dict[str, dict[str, list[str]]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    bridges = raw.get("bridges") if isinstance(raw, dict) else None
    if not isinstance(bridges, dict):
        raise ValueError("Bridge library requires a bridges mapping")
    return bridges


def load_prompt(path: Path) -> dict[str, Any]:
    prompt = json.loads(path.read_text(encoding="utf-8"))
    base_name = prompt.get("base_prompt")
    if not base_name:
        return prompt
    base = json.loads(path.with_name(base_name).read_text(encoding="utf-8"))
    addendum = prompt.get("system_prompt_addendum")
    if not isinstance(addendum, str) or not addendum.strip():
        raise ValueError("An extending prompt requires system_prompt_addendum")
    return {
        **base,
        **prompt,
        "system_prompt": base["system_prompt"] + "\n\n" + addendum,
        "user_template": base["user_template"],
    }


def candidates_for(
    library: dict[str, dict[str, list[str]]], primary_domain: str
) -> dict[str, list[str]]:
    candidates = library.get(primary_domain)
    if not isinstance(candidates, dict) or not candidates:
        raise ValueError(f"No bridge candidates for primary domain {primary_domain}")
    return candidates


def validate(
    selection: dict[str, Any], bridge_candidates: dict[str, list[str]]
) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_FIELDS - selection.keys())
    unexpected = sorted(selection.keys() - REQUIRED_FIELDS)
    if missing:
        errors.append(f"missing fields: {', '.join(missing)}")
    if unexpected:
        errors.append(f"unexpected fields: {', '.join(unexpected)}")
    decision = selection.get("decision")
    if decision not in {"select", "none"}:
        errors.append("invalid decision")
    reason = selection.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        errors.append("reason is empty or not a string")
    confidence = selection.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
        errors.append("confidence must be a number from 0 to 1")
    risks = selection.get("risk_flags")
    if not isinstance(risks, list):
        errors.append("risk_flags is not a list")
        risks = []
    else:
        invalid = [risk for risk in risks if risk not in RISKS]
        if invalid:
            errors.append(f"invalid risk flags: {invalid}")
        if len(risks) != len(set(risks)):
            errors.append("risk_flags contains duplicates")
    target = selection.get("target_domain")
    bridge = selection.get("bridge")
    dependency = selection.get("expected_new_dependency")
    if decision == "select":
        if target not in bridge_candidates:
            errors.append("target_domain is not allowed for this primary domain")
        elif bridge not in bridge_candidates[target]:
            errors.append("bridge is not allowed for the selected target domain")
        if target not in DOMAINS:
            errors.append("invalid target_domain")
        if not isinstance(dependency, str) or not dependency.strip():
            errors.append("select requires expected_new_dependency")
        if set(risks) & HARD_RISKS:
            errors.append("select cannot contain a hard risk")
    if decision == "none":
        if target is not None or bridge is not None or dependency is not None:
            errors.append("none requires null target_domain, bridge, and dependency")
    return errors


def select_one(
    item: dict[str, Any],
    prompt: dict[str, Any],
    bridge_candidates: dict[str, list[str]],
    model: str,
    max_tokens: int,
) -> dict[str, Any]:
    source = item["source"]
    user_prompt = prompt["user_template"].format(
        primary_domain=item["primary_domain"],
        question=source["question"],
        choices_json=json.dumps(source.get("choices", {}), ensure_ascii=False),
        reference_answer=source["answer"],
        bridge_candidates_json=json.dumps(bridge_candidates, ensure_ascii=False, indent=2),
    )
    try:
        client = DeepSeekJSONClient(model=model)
        selection, metadata = client.generate_json(
            [
                {"role": "system", "content": prompt["system_prompt"]},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            thinking_enabled=False,
            retries=0,
        )
        errors = validate(selection, bridge_candidates)
        return {
            "bridge_source_id": item["bridge_source_id"],
            "primary_domain": item["primary_domain"],
            "selection": selection,
            "schema_valid": not errors,
            "validation_errors": errors,
            "api_metadata": metadata,
            "prompt_version": prompt["version"],
            "error": None,
        }
    except Exception as exc:
        return {
            "bridge_source_id": item["bridge_source_id"],
            "primary_domain": item["primary_domain"],
            "selection": None,
            "schema_valid": False,
            "validation_errors": [],
            "api_metadata": None,
            "prompt_version": prompt["version"],
            "error": f"{type(exc).__name__}: {exc}",
        }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [trace for trace in traces if trace.get("schema_valid")]
    selected = [trace["selection"] for trace in valid if trace["selection"]["decision"] == "select"]
    return {
        "total": len(traces),
        "schema_valid": len(valid),
        "failed_or_invalid": len(traces) - len(valid),
        "decision": dict(Counter(trace["selection"]["decision"] for trace in valid)),
        "selected_target_domain": dict(Counter(item["target_domain"] for item in selected)),
        "selected_bridge": dict(Counter(item["bridge"] for item in selected)),
        "risk_flags": dict(
            Counter(risk for trace in valid for risk in trace["selection"]["risk_flags"])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    base = PROJECT_ROOT / "datasets/processed/bridge_guided_calibration_v1"
    parser.add_argument("--input", type=Path, default=base / "sources_25.jsonl")
    parser.add_argument(
        "--library",
        type=Path,
        default=PROJECT_ROOT / "configs/data/interdisciplinary_bridge_library.yaml",
    )
    parser.add_argument(
        "--prompt", type=Path, default=PROJECT_ROOT / "src/prompts/bridge_selector_v1.json"
    )
    parser.add_argument("--output", type=Path, default=base / "bridge_selector_traces.jsonl")
    parser.add_argument("--summary", type=Path, default=base / "bridge_selector_summary.json")
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--bridge-source-id",
        action="append",
        help="Run only this bridge source ID; may repeat.",
    )
    args = parser.parse_args()

    items = read_jsonl(args.input)
    if args.bridge_source_id:
        requested = set(args.bridge_source_id)
        items = [item for item in items if item["bridge_source_id"] in requested]
        missing = requested - {item["bridge_source_id"] for item in items}
        if missing:
            parser.error(f"Unknown bridge source IDs: {sorted(missing)}")
    if args.limit is not None:
        items = items[: args.limit]
    prompt = load_prompt(args.prompt)
    library = load_library(args.library)
    previous = read_jsonl(args.output)
    latest = {trace["bridge_source_id"]: trace for trace in previous}
    pending = [item for item in items if item["bridge_source_id"] not in latest]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for trace in latest.values():
            handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
        handle.flush()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    select_one,
                    item,
                    prompt,
                    candidates_for(library, item["primary_domain"]),
                    args.model,
                    args.max_tokens,
                ): item["bridge_source_id"]
                for item in pending
            }
            for index, future in enumerate(as_completed(futures), start=1):
                trace = future.result()
                handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"[{index}/{len(pending)}] {trace['bridge_source_id']} "
                    f"schema_valid={trace['schema_valid']} error={trace['error'] is not None}",
                    flush=True,
                )
    selected_ids = {item["bridge_source_id"] for item in items}
    traces = [trace for trace in read_jsonl(args.output) if trace["bridge_source_id"] in selected_ids]
    summary = summarize(traces)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"summary": str(args.summary), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
