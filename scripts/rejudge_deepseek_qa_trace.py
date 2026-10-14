#!/usr/bin/env python3
"""Rejudge frozen QA traces without regenerating questions or Gold answers.

Input: a QA trace JSONL, a versioned Judge prompt, and optional repeated QA IDs.
Output: resumable JSONL containing only the new Judge result and deterministic gate issues.
Example: ``python scripts/rejudge_deepseek_qa_trace.py --qa-id qa-pilot-v1-astronomy-122 --output /tmp/rejudge.jsonl``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_deepseek_qa_pilot import call_json, load_prompt, validate_judge
from src.generation.deepseek_api import DeepSeekJSONClient


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def judge_payload(trace: dict[str, Any]) -> dict[str, Any]:
    candidate = trace["transformer"]["result"]
    source = trace["source"]
    transformed = trace["route"] == "transform_candidate"
    return {
        "route": trace["route"],
        "source_question": source["question"],
        "untrusted_source_answer": source["answer"],
        "verified_source_analysis": trace["source_analysis"],
        "relation_plan": trace["relation_planner"]["result"],
        "transformed_candidate": candidate,
        "final_question": candidate["question"] if transformed else source["question"],
        "final_choices": candidate["choices"] if transformed else (source["choices"] or None),
        "blind_solver_gold": trace["blind_solver"]["result"],
    }


def rejudge(trace: dict[str, Any], prompt: dict[str, Any], model: str, max_tokens: int) -> dict[str, Any]:
    try:
        result, metadata = call_json(
            DeepSeekJSONClient(model=model), prompt, judge_payload(trace), max_tokens
        )
        gate_issues = validate_judge(trace["route"], result)
        return {
            "qa_id": trace["qa_id"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "prompt_id": prompt["id"],
            "prompt_version": prompt["version"],
            "result": result,
            "metadata": metadata,
            "gate_issues": gate_issues,
            "accepted": not gate_issues,
            "error": None,
        }
    except Exception as exc:
        return {
            "qa_id": trace["qa_id"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "prompt_id": prompt["id"],
            "prompt_version": prompt["version"],
            "result": None,
            "metadata": None,
            "gate_issues": [],
            "accepted": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    base = PROJECT_ROOT / "datasets/processed/deepseek_qa_pilot_v1"
    parser.add_argument("--trace", type=Path, default=base / "trace.jsonl")
    parser.add_argument("--prompt", type=Path, default=PROJECT_ROOT / "src/prompts/deepseek_qa_judge_v2.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qa-id", action="append")
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=6000)
    args = parser.parse_args()

    traces = read_jsonl(args.trace)
    if args.qa_id:
        requested = set(args.qa_id)
        traces = [trace for trace in traces if trace["qa_id"] in requested]
        missing = requested - {trace["qa_id"] for trace in traces}
        if missing:
            parser.error(f"Unknown QA IDs: {sorted(missing)}")
    traces = [trace for trace in traces if trace["blind_solver"]["result"] is not None]
    prompt = load_prompt(args.prompt)
    previous = read_jsonl(args.output) if args.output.exists() else []
    completed = {record["qa_id"]: record for record in previous if record.get("error") is None}
    pending = [trace for trace in traces if trace["qa_id"] not in completed]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(rejudge, trace, prompt, args.model, args.max_tokens): trace["qa_id"]
                for trace in pending
            }
            for index, future in enumerate(as_completed(futures), start=1):
                record = future.result()
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"[{index}/{len(pending)}] {record['qa_id']} accepted={record['accepted']}")


if __name__ == "__main__":
    main()
