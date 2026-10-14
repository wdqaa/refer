#!/usr/bin/env python3
"""Export verified QA after preserving and applying separate Judge retries.

Input: an append-only production trace and an append-only rejudge trace.
Output: machine-accepted verified QA JSONL using the latest successful Judge result.
Example: ``python scripts/export_rejudged_qa.py --trace trace.jsonl --rejudge retry.jsonl --output verified.jsonl``.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from generate_deepseek_qa_pilot import verified_record


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def resolved_traces(traces: list[dict], retries: list[dict]) -> list[dict]:
    latest = {item["qa_id"]: item for item in traces}
    retry_by_id = {item["qa_id"]: item for item in retries}
    output = []
    for qa_id, trace in latest.items():
        if trace["gate"]["status"] == "accepted":
            output.append(trace)
            continue
        retry = retry_by_id.get(qa_id)
        if not retry or retry.get("error") is not None or not retry.get("accepted"):
            continue
        trace = deepcopy(trace)
        trace["judge"] = {"result": retry["result"], "metadata": retry["metadata"]}
        trace["gate"]["judge_issues"] = []
        trace["gate"]["status"] = "accepted"
        trace["prompt_versions"]["judge"] = retry["prompt_version"]
        trace["error"] = None
        output.append(trace)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--rejudge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    resolved = resolved_traces(read_jsonl(args.trace), read_jsonl(args.rejudge))
    records = [verified_record(trace) for trace in resolved]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"accepted={len(records)} output={args.output}")


if __name__ == "__main__":
    main()
