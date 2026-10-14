#!/usr/bin/env python3
"""Create a unique-ID evaluation view without changing append-only raw traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-items", type=int, required=True)
    parser.add_argument(
        "--prefer-single",
        action="store_true",
        help="For duplicate IDs, require and select the unique batch-size-1 trace.",
    )
    args = parser.parse_args()

    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for row in read_jsonl(args.input):
        item_id = row["item_id"]
        if item_id not in grouped:
            grouped[item_id] = []
            order.append(item_id)
        grouped[item_id].append(row)
    if len(grouped) != args.expected_items:
        raise ValueError(
            f"expected {args.expected_items} unique IDs, found {len(grouped)}"
        )

    selected: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for item_id in order:
        candidates = grouped[item_id]
        if len(candidates) == 1:
            chosen = candidates[0]
            reason = "only_trace"
        elif args.prefer_single:
            singles = [
                row for row in candidates if row.get("config", {}).get("batch_size", 1) == 1
            ]
            if len(singles) != 1:
                raise ValueError(
                    f"{item_id} has {len(candidates)} traces but {len(singles)} single traces"
                )
            chosen = singles[0]
            reason = "unique_single_item_trace"
        else:
            raise ValueError(f"duplicate trace requires an explicit selection policy: {item_id}")
        selected.append(chosen)
        decisions.append(
            {
                "item_id": item_id,
                "candidate_count": len(candidates),
                "selection_reason": reason,
                "selected_created_at": chosen.get("created_at"),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit_path = args.output.with_name(args.output.stem + "_selection.jsonl")
    with audit_path.open("w", encoding="utf-8") as handle:
        for row in decisions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "raw_rows": sum(len(rows) for rows in grouped.values()),
                "unique_items": len(selected),
                "duplicates": sum(len(rows) - 1 for rows in grouped.values()),
                "output": str(args.output),
                "selection_audit": str(audit_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
