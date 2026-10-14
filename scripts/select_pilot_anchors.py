"""Create the deterministic 20-record pilot anchor manifest.

Purpose:
    Select moderate-length, unique records from non-training evaluation files across
    the five implemented expert domains. Assign keep/transform/synthetic modes using
    the documented pilot plan without exposing target domain combinations.

Input:
    The immutable mixed-dataset root.

Output:
    A JSONL anchor manifest used by the generation pipeline.

Example:
    python scripts/select_pilot_anchors.py \
        --input datasets/science_dataset/mixed_dataset \
        --output datasets/processed/pilot_anchors_20.jsonl \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.selection import DEFAULT_PLAN, select_pilot_anchors  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Source dataset root")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        anchors = select_pilot_anchors(args.input, seed=args.seed)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for anchor in anchors:
            handle.write(json.dumps(anchor, ensure_ascii=False) + "\n")
    mode_counts = {
        mode: sum(1 for anchor in anchors if anchor["requested_mode"] == mode)
        for mode in ("real_anchored", "synthetic")
    }
    print(f"Wrote {len(anchors)} anchors to {args.output}")
    print(f"Mode counts: {mode_counts}; plan: {DEFAULT_PLAN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
