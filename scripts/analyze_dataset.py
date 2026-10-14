"""Analyze the heterogeneous source science dataset.

Purpose:
    Stream JSONL files and inspect JSON arrays through a shared normalized loader,
    producing record counts, schemas, length statistics, coarse type/difficulty
    proxies, and bounded examples without changing source files.

Input:
    A directory laid out as DOMAIN/FILE.json or DOMAIN/FILE.jsonl.

Output:
    `source_analysis.json` and `source_analysis.md` in the requested output folder.

Example:
    python scripts/analyze_dataset.py \
        --input datasets/science_dataset/mixed_dataset \
        --output outputs/data_analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.analysis import analyze_dataset, render_markdown  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Source dataset root")
    parser.add_argument("--output", required=True, type=Path, help="Report directory")
    parser.add_argument(
        "--examples-per-file",
        type=int,
        default=1,
        help="Bounded examples retained per source file (default: 1)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = analyze_dataset(args.input, examples_per_file=args.examples_per_file)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / "source_analysis.json"
    markdown_path = args.output / "source_analysis.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

