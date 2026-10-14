"""Generate and verify a small question-pool prototype with local Qwen3-8B.

Purpose:
    Run the versioned Question Generator and Question Verifier over a deterministic
    anchor manifest. Save canonical candidates, failures, and raw model traces in
    append-only JSONL files so interrupted runs can resume.

Input:
    A pilot anchor JSONL manifest and a local Hugging Face model directory.

Output:
    `candidates.jsonl`, `failures.jsonl`, and `raw_traces.jsonl` under output-dir.

Example:
    CUDA_VISIBLE_DEVICES=0 python scripts/generate_question_pool.py \
        --anchors datasets/processed/pilot_anchors_20.jsonl \
        --output-dir outputs/question_pool/pilot_v1 \
        --model-path /home/dqwang/Model/Qwen3-8B \
        --device cuda:0 --limit 1
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.local_model import GenerationSettings, TransformersChatModel  # noqa: E402
from src.generation.question_pool_pipeline import run_question_pool_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--model-id", default="Qwen/Qwen3-8B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.anchors.is_file():
        print(f"error: anchor manifest not found: {args.anchors}", file=sys.stderr)
        return 2
    if args.limit is not None and args.limit <= 0:
        print("error: --limit must be positive", file=sys.stderr)
        return 2
    settings = GenerationSettings(
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    model = TransformersChatModel(args.model_path, device=args.device)
    counts = run_question_pool_pipeline(
        model=model,
        anchors_path=args.anchors,
        output_dir=args.output_dir,
        model_id=args.model_id,
        model_path=str(args.model_path.resolve()),
        settings=settings,
        limit=args.limit,
    )
    print(f"Pipeline summary: {counts}")
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

