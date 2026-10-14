"""Evaluate standalone Source Analyzer v3 on the 30-record gold set.

Purpose:
    Analyze original questions only, save raw/predicted JSONL, and report Stage 01A
    domain, decision, math false-positive, subfield, precision, and recall metrics.

Input:
    Source-analysis gold annotations, immutable source dataset, prompt, and local model.

Output:
    `predictions.jsonl` and `metrics.json` under the output directory.

Example:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/evaluate_source_analyzer.py \
        --gold datasets/metadata/source_analysis_gold_v1.jsonl \
        --dataset-root datasets/science_dataset/mixed_dataset \
        --prompt src/prompts/source_analyzer_v3.json \
        --model-path /home/dqwang/Model/Qwen3-8B \
        --output-dir outputs/source_analysis/v3_gold30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.source_analysis import compute_metrics, load_gold, parse_prediction  # noqa: E402
from src.generation.local_model import GenerationSettings, TransformersChatModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--prompt", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=900)
    return parser.parse_args()


def _system_prompt(config: dict) -> str:
    examples = "\n\nBoundary examples:\n" + "\n".join(
        "Input:\n"
        + json.dumps(item["input"], ensure_ascii=False)
        + "\nOutput:\n"
        + json.dumps(item["output"], ensure_ascii=False)
        for item in config["few_shot_examples"]
    )
    schema = "\n\nRequired JSON fields:\n" + json.dumps(
        config["output_schema"], ensure_ascii=False, indent=2
    )
    return config["system_prompt"] + examples + schema


def main() -> int:
    args = parse_args()
    gold = load_gold(args.gold, args.dataset_root)
    if args.limit is not None:
        gold = gold[: args.limit]
    prompt_config = json.loads(args.prompt.read_text(encoding="utf-8"))
    model = TransformersChatModel(args.model_path, device=args.device)
    settings = GenerationSettings(max_new_tokens=args.max_new_tokens, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    completed = set()
    if predictions_path.exists():
        with predictions_path.open("r", encoding="utf-8") as handle:
            completed = {json.loads(line)["id"] for line in handle if line.strip()}

    with predictions_path.open("a", encoding="utf-8") as handle:
        for item in gold:
            if item["id"] in completed:
                continue
            user_payload = {
                "question": item["source"]["question"],
                "choices": item["source"]["choices"],
            }
            raw = model.generate(
                [
                    {"role": "system", "content": _system_prompt(prompt_config)},
                    {
                        "role": "user",
                        "content": "Analyze this original question only:\n" + json.dumps(user_payload, ensure_ascii=False, indent=2),
                    },
                ],
                settings,
            )
            result = {"id": item["id"], "prompt_version": prompt_config["id"], "raw": raw}
            try:
                result["prediction"] = parse_prediction(raw)
            except (json.JSONDecodeError, ValueError) as exc:
                result["error"] = str(exc)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()

    with predictions_path.open("r", encoding="utf-8") as handle:
        predictions = [json.loads(line) for line in handle if line.strip()]
    metrics = compute_metrics(gold, predictions)
    metrics.update(
        {
            "prompt_version": prompt_config["id"],
            "model": "Qwen/Qwen3-8B",
            "seed": args.seed,
            "gold_path": str(args.gold),
        }
    )
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

