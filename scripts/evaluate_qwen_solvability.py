"""Compare Qwen3-8B direct solving with oracle-routed fixed expert collaboration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.parsing import parse_agent_output  # noqa: E402
from src.generation.local_model import GenerationSettings, TransformersChatModel  # noqa: E402
from src.prompts.loading import load_prompt_registry  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--prompt", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--thinking-max-new-tokens", type=int, default=1600)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--with-critic", action="store_true")
    parser.add_argument("--ids", nargs="*", default=None)
    return parser.parse_args()


def _decoded_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if "</think>" in candidate:
        candidate = candidate.rsplit("</think>", 1)[1].strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    try:
        value = json.loads(candidate)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    decoded: list[dict[str, Any]] = []
    for index, char in enumerate(candidate):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(candidate[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            decoded.append(value)
    if not decoded:
        raise ValueError("output contains no parseable JSON object")
    return decoded[-1]


def _json_object(text: str) -> dict[str, Any]:
    value = _decoded_object(text)
    for field in ("correct_choice", "answer", "reasoning", "confidence"):
        if field not in value:
            raise ValueError(f"solver output missing {field}")
    if not isinstance(value["reasoning"], list):
        raise ValueError("reasoning must be a list")
    return value


def _system(text: str, schema: dict[str, Any]) -> str:
    return text + "\n\nRequired JSON schema:\n" + json.dumps(schema, ensure_ascii=False, indent=2)


def _question(candidate: dict[str, Any]) -> dict[str, Any]:
    return {"question": candidate["question"], "choices": candidate["choices"]}


def main() -> int:
    args = parse_args()
    rows = [json.loads(line) for line in args.pilot.read_text(encoding="utf-8").splitlines() if line.strip()]
    review = json.loads(args.review.read_text(encoding="utf-8"))
    accepted = set(review["accepted_ids"])
    rows = [row for row in rows if row["id"] in accepted]
    if args.ids is not None:
        requested = set(args.ids)
        rows = [row for row in rows if row["id"] in requested]
    config = json.loads(args.prompt.read_text(encoding="utf-8"))
    registry = load_prompt_registry()
    model = TransformersChatModel(args.model_path, device=args.device)
    plain_settings = GenerationSettings(
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        enable_thinking=False,
    )
    thinking_settings = GenerationSettings(
        max_new_tokens=args.thinking_max_new_tokens,
        seed=args.seed,
        enable_thinking=args.enable_thinking,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "predictions.jsonl"

    results = []
    for row in rows:
        candidate = row["generator"]["candidate"]
        visible = _question(candidate)
        direct_raw = model.generate(
            [
                {"role": "system", "content": _system(config["direct_solver"], config["output_schema"])},
                {"role": "user", "content": json.dumps(visible, ensure_ascii=False, indent=2)},
            ],
            thinking_settings,
        )
        try:
            direct = _json_object(direct_raw)
            direct_error = None
        except (json.JSONDecodeError, ValueError) as exc:
            direct = None
            direct_error = str(exc)

        expert_outputs: dict[str, Any] = {}
        expert_raw: dict[str, str] = {}
        for domain in candidate["required_domains_after"]:
            role = f"{domain}_expert"
            prompt = registry[role]
            raw = model.generate(
                [
                    {
                        "role": "system",
                        "content": _system(prompt["system_prompt"], prompt["output_schema"]),
                    },
                    {
                        "role": "user",
                        "content": "Analyze only the contribution of your specialty needed to solve this question:\n"
                        + json.dumps(visible, ensure_ascii=False, indent=2),
                    },
                ],
                plain_settings,
            )
            expert_raw[domain] = raw
            try:
                expert_outputs[domain] = parse_agent_output(
                    role, json.dumps(_decoded_object(raw), ensure_ascii=False)
                )
            except ValueError as exc:
                expert_outputs[domain] = {"parse_error": str(exc)}

        synthesis_input = {"question": visible, "expert_contributions": expert_outputs}
        synthesis_raw = model.generate(
            [
                {"role": "system", "content": _system(config["fixed_synthesizer"], config["output_schema"])},
                {"role": "user", "content": json.dumps(synthesis_input, ensure_ascii=False, indent=2)},
            ],
            plain_settings,
        )
        try:
            draft_synthesis = _json_object(synthesis_raw)
            synthesis_error = None
        except (json.JSONDecodeError, ValueError) as exc:
            draft_synthesis = None
            synthesis_error = str(exc)

        critic = None
        critic_raw = None
        critic_error = None
        final_raw = synthesis_raw
        synthesis = draft_synthesis
        if args.with_critic and draft_synthesis is not None:
            critic_prompt = registry["critic"]
            critic_input = {
                "question": visible,
                "expert_contributions": expert_outputs,
                "draft_answer": draft_synthesis,
            }
            critic_raw = model.generate(
                [
                    {
                        "role": "system",
                        "content": _system(critic_prompt["system_prompt"], critic_prompt["output_schema"]),
                    },
                    {"role": "user", "content": json.dumps(critic_input, ensure_ascii=False, indent=2)},
                ],
                plain_settings,
            )
            try:
                critic = parse_agent_output(
                    "critic", json.dumps(_decoded_object(critic_raw), ensure_ascii=False)
                )
            except ValueError as exc:
                critic_error = str(exc)
                critic = {"parse_error": critic_error}
            revision_input = {
                "question": visible,
                "expert_contributions": expert_outputs,
                "draft_answer": draft_synthesis,
                "critic_report": critic,
            }
            final_raw = model.generate(
                [
                    {
                        "role": "system",
                        "content": _system(config["final_reviser"], config["output_schema"]),
                    },
                    {"role": "user", "content": json.dumps(revision_input, ensure_ascii=False, indent=2)},
                ],
                thinking_settings,
            )
            try:
                synthesis = _json_object(final_raw)
                synthesis_error = None
            except (json.JSONDecodeError, ValueError) as exc:
                synthesis = None
                synthesis_error = str(exc)

        gold = candidate["correct_choice"]
        direct_correct = direct is not None and direct.get("correct_choice") == gold
        fixed_correct = synthesis is not None and synthesis.get("correct_choice") == gold
        if direct_correct and fixed_correct:
            bucket = "direct_and_fixed_correct"
        elif not direct_correct and fixed_correct:
            bucket = "fixed_multi_agent_gain"
        elif direct_correct and not fixed_correct:
            bucket = "collaboration_harms"
        else:
            bucket = "too_hard_or_shared_failure"
        result = {
            "id": row["id"],
            "gold_choice": gold,
            "direct": {"parsed": direct, "raw": direct_raw, "error": direct_error, "correct": direct_correct},
            "fixed_multi_agent": {
                "oracle_routed_domains": candidate["required_domains_after"],
                "expert_outputs": expert_outputs,
                "expert_raw": expert_raw,
                "draft_synthesis": draft_synthesis,
                "synthesis": synthesis,
                "synthesis_raw": synthesis_raw,
                "critic": critic,
                "critic_raw": critic_raw,
                "critic_error": critic_error,
                "final_raw": final_raw,
                "error": synthesis_error,
                "correct": fixed_correct,
            },
            "bucket": bucket,
        }
        results.append(result)
        print(row["id"], "direct=", direct_correct, "fixed=", fixed_correct, bucket, flush=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    total = len(results)
    metrics = {
        "total": total,
        "direct_accuracy": sum(item["direct"]["correct"] for item in results) / total if total else 0,
        "fixed_multi_agent_accuracy": sum(item["fixed_multi_agent"]["correct"] for item in results) / total if total else 0,
        "buckets": {
            name: sum(item["bucket"] == name for item in results)
            for name in (
                "direct_and_fixed_correct",
                "fixed_multi_agent_gain",
                "collaboration_harms",
                "too_hard_or_shared_failure",
            )
        },
        "routing": "oracle fixed routing from reviewed candidate metadata; labels were not shown to solver prompts",
        "model": "Qwen/Qwen3-8B",
        "seed": args.seed,
        "enable_thinking": args.enable_thinking,
        "with_critic": args.with_critic,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
