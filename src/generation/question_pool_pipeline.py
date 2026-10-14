"""Run generator and verifier roles over an anchor manifest with resumable JSONL logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Protocol

from src.agents.parsing import AgentOutputError, parse_agent_output
from src.data.question_pool import (
    build_question_pool_record,
    validate_candidate,
    validate_verification,
)
from src.generation.local_model import GenerationSettings
from src.prompts.loading import load_prompt_registry


class ChatModel(Protocol):
    def generate(self, messages: list[dict[str, str]], settings: GenerationSettings) -> str: ...


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            records.append(value)
    return records


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _generator_request(anchor: dict[str, Any]) -> str:
    mode_guidance = {
        "real_anchored": "Analyze the source before choosing keep, transform, or reject. Keep it exactly if it is already valid and useful. Transform it only when one minimal added condition creates a genuine necessary cross-domain dependency. Do not merely paraphrase it or change the answer format. Do not choose target disciplines in advance; reject if no natural transformation exists.",
        "synthetic": "Create one new cross-domain question only if the source provides a credible style or scenario anchor. It must remain compact, independently answerable, and require at least two domains; otherwise reject.",
    }
    mode = anchor["requested_mode"]
    return (
        f"Requested mode: {mode}\n{mode_guidance[mode]}\n\n"
        "Source record:\n" + json.dumps(anchor["source"], ensure_ascii=False, indent=2)
    )


def _verifier_request(anchor: dict[str, Any], candidate: dict[str, Any]) -> str:
    return (
        "Verify this candidate independently. Treat required_domains as claims to ablate, not as facts.\n\n"
        "Source provenance:\n"
        + json.dumps(anchor["source"], ensure_ascii=False, indent=2)
        + "\n\nCandidate:\n"
        + json.dumps(candidate, ensure_ascii=False, indent=2)
    )


def _call_and_parse(
    model: ChatModel,
    *,
    role: str,
    system_prompt: str,
    user_prompt: str,
    settings: GenerationSettings,
    raw_outputs: list[str],
    semantic_validator: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    raw = model.generate(messages, settings)
    raw_outputs.append(raw)
    try:
        payload = parse_agent_output(role, raw)
        if semantic_validator is not None:
            semantic_validator(payload)
        return payload
    except (AgentOutputError, ValueError) as first_error:
        repair_messages = messages + [
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": (
                    f"Your output failed validation: {first_error}. Return one corrected JSON object only, "
                    "using the required schema from the system instruction."
                ),
            },
        ]
        repaired = model.generate(repair_messages, settings)
        raw_outputs.append(repaired)
        payload = parse_agent_output(role, repaired)
        if semantic_validator is not None:
            semantic_validator(payload)
        return payload


def _system_with_schema(entry: dict[str, Any]) -> str:
    """Join the prose contract and its machine-readable schema for model input."""

    return (
        entry["system_prompt"]
        + "\n\nRequired JSON fields and value types:\n"
        + json.dumps(entry["output_schema"], ensure_ascii=False, indent=2)
    )


def run_question_pool_pipeline(
    *,
    model: ChatModel,
    anchors_path: Path,
    output_dir: Path,
    model_id: str,
    model_path: str,
    settings: GenerationSettings,
    limit: int | None = None,
) -> dict[str, int]:
    """Run pending anchors and append successes, failures, and raw traces."""

    prompts = load_prompt_registry()
    candidates_path = output_dir / "candidates.jsonl"
    failures_path = output_dir / "failures.jsonl"
    traces_path = output_dir / "raw_traces.jsonl"
    completed_ids = {
        record["anchor_id"] for path in (traces_path,) if path.exists() for record in read_jsonl(path)
    }
    pending = [record for record in read_jsonl(anchors_path) if record["id"] not in completed_ids]
    if limit is not None:
        pending = pending[:limit]

    counts = {"attempted": 0, "written": 0, "accepted": 0, "failed": 0}
    output_dir.mkdir(parents=True, exist_ok=True)
    settings_dict = {
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "max_tokens": settings.max_new_tokens,
        "seed": settings.seed,
    }
    for anchor in pending:
        counts["attempted"] += 1
        trace: dict[str, Any] = {"anchor_id": anchor["id"], "generator_outputs": [], "verifier_outputs": []}
        try:
            candidate = _call_and_parse(
                model,
                role="question_generator",
                system_prompt=_system_with_schema(prompts["question_generator"]),
                user_prompt=_generator_request(anchor),
                settings=settings,
                raw_outputs=trace["generator_outputs"],
                semantic_validator=lambda value: validate_candidate(anchor, value),
            )
            if candidate["decision"] == "reject":
                raise ValueError(f"Generator rejected anchor: {candidate['transformation_summary']}")
            verification = _call_and_parse(
                model,
                role="question_verifier",
                system_prompt=_system_with_schema(prompts["question_verifier"]),
                user_prompt=_verifier_request(anchor, candidate),
                settings=settings,
                raw_outputs=trace["verifier_outputs"],
                semantic_validator=lambda value: validate_verification(candidate, value),
            )
            record = build_question_pool_record(
                anchor=anchor,
                candidate=candidate,
                verification=verification,
                model_id=model_id,
                model_path=model_path,
                settings=settings_dict,
                generator_prompt_version=prompts["question_generator"]["id"],
                verifier_prompt_version=prompts["question_verifier"]["id"],
            )
            append_jsonl(candidates_path, record)
            counts["written"] += 1
            counts["accepted"] += int(record["verification"]["accepted"])
            trace["status"] = "written"
        except (AgentOutputError, KeyError, TypeError, ValueError) as exc:
            counts["failed"] += 1
            trace["status"] = "failed"
            trace["error"] = str(exc)
            append_jsonl(failures_path, {"anchor": anchor, "error": str(exc)})
        append_jsonl(traces_path, trace)
    return counts
