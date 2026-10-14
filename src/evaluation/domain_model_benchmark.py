"""Single-model domain benchmark inference with format-tolerant answer parsing."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from src.evaluation.baseline_runner import decode_object, system_with_schema, validate_answer
from src.generation.local_model import GenerationSettings


def record_id(record: dict[str, Any]) -> str:
    value = record.get("native_qa_id", record.get("id"))
    if not isinstance(value, str) or not value:
        raise ValueError("benchmark record has no native_qa_id or id")
    return value


def visible_question(record: dict[str, Any]) -> dict[str, Any]:
    return {"question": record["question"], "choices": record.get("choices") or None}


def _fallback_choice(raw: str, choices: dict[str, str] | None) -> str | None:
    if not choices:
        return None
    allowed = {str(key).upper() for key in choices}
    patterns = (
        r'(?i)["\']choice["\']\s*:\s*["\']([A-J])["\']',
        r"(?i)final\s+answer\s*(?:is|:)?\s*[\[(]?([A-J])[\])]?",
        r"(?i)(?:correct\s+)?(?:answer|option|choice)\s*(?:is|:)?\s*[\[(]?([A-J])[\])]?",
    )
    for pattern in patterns:
        matches = re.findall(pattern, raw)
        if matches and matches[-1].upper() in allowed:
            return matches[-1].upper()
    stripped = raw.strip()
    if stripped.upper() in allowed:
        return stripped.upper()
    return None


def parse_benchmark_answer(
    raw: str, choices: dict[str, str] | None
) -> tuple[dict[str, Any], str, str | None]:
    """Prefer the shared JSON schema, but retain usable plain-text science answers."""

    try:
        decoded = decode_object(raw)
    except ValueError as exc:
        if not raw.strip():
            return {}, "parse_failure", str(exc)
        decoded = None
        decode_error = str(exc)
    else:
        try:
            parsed = validate_answer(decoded)
            return parsed, "json", None
        except ValueError as exc:
            decode_error = str(exc)

    # Domain checkpoints may emit a scientifically usable JSON object while omitting
    # bookkeeping fields from the requested schema, then continue with prose. Preserve
    # explicit fields and fill only absent structural values; never infer answer content.
    if isinstance(decoded, dict) and isinstance(decoded.get("answer"), str) and decoded["answer"].strip():
        explicit_choice = decoded.get("choice")
        allowed = {str(key).upper() for key in choices} if choices else set()
        choice = str(explicit_choice).strip().upper() if explicit_choice is not None else None
        if choice is not None and (not choices or choice not in allowed):
            choice = None
        reasoning = decoded.get("reasoning")
        if isinstance(reasoning, str):
            reasoning = [reasoning]
        if not isinstance(reasoning, list) or any(not isinstance(row, str) for row in reasoning):
            reasoning = []
        confidence = decoded.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0 <= confidence <= 1
        ):
            confidence = None
        parsed = {
            "answer": decoded["answer"].strip(),
            "choice": choice,
            "value": (
                decoded.get("value")
                if isinstance(decoded.get("value"), (int, float))
                and not isinstance(decoded.get("value"), bool)
                else None
            ),
            "unit": decoded.get("unit") if isinstance(decoded.get("unit"), str) else None,
            "canonical_expression": (
                decoded.get("canonical_expression")
                if isinstance(decoded.get("canonical_expression"), str)
                else None
            ),
            "reasoning": reasoning,
            "confidence": confidence,
        }
        warning = decode_error
        if choices and choice is None:
            warning += "; partial JSON MCQ has no valid explicit choice"
        return parsed, "partial_json_normalized", warning

    parse_error = decode_error
    if not raw.strip():
        return {}, "parse_failure", parse_error
    else:
        choice = _fallback_choice(raw, choices)
        parsed = {
            "answer": raw.strip(),
            "choice": choice,
            "value": None,
            "unit": None,
            "canonical_expression": None,
            "reasoning": [],
            "confidence": None,
        }
        warning = parse_error
        if choices and choice is None:
            warning += "; plain-text MCQ has no explicit parseable choice"
        return parsed, "plain_text_fallback", warning


def run_domain_model_item(
    model: Any,
    record: dict[str, Any],
    *,
    prompt: dict[str, Any],
    settings: GenerationSettings,
    model_id: str,
    prompt_renderer: str,
) -> dict[str, Any]:
    visible = visible_question(record)
    messages = [
        {"role": "system", "content": system_with_schema(prompt)},
        {"role": "user", "content": json.dumps(visible, ensure_ascii=False, indent=2)},
    ]
    renderer = getattr(model, "render_messages", None)
    rendered_prompt = (
        renderer(messages, enable_thinking=settings.enable_thinking)
        if callable(renderer)
        else None
    )
    started = time.perf_counter()
    raw = model.generate(messages, settings)
    latency = time.perf_counter() - started
    prediction, parse_mode, parse_warning = parse_benchmark_answer(raw, visible["choices"])
    tokenizer = getattr(model, "tokenizer", None)
    output_tokens = len(tokenizer.encode(raw, add_special_tokens=False)) if tokenizer else None
    token_counter = getattr(model, "count_input_tokens", None)
    input_tokens = (
        token_counter(messages, enable_thinking=settings.enable_thinking)
        if callable(token_counter)
        else None
    )
    return {
        "schema_version": "domain_model_benchmark_trace.v1",
        "item_id": record_id(record),
        "model_id": model_id,
        "prompt_id": prompt["id"],
        "prompt_renderer": prompt_renderer,
        "visible_input": visible,
        "messages": messages,
        "rendered_prompt": rendered_prompt,
        "raw": raw,
        "final_prediction": prediction or None,
        "parse_mode": parse_mode,
        "parse_warning": parse_warning,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_seconds": round(latency, 6),
        "status": "ok" if prediction else "parse_error",
    }


def run_domain_model_batch(
    model: Any,
    records: list[dict[str, Any]],
    *,
    prompt: dict[str, Any],
    settings: GenerationSettings,
    model_id: str,
    prompt_renderer: str,
) -> list[dict[str, Any]]:
    """Run an inference-only batch while preserving one complete trace per item."""

    if not records:
        return []
    batch_generator = getattr(model, "generate_batch", None)
    if not callable(batch_generator):
        return [
            run_domain_model_item(
                model,
                record,
                prompt=prompt,
                settings=settings,
                model_id=model_id,
                prompt_renderer=prompt_renderer,
            )
            for record in records
        ]

    visible_rows = [visible_question(record) for record in records]
    conversations = [
        [
            {"role": "system", "content": system_with_schema(prompt)},
            {"role": "user", "content": json.dumps(visible, ensure_ascii=False, indent=2)},
        ]
        for visible in visible_rows
    ]
    renderer = getattr(model, "render_messages", None)
    rendered_prompts = [
        renderer(messages, enable_thinking=settings.enable_thinking)
        if callable(renderer)
        else None
        for messages in conversations
    ]
    started = time.perf_counter()
    raw_rows = batch_generator(conversations, settings)
    batch_latency = time.perf_counter() - started
    if len(raw_rows) != len(records):
        raise ValueError("batch generator returned the wrong number of completions")

    tokenizer = getattr(model, "tokenizer", None)
    token_counter = getattr(model, "count_input_tokens", None)
    traces: list[dict[str, Any]] = []
    for record, visible, messages, rendered, raw in zip(
        records, visible_rows, conversations, rendered_prompts, raw_rows
    ):
        prediction, parse_mode, parse_warning = parse_benchmark_answer(
            raw, visible["choices"]
        )
        traces.append(
            {
                "schema_version": "domain_model_benchmark_trace.v1",
                "item_id": record_id(record),
                "model_id": model_id,
                "prompt_id": prompt["id"],
                "prompt_renderer": prompt_renderer,
                "visible_input": visible,
                "messages": messages,
                "rendered_prompt": rendered,
                "raw": raw,
                "final_prediction": prediction or None,
                "parse_mode": parse_mode,
                "parse_warning": parse_warning,
                "input_tokens": (
                    token_counter(messages, enable_thinking=settings.enable_thinking)
                    if callable(token_counter)
                    else None
                ),
                "output_tokens": (
                    len(tokenizer.encode(raw, add_special_tokens=False))
                    if tokenizer
                    else None
                ),
                "latency_seconds": round(batch_latency / len(records), 6),
                "batch_size": len(records),
                "batch_latency_seconds": round(batch_latency, 6),
                "status": "ok" if prediction else "parse_error",
            }
        )
    return traces
