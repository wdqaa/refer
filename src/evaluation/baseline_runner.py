"""Prompt-only Direct, routed, and oracle Multi-Agent inference primitives."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable

from src.agents.parsing import parse_agent_output
from src.generation.local_model import GenerationSettings


DOMAINS = {"physics", "chemistry", "biology", "astronomy", "geography", "mathematics"}
ARMS = {"direct", "routed_multi_agent", "oracle_multi_agent"}


def _escape_invalid_json_backslashes(candidate: str) -> str:
    """Repair only odd, invalid backslash runs commonly produced by LaTeX."""

    output: list[str] = []
    index = 0
    valid_json_escapes = set('"\\/bfnrtu')
    while index < len(candidate):
        if candidate[index] != "\\":
            output.append(candidate[index])
            index += 1
            continue
        end = index
        while end < len(candidate) and candidate[end] == "\\":
            end += 1
        run_length = end - index
        output.append("\\" * run_length)
        next_char = candidate[end] if end < len(candidate) else ""
        latex_t_command = candidate.startswith("times", end) or candidate.startswith("text", end)
        if run_length % 2 == 1 and (next_char not in valid_json_escapes or latex_t_command):
            output.append("\\")
        index = end
    return "".join(output)


def visible_question(record: dict[str, Any]) -> dict[str, Any]:
    """Return the only record fields allowed in model-visible question messages."""

    return {"question": record["question"], "choices": record.get("choices")}


def system_with_schema(prompt: dict[str, Any]) -> str:
    return prompt["system_prompt"] + "\n\nRequired JSON schema:\n" + json.dumps(
        prompt["output_schema"], ensure_ascii=False, indent=2
    )


def decode_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if "</think>" in candidate:
        candidate = candidate.rsplit("</think>", 1)[1].strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    # Models sometimes emit LaTeX such as ``\times`` inside a JSON string without
    # escaping the backslash. Preserve the raw trace, but permit this narrow,
    # deterministic serialization repair before rejecting an otherwise valid object.
    escaped = _escape_invalid_json_backslashes(candidate)
    candidates = [escaped, candidate] if escaped != candidate else [candidate]
    for serialized in candidates:
        try:
            value = json.loads(serialized)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    decoder = json.JSONDecoder()
    found = []
    for serialized in candidates:
        for index, char in enumerate(serialized):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(serialized[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                found.append(value)
    if not found:
        raise ValueError("output contains no parseable JSON object")
    return found[-1]


def validate_answer(payload: dict[str, Any]) -> dict[str, Any]:
    required = {"answer", "choice", "value", "unit", "canonical_expression", "reasoning", "confidence"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"answer output missing fields: {sorted(missing)}")
    if not isinstance(payload["answer"], str) or not payload["answer"].strip():
        raise ValueError("answer must be a non-empty string")
    if payload["choice"] is not None and not isinstance(payload["choice"], str):
        raise ValueError("choice must be a string or null")
    if payload["value"] is not None and (
        isinstance(payload["value"], bool) or not isinstance(payload["value"], (int, float))
    ):
        raise ValueError("value must be numeric or null")
    for field in ("unit", "canonical_expression"):
        if payload[field] is not None and not isinstance(payload[field], str):
            raise ValueError(f"{field} must be a string or null")
    if not isinstance(payload["reasoning"], list) or any(
        not isinstance(item, str) for item in payload["reasoning"]
    ):
        raise ValueError("reasoning must be a list of strings")
    confidence = payload["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")
    return payload


def validate_route(payload: dict[str, Any], max_domains: int = 3) -> dict[str, Any]:
    if set(payload) < {"selected_domains", "rationale"}:
        raise ValueError("router output missing selected_domains or rationale")
    domains = payload["selected_domains"]
    if not isinstance(domains, list) or any(domain not in DOMAINS for domain in domains):
        raise ValueError("router selected_domains contains an invalid domain")
    if len(domains) != len(set(domains)) or len(domains) > max_domains:
        raise ValueError("router selected_domains must be unique and bounded")
    if not isinstance(payload["rationale"], str) or not payload["rationale"].strip():
        raise ValueError("router rationale must be non-empty")
    return payload


def _call(
    model: Any,
    *,
    role: str,
    prompt_id: str,
    system_prompt: str,
    user_payload: dict[str, Any],
    settings: GenerationSettings,
    parser: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
    ]
    started = time.perf_counter()
    raw = model.generate(messages, settings)
    latency = time.perf_counter() - started
    parsed = None
    error = None
    try:
        parsed = parser(decode_object(raw))
    except ValueError as exc:
        error = str(exc)
    tokenizer = getattr(model, "tokenizer", None)
    output_tokens = len(tokenizer.encode(raw)) if tokenizer is not None else None
    counter = getattr(model, "count_input_tokens", None)
    input_tokens = (
        counter(messages, enable_thinking=settings.enable_thinking)
        if callable(counter)
        else None
    )
    return {
        "role": role,
        "prompt_id": prompt_id,
        "messages": messages,
        "raw": raw,
        "parsed": parsed,
        "error": error,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_seconds": round(latency, 6),
    }


def run_arm(
    model: Any,
    record: dict[str, Any],
    arm: str,
    baseline_prompts: dict[str, Any],
    agent_registry: dict[str, dict[str, Any]],
    role_settings: GenerationSettings,
    final_settings: GenerationSettings,
    max_router_domains: int = 3,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unsupported baseline arm: {arm}")
    visible = visible_question(record)
    calls = []
    selected_domains: list[str] = []
    routing_rationale = None

    if arm == "direct":
        prompt = baseline_prompts["direct_solver"]
        call = _call(
            model,
            role="direct_solver",
            prompt_id=prompt["id"],
            system_prompt=system_with_schema(prompt),
            user_payload=visible,
            settings=final_settings,
            parser=validate_answer,
        )
        calls.append(call)
        final = call["parsed"]
    else:
        if arm == "routed_multi_agent":
            prompt = baseline_prompts["router"]
            route_call = _call(
                model,
                role="router",
                prompt_id=prompt["id"],
                system_prompt=system_with_schema(prompt),
                user_payload=visible,
                settings=role_settings,
                parser=lambda payload: validate_route(payload, max_router_domains),
            )
            calls.append(route_call)
            route = route_call["parsed"] or {"selected_domains": [], "rationale": None}
            selected_domains = route["selected_domains"]
            routing_rationale = route["rationale"]
        else:
            selected_domains = list(record["required_domains"])

        expert_contributions = []
        for domain in selected_domains:
            role = f"{domain}_expert"
            prompt = agent_registry[role]
            expert_call = _call(
                model,
                role=role,
                prompt_id=prompt["id"],
                system_prompt=system_with_schema(prompt),
                user_payload={
                    **visible,
                    "request": "Provide only the domain-specific contribution needed to solve the visible question.",
                },
                settings=role_settings,
                parser=lambda payload, role=role: parse_agent_output(
                    role, json.dumps(payload, ensure_ascii=False)
                ),
            )
            calls.append(expert_call)
            expert_contributions.append(
                {"domain": domain, "contribution": expert_call["parsed"], "error": expert_call["error"]}
            )

        prompt = baseline_prompts["synthesizer"]
        synthesis_call = _call(
            model,
            role="synthesizer",
            prompt_id=prompt["id"],
            system_prompt=system_with_schema(prompt),
            user_payload={**visible, "expert_contributions": expert_contributions},
            settings=final_settings,
            parser=validate_answer,
        )
        calls.append(synthesis_call)
        final = synthesis_call["parsed"]

    return {
        "schema_version": "qwen_baseline_trace.v1",
        "item_id": record["id"],
        "arm": arm,
        "visible_input": visible,
        "routing": {
            "mode": "none" if arm == "direct" else "question_only" if arm == "routed_multi_agent" else "oracle_control_only",
            "selected_domains": selected_domains,
            "rationale": routing_rationale,
        },
        "calls": calls,
        "final_prediction": final,
        "status": "ok" if final is not None else "parse_error",
    }
