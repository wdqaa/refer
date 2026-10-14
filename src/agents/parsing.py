"""Parse and validate JSON outputs for the Stage 01 prompt-based agents."""

from __future__ import annotations

import json
import re
from typing import Any


class AgentOutputError(ValueError):
    """Raised when a model response violates its role contract."""


EXPERT_ROLES = {
    "physics_expert",
    "chemistry_expert",
    "mathematics_expert",
    "biology_expert",
    "astronomy_expert",
    "geography_expert",
}
PLANNER_ACTIONS = {"call_expert", "critique", "continue_reasoning", "finalize"}
EXPERT_DOMAINS = {
    "physics", "chemistry", "mathematics", "biology", "astronomy", "geography"
}


def _decode_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise AgentOutputError(f"Agent output is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise AgentOutputError("Agent output must be one JSON object")
    return payload


def _require(payload: dict[str, Any], fields: set[str], role: str) -> None:
    missing = fields - set(payload)
    if missing:
        raise AgentOutputError(f"{role} output missing fields: {sorted(missing)}")


def _require_string_list(payload: dict[str, Any], field: str, role: str) -> None:
    value = payload[field]
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise AgentOutputError(f"{role}.{field} must be a list of strings")


def _validate_planner(payload: dict[str, Any]) -> None:
    _require(payload, {"action", "rationale"}, "planner")
    action = payload["action"]
    if action not in PLANNER_ACTIONS:
        raise AgentOutputError(f"Unsupported planner action: {action!r}")
    if not isinstance(payload["rationale"], str) or not payload["rationale"].strip():
        raise AgentOutputError("planner.rationale must be a non-empty string")
    if action == "call_expert":
        if payload.get("domain") not in EXPERT_DOMAINS:
            raise AgentOutputError("call_expert requires a supported domain")
        if not isinstance(payload.get("query"), str) or not payload["query"].strip():
            raise AgentOutputError("call_expert requires a focused query")
    if action == "finalize":
        if not isinstance(payload.get("final_answer"), str) or not payload["final_answer"].strip():
            raise AgentOutputError("finalize requires final_answer")


def _validate_expert(payload: dict[str, Any], role: str) -> None:
    _require(payload, {"relevant", "contribution", "assumptions", "confidence"}, role)
    if not isinstance(payload["relevant"], bool):
        raise AgentOutputError(f"{role}.relevant must be boolean")
    if not isinstance(payload["contribution"], str):
        raise AgentOutputError(f"{role}.contribution must be a string")
    _require_string_list(payload, "assumptions", role)
    confidence = payload["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise AgentOutputError(f"{role}.confidence must be between 0 and 1")


def _validate_critic(payload: dict[str, Any]) -> None:
    fields = {"valid", "issues", "unnecessary_calls", "missing_checks", "recommendation"}
    _require(payload, fields, "critic")
    if not isinstance(payload["valid"], bool):
        raise AgentOutputError("critic.valid must be boolean")
    for field in ("issues", "unnecessary_calls", "missing_checks"):
        _require_string_list(payload, field, "critic")
    if not isinstance(payload["recommendation"], str):
        raise AgentOutputError("critic.recommendation must be a string")


def _validate_generator(payload: dict[str, Any]) -> None:
    fields = {
        "decision", "question", "answer", "answer_type", "choices", "primary_domain",
        "required_domains", "difficulty", "reasoning_type", "derivation",
        "transformation_summary", "source_analysis",
    }
    _require(payload, fields, "question_generator")
    if payload["decision"] not in {"keep", "transform", "reject", "synthetic"}:
        raise AgentOutputError("question_generator.decision is invalid")
    for field in ("question", "answer", "primary_domain", "transformation_summary"):
        if not isinstance(payload[field], str):
            raise AgentOutputError(f"question_generator.{field} must be a string")
    if payload["difficulty"] not in {"easy", "medium", "hard"}:
        raise AgentOutputError("question_generator.difficulty is invalid")
    if payload["answer_type"] not in {"multiple_choice", "short_answer", "numeric", "open_ended"}:
        raise AgentOutputError("question_generator.answer_type is invalid")
    if payload["choices"] is not None and not isinstance(payload["choices"], dict):
        raise AgentOutputError("question_generator.choices must be an object or null")
    for field in ("required_domains", "reasoning_type", "derivation"):
        _require_string_list(payload, field, "question_generator")
    analysis = payload["source_analysis"]
    if not isinstance(analysis, dict):
        raise AgentOutputError("question_generator.source_analysis must be an object")
    _require(
        analysis,
        {"scientifically_valid", "existing_required_domains", "already_cross_domain", "natural_extension"},
        "question_generator.source_analysis",
    )
    if not isinstance(analysis["scientifically_valid"], bool):
        raise AgentOutputError("source_analysis.scientifically_valid must be boolean")
    if not isinstance(analysis["already_cross_domain"], bool):
        raise AgentOutputError("source_analysis.already_cross_domain must be boolean")
    if analysis["natural_extension"] is not None and not isinstance(analysis["natural_extension"], str):
        raise AgentOutputError("source_analysis.natural_extension must be a string or null")
    _require_string_list(analysis, "existing_required_domains", "source_analysis")


def _validate_verifier(payload: dict[str, Any]) -> None:
    score_fields = {
        "scientific_validity", "naturalness", "fusion_necessity",
        "answer_correctness", "synthetic_smell",
    }
    _require(payload, score_fields | {"domain_necessity", "issues", "accepted"}, "question_verifier")
    for field in score_fields:
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
            raise AgentOutputError(f"question_verifier.{field} must be an integer from 1 to 5")
    if not isinstance(payload["domain_necessity"], list):
        raise AgentOutputError("question_verifier.domain_necessity must be a list")
    for judgment in payload["domain_necessity"]:
        if not isinstance(judgment, dict):
            raise AgentOutputError("Each domain necessity judgment must be an object")
        if (
            not isinstance(judgment.get("domain"), str)
            or not isinstance(judgment.get("necessary"), bool)
            or not isinstance(judgment.get("reason"), str)
        ):
            raise AgentOutputError(
                "Domain necessity requires string domain, boolean necessary, and string reason"
            )
    _require_string_list(payload, "issues", "question_verifier")
    if not isinstance(payload["accepted"], bool):
        raise AgentOutputError("question_verifier.accepted must be boolean")


def parse_agent_output(role: str, text: str) -> dict[str, Any]:
    """Decode one response and enforce role-specific required fields and ranges."""

    payload = _decode_object(text)
    if role == "planner":
        _validate_planner(payload)
    elif role in EXPERT_ROLES:
        _validate_expert(payload, role)
    elif role == "critic":
        _validate_critic(payload)
    elif role == "question_generator":
        _validate_generator(payload)
    elif role == "question_verifier":
        _validate_verifier(payload)
    else:
        raise AgentOutputError(f"Unknown agent role: {role}")
    return payload
