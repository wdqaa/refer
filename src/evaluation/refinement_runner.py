"""Expert-Critique and compute-matched generic Self-Refine inference primitives."""

from __future__ import annotations

import json
from typing import Any

from src.evaluation.baseline_runner import _call, system_with_schema, validate_answer, visible_question
from src.generation.local_model import GenerationSettings


ARMS = {"expert_critique", "self_refine"}
DOMAINS = {"physics", "chemistry", "biology", "astronomy", "geography", "mathematics"}


def validate_critique(payload: dict[str, Any]) -> dict[str, Any]:
    required = {
        "material_errors",
        "missing_points",
        "reasoning_issues",
        "preserve",
        "recommend_change",
        "confidence",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"critique output missing fields: {sorted(missing)}")
    if not isinstance(payload["material_errors"], list):
        raise ValueError("material_errors must be a list")
    for row in payload["material_errors"]:
        if not isinstance(row, dict) or set(row) < {"draft_claim", "problem", "correction"}:
            raise ValueError("invalid material_errors entry")
        if not all(isinstance(row[key], str) for key in ("draft_claim", "problem", "correction")):
            raise ValueError("material error fields must be strings")
    if not isinstance(payload["missing_points"], list):
        raise ValueError("missing_points must be a list")
    for row in payload["missing_points"]:
        if not isinstance(row, dict) or set(row) < {"missing", "why_needed"}:
            raise ValueError("invalid missing_points entry")
        if not all(isinstance(row[key], str) for key in ("missing", "why_needed")):
            raise ValueError("missing point fields must be strings")
    for field in ("reasoning_issues", "preserve"):
        if not isinstance(payload[field], list) or any(not isinstance(value, str) for value in payload[field]):
            raise ValueError(f"{field} must be a list of strings")
    if not isinstance(payload["recommend_change"], bool):
        raise ValueError("recommend_change must be boolean")
    confidence = payload["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")
    return payload


def critic_system(prompt: dict[str, Any], *, domain: str | None = None) -> str:
    if domain is None:
        return system_with_schema(prompt)
    if domain not in DOMAINS:
        raise ValueError(f"unsupported critic domain: {domain}")
    system_prompt = prompt["system_prompt_template"].format(
        domain=domain,
        domain_guidance=prompt["domain_guidance"][domain],
    )
    return system_prompt + "\n\nRequired JSON schema:\n" + json.dumps(
        prompt["output_schema"], ensure_ascii=False, indent=2
    )


def run_refinement_arm(
    model: Any,
    record: dict[str, Any],
    *,
    arm: str,
    direct_prediction: dict[str, Any],
    selected_domains: list[str],
    routing_rationale: str | None,
    prompts: dict[str, Any],
    critic_settings: GenerationSettings,
    final_settings: GenerationSettings,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unsupported refinement arm: {arm}")
    if any(domain not in DOMAINS for domain in selected_domains):
        raise ValueError("selected_domains contains an unsupported domain")
    visible = visible_question(record)
    draft = validate_answer(dict(direct_prediction))
    calls: list[dict[str, Any]] = []
    critiques: list[dict[str, Any]] = []

    if arm == "expert_critique":
        prompt = prompts["expert_critic"]
        for domain in selected_domains:
            call = _call(
                model,
                role=f"{domain}_critic",
                prompt_id=prompt["id_template"].format(domain=domain),
                system_prompt=critic_system(prompt, domain=domain),
                user_payload={**visible, "draft": draft},
                settings=critic_settings,
                parser=validate_critique,
            )
            calls.append(call)
            critiques.append(
                {"reviewer": domain, "critique": call["parsed"], "error": call["error"]}
            )
    else:
        prompt = prompts["generic_critic"]
        focuses = prompt["review_focuses"]
        for index in range(len(selected_domains)):
            call = _call(
                model,
                role=f"generic_critic_pass_{index + 1}",
                prompt_id=prompt["id"],
                system_prompt=critic_system(prompt),
                user_payload={
                    **visible,
                    "draft": draft,
                    "review_pass": index + 1,
                    "review_focus": focuses[index % len(focuses)],
                },
                settings=critic_settings,
                parser=validate_critique,
            )
            calls.append(call)
            critiques.append(
                {"reviewer": f"generic_pass_{index + 1}", "critique": call["parsed"], "error": call["error"]}
            )

    prompt = prompts["reviser"]
    revision_call = _call(
        model,
        role="draft_reviser",
        prompt_id=prompt["id"],
        system_prompt=system_with_schema(prompt),
        user_payload={**visible, "draft": draft, "critiques": critiques},
        settings=final_settings,
        parser=validate_answer,
    )
    calls.append(revision_call)
    final = revision_call["parsed"]
    return {
        "schema_version": "qwen_refinement_trace.v1",
        "item_id": record["id"],
        "arm": arm,
        "visible_input": visible,
        "direct_draft": draft,
        "routing": {
            "mode": "cached_question_only_count" if arm == "self_refine" else "cached_question_only",
            "selected_domains": selected_domains if arm == "expert_critique" else [],
            "critic_passes": len(selected_domains),
            "rationale": routing_rationale if arm == "expert_critique" else None,
        },
        "calls": calls,
        "final_prediction": final,
        "status": "ok" if final is not None else "parse_error",
    }
