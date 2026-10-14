"""Canonical question-pool record construction and rule-based validation."""

from __future__ import annotations

from datetime import datetime
import re
from typing import Any


ALLOWED_DOMAINS = {
    "physics", "chemistry", "mathematics", "biology", "astronomy", "geography"
}
LEAKAGE_PATTERNS = {
    "interdisciplinary": re.compile(r"\binterdisciplinary\b", re.IGNORECASE),
    "from_the_perspective_of": re.compile(r"\bfrom the perspective of\b", re.IGNORECASE),
    "combine_knowledge_from": re.compile(r"\bcombine knowledge from\b", re.IGNORECASE),
    "using_named_domains": re.compile(
        r"\busing (?:knowledge (?:of|from) )?(?:physics|chemistry|mathematics|biology|astronomy|geography)(?:\s+and\s+(?:physics|chemistry|mathematics|biology|astronomy|geography))+",
        re.IGNORECASE,
    ),
    "chinese_combine": re.compile(r"结合.{0,30}(?:物理|化学|数学|生物|天文|地理).{0,20}(?:学科|知识)"),
    "chinese_perspective": re.compile(r"从.{0,30}(?:物理|化学|数学|生物|天文|地理).{0,10}角度"),
}


def explicit_fusion_leakage(question: str) -> list[str]:
    """Return named leakage rules matched by visible question text."""

    return [name for name, pattern in LEAKAGE_PATTERNS.items() if pattern.search(question)]


def validate_candidate(anchor: dict[str, Any], candidate: dict[str, Any]) -> None:
    """Enforce source-mode and domain invariants before spending a verifier call."""

    decision = candidate["decision"]
    if decision == "reject":
        return
    analysis = candidate["source_analysis"]
    existing_domains = analysis["existing_required_domains"]
    invalid_existing = set(existing_domains) - ALLOWED_DOMAINS
    if invalid_existing:
        raise ValueError(f"Unsupported existing source domains: {sorted(invalid_existing)}")
    if len(existing_domains) != len(set(existing_domains)):
        raise ValueError("source_analysis.existing_required_domains must be unique")
    if not analysis["scientifically_valid"]:
        raise ValueError("A scientifically invalid source must be rejected")
    requested_mode = anchor["requested_mode"]
    if requested_mode == "real_anchored" and decision not in {"keep", "transform"}:
        raise ValueError(f"Real-anchored request cannot produce decision {decision!r}")
    if requested_mode == "synthetic" and decision != "synthetic":
        raise ValueError(f"Synthetic request cannot produce decision {decision!r}")

    domains = candidate["required_domains"]
    invalid_domains = set(domains) - ALLOWED_DOMAINS
    if invalid_domains:
        raise ValueError(f"Unsupported required domains: {sorted(invalid_domains)}")
    if candidate["primary_domain"] not in ALLOWED_DOMAINS:
        raise ValueError(f"Unsupported primary domain: {candidate['primary_domain']}")
    if len(domains) != len(set(domains)):
        raise ValueError("required_domains must be unique")
    if candidate["primary_domain"] not in domains:
        raise ValueError("primary_domain must be included in required_domains")
    if decision in {"transform", "synthetic"} and len(domains) < 2:
        raise ValueError(f"{decision} candidates must require at least two domains")

    if requested_mode == "real_anchored":
        if analysis["already_cross_domain"] != (len(existing_domains) >= 2):
            raise ValueError("already_cross_domain must agree with existing_required_domains")
        if analysis["already_cross_domain"] and decision != "keep":
            raise ValueError("An already cross-domain source must be kept rather than transformed")
        if decision == "keep" and set(domains) != set(existing_domains):
            raise ValueError("keep required_domains must equal analyzed existing source domains")
        if decision == "transform":
            extension = analysis["natural_extension"]
            if not isinstance(extension, str) or not extension.strip():
                raise ValueError("transform requires a concrete natural_extension")
            if not set(existing_domains).issubset(domains) or set(existing_domains) == set(domains):
                raise ValueError("transform must preserve existing domains and add a necessary domain")

    source = anchor["source"]
    if decision == "keep":
        if candidate["question"] != source["original_question"]:
            raise ValueError("keep candidates must preserve the original question exactly")
        if candidate["answer"] != source["original_answer"]:
            raise ValueError("keep candidates must preserve the original answer exactly")
        if candidate.get("choices") != source.get("original_choices"):
            raise ValueError("keep candidates must preserve original choices exactly")
    if decision == "transform":
        unchanged_solution = (
            candidate["answer"] == source["original_answer"]
            and candidate.get("choices") == source.get("original_choices")
        )
        if unchanged_solution:
            raise ValueError(
                "transform cannot preserve both the original answer and choices; this is likely a paraphrase"
            )


def validate_verification(candidate: dict[str, Any], verification: dict[str, Any]) -> None:
    """Require one unique ablation judgment for every claimed required domain."""

    domains = candidate["required_domains"]
    judgments = verification["domain_necessity"]
    judgment_domains = [item["domain"] for item in judgments]
    if len(judgment_domains) != len(set(judgment_domains)):
        raise ValueError("Verifier returned duplicate domain necessity judgments")
    if set(judgment_domains) != set(domains):
        raise ValueError(
            "Verifier domain necessity entries do not exactly match required_domains: "
            f"expected={sorted(domains)}, actual={sorted(judgment_domains)}"
        )


def build_question_pool_record(
    *,
    anchor: dict[str, Any],
    candidate: dict[str, Any],
    verification: dict[str, Any],
    model_id: str,
    model_path: str,
    settings: dict[str, Any],
    generator_prompt_version: str = "question_generator_v2",
    verifier_prompt_version: str = "question_verifier_v2",
) -> dict[str, Any]:
    """Build one canonical v1 record and apply deterministic acceptance gates."""

    decision = candidate["decision"]
    if decision == "reject":
        raise ValueError("Rejected generator output cannot become a pool record")
    type_by_decision = {"keep": "real", "transform": "transformed", "synthetic": "synthetic"}
    if decision not in type_by_decision:
        raise ValueError(f"Unsupported generator decision: {decision}")
    validate_candidate(anchor, candidate)
    validate_verification(candidate, verification)
    domains = candidate["required_domains"]
    judgments = verification["domain_necessity"]
    every_domain_necessary = all(item["necessary"] for item in judgments)

    leakage = explicit_fusion_leakage(candidate["question"])
    deterministic_pass = (
        verification["scientific_validity"] >= 4
        and verification["naturalness"] >= 4
        and verification["answer_correctness"] >= 4
        and verification["fusion_necessity"] >= 4
        and verification["synthetic_smell"] <= 2
        and every_domain_necessary
        and not leakage
    )
    source = anchor["source"]
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    return {
        "schema_version": "question_pool.v1",
        "id": anchor["id"].replace("anchor_", "qp_", 1),
        "source": {
            "type": type_by_decision[decision],
            "dataset": source["dataset"],
            "original_id": source["original_id"],
            "original_question": source["original_question"],
            "original_answer": source["original_answer"],
            "original_choices": source.get("original_choices"),
        },
        "question": candidate["question"],
        "answer": candidate["answer"],
        "answer_type": candidate["answer_type"],
        "choices": candidate.get("choices"),
        "metadata": {
            "primary_domain": candidate["primary_domain"],
            "required_domains": domains,
            "difficulty": candidate["difficulty"],
            "reasoning_type": candidate["reasoning_type"],
            "derivation": candidate["derivation"],
        },
        "generation": {
            "generator_model": model_id,
            "model_path": model_path,
            "prompt_version": generator_prompt_version,
            **settings,
            "timestamp": timestamp,
            "requested_mode": anchor["requested_mode"],
            "source_analysis": candidate["source_analysis"],
            "transformation_summary": candidate["transformation_summary"],
        },
        "verification": {
            **verification,
            "explicit_fusion_leakage": leakage,
            "deterministic_gate_passed": deterministic_pass,
            "accepted": bool(verification["accepted"] and deterministic_pass),
            "verifier_model": model_id,
            "prompt_version": verifier_prompt_version,
        },
    }
