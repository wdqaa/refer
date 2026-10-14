"""Conservative deterministic scoring for objective verified QA types."""

from __future__ import annotations

import re
from typing import Any


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def score_prediction(record: dict[str, Any], prediction: dict[str, Any] | None) -> dict[str, Any]:
    if prediction is None:
        return {"auto_correct": False, "method": "parse_failure", "reason": "No parsed prediction"}
    question_type = record["question_type"]
    gold = record["gold"]
    if question_type == "mcq":
        expected = str(gold["correct_choice"]).strip().upper()
        actual = str(prediction.get("choice") or "").strip().upper()
        return {"auto_correct": actual == expected, "method": "choice_exact", "reason": f"expected={expected} actual={actual}"}
    if question_type == "numeric":
        normalized = gold["normalized_answer"]
        expected = normalized.get("value")
        tolerance = normalized.get("tolerance")
        actual = prediction.get("value")
        if expected is None or tolerance is None or actual is None:
            return {"auto_correct": None, "method": "manual_required", "reason": "Missing numeric value or tolerance"}
        acceptable = [normalized.get("canonical") or gold["answer"]] + list(
            normalized.get("aliases") or []
        )
        answer_text = normalize_text(prediction.get("answer") or "")
        if answer_text in {
            normalize_text(value) for value in acceptable
            if isinstance(value, str) and value.strip()
        }:
            return {
                "auto_correct": True,
                "method": "normalized_numeric_alias_exact",
                "reason": "Exact normalized match to a verified numeric answer or alias",
            }
        expected_unit = normalize_text(str(normalized.get("unit") or ""))
        actual_unit = normalize_text(str(prediction.get("unit") or ""))
        if expected_unit and actual_unit != expected_unit:
            return {
                "auto_correct": None,
                "method": "numeric_unit_review",
                "reason": f"Unit conversion or mismatch requires review: expected={expected_unit} actual={actual_unit}",
            }
        correct = abs(float(actual) - float(expected)) <= float(tolerance)
        return {"auto_correct": correct, "method": "numeric_tolerance", "reason": f"expected={expected} tolerance={tolerance} actual={actual}"}
    if question_type == "short_answer":
        normalized = gold["normalized_answer"]
        acceptable = [normalized.get("canonical") or gold["answer"]] + list(normalized.get("aliases") or [])
        actual = normalize_text(prediction.get("answer") or "")
        expected = {normalize_text(value) for value in acceptable if isinstance(value, str) and value.strip()}
        if actual in expected:
            return {
                "auto_correct": True,
                "method": "normalized_alias_exact",
                "reason": "Exact normalized match to a verified answer or alias",
            }
        return {
            "auto_correct": None,
            "method": "rubric_review",
            "reason": "Non-exact short answer requires blinded rubric review",
        }
    return {"auto_correct": None, "method": "rubric_or_symbolic_review", "reason": "Requires blinded rubric or equivalence review"}
