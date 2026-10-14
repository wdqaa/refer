"""Tests for question-pool leakage rules and canonical serialization."""

from __future__ import annotations

import unittest

from src.data.question_pool import (
    build_question_pool_record,
    explicit_fusion_leakage,
    validate_candidate,
    validate_verification,
)


class QuestionPoolTest(unittest.TestCase):
    def test_explicit_fusion_wording_is_detected(self) -> None:
        hits = explicit_fusion_leakage("Using physics and mathematics, calculate the orbit.")
        self.assertIn("using_named_domains", hits)

    def test_verifier_acceptance_is_overridden_by_rule_gate(self) -> None:
        anchor = {
            "id": "anchor_123",
            "requested_mode": "real_anchored",
            "source": {
                "dataset": "physics/test.json",
                "original_id": "1",
                "original_question": "Old?",
                "original_answer": "A",
                "original_choices": None,
            },
        }
        candidate = {
            "source_analysis": {
                "scientifically_valid": True,
                "existing_required_domains": ["physics"],
                "already_cross_domain": False,
                "natural_extension": "Add a mathematical calculation",
            },
            "decision": "transform",
            "question": "Using physics and mathematics, what is the value?",
            "answer": "1",
            "answer_type": "numeric",
            "choices": None,
            "primary_domain": "physics",
            "required_domains": ["physics", "mathematics"],
            "difficulty": "medium",
            "reasoning_type": ["quantitative"],
            "derivation": ["Compute it."],
            "transformation_summary": "Added a value.",
        }
        verification = {
            "scientific_validity": 5,
            "naturalness": 5,
            "fusion_necessity": 5,
            "answer_correctness": 5,
            "synthetic_smell": 1,
            "domain_necessity": [
                {"domain": "physics", "necessary": True, "reason": "Physical relation"},
                {"domain": "mathematics", "necessary": True, "reason": "Calculation"}
            ],
            "issues": [],
            "accepted": True,
        }
        record = build_question_pool_record(
            anchor=anchor,
            candidate=candidate,
            verification=verification,
            model_id="test",
            model_path="/model",
            settings={"temperature": 0.0, "top_p": 1.0, "max_tokens": 10, "seed": 1},
        )
        self.assertFalse(record["verification"]["accepted"])
        self.assertTrue(record["verification"]["explicit_fusion_leakage"])

    def test_transform_paraphrase_with_unchanged_solution_is_rejected(self) -> None:
        anchor = {
            "requested_mode": "real_anchored",
            "source": {
                "original_question": "Original question?", "original_answer": "A",
                "original_choices": {"A": "yes", "B": "no"},
            },
        }
        candidate = {
            "source_analysis": {
                "scientifically_valid": True,
                "existing_required_domains": ["physics"],
                "already_cross_domain": False,
                "natural_extension": "Add a calculation",
            },
            "decision": "transform", "question": "Rephrased question?", "answer": "A",
            "choices": {"A": "yes", "B": "no"}, "primary_domain": "physics",
            "required_domains": ["physics", "mathematics"],
        }
        with self.assertRaisesRegex(ValueError, "likely a paraphrase"):
            validate_candidate(anchor, candidate)

    def test_verifier_domains_must_match_candidate(self) -> None:
        candidate = {"required_domains": ["physics", "biology"]}
        verification = {
            "domain_necessity": [
                {"domain": "physics", "necessary": True, "reason": "x"},
                {"domain": "astronomy", "necessary": True, "reason": "y"},
            ]
        }
        with self.assertRaisesRegex(ValueError, "exactly match"):
            validate_verification(candidate, verification)


if __name__ == "__main__":
    unittest.main()
