"""Tests for strict agent JSON output parsing."""

from __future__ import annotations

import json
import unittest

from src.agents.parsing import AgentOutputError, parse_agent_output


class AgentParsingTest(unittest.TestCase):
    def test_valid_planner_call(self) -> None:
        payload = {
            "action": "call_expert",
            "domain": "physics",
            "query": "Check the energy balance.",
            "rationale": "The answer depends on a conservation law.",
            "final_answer": None,
        }
        self.assertEqual(parse_agent_output("planner", json.dumps(payload)), payload)

    def test_planner_accepts_geography(self) -> None:
        payload = {
            "action": "call_expert",
            "domain": "geography",
            "query": "Check the spatial pattern.",
            "rationale": "The conclusion depends on GIS interpretation.",
        }
        self.assertEqual(parse_agent_output("planner", json.dumps(payload)), payload)

    def test_planner_rejects_unknown_domain(self) -> None:
        payload = {
            "action": "call_expert",
            "domain": "earth_science",
            "query": "Help",
            "rationale": "Needed",
        }
        with self.assertRaisesRegex(AgentOutputError, "supported domain"):
            parse_agent_output("planner", json.dumps(payload))

    def test_fenced_expert_json_is_accepted(self) -> None:
        text = """```json
        {"relevant": false, "contribution": "Not needed.", "assumptions": [], "confidence": 0.9}
        ```"""
        result = parse_agent_output("chemistry_expert", text)
        self.assertFalse(result["relevant"])

    def test_verifier_rejects_out_of_range_score(self) -> None:
        payload = {
            "scientific_validity": 6,
            "naturalness": 4,
            "fusion_necessity": 4,
            "answer_correctness": 4,
            "synthetic_smell": 1,
            "domain_necessity": [],
            "issues": [],
            "accepted": True,
        }
        with self.assertRaisesRegex(AgentOutputError, "integer from 1 to 5"):
            parse_agent_output("question_verifier", json.dumps(payload))

    def test_generator_rejects_scalar_reasoning_type(self) -> None:
        payload = {
            "source_analysis": {
                "scientifically_valid": True,
                "existing_required_domains": ["physics"],
                "already_cross_domain": False,
                "natural_extension": "Add a calculation",
            },
            "decision": "transform",
            "question": "Question",
            "answer": "Answer",
            "answer_type": "short_answer",
            "choices": None,
            "primary_domain": "physics",
            "required_domains": ["physics"],
            "difficulty": "easy",
            "reasoning_type": "conceptual",
            "derivation": ["Step"],
            "transformation_summary": "None",
        }
        with self.assertRaisesRegex(AgentOutputError, "reasoning_type"):
            parse_agent_output("question_generator", json.dumps(payload))


if __name__ == "__main__":
    unittest.main()
