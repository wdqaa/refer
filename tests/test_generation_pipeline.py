"""Tests for schema injection, serialization, and raw failure trace preservation."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.generation.local_model import GenerationSettings
from src.generation.question_pool_pipeline import read_jsonl, run_question_pool_pipeline


class FakeModel:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = iter(outputs)
        self.messages: list[list[dict[str, str]]] = []

    def generate(self, messages, settings):  # type: ignore[no-untyped-def]
        self.messages.append(messages)
        return next(self.outputs)


def _anchor() -> dict:
    return {
        "anchor_version": "pilot_anchor.v1",
        "id": "anchor_test",
        "requested_mode": "real_anchored",
        "source": {
            "dataset": "physics/test.json",
            "original_id": "1",
            "source_domain": "physics",
            "original_question": "What is energy?",
            "original_answer": "A",
            "original_choices": {"A": "capacity to do work"},
        },
    }


class GenerationPipelineTest(unittest.TestCase):
    def test_schema_is_sent_and_valid_record_is_written(self) -> None:
        candidate = {
            "source_analysis": {
                "scientifically_valid": True, "existing_required_domains": ["physics"],
                "already_cross_domain": False, "natural_extension": None,
            },
            "decision": "keep", "question": "What is energy?", "answer": "A",
            "answer_type": "multiple_choice", "choices": {"A": "capacity to do work"},
            "primary_domain": "physics", "required_domains": ["physics"],
            "difficulty": "easy", "reasoning_type": ["conceptual"],
            "derivation": ["Definition"], "transformation_summary": "Unchanged",
        }
        verification = {
            "scientific_validity": 5, "naturalness": 5, "fusion_necessity": 5,
            "answer_correctness": 5, "synthetic_smell": 1,
            "domain_necessity": [{"domain": "physics", "necessary": True, "reason": "Definition"}],
            "issues": [], "accepted": True,
        }
        model = FakeModel([json.dumps(candidate), json.dumps(verification)])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anchors = root / "anchors.jsonl"
            anchors.write_text(json.dumps(_anchor()) + "\n", encoding="utf-8")
            counts = run_question_pool_pipeline(
                model=model, anchors_path=anchors, output_dir=root / "out",
                model_id="test", model_path="/model", settings=GenerationSettings(), limit=1,
            )
            self.assertEqual(counts["written"], 1)
            self.assertIn("Required JSON fields", model.messages[0][0]["content"])
            self.assertTrue(read_jsonl(root / "out" / "candidates.jsonl")[0]["verification"]["accepted"])

    def test_invalid_repair_outputs_remain_in_trace(self) -> None:
        model = FakeModel(['{"bad": 1}', '{"still_bad": 2}'])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anchors = root / "anchors.jsonl"
            anchors.write_text(json.dumps(_anchor()) + "\n", encoding="utf-8")
            counts = run_question_pool_pipeline(
                model=model, anchors_path=anchors, output_dir=root / "out",
                model_id="test", model_path="/model", settings=GenerationSettings(), limit=1,
            )
            self.assertEqual(counts["failed"], 1)
            trace = read_jsonl(root / "out" / "raw_traces.jsonl")[0]
            self.assertEqual(trace["generator_outputs"], ['{"bad": 1}', '{"still_bad": 2}'])


if __name__ == "__main__":
    unittest.main()
