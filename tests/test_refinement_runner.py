import json
from pathlib import Path
import unittest

from src.evaluation.refinement_runner import run_refinement_arm, validate_critique
from src.generation.local_model import GenerationSettings


ROOT = Path(__file__).resolve().parents[1]


class _Tokenizer:
    @staticmethod
    def encode(text):
        return text.split()


class _FakeModel:
    tokenizer = _Tokenizer()

    @staticmethod
    def count_input_tokens(messages, enable_thinking=False):
        return sum(len(message["content"].split()) for message in messages)

    @staticmethod
    def generate(messages, settings):
        if '"material_errors"' in messages[0]["content"]:
            return json.dumps(
                {
                    "material_errors": [],
                    "missing_points": [],
                    "reasoning_issues": [],
                    "preserve": ["draft"],
                    "recommend_change": False,
                    "confidence": 0.8,
                }
            )
        return json.dumps(
            {
                "answer": "Revised answer",
                "choice": None,
                "value": None,
                "unit": None,
                "canonical_expression": None,
                "reasoning": ["Checked the draft"],
                "confidence": 0.8,
            }
        )


class RefinementRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prompts = json.loads(
            (ROOT / "src/prompts/qwen_refinement_v1.json").read_text(encoding="utf-8")
        )
        cls.settings = GenerationSettings(max_new_tokens=32, seed=42, enable_thinking=False)
        cls.draft = {
            "answer": "Draft answer",
            "choice": None,
            "value": None,
            "unit": None,
            "canonical_expression": None,
            "reasoning": ["Draft reasoning"],
            "confidence": 0.5,
        }
        cls.record = {
            "id": "item-1",
            "question": "Explain the mechanism.",
            "choices": None,
            "required_domains": ["physics", "chemistry"],
            "gold": {"answer": "HIDDEN_GOLD"},
        }

    def test_critique_schema(self):
        value = {
            "material_errors": [],
            "missing_points": [],
            "reasoning_issues": [],
            "preserve": [],
            "recommend_change": False,
            "confidence": 1.0,
        }
        self.assertEqual(validate_critique(value), value)

    def test_matched_new_call_count_and_field_isolation(self):
        outputs = {}
        for arm in ("expert_critique", "self_refine"):
            outputs[arm] = run_refinement_arm(
                _FakeModel(),
                self.record,
                arm=arm,
                direct_prediction=self.draft,
                selected_domains=["physics", "chemistry"],
                routing_rationale="visible-question route",
                prompts=self.prompts,
                critic_settings=self.settings,
                final_settings=self.settings,
            )
            self.assertEqual(len(outputs[arm]["calls"]), 3)
            serialized = json.dumps(outputs[arm]["calls"])
            self.assertNotIn("HIDDEN_GOLD", serialized)
            self.assertNotIn("required_domains", serialized)
        self.assertEqual(outputs["self_refine"]["routing"]["selected_domains"], [])
        self.assertEqual(outputs["expert_critique"]["routing"]["selected_domains"], ["physics", "chemistry"])

    def test_zero_route_uses_only_shared_reviser(self):
        trace = run_refinement_arm(
            _FakeModel(),
            self.record,
            arm="self_refine",
            direct_prediction=self.draft,
            selected_domains=[],
            routing_rationale=None,
            prompts=self.prompts,
            critic_settings=self.settings,
            final_settings=self.settings,
        )
        self.assertEqual(len(trace["calls"]), 1)
        self.assertEqual(trace["calls"][0]["role"], "draft_reviser")

    def test_prompt_and_adjudication_versions_are_well_formed(self):
        self.assertEqual(self.prompts["id"], "qwen_refinement_v1")
        self.assertEqual(set(self.prompts["expert_critic"]["domain_guidance"]), {
            "physics", "chemistry", "biology", "astronomy", "geography", "mathematics"
        })
        adjudications = json.loads(
            (ROOT / "configs/evaluation/native_eval48_refinement_adjudications_v1.json").read_text(
                encoding="utf-8"
            )
        )
        coverage_keys = [
            (row["item_id"], row["arm"], row["criterion_id"])
            for row in adjudications["coverage_overrides"]
        ]
        error_keys = [
            (row["item_id"], row["arm"], row["error_id"])
            for row in adjudications["scientific_errors"]
        ]
        self.assertEqual(len(coverage_keys), len(set(coverage_keys)))
        self.assertEqual(len(error_keys), len(set(error_keys)))
        self.assertTrue(all(key[1] in {"expert_critique", "self_refine"} for key in coverage_keys))
        self.assertTrue(all(key[1] in {"expert_critique", "self_refine"} for key in error_keys))


if __name__ == "__main__":
    unittest.main()
