import json
from pathlib import Path
import unittest

from src.evaluation.answer_scoring import score_prediction
from src.evaluation.baseline_runner import decode_object, run_arm, validate_route, visible_question
from src.generation.local_model import GenerationSettings
from src.prompts.loading import load_prompt_registry


class FakeModel:
    tokenizer = None

    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.messages = []

    def generate(self, messages, settings):
        self.messages.append(messages)
        return json.dumps(next(self.outputs))


class BaselineRunnerTest(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.prompts = json.loads((root / "src/prompts/qwen_baselines_v1.json").read_text())
        self.registry = load_prompt_registry()
        self.settings = GenerationSettings(max_new_tokens=32)
        self.record = {
            "id": "secret-biology-id",
            "question": "What is moving energy called?",
            "choices": {"A": "kinetic", "B": "potential"},
            "question_type": "mcq",
            "required_domains": ["physics"],
            "gold": {"correct_choice": "A", "answer": "A", "normalized_answer": {}},
            "source": {"original_answer": "hidden"},
        }

    def answer(self):
        return {"answer": "kinetic", "choice": "A", "value": None, "unit": None, "canonical_expression": None, "reasoning": ["motion"], "confidence": 0.9}

    def test_visible_question_is_strict_whitelist(self):
        self.assertEqual(set(visible_question(self.record)), {"question", "choices"})

    def test_router_rejects_duplicate_domains(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_route({"selected_domains": ["physics", "physics"], "rationale": "x"})

    def test_decode_repairs_unescaped_latex_backslashes(self):
        raw = r'{"answer":"x","canonical_expression":"x = 2 \times 3 \text{ m}"}'
        parsed = decode_object(raw)
        self.assertEqual(parsed["canonical_expression"], r"x = 2 \times 3 \text{ m}")

    def test_decode_preserves_valid_pairs_while_repairing_mixed_latex(self):
        raw = r'{"answer":"x","canonical_expression":"-\\frac{\\pi^2 \hbar c}{d^4}"}'
        parsed = decode_object(raw)
        self.assertEqual(parsed["canonical_expression"], r"-\frac{\pi^2 \hbar c}{d^4}")

    def test_direct_messages_hide_metadata(self):
        model = FakeModel([self.answer()])
        trace = run_arm(model, self.record, "direct", self.prompts, self.registry, self.settings, self.settings)
        serialized = json.dumps(model.messages)
        self.assertNotIn("required_domains", serialized)
        self.assertNotIn("secret-biology-id", serialized)
        self.assertEqual(trace["final_prediction"]["choice"], "A")

    def test_oracle_labels_only_control_calls(self):
        expert = {"relevant": True, "contribution": "Use kinetic energy.", "assumptions": [], "confidence": 0.9}
        model = FakeModel([expert, self.answer()])
        trace = run_arm(model, self.record, "oracle_multi_agent", self.prompts, self.registry, self.settings, self.settings)
        self.assertEqual(trace["routing"]["selected_domains"], ["physics"])
        for messages in model.messages:
            user = messages[-1]["content"]
            self.assertNotIn("required_domains", user)
            self.assertNotIn("secret-biology-id", user)

    def test_mcq_scoring(self):
        score = score_prediction(self.record, self.answer())
        self.assertTrue(score["auto_correct"])

    def test_non_exact_short_answer_is_deferred_to_review(self):
        record = {
            "question_type": "short_answer",
            "gold": {
                "answer": "photosynthesis",
                "normalized_answer": {"canonical": "photosynthesis", "aliases": []},
            },
        }
        prediction = {"answer": "The process is photosynthesis."}
        score = score_prediction(record, prediction)
        self.assertIsNone(score["auto_correct"])
        self.assertEqual(score["method"], "rubric_review")

    def test_numeric_unit_conversion_is_deferred_to_review(self):
        record = {
            "question_type": "numeric",
            "gold": {
                "answer": "5.2 mm",
                "normalized_answer": {
                    "canonical": "5.2 mm",
                    "value": 5.2,
                    "unit": "mm",
                    "tolerance": 0.1,
                    "aliases": ["0.0052 m"],
                },
            },
        }
        prediction = {"answer": "0.00519 m", "value": 0.00519, "unit": "m"}
        score = score_prediction(record, prediction)
        self.assertIsNone(score["auto_correct"])
        self.assertEqual(score["method"], "numeric_unit_review")


if __name__ == "__main__":
    unittest.main()
