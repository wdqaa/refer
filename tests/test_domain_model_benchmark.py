import json
from pathlib import Path
import unittest

from src.evaluation.domain_model_benchmark import (
    parse_benchmark_answer,
    run_domain_model_batch,
    run_domain_model_item,
    visible_question,
)
from src.generation.local_model import GenerationSettings, TransformersCompletionModel
from scripts.judge_astronomy_domain_answers import gold_payload


ROOT = Path(__file__).resolve().parents[1]


class _Tokenizer:
    @staticmethod
    def encode(text, add_special_tokens=False):
        return text.split()


class _FakeModel:
    tokenizer = _Tokenizer()

    def __init__(self, output):
        self.output = output

    @staticmethod
    def render_messages(messages, enable_thinking=False):
        return "\n".join(row["content"] for row in messages)

    @staticmethod
    def count_input_tokens(messages, enable_thinking=False):
        return sum(len(row["content"].split()) for row in messages)

    def generate(self, messages, settings):
        return self.output

    def generate_batch(self, conversations, settings):
        return [self.output for _ in conversations]


class DomainModelBenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prompt = json.loads(
            (ROOT / "src/prompts/astronomy_domain_benchmark_v1.json").read_text()
        )
        cls.record = {
            "native_qa_id": "astro-1",
            "question": "Which option is correct?",
            "choices": {"A": "first", "B": "second"},
            "answer": "B",
            "required_domains": ["astronomy"],
            "benchmark_freeze": {"evaluation_only": True},
        }

    def test_visible_question_hides_gold_and_metadata(self):
        self.assertEqual(
            visible_question(self.record),
            {"question": "Which option is correct?", "choices": {"A": "first", "B": "second"}},
        )

    def test_plain_text_mcq_fallback_requires_explicit_choice(self):
        parsed, mode, warning = parse_benchmark_answer(
            "The correct option is B because it matches the observation.",
            self.record["choices"],
        )
        self.assertEqual(mode, "plain_text_fallback")
        self.assertEqual(parsed["choice"], "B")
        self.assertIsNotNone(warning)
        ambiguous, _, _ = parse_benchmark_answer(
            "The second statement matches the observation.", self.record["choices"]
        )
        self.assertIsNone(ambiguous["choice"])

    def test_json_answer_uses_shared_schema(self):
        value = {
            "answer": "second",
            "choice": "B",
            "value": None,
            "unit": None,
            "canonical_expression": None,
            "reasoning": ["observation"],
            "confidence": 0.8,
        }
        parsed, mode, warning = parse_benchmark_answer(json.dumps(value), self.record["choices"])
        self.assertEqual(parsed, value)
        self.assertEqual(mode, "json")
        self.assertIsNone(warning)

    def test_partial_json_preserves_explicit_choice(self):
        raw = (
            '{"answer":"B","choice":"B","reasoning":["dark matter is about 27 percent"],'
            '"confidence":1.0}\nAdditional prose after the object.'
        )
        parsed, mode, warning = parse_benchmark_answer(raw, self.record["choices"])
        self.assertEqual(mode, "partial_json_normalized")
        self.assertEqual(parsed["answer"], "B")
        self.assertEqual(parsed["choice"], "B")
        self.assertIsNone(parsed["value"])
        self.assertIsNotNone(warning)

    def test_truncated_json_fallback_preserves_explicit_choice(self):
        raw = '{"answer":"second","choice":"B","reasoning":["observation"'
        parsed, mode, warning = parse_benchmark_answer(raw, self.record["choices"])
        self.assertEqual(mode, "plain_text_fallback")
        self.assertEqual(parsed["choice"], "B")
        self.assertIsNotNone(warning)

    def test_trace_contains_no_hidden_fields_in_prompt(self):
        model = _FakeModel("Final answer: B")
        trace = run_domain_model_item(
            model,
            self.record,
            prompt=self.prompt,
            settings=GenerationSettings(max_new_tokens=32),
            model_id="test/model",
            prompt_renderer="plain_completion",
        )
        serialized = json.dumps(trace["messages"])
        self.assertNotIn("required_domains", serialized)
        self.assertNotIn("benchmark_freeze", serialized)
        self.assertNotIn('"answer": "B"', serialized)
        self.assertEqual(trace["final_prediction"]["choice"], "B")

    def test_batch_trace_preserves_one_isolated_prompt_per_item(self):
        model = _FakeModel("Final answer: B")
        records = [self.record, {**self.record, "native_qa_id": "astro-2"}]
        traces = run_domain_model_batch(
            model,
            records,
            prompt=self.prompt,
            settings=GenerationSettings(max_new_tokens=32),
            model_id="test/model",
            prompt_renderer="plain_completion",
        )
        self.assertEqual([row["item_id"] for row in traces], ["astro-1", "astro-2"])
        self.assertTrue(all(row["batch_size"] == 2 for row in traces))
        self.assertTrue(all(row["final_prediction"]["choice"] == "B" for row in traces))

    def test_plain_completion_renderer_is_explicit(self):
        backend = TransformersCompletionModel.__new__(TransformersCompletionModel)
        rendered = backend.render_messages(
            [
                {"role": "system", "content": "Follow the instructions."},
                {"role": "user", "content": "What is a star?"},
            ]
        )
        self.assertIn("Instructions:\nFollow the instructions.", rendered)
        self.assertIn("Question input:\nWhat is a star?", rendered)
        self.assertTrue(rendered.endswith("Response:\n"))
        with self.assertRaisesRegex(ValueError, "thinking"):
            backend.render_messages([], enable_thinking=True)

    def test_short_answer_may_use_one_question_entailed_criterion(self):
        record = {
            "native_qa_id": "astro-short-1",
            "question_type": "short_answer",
            "gold_enrichment": {
                "accepted": True,
                "gold_answer": {"correct_choice": None, "answer_text": "spiral galaxy"},
                "standard_solution": ["A rotating stellar disk is a spiral galaxy."],
                "scoring_rubric": [
                    {"criterion": "Answers spiral galaxy.", "points": 10}
                ],
            },
        }
        payload = gold_payload(record)
        self.assertEqual(len(payload["rubric"]), 1)

    def test_qualitative_still_requires_three_criteria(self):
        record = {
            "native_qa_id": "astro-qual-1",
            "question_type": "qualitative",
            "gold_enrichment": {
                "accepted": True,
                "gold_answer": {"correct_choice": None, "answer_text": "answer"},
                "standard_solution": ["step"],
                "scoring_rubric": [
                    {"criterion": "Only one criterion.", "points": 10}
                ],
            },
        }
        with self.assertRaisesRegex(ValueError, "3-5"):
            gold_payload(record)


if __name__ == "__main__":
    unittest.main()
