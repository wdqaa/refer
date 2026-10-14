"""Lightweight tests for heterogeneous source loading and normalization."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.data.loading import DatasetFormatError, iter_normalized_records


class DataLoadingTest(unittest.TestCase):
    def test_jsonl_prompt_response_is_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            domain = root / "physics"
            domain.mkdir()
            path = domain / "sample.jsonl"
            record = {
                "prompt": (
                    "<|start_header_id|>user<|end_header_id|>\n"
                    "Solve the following physics problem.\n\nWhat is force?"
                    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
                ),
                "response": "mass times acceleration",
            }
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            result = list(iter_normalized_records(path, root=root))
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].question, "What is force?")
            self.assertEqual(result[0].answer, "mass times acceleration")
            self.assertEqual(result[0].source_domain, "physics")

    def test_json_array_stem_choices_is_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            domain = root / "chemistry"
            domain.mkdir()
            path = domain / "sample.json"
            path.write_text(
                json.dumps(
                    [{"id": 7, "stem": "Pick one", "choices": ["x", "y"], "answer": "B"}]
                ),
                encoding="utf-8",
            )

            result = list(iter_normalized_records(path, root=root))[0]
            self.assertEqual(result.source_id, "7")
            self.assertEqual(result.choices, {"A": "x", "B": "y"})

    def test_inline_prompt_choices_are_extracted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            domain = root / "astronomy"
            domain.mkdir()
            path = domain / "sample.jsonl"
            record = {
                "prompt": (
                    "<|start_header_id|>user<|end_header_id|>\n"
                    "Solve the following astronomy problem.\n\nWhich object?\n\n"
                    "Choices:\nA. A star\nB. A planet\n\n"
                    "Please answer with the correct option letter.<|eot_id|>"
                ),
                "response": "B",
            }
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            result = list(iter_normalized_records(path, root=root))[0]
            self.assertEqual(result.question, "Which object?")
            self.assertEqual(result.choices, {"A": "A star", "B": "A planet"})

    def test_missing_answer_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            domain = root / "math"
            domain.mkdir()
            path = domain / "sample.jsonl"
            path.write_text('{"question": "Incomplete"}\n', encoding="utf-8")

            with self.assertRaisesRegex(DatasetFormatError, "No answer"):
                list(iter_normalized_records(path, root=root))


if __name__ == "__main__":
    unittest.main()
