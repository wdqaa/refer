"""Tests for deterministic pilot anchor selection."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.data.selection import select_pilot_anchors


class AnchorSelectionTest(unittest.TestCase):
    def test_selection_is_deterministic_and_excludes_train_files(self) -> None:
        plan = {"physics": {"real_anchored": 2, "synthetic": 0}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "physics"
            folder.mkdir()
            rows = [
                {"id": i, "question": f"Which physical process best explains the observed event number {i}?", "mcq_answer": "A", "choices": {"A": "x", "B": "y"}}
                for i in range(5)
            ]
            (folder / "physics_test.json").write_text(json.dumps(rows), encoding="utf-8")
            (folder / "physics_train.jsonl").write_text(
                '{"question":"Training record should never appear here","answer":"x"}\n',
                encoding="utf-8",
            )
            first = select_pilot_anchors(root, seed=9, plan=plan)
            second = select_pilot_anchors(root, seed=9, plan=plan)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 2)
            self.assertTrue(all("_train" not in item["source"]["dataset"] for item in first))


if __name__ == "__main__":
    unittest.main()
