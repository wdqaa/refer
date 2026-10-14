"""Tests for deterministic source-analysis summaries."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from src.data.analysis import analyze_dataset, length_summary


class DataAnalysisTest(unittest.TestCase):
    def test_length_summary(self) -> None:
        self.assertEqual(length_summary([1, 2, 3, 4])["median"], 2.5)
        self.assertEqual(length_summary([])["count"], 0)

    def test_mixed_is_excluded_from_non_mixed_total(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for domain in ("physics", "mixed"):
                folder = root / domain
                folder.mkdir()
                (folder / "data.jsonl").write_text(
                    '{"question":"What is this?","answer":"x"}\n', encoding="utf-8"
                )

            report = analyze_dataset(root, examples_per_file=0)
            self.assertEqual(report["totals"]["non_mixed_total"], 1)
            self.assertEqual(report["totals"]["mixed_total"], 1)
            self.assertEqual(report["totals"]["all_records_including_mixed"], 2)

    def test_hidden_configuration_directory_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            domain = root / "physics"
            domain.mkdir()
            (domain / "data.jsonl").write_text(
                '{"question":"What is this?","answer":"x"}\n', encoding="utf-8"
            )
            hidden = root / ".claude"
            hidden.mkdir()
            (hidden / "settings.json").write_text('{"not":"data"}\n', encoding="utf-8")

            report = analyze_dataset(root, examples_per_file=0)
            self.assertEqual(report["totals"]["files"], 1)


if __name__ == "__main__":
    unittest.main()
