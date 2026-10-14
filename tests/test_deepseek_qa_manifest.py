import importlib.util
from pathlib import Path
import unittest


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


selector = load_script("build_deepseek_qa_manifest")


class DeepSeekQAManifestTest(unittest.TestCase):
    def source(self):
        return {"protocol_id": "protocol-v1-biology-1", "question": "Q"}

    def analysis(self, decision="transform_candidate"):
        return {
            "protocol_id": "protocol-v1-biology-1",
            "schema_valid": True,
            "analysis": {
                "scientifically_valid": True,
                "decision": decision,
                "reference_answer_role": "qualitative_reference",
            },
        }

    def test_builds_requested_manifest(self):
        records = selector.build_manifest(
            [self.source()], [self.analysis()], ["protocol-v1-biology-1"]
        )
        self.assertEqual(records[0]["route"], "transform_candidate")
        self.assertEqual(records[0]["qa_id"], "qa-validation-v2-biology-1")

    def test_rejects_source_analyzer_reject(self):
        with self.assertRaisesRegex(ValueError, "non-production"):
            selector.build_manifest(
                [self.source()], [self.analysis("reject")], ["protocol-v1-biology-1"]
            )


if __name__ == "__main__":
    unittest.main()
