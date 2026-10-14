import importlib.util
from pathlib import Path
import unittest

from src.data.loading import NormalizedRecord


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


sampler = load_script("build_stage015_source_batch")
selector = load_script("select_stage015_calibration_qa")


def record(index, question, answer="brief", choices=None):
    return NormalizedRecord(
        source_file="astronomy/astronomy_train.jsonl",
        source_index=index,
        source_domain="astronomy",
        source_id=f"index:{index}",
        question=question,
        answer=answer,
        choices=choices or {},
        schema_fields=("question", "answer"),
    )


class Stage015SamplingTest(unittest.TestCase):
    def test_balanced_sampler_is_deterministic_and_excludes_old_source(self):
        records = [
            record(0, "What maximum limit constrains the system?"),
            record(1, "What spectrum is observed by the sensor?"),
            record(2, "Calculate the rate using this equation."),
            record(3, "Choose one.", choices={"A": "x", "B": "y", "C": "z"}),
            record(4, "What is chlorophyll?"),
            record(5, "What is another short fact?"),
        ]
        kwargs = {
            "records_by_domain": {"astronomy": records},
            "quotas": {"astronomy": {"mcq": 1, "likely_objective_open": 1, "non_mechanism_affordance": 3}},
            "excluded": {(records[5].source_file, records[5].source_index)},
            "batch_id": "test",
            "seed": 7,
            "relation_quotas": {
                "quantitative_modeling": 1,
                "constraint_coupling": 1,
                "measurement_observation": 1,
            },
        }
        first = sampler.build_batch(**kwargs)
        second = sampler.build_batch(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertNotIn(5, {item["source_index"] for item in first})
        self.assertEqual(
            {item["affordance_hint"] for item in first if item["affordance_hint"]},
            {"quantitative_modeling", "constraint_coupling", "measurement_observation"},
        )

    def test_selector_preserves_target_relation(self):
        sources = []
        traces = []
        for index, relation in enumerate(selector.NON_MECHANISM_RELATIONS):
            protocol_id = f"protocol-test-astronomy-{index}"
            sources.append(
                {
                    "protocol_id": protocol_id,
                    "source_domain": "astronomy",
                    "sampling_stratum": "non_mechanism_affordance",
                    "affordance_hint": relation,
                }
            )
            traces.append(
                {
                    "protocol_id": protocol_id,
                    "schema_valid": True,
                    "analysis": {
                        "scientifically_valid": True,
                        "reference_answer_role": "qualitative_reference",
                        "decision": "transform_candidate",
                    },
                }
            )
        selected = selector.select_manifest(sources, traces, objective_limit=0, per_relation_limit=1)
        self.assertEqual(len(selected), 3)
        self.assertEqual(
            {item["preferred_fusion_relation"] for item in selected},
            set(selector.NON_MECHANISM_RELATIONS),
        )


if __name__ == "__main__":
    unittest.main()
