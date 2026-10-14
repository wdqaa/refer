# Data Protocol

## Source policy

`datasets/science_dataset/mixed_dataset/` is immutable. Loaders normalize source
records in memory or write derived records under `datasets/processed/`. The mixed
training set is analyzed separately from domain source files to avoid double counts.

## Canonical question-pool record

Required fields are shown below. Nullable verification values are allowed before
verification; finalized accepted records require all verification axes.

```json
{
  "schema_version": "question_pool.v1",
  "id": "qp_000001",
  "source": {
    "type": "real",
    "dataset": "astronomy/astronomy_train.jsonl",
    "original_id": "line:1",
    "original_question": "...",
    "original_answer": "...",
    "original_choices": null
  },
  "question": "...",
  "answer": "...",
  "answer_type": "multiple_choice",
  "choices": {"A": "..."},
  "metadata": {
    "primary_domain": "astronomy",
    "required_domains": ["astronomy"],
    "difficulty": "easy",
    "reasoning_type": ["conceptual"],
    "derivation": ["..."]
  },
  "generation": {
    "generator_model": "Qwen/Qwen3-8B",
    "model_path": "/home/dqwang/Model/Qwen3-8B",
    "prompt_version": "question_generator_v2",
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 1024,
    "seed": 42,
    "timestamp": "2026-09-27T00:00:00+08:00",
    "source_analysis": {
      "scientifically_valid": true,
      "existing_required_domains": ["physics"],
      "already_cross_domain": false,
      "natural_extension": null
    }
  },
  "verification": {
    "scientific_validity": null,
    "naturalness": null,
    "fusion_necessity": null,
    "answer_correctness": null,
    "synthetic_smell": null,
    "domain_necessity": [],
    "explicit_fusion_leakage": [],
    "deterministic_gate_passed": null,
    "issues": [],
    "accepted": null,
    "verifier_model": null,
    "prompt_version": "question_verifier_v2"
  }
}
```

## Controlled values

- `source.type`: `real`, `transformed`, `synthetic`
- domains: `physics`, `chemistry`, `mathematics`, `biology`, `astronomy`
- `difficulty`: `easy`, `medium`, `hard`
- common reasoning types: `conceptual`, `quantitative`, `causal`, `comparison`

Geography may be retained as source provenance but is not initially an expert role.
Every required domain must be justified by a necessary reasoning step. Scenario or
vocabulary alone is insufficient.

For Stage 01A, the gold record stores a source pointer, one of six boundary
categories, and the analyzer JSON under `gold`. Required domains use evidence plus
an indispensability reason. Directory labels and source answers are not sent to the
model. The evaluator performs no repair or hidden label canonicalization.

`domain_necessity` is a list with one `{domain, necessary, reason}` item per claimed
required domain. Subfield aliases such as `astrophysics` and `developmental_biology`
are mapped to the five controlled domain labels rather than added as new labels.

## Leakage rule

Question text is checked for explicit fusion phrases. Hidden routing labels and
verification results must never enter planner or answer-model input.
