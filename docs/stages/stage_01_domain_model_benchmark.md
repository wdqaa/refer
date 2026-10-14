# Stage 01.5 Domain-Model Benchmark

## Question

Does a real astronomy-specialized checkpoint provide complementary scientific
knowledge beyond Qwen3-8B, especially on items Qwen answers incorrectly?

This is a single-model capability comparison. It does not modify or invoke the
Multi-Agent system. A critique/revision experiment is allowed only if this first
stage finds a useful AstroSage-only correctness slice.

## Frozen astronomy set

astronomy_domain_eval30.v1 contains exactly 30 unchanged, previously reviewed
pilot-500 records whose minimum required domains include astronomy and whose source
identities do not occur in native eval50:

- 16 MCQ, 12 qualitative, and 2 short-answer questions;
- 29 single-domain astronomy items and one astronomy-plus-physics item;
- 28 high-priority and two medium-priority items.

The set is stored in
datasets/processed/domain_model_benchmark_v1/astronomy_eval30/source_items.jsonl.
All 30 records are now evaluation-only and must be excluded from SFT, cold-start, RL,
prompt tuning, and training-data exports.
The historical train_candidates_164.jsonl is preserved for provenance rather than
overwritten. Future training exports must subtract training_exclusions.jsonl, leaving
134 currently eligible records.

Gold audit reduced this source pool to the evaluation-ready astronomy eval24:

- 11 MCQ, 12 qualitative, and one short-answer item;
- 23 single-domain items and one native cross-domain item;
- 16 Gold records accepted unchanged and eight accepted with explicit edits.

Six records were excluded rather than repaired: one internally inconsistent helium-
burning calculation, the non-unique Venus rotation question, a Blandford-Znajek stem
that conflated disk and black-hole rotational energy, a non-unique CME mechanism MCQ,
and two unsupported planet-orbit/gravitational-wave MCQs. These six remain excluded
from training as well as evaluation.

## Gold gate

The original MCQ key is sufficient for preliminary exact-choice scoring, but every
item must receive a scientific Gold audit before final reporting. The 14 non-MCQ
items additionally need a compact standard solution and a fixed 3--5 point core
rubric. Gold is produced before inspecting model outputs:

1. Run the existing non-thinking DeepSeek Gold enricher in six batches of five.
2. Codex audits all 30 records against the visible question and reference answer.
3. Any correction is recorded in a versioned review policy.
4. Freeze astronomy_eval30_gold_curated.jsonl; do not tune prompts afterward.

Gold was generated with:

    .venv/bin/python scripts/enrich_native_qa_gold.py \
      --input datasets/processed/domain_model_benchmark_v1/astronomy_eval30/source_items.jsonl \
      --output-dir datasets/processed/domain_model_benchmark_v1/astronomy_eval30 \
      --prompt src/prompts/astronomy_eval_gold_enricher_v1.json \
      --artifact-prefix astronomy_eval30 --workers 3

All six batches returned without API error. Usage was 13,535 prompt and 10,345
completion tokens, costing about USD 0.00812 off peak or USD 0.01625 peak. DeepSeek
produced 29 schema-valid enrichments and one deterministic schema failure. Human audit
then retained 24 records, explicitly edited eight, rejected five, and separately
rejected the failed item. No model answer was generated or inspected before this Gold
freeze.

## Inference protocol

Both models receive the same visible question, choices, semantic instruction, greedy
decoding, seed 42, and 1024-new-token cap. Gold, domains, source identity, and model
labels are absent from the prompt.

- Qwen3-8B uses its released tokenizer chat template with enable_thinking=false.
- AstroSage-8B uses a plain completion prompt. Its tokenizer has no chat_template,
  and the released model card demonstrates direct tokenizer(prompt) generation.
- A plain-text fallback parser preserves scientifically usable answers when a model
  fails the requested JSON format. For MCQ it accepts only an explicit answer/option/
  final-answer letter; it never infers a choice from free prose.

Run each checkpoint once, preferably in parallel on separate GPUs:

    .venv/bin/python scripts/evaluate_astronomy_domain_models.py \
      --model qwen3_8b --device cuda:0

    .venv/bin/python scripts/evaluate_astronomy_domain_models.py \
      --model astrosage_8b --device cuda:1

The runner is append-only and treats every recorded item, including a parse failure,
as completed. This prevents silent answer-quality retries.

## Evaluation ownership

MCQ is deterministic exact-choice scoring. Non-MCQ evaluation uses a fixed audited
rubric:

1. DeepSeek performs a blinded first pass, seeing anonymous candidates A and B in a
   deterministic shuffled order. It marks criterion coverage and separate scientific
   errors. Model names are not sent.
2. Codex reviews every Gold/rubric problem, A/B coverage disagreement, material-error
   claim, and low-confidence judgment without using overall model reputation.
3. The reviewed rows are marked review_status=codex_audited.
4. Final scoring refuses unaudited judgments unless an explicit preliminary flag is
   used.

DeepSeek therefore supplies scalable semantic matching, not the final experimental
verdict. It must not create a rubric after seeing candidate answers.

Planned blind-judge call after both traces exist (13 API calls, one per non-MCQ item):

    .venv/bin/python scripts/judge_astronomy_domain_answers.py \
      --dataset datasets/processed/domain_model_benchmark_v1/astronomy_eval30/astronomy_eval30_gold_curated.jsonl \
      --trace qwen3_8b=outputs/domain_model_benchmark_v1/astronomy_eval30/qwen3_8b/traces.jsonl \
      --trace astrosage_8b=outputs/domain_model_benchmark_v1/astronomy_eval30/astrosage_8b/traces.jsonl \
      --output-dir outputs/domain_model_benchmark_v1/astronomy_eval30/evaluation

## Reported metrics

- MCQ correct/total and accuracy;
- non-MCQ mean weighted rubric coverage;
- material scientific-error responses and claims;
- actual input/output tokens and latency;
- both-correct, Qwen-only, AstroSage-only, and both-wrong quadrants;
- per-item AstroSage-minus-Qwen rubric-coverage changes.

For a qualitative or short-answer quadrant only, strict correct means 100% rubric
coverage with no material scientific error. Coverage remains the primary non-MCQ
metric.

## Current status

- The 30-item source pool is preserved and its 24-item curated Gold benchmark is frozen.
- Qwen3-8B and AstroSage-8B each completed all 24 records once. No answering call was
  retried or repaired.
- Both scored 10/11 MCQ. Audited non-MCQ mean rubric coverage was 45.38% for Qwen and
  38.46% for AstroSage; material-error responses were 4 and 6 respectively.
- There were zero AstroSage-only strict-correct items. AstroSage's two coverage gains
  both retained material errors, while seven non-MCQ answers degraded and four tied.
- AstroSage used 22,942 output tokens and 1,064.34 summed generation seconds versus
  Qwen's 5,216 tokens and 347.65 seconds.
- AstroSage is fully downloaded: all four indexed safetensors shards are present and
  all 291 declared parameter keys match their stored shard contents.
- Blind-Judge v1 failed because the runner omitted the prompt's existing JSON schema;
  all failure traces remain archived. The corrected v2 used 26,227 prompt and 9,278
  completion tokens, and every judgment was audited through the versioned adjudication
  file before final scoring.
- Gate decision: no-go. Do not proceed to AstroSage critique integration on this set.

## Interpretation boundary

This experiment compares released systems, not domain training in isolation.
AstroSage is Llama-3.1-based and Qwen3-8B is a different model generation; their
documented prompt interfaces also differ. A useful AstroSage-only slice supports using
it as a specialist. It does not by itself prove that astronomy continued pretraining
caused the gain.
