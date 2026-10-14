# Project Status

## Current Stage

Stage 01.5 -- parallel verified-data expansion and prompt Multi-Agent go/no-go gate.

## Current Goal

Hold the verified 49-record QA seed and baseline-v2 prompts fixed. Synthetic
cross-domain generation is closed. The active data route is lightweight mining of
unchanged native single-domain and native cross-domain QA. The 500-item staged pilot is
complete: v1.3 is useful for broad candidate filtering, but autonomous cross-domain
precision is too low for a full scan. A balanced unseen 50-item split now has
human-reviewed canonical answers, standard solutions, and scoring rubrics. Frozen
baseline v2 has now been evaluated on it. A separate audit of the 32 qualitative
rubrics recovers a modest Multi-Agent coverage advantage, but the earlier strict Oracle
gain still did not reproduce meaningfully. A fixed Expert-Critique versus matched
Self-Refine diagnostic is also complete: both reach 75.52% qualitative coverage, while
Expert loses one MCQ and reduces one additional error response. No cold-start or RL is
active. The next bounded gate is a direct Qwen3-8B versus AstroSage-8B astronomy
benchmark. Its scripts are ready and Gold audit reduced the 30-item evaluation-only
source pool to 24 reliable records; model inference has not started.

## Completed

- Inspected the initial repository and source-data layout.
- Confirmed that the directory is not currently a Git repository.
- Identified 20 source data files across six domain directories plus one mixed set.
- Identified heterogeneous JSON/JSONL schemas and a duplicate-counting risk from
  `mixed/mixed_train.jsonl`.
- Created the Stage 00 directory and documentation structure.
- Added a schema-aware loader and reproducible source-analysis command with tests.
- Counted 180,029 non-mixed source records and proved that the 158,985-record
  `mixed_train.jsonl` is an exact concatenation of the six domain training files.
- Wrote per-file schemas, length summaries, type/difficulty proxies, and bounded
  examples to `outputs/data_analysis/`.
- Defined the current ten roles in one versioned prompt registry, bound them to
  the shared local Qwen3-8B configuration, and added strict JSON output validation.
- Confirmed the local model path and successfully loaded Qwen3-8B for bounded tests.
- Built a deterministic 20-anchor manifest, resumable generator/verifier pipeline,
  raw trace preservation, explicit leakage checks, and deterministic acceptance gates.
- Created an isolated Qwen3-compatible runtime and ran single-item plus five-item
  prompt-generator pilots on one RTX 3090.
- Preserved prompt v1/v2 negative results and split source analysis from generation.
- Built a 30-item gold protocol with six boundary categories, five records each.
- Added Source Analyzer v3, four boundary few-shots, a no-repair evaluator, and two
  evaluation tests.
- Ran the audited 30-item evaluation: domain exact match 90%, cross-domain precision
  100%, cross-domain recall 80%, and zero math/subfield false positives.
- Ran a five-item DeepSeek Flash transformation pilot. Four survived Codex protocol
  review; one parallel-observation conjunction was rejected. None is final gold yet.
- Tested the four retained items with Qwen3-8B. Choice accuracy was 75% direct and
  100% fixed multi-agent, but strict reasoning accuracy was only 25% for both.
- A thinking-plus-Critic smoke showed that Qwen can internally solve the hard QCD
  item, but thinking exhausts the output budget before JSON and the same-model Critic
  approves a dimensionally invalid draft.
- Replaced mathematics with geography only in the Stage 01 source-data pool. The
  current math source data remains excluded, but mathematics is a callable zero-shot
  prompt expert alongside the five trained-data domains.
- Added Source Analyzer v2.1 with separate minimum-required and optional-helpful domains;
  independent mathematical methods are retained rather than rejected.
- Ran a five-item v2.1 boundary smoke: 5/5 schema-valid; astronomy/physics over-splitting
  decreased, while two math-dependent but underspecified questions remained rejected
  for missing data rather than for using mathematics.
- Completed the 30-item DeepSeek QA pilot with isolated blind solving and judging.
  Machine gates accepted 25; the original semantic review retained 17 records before
  prompt-v2 regression removed one additional false-boundary transformation.
- Built a 150-source protocol batch and obtained 150/150 valid DeepSeek source
  analyses covering MCQ, objective open, and qualitative open questions.
- Validated Transformer/Judge v2 on 12 frozen records and six fresh transformations:
  all eight known failures were rejected, three of four boundary positives remained,
  and two of six fresh transformations passed.
- The corrected first-pilot curated seed now contains 16 records.
- Prepared and executed a 78-source second-batch manifest: 47 controls, nine native
  cross-domain items, and 22 transform candidates.
- Completed the 78-source second batch: 37 machine accepts and 31 semantic-review
  accepts (18 native single, 5 native cross-domain, 8 transformed).
- Exported a unified 49-record verified QA set from the 16-record pilot seed, two
  validation records, and 31 second-batch records.
- Implemented prompt-only Direct, question-routed Multi-Agent, and oracle-domain
  Multi-Agent paths with strict model-visible field isolation and separate scoring.
- Completed a three-question Qwen3-8B baseline smoke: all nine arm-item traces parsed,
  no `required_domains` label entered model messages, and call/token/latency data were
  retained. This smoke is not a go/no-go accuracy result.
- Built the first fresh 50-source expansion calibration wave without API calls: 20
  astronomy sources, 35 objective-surface sources, and 15 non-mechanism-affordance
  sources balanced five each across three target relations.
- Completed the full 49-item three-arm Qwen3-8B baseline after GPU capacity returned.
  Strict complete-rubric accuracy was 17/49 Direct, 16/49 routed, and 16/49 Oracle;
  no hidden labels entered model messages. Baseline v1 therefore fails the Oracle
  collaboration gate and does not authorize Planner training.
- Completed the fixed-data v2 answer-completeness ablation without changing the
  Router, decoding, Gold, evaluator, or answer schema. Strict accuracy was 16/49
  Direct, 20/49 routed, and 20/49 Oracle. Oracle rescued five Direct failures and
  damaged one Direct success, a paired gain of 4/49; qualitative accuracy remained
  low at 2/25 for Oracle. No hidden labels entered model messages, and all 50
  repository tests pass.
- Ran the frozen 50-source Stage 01.5 data calibration wave. Source Analyzer produced
  49 valid analyses; one time-sensitive PBH item exhausted 6000 and 10000 reasoning
  tokens without visible JSON and remains a preserved failure. Strict offline selection
  yielded only 14 QA candidates, so no quota was force-filled.
- Completed the bounded 14-item QA pass with Transformer/Judge v2: five machine
  accepts became four semantically curated native-single QA records. The sole
  machine-accepted non-mechanism transformation was rejected during semantic review
  for falsely splitting astronomy from physics. The required two-transform calibration
  gate therefore failed, and these four records remain separate from the frozen 49.
- Implemented `source_transformability_selector_v1` for quantitative modeling,
  constraint coupling, and measurement inference. Its strict schema detects nine
  explicit source risks, including supplied answers and false mature-field boundaries.
- Built a deterministic keyword-free 150-source candidate pool that excludes all 200
  previously sampled sources. It contains 40 astronomy, 28 biology, 28 chemistry,
  27 physics, and 27 geography records, split into 44 MCQ and 106 open questions.
- Added a non-thinking, one-request-per-source Selector runner and a strict human-review
  shortlist gate.
- Ran the Selector once on all 150 sources: 137 schema-valid results, 13 preserved
  invalid outputs, and no API errors. Valid results were 4 high, 55 medium, and 78 low.
  Only one of four strict-high records survived source-level semantic review; the other
  three reflected mature-field boundaries or a noisy directory label. No records were
  sent to the Transformer and no QA were added.
- Audited all 15 clean-medium records offline to test whether the strict-high gate was
  hiding useful sources. Three survived; combined with the strict-high review, four of
  19 sources are credible. This improves recall but remains too small for the planned
  10--15 source QA calibration, so no downstream call was started.
- Implemented a 49-entry interdisciplinary bridge library and a 25-source bridge-guided
  calibration without retrieval, Top-k matching, knowledge signatures, or scoring.
- Bridge Selector v1 returned zero selections among 24 valid results. A single v1.1
  clarification was tested on four human-positive and four boundary-negative sources;
  it again selected zero bridges. The all-NONE policy rejected every positive together
  with the intended negatives, so the pipeline stopped before Transformer/Solver/Judge
  and added zero QA.
- Bypassed Bridge Selector for one final eight-pair human-specified bridge smoke. All
  four positive pairs were transformed, but zero passed machine gates or semantic
  review. Of four negatives, one was correctly rejected, two became pseudo-fusions,
  and one returned malformed JSON. The final accepted cross-domain QA count is zero,
  so synthetic single-source transformation is closed as a no-go route.
- Scanned the read-only source universe for native QA mining: 180,029 non-mixed records
  across 19 files become 178,814 exact-unique question-answer pairs. The known mixed
  concatenation is excluded.
- Implemented a no-rewrite, no-reasoning DeepSeek classifier for validity, question
  type, minimum required domains, native domain type, answer verifiability, and keep
  priority. The model does not see source-directory labels.
- Completed a 60-item calibration with 60/60 schema-valid results. Validity accuracy on
  30 existing human-gold boundary records is 93.33%, but cross-domain precision is only
  40%. Human review retained four of eight machine high-priority cross-domain items, so
  the full native-QA scan remains paused.
- Clarified native-classifier retention once and evaluated frozen prompt v1.3 on a new
  30-item held-out set. It retained 22/24 valid QA, rejected 4/6 invalid controls, and
  found six of nine valid native-cross-domain QA with no false cross-domain label among
  valid items. Human review accepted 5/6 high-priority cross-domain candidates (83.33%)
  and 6/8 cross-domain candidates overall. This passes only a staged medium-pilot gate.
- Completed the approved 500-item stratified native-QA pilot with 100/100 API calls
  successful and 496 exportable labels. Semantic review of all 67 machine high-cross
  candidates accepted 17, reclassified 43 as single-domain, and rejected seven QA.
  A 12-item high-single audit was 12/12, while an invalid/low audit was 9/12. The pilot
  cost about $0.04276 off peak or $0.08552 peak; the full scan remains stopped.
- Completed offline review of all 141 machine high-single candidates: 116 stayed high,
  12 were downgraded to medium, and 13 materially defective QA were rejected. Together
  with 43 reviewed cross-to-single corrections and 17 reviewed cross-domain QA, the
  fixed pilot now provides 188 reviewed records without modifying the frozen 49.
- Reviewed all 31 medium-cross candidates, retaining 18 as cross-domain, correcting
  eight to single-domain, and rejecting five. The pilot now has 214 reviewed QA. A
  deterministic unseen evaluation split is frozen at 25 cross-domain plus 25 matched
  single-domain controls; the other 164 records remain training candidates.
- Enriched the frozen evaluation split with DeepSeek Flash in ten batches: 50/50
  records were structurally valid after deterministic normalization of harmless JSON
  shape variants. The run used 43,884 tokens and cost about $0.01512 off peak or
  $0.03025 peak; raw traces are retained.
- Semantically reviewed all 50 enriched Gold records. Forty-four were accepted as
  generated and six received explicit human-curated scientific or completeness edits;
  none was rejected. The final evaluation artifact remains exactly 25 native-cross
  plus 25 single-domain controls and is isolated from the 164 training candidates.
- Ran frozen Qwen baseline v2 on the independent native eval50 using three GPUs, one
  arm per GPU. All 150 arm-item traces parsed and passed field-isolation checks. Strict
  complete-rubric results were 14/50 Direct and 15/50 for both Routed and Oracle;
  Oracle rescued one Direct MCQ, damaged none, and was outcome-identical to Routed.
  All arms scored 0/32 on qualitative completeness.
- Post-run scientific audit flagged two ambiguous/non-unique MCQ Golds while preserving
  the original 50-item results. The audited-48 sensitivity result is 14/48 Direct and
  15/48 Routed/Oracle, so the conclusion is unchanged.
- Audited all 32 qualitative rubrics into 124 question-entailed core criteria, then
  rescored the existing traces offline without Qwen inference. Mean item coverage is
  71.77% Direct, 78.85% Routed, and 77.08% Oracle. Routed improves 10 and degrades 5
  items versus Direct; Oracle improves 9 and degrades 6. Material-error responses are
  8, 7, and 7, while the historical strict result remains 0/32 for every arm.
- Completed the fixed Expert-Critique versus matched Self-Refine diagnostic on audited
  native eval48. Both arms reused identical Direct drafts, matched 132-call structures,
  and the same reviser. MCQ results were 14/16 Direct, 13/16 Expert, and 14/16 Self;
  qualitative coverage was 71.77%, 75.52%, and 75.52%. Error responses were 8, 6, and
  7. Expert provided isolated unique corrections but no aggregate advantage over Self.
- Prepared the direct astronomy domain-model gate without changing Multi-Agent. Thirty
  reviewed astronomy-required items outside native eval50 are frozen as evaluation
  only: 16 MCQ, 12 qualitative, and two short-answer. Added correct model-specific
  rendering for Qwen chat/non-thinking and AstroSage plain completion, a shared prompt,
  append-only traces, blinded rubric judging, complementarity scoring, and an
  incomplete-checkpoint guard. All 93 tests pass; no answering-model inference has run.
- Completed astronomy Gold generation and audit before viewing model answers. Six
  DeepSeek batches used 23,880 tokens and cost about USD 0.00812 off peak or USD
  0.01625 peak. Of 30 source records, 24 are now evaluation-ready: 16 Gold records were
  accepted unchanged, eight explicitly corrected, and six scientifically ambiguous or
  inconsistent items were rejected. The final set has 11 MCQ, 12 qualitative, and one
  short-answer item.

## In Progress

- Route A: synthetic transformation is closed. Native-QA classifier v1.3 is frozen and
  the 500-item pilot is complete. It performs well on sampled high-single records but
  over-labels mature fields as cross-domain; the full traversal is stopped. Thirty-five
  reviewed cross-domain and 179 reviewed single-domain records are preserved. The
  frozen 50-item evaluation subset now has curated Gold. The historical pool contains
  164 training candidates; 30 astronomy records are newly reserved for the domain-model
  benchmark, leaving 134 eligible for future training exports.
- Route B: v1 and v2 full 49-record executions, deterministic scoring, and arm-blinded
  rubric reviews are complete. Independent native eval50 execution and blinded review
  are also complete. The matched refinement diagnostic is complete: Expert-Critique
  ties Self-Refine on qualitative coverage, trails by one MCQ, and improves the error-
  response count by one. This does not establish a specialization advantage and does
  not authorize Planner training.
- Route C: the fixed 24-item astronomy comparison is complete. Qwen3-8B and
  AstroSage-8B both scored 10/11 on MCQ, while audited non-MCQ rubric coverage was
  45.38% and 38.46% respectively. AstroSage produced no strict-correct/Qwen-wrong
  item, had more material-error responses (6 versus 4), used 4.40 times as many output
  tokens, and took 3.06 times as much summed generation time. Do not proceed to the
  planned AstroSage critique integration on this evidence.

## Next Steps

1. Preserve the completed astronomy benchmark and its no-go result. Do not add
   AstroSage to the Multi-Agent system or run the Stage-2 specialist-critique comparison
   unless a new held-out benchmark first demonstrates reliable specialist-only wins.
2. Keep the 50-item evaluation IDs and curated Gold frozen and hidden from all answering
   prompts; never use these records for prompt tuning or training.
3. Freeze the v2 prompts; do not tune them again on the original 49 or the new 50.
4. Do not full-scan with v1.3. Treat machine high-single output as candidates and require
   semantic review for every proposed cross-domain record; do not tune again on the
   current calibration, holdout, or pilot.
5. Do not start Planner cold-start or RL: the qualitative audit shows modest information
   gain but not a decisive Oracle correctness advantage.
6. Do not iterate the Expert-Critique or Self-Refine prompts on eval48. If Multi-Agent
   collaboration is revisited, first define a critic-reliability mechanism and evaluate
   it once on a new held-out set. Do not begin cold-start, expert SFT, RL, or OPD.

## Known Problems

- Source formats and field names differ across files.
- Pretty-printed JSON array line counts are not sample counts.
- Directory labels do not always match the knowledge required by each question.
- Prompt/response training files mix answer styles: some answers are option letters,
  while others contain long rationales.
- Source directory labels are noisy (for example, chemistry content under physics).
- Qwen3-8B generator v1 produced zero valid records in the five-item mini-pilot: it
  preferred forced transformations, invented subdomain labels, and paraphrased items.
- Generator v2 still missed an existing physics-biology dependency on the first
  smoke item. Same-model self-verification was over-permissive before hard gates.
- Source Analyzer v3 still treats one pure percentage word problem as biology and
  misses chemistry as independently required in one olfaction item.
- Two underspecified chemistry records are judged valid by the model; source
  validity therefore remains less reliable than domain classification.

## Important Paths

- Research brief: `how_to_start.md`
- Raw data: `datasets/science_dataset/mixed_dataset/` (read-only)
- Model: `/home/dqwang/Model/Qwen3-8B`
- Data protocol: `docs/data_protocol.md`
- Stage 01 log: `docs/stages/stage_01_question_pool.md`
- Dataset analysis: `outputs/data_analysis/source_analysis.md`
- Pilot manifest: `datasets/processed/pilot_anchors_20.jsonl`
- Mini-pilot traces: `outputs/question_pool/pilot_v1_mini5/`
- Gold protocol: `datasets/metadata/source_analysis_gold_v1.jsonl`
- Source Analyzer v3: `src/prompts/source_analyzer_v3.json`
- Audited evaluation: `outputs/source_analysis/v3_gold30_audited/`
- DeepSeek v3 pilot: `outputs/question_pool/deepseek_flash_pilot_v3.jsonl`
- Pilot human review: `outputs/question_pool/deepseek_flash_pilot_v3_review.json`
- Qwen solvability: `outputs/solvability/qwen3_8b_deepseek_pilot_v3/`
- DeepSeek protocol sources: `datasets/processed/deepseek_protocol_v1/protocol_sources_150.jsonl`
- DeepSeek source analyses: `datasets/processed/deepseek_protocol_v1/source_analysis_v1.jsonl`
- Source Analyzer v2.1: `src/prompts/deepseek_source_analyzer_v2_1.json`
- Six-expert prompt config: `configs/agents/prompt_agents_v3.json`
- Curated QA seed: `datasets/processed/deepseek_qa_pilot_v1/verified_qa_curated.jsonl`
- QA pilot summary: `datasets/processed/deepseek_qa_pilot_v1/curated_summary.json`
- Transformer v2: `src/prompts/deepseek_question_transformer_v2.json`
- QA Judge v2: `src/prompts/deepseek_qa_judge_v2.json`
- Prompt-v2 validation: `datasets/processed/deepseek_qa_validation_v2/validation_summary.json`
- Second-batch manifest: `datasets/processed/deepseek_qa_batch_v2/manifest_78.jsonl`
- Second-batch curated QA: `datasets/processed/deepseek_qa_batch_v2/verified_qa_curated.jsonl`
- Unified verified QA: `datasets/processed/deepseek_qa_curated_v2/verified_qa_curated_49.jsonl`
- Stage 01.5 plan: `docs/stages/stage_01_5_parallel_validation.md`
- Stage 01.5 source wave: `datasets/processed/stage015_calibration_v1/sources_50.jsonl`
- Stage 01.5 source analyses: `datasets/processed/stage015_calibration_v1/source_analysis_v2_1.jsonl`
- Stage 01.5 QA manifest: `datasets/processed/stage015_calibration_v1/qa_manifest.jsonl`
- Stage 01.5 curated calibration QA: `datasets/processed/stage015_calibration_v1/verified_qa_curated.jsonl`
- Transformability Selector prompt: `src/prompts/source_transformability_selector_v1.json`
- Transformability calibration design: `docs/stages/stage_01_source_selector_calibration.md`
- Keyword-free Selector source pool: `datasets/processed/transformability_selector_calibration_v1/candidate_sources_150.jsonl`
- Selector traces: `datasets/processed/transformability_selector_calibration_v1/selector_traces.jsonl`
- Selector human review: `datasets/processed/transformability_selector_calibration_v1/human_review.jsonl`
- Clean-medium human review: `datasets/processed/transformability_selector_calibration_v1/clean_medium_human_review.jsonl`
- Bridge library: `configs/data/interdisciplinary_bridge_library.yaml`
- Bridge-guided stage log: `docs/stages/stage_01_bridge_guided_transformation.md`
- Bridge calibration summary: `datasets/processed/bridge_guided_calibration_v1/calibration_summary.json`
- Human-bridge smoke log: `docs/stages/stage_01_human_bridge_transform_smoke.md`
- Human-bridge traces and review: `datasets/processed/human_bridge_transform_smoke_v1/`
- Native-QA mining stage log: `docs/stages/stage_01_native_qa_mining.md`
- Native-QA classifier prompt: `src/prompts/native_qa_classifier_v1.json`
- Native-QA calibration: `datasets/processed/native_qa_mining_v1/calibration_60/`
- Native-QA v1.3 holdout: `datasets/processed/native_qa_mining_v1/holdout_30/`
- Native-QA 500-item pilot and review: `datasets/processed/native_qa_mining_v1/pilot_500/`
- Frozen native-QA evaluation Gold: `datasets/processed/native_qa_mining_v1/pilot_500/eval_split_v1/eval_50_gold_curated.jsonl`
- Independent native-QA baseline v2: `outputs/baselines/qwen3_8b_native_eval50_v2/`
- Audited qualitative rubric: `configs/evaluation/native_eval50_qualitative_rubric_v2.json`
- Qualitative coverage result: `outputs/baselines/qwen3_8b_native_eval50_v2/qualitative_rubric_v2/`
- Refinement prompts: `src/prompts/qwen_refinement_v1.json`
- Expert-Critique versus Self-Refine result: `outputs/baselines/qwen3_8b_native_eval48_refinement_v1/`
- Astronomy domain-model protocol: `docs/stages/stage_01_domain_model_benchmark.md`
- Astronomy eval30 source freeze: `datasets/processed/domain_model_benchmark_v1/astronomy_eval30/`
- Astronomy curated Gold: `datasets/processed/domain_model_benchmark_v1/astronomy_eval30/astronomy_eval30_gold_curated.jsonl`
- Domain-model runner: `scripts/evaluate_astronomy_domain_models.py`
- Domain-model scorer: `scripts/score_astronomy_domain_models.py`
- Astronomy domain-model results: `outputs/domain_model_benchmark_v1/astronomy_eval24/evaluation_v1/`
- Astronomy judgment audit: `configs/evaluation/astronomy_eval24_domain_model_adjudications_v1.json`
- Selector calibration summary: `datasets/processed/transformability_selector_calibration_v1/calibration_summary.json`
- Baseline smoke: `outputs/baselines/qwen3_8b_smoke3_v1/`
- Full baseline and review: `outputs/baselines/qwen3_8b_full49_v1/`
- Completeness baseline v2 and review: `outputs/baselines/qwen3_8b_full49_v2/`
- Baseline v2 prompts: `src/prompts/qwen_baselines_v2.json`
- Expert v2 registry: `src/prompts/agent_prompts.v4.json`

## Latest Experiments

The latest completed result is `EXP-037`: Qwen3-8B and AstroSage-8B were compared once
on the frozen astronomy eval24. Both scored 10/11 MCQ; Qwen led audited non-MCQ rubric
coverage 45.38% to 38.46%, had fewer material-error responses (4 versus 6), and was far
more efficient. There were zero AstroSage-only strict-correct records. This fails the
predeclared gate for a domain-model critique experiment. Cold-start, expert SFT, RL,
and OPD remain unauthorized.
