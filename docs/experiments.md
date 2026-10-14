# Experiment Log

## EXP-000 -- Initial repository inspection

- Date: 2026-09-27
- Purpose: establish the starting state without changing source data.
- Model: none
- Dataset: `datasets/science_dataset/mixed_dataset/`
- Config: read-only file/tree inspection
- Command: `find`, `rg --files`, `wc -l`, `du`, and first-record inspection
- Output path: this log and `docs/PROJECT.md`
- Main results: 20 source data files, heterogeneous JSON/JSONL schemas, about 479 MB;
  only the research brief and source dataset existed initially.
- Observations: JSON array line counts are not record counts; mixed training data likely
  duplicates domain data; topic boundaries are noisy.
- Conclusion: implement schema-aware streaming analysis before selecting anchors.

## EXP-001 -- Source analysis, first run

- Date: 2026-09-27
- Purpose: validate the source analyzer on the real directory.
- Model: none
- Dataset: `datasets/science_dataset/mixed_dataset/`
- Config: `examples_per_file=1`, analysis schema `source_analysis.v1`
- Command: `python scripts/analyze_dataset.py --input datasets/science_dataset/mixed_dataset --output outputs/data_analysis --examples-per-file 1`
- Output path: none; validation stopped before writing output
- Main results: failed fast because `.claude/settings.local.json` is a configuration
  object rather than a dataset array.
- Observations: extension-only discovery is too broad when metadata is nested under
  the dataset root.
- Conclusion: exclude hidden top-level directories and cover the case with a test.

## EXP-002 -- Corrected full source analysis

- Date: 2026-09-27
- Purpose: characterize all source files and validate loader behavior.
- Model: none
- Dataset: `datasets/science_dataset/mixed_dataset/`
- Config: `source_analysis.v1`, one bounded example per file
- Command: `python scripts/analyze_dataset.py --input datasets/science_dataset/mixed_dataset --output outputs/data_analysis --examples-per-file 1`
- Output path: `outputs/data_analysis/source_analysis.{json,md}`
- Main results: 180,029 non-mixed raw records; 158,985 mixed records; 20 files;
  seven loader/analysis tests passed. Per-file schemas and length distributions are
  in the JSON report.
- Observations: the mixed file is an exact byte-for-byte concatenation of all six
  domain training files. Source domains contain 4,585 astronomy, 27,540 biology,
  24,391 chemistry, 36,426 geography, 66,858 math, and 20,229 physics records.
- Conclusion: exclude mixed from source totals and use normalized stems for anchor
  selection. Treat rule-derived reasoning and difficulty values only as proxies.

## EXP-003 -- Prompt registry and output-contract validation

- Date: 2026-09-27
- Purpose: define all Stage 01 prompt roles before model execution.
- Model: configured as `Qwen/Qwen3-8B` at `/home/dqwang/Model/Qwen3-8B`; not loaded
- Dataset: none
- Config: `configs/agents/prompt_agents_v1.json`
- Command: `python -m unittest discover -s tests -v`
- Output path: `src/prompts/agent_prompts.v1.json`, `src/agents/parsing.py`
- Main results: all nine requested roles have versioned prompts and explicit JSON
  contracts; 13 total repository tests pass.
- Observations: planner actions are intentionally limited to implemented Stage 01
  roles; retrieval is deferred. Output parsing accepts fenced JSON but enforces role
  fields, action/domain values, confidence ranges, and verifier score ranges.
- Conclusion: contracts are ready for a small inference-backed pipeline.

## EXP-004 -- Generator smoke iterations

- Date: 2026-09-27
- Purpose: validate local Qwen3 execution, logging, parsing, and semantic gates.
- Model: Qwen/Qwen3-8B, greedy decoding, seed 42
- Dataset: first record of `pilot_anchors_20.jsonl`
- Config: question generator/verifier v1, up to one repair
- Command: `generate_question_pool.py --limit 1` (four incremental smoke outputs)
- Output path: `outputs/question_pool/pilot_v1_smoke*/`
- Main results: model/runtime loading succeeded after creating the isolated `.venv`.
  Early schema/logging bugs were fixed. The final v1 smoke was rejected before the
  verifier because it preserved the original answer and choices while paraphrasing.
- Observations: an intermediate verifier incorrectly accepted the paraphrase and
  added a domain not claimed by the generator; hard gates caught both issues.
- Conclusion: engineering path works, but same-model semantic judgment is too lenient.

## EXP-005 -- Five-item generator v1 mini-pilot

- Date: 2026-09-27
- Purpose: test whether the first failure generalizes across domains and modes.
- Model: Qwen/Qwen3-8B, greedy decoding, seed 42
- Dataset: first five pilot anchors
- Config: question_generator_v1, candidate pre-validation, one repair
- Command: `generate_question_pool.py --limit 5`
- Output path: `outputs/question_pool/pilot_v1_mini5/`
- Main results: 0 written, 0 accepted, 5 failed. Failure counts: two unchanged-solution
  paraphrases, two transforms with only one required domain, and one unsupported
  domain label.
- Observations: repairs often repeated the invalid response. The model invented
  `astrophysics` and `developmental_biology` instead of using controlled labels.
- Conclusion: do not scale; add explicit source analysis and closed domain labels.

## EXP-006 -- Generator v2 source-analysis smoke

- Date: 2026-09-27
- Purpose: test explicit source analysis and a stricter scenario-first rubric.
- Model: Qwen/Qwen3-8B, greedy decoding, seed 42
- Dataset: first pilot anchor (cosmic-ray energy, tissue penetration, health risk)
- Config: question_generator_v2, one repair
- Command: `generate_question_pool.py --limit 1`
- Output path: `outputs/question_pool/pilot_v2_smoke/`
- Main results: 0 written, 0 accepted, 1 failed.
- Observations: v2 labeled the existing question as physics-only despite explicit
  tissue/cellular/cancer content, then claimed biology was newly introduced and
  retained the original answer and choices in both attempts.
- Conclusion: isolate and benchmark source-domain analysis before another generator
  revision; do not run the 20-item pool or multi-agent answering baseline.

## EXP-007 -- Source Analyzer v3 preliminary gold run

- Date: 2026-09-27
- Purpose: test source dependency classification independently from generation.
- Model: Qwen/Qwen3-8B, greedy decoding, seed 42
- Dataset: 30 source records, six boundary categories with five records each
- Config: `source_analyzer_v3`, four few-shot boundary examples, no repair
- Output path: `outputs/source_analysis/v3_gold30/`
- Main results: 30/30 parsed; 83.33% exact domain match; 100% cross-domain precision;
  40% cross-domain recall; zero math and subfield false positives.
- Observations: three of the five missed/counted cross-domain cases were not strong
  dependency tests: cosmic-ray atmospheric chemistry, respiration versus combustion,
  and CMB blackbody interpretation can reasonably be handled inside one main field.
- Conclusion: audit the gold protocol before attributing low recall to the model.

## EXP-008 -- Source Analyzer v3 audited gold run

- Date: 2026-09-27
- Purpose: rerun 01A after replacing three ambiguous cross-domain items with clear
  plasma-medicine, hearing, and olfaction dependency cases.
- Model: Qwen/Qwen3-8B, greedy decoding, seed 42
- Dataset: `datasets/metadata/source_analysis_gold_v1.jsonl`
- Config: `source_analyzer_v3`, four few-shot boundary examples, no repair
- Command: `evaluate_source_analyzer.py --gold datasets/metadata/source_analysis_gold_v1.jsonl --dataset-root datasets/science_dataset/mixed_dataset --prompt src/prompts/source_analyzer_v3.json --model-path /home/dqwang/Model/Qwen3-8B --output-dir outputs/source_analysis/v3_gold30_audited`
- Output path: `outputs/source_analysis/v3_gold30_audited/`
- Main results: 30/30 parsed; 93.33% validity and primary-domain accuracy; 90% exact
  domain match; 66.67% decision accuracy; 100% cross-domain precision; 80% recall;
  zero math/subfield false positives and zero invalid labels.
- Observations: remaining domain errors are a frog-percentage word problem classified
  as biology, an olfaction item missing chemistry, and a molecular-orbital boundary
  classified as physics rather than chemistry. Both intentionally underspecified
  chemistry items were judged valid. Most decision errors are conservative
  `keep_control` predictions rather than domain-routing failures.
- Conclusion: the precision-first 01A pilot is good enough to proceed after a final
  protocol spot review. Decision accuracy is secondary and should not block 01B.

## EXP-009 -- DeepSeek Flash transformation and blind-solver pilot

- Date: 2026-09-27
- Purpose: construct transformed questions with an external strong model while
  preserving both the original and one newly added domain dependency.
- Model: `deepseek-flash`; separate generator and blind-solver calls
- Dataset: five verified `transform_candidate` records from the 01A gold protocol
- Config: generator v1-v3 iteration; final v3 with low reasoning, 8192 max tokens;
  blind solver v1; no repair
- Output path: `outputs/question_pool/deepseek_flash_pilot_v3.jsonl`
- Main results: 5/5 parsed and passed machine gates; Codex protocol review retained
  4/5 for user/human confirmation.
- Observations: v1 stated the chemistry fact directly; v2 introduced physics but
  supplied away the original biology dependency; v3 preserved both under blind
  solving. One v3 item still concatenated independent collider and galaxy evidence.
- Usage: the frozen five-item v3 run used 7,245 prompt and 35,523 completion tokens;
  estimated off-peak cost $0.022099, excluding prompt-development calls.
- Conclusion: DeepSeek Flash is cost-effective and materially stronger for proposal
  generation, but answer agreement alone does not detect non-integrated conjunctions.

## EXP-010 -- Qwen3-8B solvability check on retained DeepSeek candidates

- Date: 2026-09-27
- Purpose: determine whether retained DeepSeek transformations remain solvable by the
  local 8B model directly and with oracle-routed fixed domain experts.
- Model: Qwen/Qwen3-8B, non-thinking greedy decoding, seed 42
- Dataset: four candidates retained by the EXP-009 protocol review
- Config: direct solver versus two known-domain experts plus a synthesizer; hidden
  answers and DeepSeek derivations were never shown to Qwen
- Output path: `outputs/solvability/qwen3_8b_deepseek_pilot_v3/`
- Choice-only results: direct 3/4, fixed multi-agent 4/4.
- Strict reasoning results: direct 1/4, fixed multi-agent 1/4.
- Observations: three correct choices were supported by contradictory, dimensionally
  invalid, or arithmetically inconsistent reasoning. Fixed experts rescued the plant
  inhibitor item but damaged the otherwise-correct 21-cm distance solution.
- Conclusion: multiple-choice match is not a sufficient solvability test. Under the
  current non-thinking configuration, the four candidates are not yet suitable as
  OPD trajectories; reasoning validity needs a gate and the Qwen inference protocol
  needs improvement before blaming difficulty alone.

## EXP-011 -- Qwen3-8B thinking plus Critic smoke

- Date: 2026-09-27
- Purpose: distinguish genuine question difficulty from non-thinking inference and
  missing-critic limitations on the frozen QCD-redshift item.
- Model: Qwen/Qwen3-8B, greedy; hybrid thinking/non-thinking roles
- Config: Direct and final Reviser in thinking mode with 1600-token caps; fixed
  experts, draft Synthesizer, and Critic in non-thinking mode
- Output path: `outputs/solvability/qwen3_8b_hybrid_critic_smoke/`
- Main results: Direct hidden reasoning correctly obtained about 6.55e11 and option A
  but hit the token cap before JSON. The draft selected C; Critic falsely approved it;
  the final Reviser also failed to emit structured output within budget.
- Efficiency observation: enabling 4096-token thinking for all six role calls was
  stopped during the expert stage because single-item latency was already excessive.
- Conclusion: the question is within Qwen3-8B's reasoning capacity, but the current
  thinking/JSON interface and same-model Critic are not a viable batch pipeline.
  Do not run this configuration on the other three items.

## EXP-012 -- DeepSeek source-protocol batch v1

- Date: 2026-09-27
- Purpose: classify the existing source distribution before generating any new data.
- Model: `deepseek-flash`, low reasoning effort; no repair.
- Dataset: 150 train sources: 100 random representative and 50 type-balanced stress
  items across astronomy, biology, chemistry, physics, and geography; math excluded.
- Output path: `datasets/processed/deepseek_protocol_v1/`.
- Main results: 150/150 valid JSON analyses after retrying token-truncated calls;
  29 A1, 1 A2, 28 B1, 0 B2, 66 C1, and 26 C2. Decisions were 66 keep-control,
  42 transform-candidate, 25 keep-cross-domain, and 17 reject.
- Observations: 39/150 predicted primary domains differ from source directories,
  including 18/35 geography sources. The representative subset contains 24/100
  predicted native cross-domain items. Astronomy+physics accounts for 10 cases and
  needs a boundary audit because astrophysics may be over-split.
- Conclusion: the protocol covers all answer formats except no B2 appeared in this
  batch, and directory labels are too noisy to use as gold. Do not transform yet;
  audit domain necessity and tighten the astronomy/physics boundary first.

## EXP-013 -- Six-expert Source Analyzer v2 boundary smoke

- Date: 2026-09-27
- Purpose: retain a zero-shot mathematics expert and separate required from helpful
  domains without weakening source-validity checks.
- Model: `deepseek-flash`; five final v2.1 boundary items, no generation or repair.
- Output path: `datasets/processed/deepseek_protocol_v1/source_analysis_v2_1_smoke5.jsonl`.
- Main results: 5/5 schema-valid. Three astronomy sources used astronomy as the only
  required domain; physics was optional-helpful in two. ANOVA and reactor-optimization
  sources included independent mathematics in their dependency analysis but were
  still rejected because the observations, kinetics, objectives, and constraints
  needed to answer them were absent.
- Observation: the first v2.0 smoke incorrectly retained both underspecified items;
  v2.1 added an explicit distinction between mathematical dependency and sufficient
  problem specification.
- Conclusion: mathematics no longer causes rejection, and optional expertise no
  longer inflates the minimum route. Keep v2.0 only as a failure record; use v2.1.

## EXP-014 -- DeepSeek QA production pilot v1

- Date: 2026-09-27
- Purpose: produce the first question-plus-answer data before any Qwen agent-trajectory
  collection, cold-start training, or RL.
- Model: `deepseek-flash`; Relation Planner and Transformer only for transform
  candidates, isolated Blind Solver for every final question, then QA Judge; no repair.
- Dataset: 30 selected v2.1 sources: 10 single-domain controls, 8 native cross-domain,
  and 12 transform candidates.
- Output path: `datasets/processed/deepseek_qa_pilot_v1/`.
- Machine result: 25 accepted, four scientifically rejected, and one final-Judge
  length failure retained as failed. Machine accepts comprised 9 native single,
  6 native cross-domain, and 10 transformed items.
- Semantic review: 17/25 retained: 9 native single, 4 native cross-domain, and 4
  transformed. Answer types are 3 MCQ, 2 numeric, 3 short answer, 1 symbolic, and
  8 qualitative; difficulty is 6 easy, 7 medium, and 4 hard.
- Failure analysis: same-model judging remained too permissive for transformations
  that supplied away the source dependency, treated physical chemistry as a separate
  physics dependency, assumed disputed taxonomy, or invented study-specific values.
- Conclusion: the answer-generation and blind-solving chain works, but semantic review
  is still necessary. Use `verified_qa_curated.jsonl`, not the 25 machine accepts, as
  the seed QA dataset.

## EXP-015 -- Stage 01 archive audit and prompt-v2 local gate

- Date: 2026-09-28
- Purpose: verify the archived pilot and convert the eight semantic-review failures
  into concise Transformer/Judge safeguards before any further API production.
- Model: none; local validation only.
- Dataset: the 150-record Source Analyzer v2.1 output and 30-record QA pilot archive.
- Output path: `src/prompts/deepseek_question_transformer_v2.json`,
  `src/prompts/deepseek_qa_judge_v2.json`.
- Main results: all 150 source analyses are present; the 30 traces, 25 machine accepts,
  17 curated records, and eight semantic rejects reconcile exactly. Source identity is
  preserved. All 36 repository tests pass.
- Change: add per-domain counterfactual necessity, established-subfield boundaries,
  missing-evidence rejection, and a deterministic block on nonempty material issues.
- Conclusion: local contracts are ready for a bounded API regression. No second-batch
  production is authorized by this result alone.

## EXP-016 -- Transformer/Judge v2 bounded API validation

- Date: 2026-09-28
- Purpose: test whether v2 rejects known semantic failures without indiscriminate
  rejection.
- Model: `deepseek-flash`, low reasoning effort; 10000-token retry cap when needed.
- Dataset: 12 frozen first-pilot records and six fresh transform candidates.
- Output path: `datasets/processed/deepseek_qa_validation_v2/`.
- Frozen replay: all eight known failures were rejected; three of four prior boundary
  positives were retained. The fourth exposed a real semiconductor/materials
  false-boundary split and was removed, reducing the curated seed to 16.
- Fresh result: two of six transformations passed and four were rejected for supplied
  or non-indispensable dependencies. Both accepted items passed manual review.
- Failure trace: five initial 6000-token Judge truncations and two fresh final-Judge
  errors were preserved; retry produced final JSON for all items.
- Conclusion: v2 is precise enough for bounded production, but transformation yield
  remains about one third. Semantic review remains mandatory.

## EXP-017 -- DeepSeek QA production batch v2

- Date: 2026-09-28
- Purpose: produce the bounded second QA batch with the validated precision-first
  Transformer/Judge v2 protocol.
- Model: `deepseek-flash`, low reasoning effort, 10000 max tokens, six workers.
- Dataset: 78 Source Analyzer v2.1 records: 47 controls, 9 native cross-domain, and
  22 transform candidates.
- Output path: `datasets/processed/deepseek_qa_batch_v2/`.
- Calls: 191 first-pass pipeline calls plus one targeted final-Judge retry; no Qwen
  trajectory, training, or RL calls.
- Machine result: 37 accepted, 32 Judge rejects, 7 Solver-gate rejects, 1 Transformer
  reject, and 1 final-Judge length failure resolved as rejected.
- Semantic result: 31/37 retained: 18 native single, 5 native cross-domain, and 8
  transformed. Six were removed for ambiguity, unsupported causality, wrong objective
  Gold, underdetermination, or a study-specific question answered generically.
- Combined result: the 16-record pilot seed, two reviewed validation records, and 31
  batch-v2 records form a 49-record verified set at
  `datasets/processed/deepseek_qa_curated_v2/verified_qa_curated_49.jsonl`.
- Conclusion: precision is acceptable but yield is below the requested batch size.
  Do not recover rejected records; use a fresh bounded source sample for any top-up.

## EXP-018 -- Qwen3-8B three-arm baseline skeleton smoke

- Date: 2026-09-28
- Purpose: verify the Direct, question-routed Multi-Agent, and oracle-domain
  Multi-Agent execution paths before a 49-item comparison.
- Model: local Qwen/Qwen3-8B, greedy non-thinking decoding, seed 42; no training.
- Dataset: three verified records: one native-single MCQ, one transformed-cross MCQ,
  and one native-cross qualitative item.
- Output path: `outputs/baselines/qwen3_8b_smoke3_v1/`.
- Main result: 9/9 item-arm traces completed and parsed. Explicit trace inspection
  found no `required_domains` field in any model message. The objective items were one
  all-arms correct and one all-arms wrong; the qualitative item requires rubric review.
- Cost observation: Direct used one call per item. Routed used three or four calls and
  381--785 output tokens; Oracle used two or three calls and 363--595 output tokens.
- Conclusion: the skeleton is executable and labels are isolated. Three examples do
  not establish a Multi-Agent gain; use the frozen 49-item set and blinded review.

## EXP-019 -- Stage 01.5 expansion calibration sampling

- Date: 2026-09-28
- Purpose: create a fresh, bounded source wave targeting current dataset gaps without
  reusing rejected records or invoking DeepSeek.
- Dataset: five active train files only; all 150 prior protocol identities excluded.
- Config: seed 43; astronomy/biology/chemistry/physics/geography quotas 20/7/8/8/7.
- Output path: `datasets/processed/stage015_calibration_v1/sources_50.jsonl`.
- Main result: 50 unique fresh sources: 16 MCQ, 19 likely-objective-open, and 15
  non-mechanism-affordance records. The last group is balanced five each across
  quantitative modeling, constraint coupling, and measurement/observation hints.
- Failure retained: the first offline sampler version produced a 12/3/0 imbalance for
  measurement/quantitative/constraint. No API call was made; explicit relation quotas
  replaced the biased preference rule before freezing the batch.
- Planned ceiling: 50 Analyzer calls plus at most 88 QA-chain calls, or 138 first-pass
  calls total. Stop unless at least two non-mechanism transformations survive semantic
  review. The environment currently contains no `DEEPSEEK_API_KEY`.

## EXP-020 -- Full 49-item baseline launch failure

- Date: 2026-09-28
- Purpose: run the three frozen non-thinking Qwen3-8B baseline arms in parallel.
- Planned execution: Direct on GPU 0, routed Multi-Agent on GPU 1, and Oracle on GPU 2.
- Failure: all three processes encountered CUDA OOM while loading checkpoint shards.
  A follow-up process query showed unrelated Python jobs using about 11--13 GiB on
  every one of the seven GPUs. The Qwen jobs exited during loading.
- Output impact: no 49-item prediction or partial trace was produced; the existing
  three-item smoke remains intact. No external process was interrupted.
- Next action: use the resumable scripts after at least one GPU has enough free memory;
  run arms sequentially if only one card is available.

## EXP-021 -- Full 49-item prompt Multi-Agent go/no-go baseline

- Date: 2026-09-28
- Purpose: test whether correct domain routing makes prompted expert collaboration
  more accurate than direct Qwen before any Planner training.
- Model/config: one local Qwen3-8B checkpoint for every role; greedy non-thinking
  decoding, seed 42; identical final-answer schema; no training or trajectories.
- Dataset: all 49 semantically reviewed QA records.
- Output path: `outputs/baselines/qwen3_8b_full49_v1/`.
- Integrity: 147/147 final item-arm predictions parsed after a narrow deterministic
  repair for unescaped LaTeX backslashes and two append-only replays. Direct used 49
  calls, routed used 172, and Oracle used 120. No item ID or `required_domains` field
  appeared in model messages.
- Scoring: MCQ/exact aliases/same-unit numeric cases used deterministic rules. The
  remaining 111 predictions were evaluated in a candidate-code-blinded strict rubric
  pass by one reviewer. All required rubric points and absence of fatal errors were
  needed for a correct label.
- Accuracy: Direct 17/49 (34.7%), routed 16/49 (32.7%), Oracle 16/49 (32.7%). Oracle
  minus Direct was -1/49 (-2.0 points), bootstrap 95% interval [-14.3, +10.2] points;
  Oracle and routed were tied. Oracle rescued four Direct failures but damaged five
  Direct successes.
- Router: exact domain-set match 65.3%, micro precision 81.1%, micro recall 84.5%; 13
  items were over-routed and 11 under-routed.
- Efficiency: mean calls per item were 1.00/3.51/2.45 and mean total tokens per item
  were approximately 459/1712/1395 for Direct/Routed/Oracle. Mean summed generation
  latency was 11.6/34.5/28.8 seconds per item.
- Failure pattern: all arms scored only 1/25 on qualitative questions under the full
  required-point rubric. Many outputs were directionally correct but too concise;
  expert/synthesizer calls sometimes removed details that Direct retained.
- Conclusion: baseline v1 is a no-go for Planner cold-start and RL. Correct routing did
  not create a gain and cost substantially more. This is evidence against the current
  concise non-thinking pipeline, not against all Multi-Agent designs. A second reviewer
  and a prompt-completeness revision developed on a separate split would strengthen
  any future reevaluation.

## EXP-022 -- Fixed-49 Qwen baseline v2 answer-completeness ablation

- Date: 2026-09-28
- Purpose: test whether fuller Expert contributions and a problem-solving Synthesizer
  create information gain when Oracle routing is correct.
- Fixed controls: same 49 verified records, Qwen3-8B, v1 Router text and route choices,
  Oracle labels, greedy non-thinking decoding, seed 42, Gold, evaluator, output schema,
  and three arms. No data generation, trajectories, training, Critic, or Debate.
- Prompt change: `qwen_baselines_v2.json` asks Direct to enumerate the problem's needs
  and self-check completeness; `agent_prompts.v4.json` upgrades all six Experts to
  complete domain contributions; Synthesizer v2 is explicitly the final problem solver
  and may assess, reject, and supplement expert content.
- Output path: `outputs/baselines/qwen3_8b_full49_v2/`.
- Integrity: 147/147 latest item-arm predictions are `ok`. One Direct JSON/LaTeX parse
  failure was preserved and replayed append-only after a narrow mixed-backslash parser
  repair. No item ID or `required_domains` appeared in model messages, and routed v2
  selected exactly the same domains as v1. All 50 repository tests pass.
- Scoring: unchanged deterministic evaluator plus one candidate-code-blinded strict
  rubric pass for 116 deferred predictions; every required point and no fatal error
  were needed for correctness.
- V1 to v2 overall: Direct 17/49 to 16/49; routed 16/49 to 20/49; Oracle 16/49 to
  20/49. V2 Oracle minus Direct is +4/49 (+8.2 points), paired bootstrap 95% interval
  [0.0, +18.4] points; Oracle rescued five Direct failures and damaged one success.
- V2 by surface: Direct qualitative/objective was 1/25 and 15/24; routed was 3/25 and
  17/24; Oracle was 2/25 and 18/24. Thus most gain came from objective items, although
  two qualitative Oracle answers now satisfy their full rubrics.
- Representative improvement: on `qa-batch-v2-biology-16775`, Oracle v2 added the
  nested taxonomy, homology/analogy distinction, and multiple valid examples omitted
  by v1, changing the strict score from false to true.
- Representative failures: on `qa-batch-v2-physics-14546`, all arms correctly described
  bounded Lorenz chaos but failed to derive beta=8/3. On `qa-batch-v2-geography-19886`,
  Oracle v2 was less explicit than Direct about the observed sulfate/nitrate fractions,
  producing the sole Oracle damage relative to Direct.
- Conclusion: completeness prompting creates a positive Oracle trend and justifies
  retaining the prompt-only Multi-Agent research line. It does not yet justify
  cold-start/RL: this was adaptive reuse of the fixed 49, the uncertainty interval
  touches zero, and qualitative coverage remains poor. Freeze v2 and validate on unseen
  verified QA with a matched-compute Direct/self-refine control.

## EXP-023 -- Stage 01.5 frozen-source and QA calibration wave

- Date: 2026-09-28
- Purpose: obtain a small unseen QA batch emphasizing astronomy, objective questions,
  and non-mechanism fusion without changing the frozen baseline-v2 prompts.
- Source input: 50 deterministic fresh sources: 20 astronomy, 35 objective-surface,
  and 15 lexical non-mechanism-affordance sources balanced across quantitative modeling,
  constraint coupling, and measurement/observation.
- Source Analyzer: 49/50 schema-valid. One time-sensitive primordial-black-hole MCQ
  exhausted both 6000- and 10000-token reasoning limits without visible JSON; both
  failures are preserved and the item was excluded. Including one successful retry,
  Source Analyzer used 52 API requests.
- Strict manifest: 14 items rather than force-filling the ceiling: eight controls, two
  native cross-domain items, and four transformations (2 quantitative, 1 constraint,
  1 measurement). The planned QA ceiling was 36 calls.
- QA run: all 14 completed without API errors using Relation Planner v1.1, Transformer
  v2, Blind Solver v1, and Judge v2. Five were machine accepted, eight rejected by the
  Judge, and one rejected by the Solver; the run used 36 calls.
- Semantic review: four of five machine accepts were retained, all native-single (two
  MCQ and two short-answer). The measurement/observation cosmic-ray transformation was
  scientifically plausible but rejected because its astronomy-plus-physics label was
  a false subfield split. Final artifact:
  `datasets/processed/stage015_calibration_v1/verified_qa_curated.jsonl`.
- Failure pattern: objective rejects exposed ambiguous or incorrect source Gold and
  reading-comprehension-only questions; transformed rejects exposed underdetermined
  numerical constraints, dependencies supplied directly in the prompt, and artificial
  domain separation. Zero non-mechanism transformations survived semantic review.
- Conclusion: the preregistered minimum of two valid non-mechanism transformations was
  not met. Preserve all traces, keep the four reliable controls separate, revise source
  selection, and do not scale data production or start training/RL.

## EXP-024 -- Transformability Selector v1 and keyword-free candidate pool

- Date: 2026-09-28
- Purpose: replace lexical fusion-affordance sampling with an explicit judgment of
  whether a source has a natural structure for quantitative modeling, constraint
  coupling, or measurement inference.
- Model plan: `deepseek-flash` with thinking disabled, strict JSON, maximum 600 output
  tokens, one request per source, and no repair loop. No API request was made in this
  implementation experiment.
- Source input: five read-only domain train files. The builder excludes the prior
  150-source protocol and 50-source Stage 01.5 wave by `(source_file, source_index)`.
- Sampling: deterministic seed 44, stratified only by source domain and MCQ/open format.
  No question keyword, proposed fusion relation, or affordance hint affects selection.
- Output: 150 unused sources at
  `datasets/processed/transformability_selector_calibration_v1/candidate_sources_150.jsonl`:
  astronomy 40, biology 28, chemistry 28, physics 27, geography 27; 44 MCQ and
  106 open-response items.
- Implementation: `source_transformability_selector_v1` emits self-containment,
  answer grounding, high/medium/low transformability, relation priors, possible new
  domains, a concise reason, and nine controlled risk flags. Schema invariants prevent
  risky or false-boundary records from being treated as strict-high candidates.
- Gate: the shortlist accepts only schema-valid records with `self_contained=true`,
  `answer_grounded=true`, `transformability=high`, nonempty relation/domain priors,
  and no risk flags. It never fills from medium/low records to meet a quota.
- Current result: tooling and offline pool construction are complete; Selector yields,
  human-review precision, QA-pipeline entries, and final valid QA count are not yet
  measured because no model call or downstream generation was run.
- Conclusion: the revised source-selection protocol is ready for a separately approved,
  bounded calibration. Do not start Transformer calls, data scaling, cold-start, or RL.

## EXP-025 -- Transformability Selector v1 full-pool calibration

- Date: 2026-09-28
- Purpose: test whether explicit structural selection substantially enriches a random,
  keyword-free source pool for natural single-source cross-domain transformation.
- Input: the fixed 150-source EXP-024 pool, with zero overlap against the prior 200
  sampled sources.
- Model/config: `deepseek-flash`, thinking disabled, strict JSON, maximum 600 output
  tokens, four workers, exactly one request per source, and no repair or retry loop.
- API result: 150/150 responses and zero API errors. There were 137 schema-valid outputs
  and 13 preserved invalid outputs. Nine invalid records omitted `reason`; five violated
  the `no_natural_extension` invariants, with one record in both groups.
- Valid distribution: 4 high, 55 medium, and 78 low. Relation priors occurred 55 times
  for quantitative modeling, 30 for constraint coupling, and 30 for measurement
  inference. Dominant risks were no natural extension (75), fact lookup only (57), too
  broad (52), answer already supplied (28), and underspecified (22).
- Usage: 194,169 prompt tokens and 20,125 completion tokens, 214,294 total. Prompt-cache
  hits were 128,256 tokens. Estimated official cost is about $0.022 off-peak or $0.045
  peak.
- Human review: only four records met the strict automatic gate. One polymer-chemistry
  source was retained as worth transforming. An accretion-disk source was rejected as
  ordinary astrophysics, a javelin item as physics mislabeled biology, and a gas-kinetics
  item as routine physical chemistry with the key relation already supplied.
- QA gate: failed before Transformer execution. One human-approved source cannot form
  the planned 10--15 source batch, and promoting medium/risky records would invalidate
  the calibration. Thus zero sources entered the QA pipeline and zero high-quality QA
  were generated.
- Conclusion: Selector v1 did not provide enough enrichment to justify another
  single-source transformation wave. Preserve every trace and move the design question
  to a bounded two-source grounded-synthesis protocol; do not start it automatically.

## EXP-026 -- Clean-medium recall audit

- Date: 2026-09-28
- Purpose: determine whether the strict-high gate, rather than source quality, caused
  the four-item shortlist.
- Input: all 15 schema-valid medium records with `self_contained=true`,
  `answer_grounded=true`, and no risk flags. No model call was made.
- Human result: three were retained as genuinely worth transformation: nephron membrane
  transport with a physical mass-balance model, insulin feedback with nonlinear dynamic
  analysis, and cosmic-ray physical dose with a biological risk model.
- Rejections: 12. Dominant causes were mature-field false boundaries (astrophysics,
  physical chemistry, NMR), unreliable or source-specific reference claims, and an
  extension that would need to invent nearly all quantitative grounding.
- Combined result: the strict-high and clean-medium reviews cover 19 records and retain
  four total sources. Thus requiring high was indeed too conservative for recall, but
  accepting clean medium still does not supply the planned 10--15 source QA batch.
- Conclusion: retain risk flags for audit, but future gates should distinguish fatal
  source risks from repairable formulation issues. Do not run the four-source QA
  pipeline merely to produce volume; keep the current zero-QA outcome and design the
  next grounded synthesis strategy separately.

## EXP-027 -- Bridge Selector v1 full calibration

- Date: 2026-09-28
- Purpose: constrain single-source transformations with a small bridge library instead
  of letting DeepSeek invent a target domain freely.
- Library: 49 named bridges in
  `configs/data/interdisciplinary_bridge_library.yaml`; no retrieval, Top-k, embeddings,
  knowledge signatures, graph, or matching score.
- Input: 25 unused sources with audited semantic primary domains: chemistry 8, biology
  7, astronomy 4, physics 4, and geography 2.
- Model/config: `deepseek-flash`, thinking disabled, maximum 800 output tokens, one
  request per source, no retry or repair.
- Result: 25 responses, 24 schema-valid and one invalid. All 24 valid decisions were
  NONE; no bridge was selected. The invalid output also semantically said NONE but
  added an unexpected `type` field.
- Failure pattern: the model treated absence of an already-specified quantitative model
  as evidence that any bridge would invent a new problem. This rejected the same four
  source structures previously retained by human review.
- Conclusion: run one bounded clarification smoke before abandoning the route; do not
  run the Transformer from an empty selection.

## EXP-028 -- Bridge Selector v1.1 positive/negative smoke

- Date: 2026-09-28
- Purpose: test whether allowing a future Transformer to add minimal hypothetical raw
  values and boundary conditions repairs Selector recall without weakening the mature-
  field boundary rule.
- Input: eight frozen records: four human-positive bridge cases (polymer optimization,
  insulin nonlinear dynamics, nephron membrane transport, radiation health effect) and
  four negative controls (accretion-disk astrophysics, colloid physical chemistry,
  galaxy rotation curves, and a mislabeled projectile problem).
- Result: eight responses, seven schema-valid and one invalid. All seven valid outputs
  were NONE. The four negative decisions were reasonable, but all four positive bridges
  were false negatives. Reasonable selected bridges: zero.
- Downstream result: zero Transformer entries and zero accepted cross-domain QA. No
  downstream API call was made.
- Qualitative comparison: the prior free-form wave proposed transformations but retained
  zero after semantic review; bridge guidance proposed none at all. It avoids false
  boundary outputs through zero recall, not through successful bridge discrimination.
- Conclusion: the success gate failed at Bridge selection. Do not add a third prompt,
  force known positives downstream, expand calls, start two-source retrieval, cold-start,
  or RL. Reassess the transformation premise and the division of human/model control.

## EXP-029 -- Human-specified source--bridge Transformer smoke

- Date: 2026-09-28
- Purpose: isolate whether the Transformer can produce genuinely integrated QA when a
  correct bridge is supplied by a human, without using or modifying Bridge Selector.
- Input: eight frozen source--bridge pairs from existing calibration data: four
  human-positive pairs and four fixed negative controls. No source was newly authored.
- Pipeline: fixed pair to Transformer; transformed items only then went to Blind Solver,
  Judge, deterministic gate, and semantic review. Retry and repair were disabled.
- Runtime note: an initial thinking-enabled execution yielded eight technical failures
  because completions exhausted the 6000-token budget or ended as truncated JSON. The
  failures remain in `trace.jsonl`. A single non-thinking execution, matching the prior
  DeepSeek Flash protocol, supplied the evaluable result in `trace_nonthinking.jsonl`.
- Positive result: 4/4 were transformed, 0/4 passed the machine pipeline, and 0/4 passed
  human semantic review. Chemistry or biology was decorative in two cases, one case was
  underdetermined, and one supplied away the cross-domain inference.
- Negative result: one of four was correctly rejected, two were transformed into
  pseudo-fusions, and one produced malformed JSON. No negative was accepted.
- Final data result: zero machine-accepted and zero human-accepted cross-domain QA.
- Conclusion: no-go. Human bridge assignment removes Selector recall failure but does
  not produce reliable two-domain necessity. Stop this synthetic transformation route
  rather than adding prompts, samples, retries, or retrieval.

## EXP-030 -- Native QA classifier calibration

- Date: 2026-10-08
- Purpose: stop synthetic generation and test lightweight mining of unchanged native
  single-domain and native cross-domain QA from the existing source dataset.
- Source universe: 180,029 records in 19 non-mixed files; the known 158,985-record mixed
  concatenation is excluded. Exact normalized question-answer deduplication leaves
  178,814 unique pairs.
- Calibration: 60 sources, comprising the existing 30-item human boundary gold set and
  30 fresh items stratified across six source directories. Directory labels were hidden
  from the model. Requests used `deepseek-flash`, non-thinking mode, batches of five,
  12 calls, no answer or question rewriting, and no reasoning generation.
- Preserved failures: prompt v1 produced 30 parseable but schema-invalid labels because
  it emitted `single_domain` and ignored the MCQ enum. Prompt v1.1 incurred 12 HTTP 400
  errors and zero tokens because the JSON-mode prompt omitted the literal word `JSON`.
  Both trace sets are retained. Prompt v1.2 is the evaluable run.
- Structural result: 60/60 schema-valid labels; 50 valid and 10 invalid; 49 single and
  11 proposed native cross-domain; priorities were 26 high, 24 medium, and 10 low.
- Gold quality: 93.33% validity accuracy, 76.67% required-domain exact match, 76.67%
  domain-type accuracy, 40% cross-domain precision, and 80% recall.
- Human high-cross audit: four of eight high-priority cross-domain candidates were
  accepted. The other four were mature-field or context-only false splits involving
  astrophysics, molecular physics, or positronium.
- Usage and cost: 22,091 prompt plus 3,668 completion tokens; approximately $0.0050
  off peak or $0.0100 peak for the evaluable run. Including the paid schema smoke,
  total calibration spend was about $0.0103 off peak or $0.0205 peak; the HTTP 400
  launch used zero tokens. Scaling the evaluable usage to 178,814 exact-unique pairs
  projects about $14.92 off peak or $29.84 peak.
- Conclusion: do not scale. The classifier is inexpensive and structurally sound, but
  cross-domain precision is insufficient for high-quality data mining. Keep all outputs
  as calibration artifacts, not training-ready data.

## EXP-031 -- Native QA classifier v1.3 held-out calibration

- Date: 2026-10-08
- Purpose: test one less restrictive retention policy without weakening the rule that
  every declared domain must be necessary.
- Input: 30 records with zero overlap with EXP-030: 15 verified native-single QA, all
  nine available verified native-cross-domain QA, and six audited invalid controls.
  Questions were neither rewritten nor supplied with generated reasoning.
- Prompt change: clear factual, qualitative, and single-domain routing controls may be
  high or medium priority when reliable. A deletion test and explicit mature-subfield
  rule constrain `required_domains`. No retry or repair was added.
- Structural result: six calls produced 30/30 schema-valid labels: 24 valid, 22 single,
  eight proposed cross-domain, 15 high, nine medium, and six low.
- Held-out quality: valid-item retention was 22/24 (91.67%); invalid-control rejection
  was 4/6 (66.67%). On the 24 valid items, required-domain exact match was 19/24
  (79.17%), domain-type accuracy 21/24 (87.50%), and cross-domain precision/recall
  6/6 (100%) and 6/9 (66.67%).
- Human audit: six of all eight final cross-domain candidates were accepted (75%). Five
  of six high-priority cross-domain candidates were accepted (83.33%). The two rejected
  candidates were invalid controls that lacked data for a radial-velocity limit or
  ANOVA. The three cross-domain false negatives involved polymer nanocomposites, lipid
  bilayers, and plasma adhesion.
- Usage and cost: 14,844 prompt plus 1,951 completion tokens; approximately $0.00317
  off peak or $0.00634 peak. Linear full-scan projection is $18.90/$37.81, but no full
  scan was launched.
- Conclusion: v1.3 passes only the gate for a staged medium pilot with sampled human
  audits. Freeze the prompt; do not tune on this holdout or start the full traversal,
  reasoning generation, cold-start, RL, or OPD.

## EXP-032 -- Native QA mining 500-item staged pilot

- Date: 2026-10-08
- Purpose: test frozen classifier v1.3 on a larger unchanged-source distribution before
  considering any full traversal.
- Sampling: 500 exact-unique QA with zero overlap with the prior 90 calibration/holdout
  records, balanced across six source directories (83--84 each) and MCQ/open surfaces
  (252/248). Directory labels were used only for sampling and hidden from the model.
- Runtime: 100 batches of five with `deepseek-flash`, non-thinking mode, no retry or
  repair. All 100 requests succeeded. There were 496 exportable labels and four strict
  schema failures caused by empty required-domain lists; all raw outputs are preserved.
- Machine result: 452 valid, 395 single, 101 proposed cross-domain, 208 high, 243
  medium, and 45 low. High-priority outputs comprised 141 single and 67 cross-domain.
- High-cross semantic review: 17/67 were accepted as real cross-domain QA (25.37%), 43
  were retained only after reclassification as single-domain, and seven QA were
  rejected. Two accepted items required corrections to the machine domain list.
  Astronomy+physics dominated with 26 candidates and was the main mature-field split.
- Bounded controls: a deterministic two-per-directory sample upheld 12/12 high-single
  decisions. A matching invalid/low sample upheld 9/12 rejections; the three false
  rejections were a chemistry-education qualitative item, a precipitation-nowcasting
  MCQ, and a probability MCQ.
- Usage and cost: 211,186 prompt plus 31,016 completion tokens; approximately $0.04276
  off peak or $0.08552 peak. The current linear full-scan projection is $15.42/$30.83.
- Conclusion: stop before the full scan. v1.3 is useful as a broad single-domain and
  candidate-recall filter, not as an autonomous high-precision cross-domain curator.
  Preserve the 17 reviewed cross-domain and 43 reclassified single-domain records;
  do not generate reasoning or start training.
- Follow-up offline review: all 141 machine high-single candidates were inspected.
  116 remain high, 12 valid but broad/awkward records were downgraded to medium, and
  13 were rejected for missing figures or parameters, incorrect keyed choices,
  malformed statements, or scientifically incorrect references. No item was rejected
  merely for being simple. Combining these 128 retained records with the 43 reviewed
  reclassifications yields 171 reviewed native-single QA; with the 17 reviewed cross-
  domain QA, EXP-032 contributes 188 reviewed unchanged-source records.
- Medium-cross follow-up: among 31 valid medium-cross candidates, 18 were accepted as
  cross-domain, eight were reclassified as single-domain, and five were rejected. This
  raises the final reviewed pilot pool to 214 records: 179 single and 35 cross. A
  deterministic seed-131 split freezes 25 cross plus 25 matched single controls for
  unseen evaluation and leaves 164 training candidates.

## EXP-033 -- Frozen native-QA evaluation Gold enrichment

- Date: 2026-10-08
- Purpose: make the independent 50-item native-QA split evaluable without modifying
  its questions, source answers, domain decisions, or membership.
- Runtime: `native_qa_gold_enricher_v1`, DeepSeek Flash non-thinking mode, ten batches
  of five, three workers, 6000 maximum output tokens, and no retry or semantic repair.
  All raw request results and validation metadata are retained.
- Structural result: 50/50 records exported, zero model rejects, zero API errors, and
  zero remaining schema failures. Twenty solution strings were deterministically
  wrapped as one-element lists; zero-point explanatory rubric rows were removed. These
  operations changed JSON shape only, not answer content.
- Usage and cost: 24,612 prompt tokens plus 19,272 completion tokens, 43,884 total;
  approximately $0.015123 off peak or $0.030247 peak.
- Semantic review: all 50 records were inspected against the unchanged question and
  source reference. Forty-four were accepted unchanged and six accepted with explicit
  edits. No record was dropped, so the final set remains 25 native-cross and 25 single
  controls, with 32 qualitative and 18 MCQ items.
- Curated corrections: distinguish bilayer moduli from deformation, remove unsupported
  Lake Tai hydrodynamics, make sulfur-doping effects configuration-dependent, correct
  carbonic-anhydrase zinc coordination, restore dimensional consistency in the helium
  permeation calculation, and expand the tokamak comparison answer.
- Artifacts: `gold_enrichment_traces.jsonl` preserves raw model output;
  `eval_50_enriched.jsonl` preserves normalized model Gold;
  `gold_human_review.jsonl` records review decisions; and
  `eval_50_gold_curated.jsonl` is the frozen evaluation-ready file.
- Conclusion: Gold construction is complete. Do not enrich the 164 training candidates
  yet. The next useful bounded experiment is a no-tuning run of frozen Qwen baseline-v2
  prompts on this split, not additional source mining or training.

## EXP-034 -- Frozen baseline v2 on independent native eval50

- Date: 2026-10-08
- Purpose: test whether the +4/49 Oracle-domain trend from the prompt-development set
  reproduces on the independent native-QA evaluation split before any training.
- Frozen setup: Qwen3-8B, `qwen_baselines_v2`, expert registry v4, seed 42, greedy
  non-thinking decoding, 768 role tokens, and 1536 final tokens. No prompt, Router,
  expert, Synthesizer, Gold, or decoding parameter was changed.
- Runtime: one-item three-arm smoke passed. Full inference ran Direct, Routed, and
  Oracle on physical GPUs 0, 1, and 2 respectively. All 150 arm-item traces parsed;
  exact inspection of every model-visible user payload found no Gold or
  `required_domains` leakage.
- Raw strict result: Direct 14/50 (28%), Routed 15/50 (30%), and Oracle 15/50 (30%).
  Oracle rescued one Direct failure and damaged none. Routed made the same rescue, and
  Oracle and Routed had identical binary outcomes on all 50 records.
- By type: on 18 MCQ, results were 14/18 Direct and 15/18 for Routed/Oracle. All three
  arms were 0/32 on qualitative complete-rubric scoring. The recurring failure was a
  generally correct high-level answer that omitted source-specific evidence, required
  mechanisms, boundary conditions, or one final rubric component.
- Routing: exact required-domain set match 50%; micro precision 74.42%; micro recall
  84.21%; 21 over-routed and 11 under-routed items. This did not create a performance
  gap because Routed and Oracle outcomes were identical.
- Efficiency: Direct averaged 1 call, 313 input tokens, 251 output tokens, and 14.54 s.
  Oracle averaged 2.52 calls, 1322 input tokens, 760 output tokens, and 45.57 s. Routed
  averaged 3.72 calls, 1713 input tokens, 917 output tokens, and 54.46 s.
- Representative rescue: on the cytochalasin-B growth-cone MCQ, Direct incorrectly
  chose irreversible growth-cone depolymerization, while both multi-agent arms used the
  biology contribution to select reversible microspike depolymerization.
- Representative failures: all arms answered the Qionghai TP item generically and
  omitted its quantitative/spatiotemporal evidence; DPC/POLDER answers hallucinated
  instrument properties instead of giving the 12 angles and 3.3-km resolution; the
  otherwise strong niobium answers omitted above-Tc pair breaking and finite resistance.
- Interpretation limit: 0/32 is a strict complete-rubric result, not evidence that all
  qualitative answers were useless. Several rubrics are arguably over-specific for the
  visible question: the permafrost rubric demands a valley-bottom observation, the
  microplastics rubric demands one study's exact strain/time/percentage, and the solar-
  cell rubric accepts only an LSC pathway despite valid spectral-conversion approaches.
  Before treating qualitative accuracy as an architectural measurement, audit rubric
  entailment and optionally report point coverage without changing the preserved binary
  result or tuning prompts on these answers.
- Evaluation audit: two MCQs lack a unique defensible answer. Raw eval50 is preserved.
  Excluding only those two yields Direct 14/48 (29.17%), Routed 15/48 (31.25%), and
  Oracle 15/48 (31.25%); Oracle still rescues one and damages none.
- Comparison with the development set: the former Oracle advantage of +4/49 shrank to
  +1/50 raw or +1/48 audited. The independent result therefore does not establish
  useful Oracle-domain information gain.
- Conclusion: no-go for Planner cold-start or RL. More routing supervision is not
  justified when perfect routing and learned routing have identical binary outcomes.
  This does not yet prove the architecture intrinsically useless because qualitative
  rubric validity also needs audit. Reconsider evaluation and collaboration, then use a
  pre-registered matched-compute control on unseen data before training.

## EXP-035 -- Native eval50 qualitative rubric audit and offline coverage scoring

- Date: 2026-10-08
- Purpose: determine whether the 0/32 qualitative result was caused by uniformly poor
  answers or by strict rubrics that demanded non-entailed paper details and complete
  coverage of every point.
- Frozen inputs: the 32 qualitative questions, curated Gold, Direct/Routed/Oracle
  traces, prompts, and historical strict scores from EXP-034. No Qwen inference, model
  call, prompt change, or data rewrite occurred.
- Rubric audit: each item now has three to five evaluator-side core criteria, 124 in
  total. Necessary mechanisms, causal links, conditions, and limitations remain.
  Paper-specific numbers/observations and non-unique implementation paths are optional
  unless requested by the visible question.
- Scoring: deterministic regex concept anchors produced criterion-level evidence.
  Fifty-nine ambiguous/contradictory decisions were resolved in one versioned
  adjudication file; no review flags remain. Explicit scientific errors are recorded
  separately from missing coverage. All arms use the identical per-item rubric.
- Result: mean item coverage is 71.77% Direct (89/124 micro criteria), 78.85% Routed
  (98/124), and 77.08% Oracle (96/124). The preserved complete-rubric score remains
  0/32 for every arm.
- Paired result: Routed versus Direct improves 10 items, degrades 5, and ties 17,
  yielding +7.08 mean coverage points. Oracle versus Direct improves 9, degrades 6,
  and ties 17, yielding +5.31 points.
- Scientific errors: Direct has 10 material error claims across 8 responses; Routed
  has 8 across 7; Oracle has 8 across 7. Recurring errors include confusing total
  natural background with the cosmic-ray dose component, misidentifying DPC/POLDER,
  direct-ion-to-CCN oversimplification, incorrect plant-protein roles, and an invalid
  quantum-Zeno formula.
- Interpretation: Multi-Agent produces a real but modest completeness improvement and
  a one-response reduction in scientific errors. Oracle does not outperform Routed,
  and regressions show that expert/synthesis can discard useful Direct reasoning.
  This supports one bounded, pre-registered Expert--Synthesizer interaction ablation
  only if it is compared with matched-compute Direct Self-Refine on unseen data. It
  does not authorize cold-start, RL, OPD, or tuning/rerunning on eval50.
- Artifacts: `configs/evaluation/native_eval50_qualitative_rubric_v2.json`,
  `configs/evaluation/native_eval50_qualitative_adjudications_v2.json`, and
  `outputs/baselines/qwen3_8b_native_eval50_v2/qualitative_rubric_v2/`.

## EXP-036 -- Expert-Critique versus matched Self-Refine on audited native eval48

- Date: 2026-10-08
- Purpose: test whether domain-specialized critique adds correction value beyond extra
  inference compute when both methods revise the exact same cached Direct draft.
- Frozen inputs: the audited 48 items (16 MCQ, 32 qualitative), cached Direct answers,
  cached question-only Router choices, qualitative rubric v2, Qwen3-8B, seed 42, and
  greedy non-thinking decoding. No question, Gold, rubric, Direct answer, or Router
  output was regenerated or exposed to an answering role.
- Arms: Expert-Critique used one domain critic per cached routed domain; Self-Refine
  used the same number of generic critique passes without domain labels. Both then used
  the identical reviser. Each arm made 84 critique calls plus 48 revision calls.
- Runtime: one dual-domain smoke passed before the fixed full run. The 48 records were
  deterministically sharded across six GPUs only to reduce wall time. All 96 final
  answers parsed. Two Expert critic responses reached the 512-token cap with incomplete
  JSON; both failures remain in trace and were not retried or repaired.
- Objective result: MCQ exact-choice accuracy was 14/16 Direct, 13/16 Expert-Critique,
  and 14/16 Self-Refine. Neither refinement arm rescued a Direct MCQ. Expert damaged
  `native-4a0fb05a96efa5`: a mathematics critic falsely replaced Seurat's default
  Wilcoxon rank-sum answer with a negative-binomial GLM. Self made the same factual
  criticism but happened to retain choice B, leaving an internally inconsistent answer
  that the required choice-only metric still counts correct.
- Qualitative result: mean per-item rubric coverage was 71.77% Direct (89/124), 75.52%
  Expert (94/124), and 75.52% Self (94/124). Expert improved four items and degraded
  none; Self improved five and degraded none. Both gained exactly +3.75 mean points.
- Scientific errors: Direct had 10 material claims across eight qualitative responses;
  Expert had seven across six; Self had eight across seven. Expert uniquely repaired
  the cosmic-ray ionization-to-aerosol/CCN distinction and gave a properly qualified
  cosmic-background health answer. It also introduced an incorrect claim that the
  Ilkovic polarographic diffusion-current condition requires stirring. Self retained
  the direct-ion-as-CCN error and overstated health effects at ordinary cosmic dose.
- Shared failures: both methods retained the propene rearrangement/stereochemistry
  errors, DPC/POLDER misidentification, the solar-cell non-radiative-loss claim, the
  geomagnetic field-strength explanation, and the absolute claim that MRI lacks
  functional information. More critique tokens therefore did not reliably overcome
  confident errors shared by the initial draft and same-model critics.
- Compute: Direct is cached at 48 calls, 14,970 input and 12,116 output tokens. Each
  refinement arm used 132 new calls. Expert used 107,245 input and 38,296 output tokens
  with 2,335.08 summed model-seconds; Self used 104,840 input and 34,399 output tokens
  with 2,268.78 seconds. Expert used 2.29% more input, 11.33% more output, and 2.92%
  more model time, so it did not obtain its error-count edge with less compute.
- Interpretation: specialization produced two useful corrections that generic review
  missed, but no coverage advantage and one objective regression. This is a no-go for
  claiming that the current Expert-Critique architecture outperforms matched-compute
  Self-Refine. Because eval48 informed the experiment design, the result is diagnostic,
  not an independent generalization estimate.
- Artifacts: `src/prompts/qwen_refinement_v1.json`,
  `configs/evaluation/native_eval48_refinement_adjudications_v1.json`, and
  `outputs/baselines/qwen3_8b_native_eval48_refinement_v1/`.
- Conclusion: do not begin expert SFT, Planner cold-start, RL, or OPD. Keep the traces
  as failure evidence; any future collaboration test must use a new held-out set and a
  stronger reliability mechanism rather than another prompt iteration on eval48.

## EXP-037 -- Astronomy domain-model benchmark preparation and Gold audit

- Date: 2026-10-09
- Purpose: prepare a clean Stage-1 comparison of Qwen3-8B and AstroSage-8B before
  changing the Multi-Agent system.
- Data: froze 30 unchanged reviewed astronomy-required records outside native eval50:
  16 MCQ, 12 qualitative, and two short-answer; 29 are single-domain and one is
  native cross-domain. All are now evaluation-only.
- Implementation: added model-specific chat/plain-completion rendering, a shared
  astronomy solver prompt, format-tolerant parsing, append-only inference traces,
  blinded DeepSeek first-pass judging, and an offline complementarity scorer. Qwen
  thinking is disabled and both arms use greedy decoding with a 1024-token cap.
- Evaluation policy: deterministic MCQ scoring; fixed Gold-rubric coverage and
  scientific-error review for non-MCQ; DeepSeek first pass plus Codex audit of Gold and
  flagged judgments. Final scoring rejects unaudited judgments by default.
- Verification: the source freeze is byte-stable with SHA-256
  7e476410b80d740b9cdd974bbb2d955afb24b680408c8675a9fe88e359f17942;
  93 repository tests pass. After download completion, all four AstroSage weight shards
  opened successfully and all 291 indexed parameter keys matched their declared files.
- Gold run: six non-thinking DeepSeek calls processed the fixed source30 with no API
  errors, using 13,535 prompt and 10,345 completion tokens. Estimated cost was USD
  0.00812 off peak or USD 0.01625 peak. One output failed deterministic validation
  because it simultaneously marked an inconsistent helium-burning problem accepted
  and reported the inconsistency.
- Human audit: 16 Gold records were accepted unchanged, eight received explicit
  scientific or rubric corrections, and six questions were excluded. The curated
  benchmark therefore has 24 items: 11 MCQ, 12 qualitative, and one short-answer.
  Major exclusions were the helium calculation, the non-unique Venus rotation item,
  Blandford-Znajek disk/black-hole conflation, CME trigger ambiguity, and two
  unsupported gravitational-wave/planet-orbit items.
- Inference: both models completed all 24 items once on the same RTX 3090, with greedy
  decoding, seed 42, and a 1024-token cap. Qwen produced valid JSON on all 24 records;
  AstroSage produced four full JSON, four deterministically normalized partial JSON,
  and 16 usable plain-text outputs. No answer was retried or repaired.
- Objective result: both models scored 10/11 MCQ. The single missed MCQ was shared, so
  there was no objective specialist-only rescue.
- Audited non-MCQ result: mean weighted rubric coverage was 45.38% Qwen versus 38.46%
  AstroSage. Strict complete-rubric correctness was 1/13 versus 0/13. Responses with a
  material scientific error were 4 versus 6. AstroSage improved two items, degraded
  seven, and tied four; its two improvements still contained material errors.
- Complementarity: across all 24 items there were ten both-correct, one Qwen-only,
  thirteen both-wrong, and zero AstroSage-only strict-correct records. The Qwen-only
  record was the Chandrasekhar-limit short answer.
- Compute: Qwen used 5,216 output tokens and 347.65 summed generation seconds;
  AstroSage used 22,942 and 1,064.34 seconds, or 4.40 times the output and 3.06 times
  the latency.
- Judge incident: blind-Judge v1 made 13 paid calls without receiving the prompt's
  defined output schema because the runner failed to append it. All outputs failed
  validation and the old runner also discarded their usage. The failure traces are
  retained. After fixing both defects, one v2 smoke plus 12 resumed calls used 26,227
  prompt and 9,278 completion tokens, costing about USD 0.00927 off peak or USD
  0.01854 peak. Twelve results were schema-valid; one response added a harmless
  `type` field and was recovered offline. Codex audited all disagreements, error claims,
  and rubric-status flags through a versioned adjudication file.
- Conclusion: Stage 1 shows no reliable AstroSage knowledge complementarity and worse
  correctness/completeness efficiency. Do not run the planned specialist-critique
  Stage 2 or integrate AstroSage into Multi-Agent on this evidence.
- Artifacts: `docs/stages/stage_01_domain_model_benchmark.md`,
  `datasets/processed/domain_model_benchmark_v1/astronomy_eval30/`,
  `outputs/domain_model_benchmark_v1/astronomy_eval24/`,
  `configs/evaluation/astronomy_eval24_domain_model_adjudications_v1.json`,
  `scripts/audit_astronomy_domain_judgments.py`, and
  `scripts/score_astronomy_domain_models.py`.
