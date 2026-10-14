# Design Decisions

## D-001 -- Prompt experts before domain LoRAs

- Date: 2026-09-27
- Decision: use one Qwen3-8B checkpoint with role prompts in Stage 01.
- Reason: isolate question-pool and workflow quality from expert training effects.
- Revisit when: prompt experts show inadequate role differentiation after controlled
  baseline evaluation.

## D-002 -- Preserve and separate the mixed aggregate

- Date: 2026-09-27
- Decision: do not combine `mixed/mixed_train.jsonl` with per-domain counts.
- Reason: byte comparison proves it is the exact concatenation of the math, physics,
  chemistry, astronomy, geography, and biology training files (in that order).
- Revisit when: the source dataset is replaced or revised.

## D-003 -- Start with a 20--50 item prototype

- Date: 2026-09-27
- Decision: require diagnostics and human review before scaling to 100--200 items.
- Reason: generation volume is not useful until naturalness, correctness, and true
  domain necessity are demonstrated.

## D-004 -- Let real-anchor analysis choose keep or transform

- Date: 2026-09-27
- Decision: select 16 `real_anchored` and 4 `synthetic` pilot inputs; do not assign
  keep/transform quotas before inspecting the scenario.
- Reason: preassigning transformation contradicts scenario-first construction and
  encourages forced fusion.

## D-005 -- Hard gates override same-model acceptance

- Date: 2026-09-27
- Decision: validate candidates before verifier calls and reject paraphrases,
  unsupported labels, missing primary domains, and mismatched domain ablations.
- Reason: the first smoke verifier accepted a rephrased question and treated a topic
  domain as necessary. Self-reported `accepted` is not reliable on its own.

## D-006 -- Do not scale after the five-item v1 failure

- Date: 2026-09-27
- Decision: stop before 20-item generation, diagnostics, and answering baselines.
- Reason: v1 produced zero valid candidates in five attempts; v2 still misclassified
  the first source dependency. The next variable to isolate is source analysis.

## D-007 -- Split Stage 01 into 01A through 01D

- Date: 2026-09-27
- Decision: separate source analysis, transformation, pool audit, and answering baseline.
- Reason: a single prompt made an early classification error propagate through an
  otherwise self-consistent generation and verification trace.

## D-008 -- Optimize source-analysis precision before recall

- Date: 2026-09-27
- Decision: treat cross-domain precision, math false positives, and subfield false
  positives as primary gates; do not maximize the number of transform candidates.
- Reason: false cross-domain labels contaminate both the pool and future routing.

## D-009 -- Remove repair from the current pilot path

- Date: 2026-09-27
- Decision: reject invalid structured or semantic outputs instead of asking the same
  deterministic model to repair them.
- Reason: v1 repairs repeated the original failure and obscured true yield.

## D-010 -- Use five source-data domains and six callable experts

- Date: 2026-09-27
- Decision: source data domains are physics, chemistry, biology, astronomy, and
  geography, while the callable experts additionally include mathematics. Exclude
  `math/math_train.jsonl`; run mathematics as a zero-shot prompt expert.
- Reason: the available math training set is dominated by elementary arithmetic and
  simple algebra, so it is not useful as specialist training data. Ordinary calculation
  remains inside each scientific expert, while genuinely independent advanced
  mathematics is retained and may route to the untrained mathematics expert.

## D-011 -- Use 150 items only for source-protocol diagnosis

- Date: 2026-09-27
- Decision: supersede D-003 only for read-only source analysis: sample 100 random
  representative sources and 50 stress sources. This does not authorize 150-item
  transformation or training.
- Reason: the revised source taxonomy must cover MCQ, objective open, and qualitative
  open questions across all five active domains before any generated pool is built.

## D-012 -- Separate minimum-required from optional-helpful experts

- Date: 2026-09-27
- Decision: use `required_domains` only for the minimal indispensable route and add
  `helpful_domains` for optional expertise. Extra helpful calls remain legal and are
  later controlled by a small RL efficiency cost rather than treated as wrong.
- Reason: borderline cases such as astrophysics do not need a brittle forced label,
  while cold-start supervision should still avoid teaching systematic over-routing.

## D-013 -- Require semantic review after same-model QA judging

- Date: 2026-09-27
- Decision: DeepSeek Judge acceptance is a machine gate, not final Gold. Before a QA
  pilot enters the curated set, independently review transformed dependency necessity
  and study-specific factual claims; reject rather than repair failures.
- Reason: the 30-item QA pilot produced 25 machine accepts, but eight still contained
  supplied-away dependencies, subfield over-splitting, ambiguity, or unverified
  study-specific claims. Seventeen survived the precision-first review.

## D-014 -- Make material Judge issues machine-blocking

- Date: 2026-09-28
- Decision: Transformer v2 must pass per-domain counterfactual necessity checks, and
  Judge v2 must list only material training-safety issues. Any listed issue blocks the
  deterministic acceptance gate.
- Reason: Judge v1 identified missing conditions, disputed premises, supplied-away
  dependencies, and study-specific uncertainty but still accepted several records.
- Scope: QA data production only; no Qwen trajectories, training, or RL are enabled.

## D-015 -- Stage the second QA batch at 78 sources

- Date: 2026-09-28
- Decision: after excluding prior items and obvious study-specific or false-boundary
  risks, use 47 controls, 9 native cross-domain items, and 22 transform candidates.
- Reason: v2 validation accepted only two of six fresh transformations. The remaining
  source pool can yield roughly 50 curated records only if reliable controls are kept
  and transformation yield is not inflated artificially.
- Execution gate: report 200 planned first-pass calls plus up to about 10 targeted
  retries; if fewer than 50 survive review, separately propose at most eight held-out
  transform candidates for a top-up.

## D-016 -- Do not fill the target from known high-risk leftovers

- Date: 2026-09-28
- Decision: keep the 31 semantically reviewed records from batch v2 and merge them
  with the 16-record seed and two reviewed validation records. Do not use rejected or
  pre-screened high-risk candidates merely to reach a round number.
- Reason: the merged verified set contains 49 records. The eight held-out transformations
  were excluded for study-specific dependence or artificial domain boundaries and are
  unlikely to improve precision.
- Next: if more records are required, draw a fresh bounded source sample and run the
  same Analyzer v2.1 and QA v2 gates.

## D-017 -- Add a Stage 01.5 oracle-routing go/no-go gate

- Date: 2026-09-28
- Decision: expand verified QA and build the prompt-only Qwen3-8B baseline skeleton in
  parallel. Before cold-start or RL, compare direct answering, question-only routed
  multi-agent answering, and oracle-domain multi-agent answering.
- Reason: oracle routing isolates whether correct expert collaboration has intrinsic
  value. If it cannot beat direct Qwen under the same final-answer protocol, training
  a router is not justified.
- Isolation: `required_domains` is read only by the evaluation orchestrator for the
  oracle condition. It is never placed in Direct, Router, Expert, or Synthesizer model
  messages. All three conditions use the same Qwen3-8B checkpoint.
- Scope: prompt inference and evaluation only; no trajectory SFT, LoRA, GRPO, or OPD.

## D-018 -- Calibrate non-mechanism fusion before scaling data production

- Date: 2026-09-28
- Decision: draw fresh train sources in deterministic 50-source waves. The first wave
  prioritizes astronomy and objective questions, while reserving 15 sources with a
  surface affordance for `quantitative_modeling`, `constraint_coupling`, or
  `measurement_observation`. A transform candidate receives one preferred relation;
  the Relation Planner must either use that relation or reject the transformation.
- Gate: attempt at most 12 balanced non-mechanism transformations in the calibration
  wave. Continue production only if at least two survive semantic review. Otherwise
  revise the selector or prompts before another wave.
- Reason: 12 of the current 13 transformed records use `mechanism_explanation`.
  Forcing a different relation after a plan is written would create artificial fusion,
  so relation targeting is an input constraint and rejection remains valid.

## D-019 -- Do not start Planner training after baseline v1

- Date: 2026-09-28
- Decision: treat the first 49-item prompt-only comparison as a no-go for Planner
  cold-start or RL under the current expert/synthesizer protocol. Continue verified
  data expansion, but do not train routing until an improved prompt-only Oracle path
  demonstrates a practically meaningful paired gain over Direct.
- Evidence: strict rubric scoring gave Direct 17/49, routed Multi-Agent 16/49, and
  oracle-domain Multi-Agent 16/49. Oracle minus Direct was -1/49, with a paired
  bootstrap 95% interval spanning approximately -7/49 to +5/49. Oracle required
  120 calls versus 49 Direct calls and about three times as many total tokens.
- Qualification: this rejects the current non-thinking, concise-answer pipeline, not
  the general possibility of expert collaboration. The 25 qualitative questions were
  especially demanding, and only one per arm satisfied every required rubric point.
  Any prompt revision must use a separate development split before reevaluation.

## D-020 -- Run one fixed-data answer-completeness prompt ablation

- Date: 2026-09-28
- Decision: on the same 49 records, create `qwen_baselines_v2` and six Expert v2
  prompts that require complete problem coverage rather than terse contributions.
  Keep the Router text, Oracle control logic, model, decoding, Gold, evaluator, answer
  schema, and three baseline arms unchanged.
- Reason: baseline v1 failed mainly because directionally plausible qualitative
  answers omitted required mechanisms, conditions, or comparisons. One controlled
  prompt ablation can test whether correct expert routing provides information gain
  once answer completeness is explicitly required.
- Gate: do not start cold-start or RL. Continue the Multi-Agent training path only if
  Oracle v2 shows a clear positive trend over Direct v2; otherwise reconsider expert
  interaction or architecture.
- Outcome: Oracle v2 reached 20/49 versus Direct v2 at 16/49, rescuing five items and
  damaging one. This is a positive trend, but its paired bootstrap 95% interval is
  [0/49, 9/49], qualitative Oracle accuracy is only 2/25, and the prompt was revised
  after v1 errors on these same records. Treat it as evidence to continue prompt-only
  validation, not as authorization for training.

## D-021 -- Freeze v2 and require an unseen-data gate before training

- Date: 2026-09-28
- Decision: freeze the v2 Direct, Expert, and Synthesizer prompts. Do not start Planner
  cold-start or RL from the fixed-49 result alone.
- Reason: the +4/49 Oracle gain is practically encouraging but adaptively measured on
  the same set used to diagnose v1. Routed and Oracle also tie at 20/49, while Oracle
  still satisfies only 2/25 qualitative rubrics.
- Gate: first evaluate frozen v2 on newly verified unseen QA and compare against a
  matched-compute Direct/self-refine control. Continue the Multi-Agent training mainline
  only if the Oracle advantage persists and is not explained by extra inference compute.

## D-022 -- Do not scale after the first non-mechanism calibration misses its gate

- Date: 2026-09-28
- Decision: keep the four semantically reliable native-single QA records in an isolated
  calibration artifact, but do not merge them into the frozen 49 and do not launch a
  second API wave yet.
- Evidence: 49 valid source analyses yielded 14 strict candidates. Five of 14 passed
  machine QA gates; semantic review retained four native-single items and rejected the
  only accepted transformation because astronomy and physics were falsely split inside
  one astrophysics reasoning chain. Zero non-mechanism transformations survived versus
  the preregistered minimum of two.
- Next: revise source selection toward questions with stable, independently necessary
  domain pairs and explicit but not pre-supplied quantitative, constraint, or
  measurement dependencies. Avoid broad qualitative prompts, source-specific facts,
  and astronomy/physics subfield relabeling before another bounded wave.

## D-023 -- Replace lexical affordance sampling with a transformability selector

- Date: 2026-09-28
- Decision: sample a broad deterministic 150-source pool without relation keywords,
  then use a dedicated `source_transformability_selector_v1` to judge whether each
  source has a natural structure for quantitative modeling, constraint coupling, or
  measurement inference. The Selector proposes only relation/domain priors and never
  writes a transformed question or answer.
- Model: use `deepseek-flash` with thinking disabled, strict JSON, a bounded output,
  one request per source, and no repair loop. Existing evidence (49/50 valid Source
  Analyzer results) indicates
  the model is sufficient; the failed calibration arose from lexical preselection, not
  a demonstrated need for a larger model.
- Gate: only `high` records that are self-contained, answer-grounded, and free of risk
  flags may enter a 20--30 item human shortlist. Human approval is required before any
  10--15 item QA pipeline. No Selector API run or QA generation is authorized by this
  implementation decision alone.
- Boundary rule: mature interdisciplinary fields such as astrophysics, physical
  chemistry, biochemistry, and geophysics are not split mechanically. A proposed new
  domain must add an independently necessary method or constraint, not merely rename
  knowledge a competent specialist in the mature field should already possess.

## D-024 -- Stop single-source transformation after Selector v1 misses its gate

- Date: 2026-09-28
- Decision: do not send medium/low records or the four-item strict-high shortlist into
  the Transformer, and do not draw another random single-source pool.
- Evidence: 150 keyword-free sources produced 137 schema-valid Selector results:
  4 high, 55 medium, and 78 low. Human review retained only one of the four strict-high
  records as genuinely worth transformation. The other three were standard mature-field
  reasoning or a noisy source-directory label, not independent cross-domain structure.
- Secondary audit: because the strict-high rule might suppress recall, all 15 medium
  records that were self-contained, answer-grounded, and risk-free were also reviewed
  offline. Three survived. Combined high plus clean-medium review therefore retained
  four of 19 sources, still below the planned 10--15 source pipeline size.
- Gate: the planned 10--15 source QA calibration cannot be formed without relaxing the
  preregistered high/self-contained/grounded/no-risk rule. Consequently zero sources
  enter the QA pipeline and zero QA are added to training data.
- Next direction: design a separate, bounded `two-source grounded synthesis` protocol
  using two real disciplinary concepts as anchors. Do not implement or call it until
  its pairing and grounding criteria have been reviewed.

## D-025 -- Stop bridge-guided calibration at the all-NONE Selector gate

- Date: 2026-09-28
- Decision: do not run the Bridge-guided Transformer, Blind Solver, or Judge, because
  neither Bridge Selector version selected a source. Do not add a third prompt revision
  or expand the calibration.
- Evidence: v1 returned zero selections among 24 schema-valid outputs from 25 sources.
  One bounded v1.1 clarification was then tested on four human-positive and four
  boundary-negative records; its seven valid outputs also all returned NONE. Human
  review expected four selections and four rejections in that smoke.
- Interpretation: the small bridge library prevents arbitrary target invention, but
  the model currently uses NONE as a degenerate precision strategy. Zero false-boundary
  proposals do not demonstrate useful bridge selection when recall is also zero.
- Scope: this supersedes D-024's tentative two-source next step. Do not build two-source
  retrieval, complex labels, or matching infrastructure. Reassess the data-construction
  premise and whether bridge choice should be human-authored before more API work.

## D-026 -- Stop synthetic single-source cross-domain transformation

- Date: 2026-09-28
- Decision: do not continue tuning the Transformer, expand fixed-bridge samples, restart
  automatic Bridge selection, or build source-pair retrieval. Redirect future data work
  toward native single-domain QA, native cross-domain QA, existing verified transformed
  QA, and externally grounded real cross-domain sources.
- Evidence: the user-authorized direct test bypassed the all-NONE Selector and fixed four
  human-positive bridges. The Transformer attempted all four, but downstream gates and
  semantic review accepted zero. It also transformed two of four negative controls into
  pseudo-fusions, correctly rejected one, and returned malformed JSON on one.
- Interpretation: Bridge selection was not the only bottleneck. Even when the bridge is
  supplied as fact, the current Transformer tends to replace the source dependency,
  supply away the new dependency, invent missing quantities, or split a mature field.
- Gate: the preregistered continuation threshold was about three accepted positives; the
  observed result is zero. This is a no-go for further synthetic single-source
  cross-domain transformation. It does not authorize cold-start, RL, OPD, or baseline
  changes.

## D-027 -- Mine native QA, but do not scale before cross-domain precision passes

- Date: 2026-10-08
- Decision: replace synthetic question generation with read-only classification of
  existing QA into native single-domain and native cross-domain pools. Preserve good
  single-domain controls rather than penalizing them.
- Source handling: exclude the known concatenated `mixed_train.jsonl`, keep all six
  allowed source domains including mathematics, and exact-deduplicate question-answer
  pairs before any future full scan. Never use the directory label as domain gold.
- Calibration: use non-thinking `deepseek-flash`, batches of five, and only the six
  requested labels. First evaluate 30 existing human-gold boundary items plus 30 fresh
  stratified sources. Do not generate reasoning or rewrite any source.
- Evidence: prompt v1.2 produced 60/60 schema-valid results and 93.33% validity accuracy,
  but only 40% native-cross-domain precision on human gold. Manual review accepted only
  four of eight machine high-priority cross-domain candidates.
- Gate: do not launch the 178,814-pair full scan. A bounded prompt clarification may be
  tested only on a newly held-out 20--50 item boundary set; do not claim improvement by
  retesting only the same gold records. No cold-start, RL, OPD, or large-scale job is
  authorized.

## D-028 -- Relax retention, not domain necessity, in one held-out recalibration

- Date: 2026-10-08
- Decision: do not require a retained question to be difficult or to contain a long
  fusion chain. Clear qualitative questions, simple factual QA, and ordinary
  single-domain routing controls may remain high or medium priority when their answer
  is reliable.
- Boundary: this relaxation applies to `keep_priority`, not to inventing extra required
  domains. A native cross-domain label needs two genuinely useful knowledge
  dependencies, but they need not be unusually sophisticated. Pure context, routine
  arithmetic, and name-only links still do not justify another expert.
- Calibration: make one concise false-boundary clarification and evaluate it on 30
  newly held-out sources assembled from already verified native QA plus a small set of
  audited invalid controls. Do not use synthetic or transformed questions and do not
  reuse the inspected gold30 as the evaluation set.
- Gate: a materially better high-cross audit and approximately 80% cross-domain
  precision are sufficient to discuss a larger pilot. This decision authorizes only
  the 30-item held-out run, not the full scan, reasoning generation, cold-start, or RL.
- Outcome: the held-out run passed only the bounded-pilot gate. Five of six
  high-priority cross-domain candidates survived human review (83.33%); all valid
  cross-domain predictions were true positives, but three of nine true cross-domain
  items were missed and two of six invalid controls were retained. Freeze v1.3 and
  proceed, if authorized, through a staged medium pilot with sampled audits. Do not
  launch the 178,814-pair traversal.

## D-029 -- Run a 500-item staged native-QA mining pilot

- Date: 2026-10-08
- Decision: after the user-approved EXP-031 gate, run frozen classifier v1.3 on exactly
  500 unchanged source QA. This is a bounded pilot, not authorization for a full scan.
- Sampling: deterministically balance the six source directories and MCQ/open surfaces,
  exclude all EXP-030 and EXP-031 source identities and exact question-answer pairs,
  and exact-deduplicate the pilot. Directory labels are sampling strata only and remain
  hidden from the classifier.
- Runtime: use `deepseek-flash`, non-thinking mode, batches of five, no retry or repair,
  and preserve all raw traces. Expected scale is 100 requests and roughly $0.05 off
  peak or $0.11 peak based on EXP-031.
- Review gate: audit every high-priority cross-domain candidate, plus bounded samples
  of retained single-domain and invalid/low decisions. Stop after reporting the pilot;
  do not generate solutions, reasoning, trajectories, or start cold-start, RL, or OPD.
- Outcome: stop before the full scan. The 500-item pilot produced 67 machine high-cross
  candidates, but semantic review retained only 17 (25.37%), reclassified 43 as valid
  single-domain candidates, and rejected seven QA. Twenty-six machine candidates used
  the astronomy+physics pair, overwhelmingly as mature-field splits. In contrast, all
  12 sampled machine high-single items were upheld. Future native mining may use v1.3
  as a broad candidate filter, but cross-domain labels require semantic review.

## D-030 -- Finish offline review of the fixed pilot's high-single candidates

- Date: 2026-10-08
- Decision: without further API calls or source sampling, semantically review all 141
  machine high-priority single-domain candidates from EXP-032. Preserve the machine
  labels and export reviewed records separately.
- Review outcomes: keep a reliable and useful item at high, downgrade a valid but
  simple/broad item to medium, or reject an unsupported/incorrect item. Single-domain
  status remains valid only when one normal domain expert can provide the complete
  answer. This review does not generate reasoning or modify source questions/answers.
- Scope: combine neither these records nor the 17 reviewed cross-domain records with
  the frozen 49 automatically. Report the reviewed counts first; no training, full
  scan, or additional model call is authorized.
- Outcome: of 141 machine high-single candidates, retain 116 at high, downgrade 12
  usable but broad or awkward items to medium, and reject 13 for material defects.
  Together with the 43 high-cross false positives that were reviewed and reclassified,
  the fixed pilot now supplies 171 reviewed native-single and 17 reviewed native-cross
  records. They remain separate from the frozen 49.

## D-031 -- Freeze a balanced unseen 50-item native-QA evaluation split

- Date: 2026-10-08
- Decision: after reviewing the medium-cross candidates, freeze 25 reviewed native-
  cross-domain QA and 25 question-type/domain-matched single-domain controls from the
  fixed pilot. Keep the remaining reviewed records as training candidates.
- Boundary: the evaluation IDs must never enter future SFT, cold-start, RL, or prompt
  tuning. Gold enrichment may use DeepSeek, but Qwen baseline prompts remain frozen.

## D-032 -- Enrich only the frozen evaluation split with auditable Gold

- Date: 2026-10-08
- Decision: add a canonical answer, short standard solution, and 10-point atomic rubric
  to the frozen 50-item evaluation split. Preserve the original question, choices,
  reference answer, and source identity unchanged.
- Runtime: use versioned `native_qa_gold_enricher_v1`, non-thinking DeepSeek Flash,
  batches of five, no retry or model repair, and retain every raw trace. The source
  reference is evidence, not authority; human review must reject an invalid question or
  explicitly record any scientific correction rather than silently accepting bad Gold.
- Boundary: enrichment may see reviewed required domains because it creates hidden Gold,
  but those labels must never enter Direct, Router, Expert, or Synthesizer messages.
  Do not generate agent trajectories or use evaluation records for training.
- Outcome: ten API calls produced 50/50 exportable enrichments with no rejected items
  or API failures. Deterministic normalization only wrapped solution strings as lists
  and removed zero-point explanatory rubric rows; it did not alter semantic content.
  Usage was 43,884 tokens, costing about $0.01512 off peak or $0.03025 peak.

## D-033 -- Add an explicit human curation layer after Gold enrichment

- Date: 2026-10-08
- Decision: preserve the original DeepSeek traces and enriched export, then apply a
  versioned human-review policy to produce the evaluation-ready Gold artifact. Accept
  reliable records directly, record every substantive edit, and validate the edited
  solution and 10-point rubric with the same deterministic gates.
- Outcome: all 50 records were reviewed; 44 were accepted unchanged and six were
  accepted with edits. The edits corrected a membrane-mechanics interpretation, an
  unsupported Lake Tai mechanism, sulfur-doping overgeneralization, carbonic-anhydrase
  coordination, helium-permeation units, and an overly terse tokamak answer. No source
  question, choice, reference answer, domain label, or raw API trace was changed.
- Boundary: `eval_50_gold_curated.jsonl` is evaluation-only. Its questions, Gold,
  rubrics, IDs, and required-domain labels must not be used for prompt tuning, SFT,
  cold-start, RL, or any answering-model input.

## D-034 -- Evaluate frozen baseline v2 on the independent native-QA split

- Date: 2026-10-08
- Decision: convert the curated 50-item native-QA file into the existing baseline
  record schema through a deterministic, lossless evaluation view, then run the frozen
  Direct, Routed Multi-Agent, and Oracle-domain Multi-Agent v2 paths on GPU 2 only.
- Invariants: do not edit prompts, model, seed, greedy non-thinking decoding, Router,
  expert registry, or answer schema. Direct and routed answering messages may contain
  only the question and choices. `required_domains` may control the Oracle expert list
  but must not appear in answering messages; Gold and rubrics are evaluator-only.
- Execution gate: first run one record through all three arms and verify parseability,
  GPU placement, and field isolation. Only then run the remaining records. Save output
  separately from all full49 runs and do not begin training or prompt tuning.
- Outcome: the smoke and all 150 full arm-item executions parsed successfully. Strict
  complete-rubric accuracy was 14/50 Direct and 15/50 for both Routed and Oracle.
  Oracle rescued one Direct MCQ, damaged none, and was identical to Routed on every
  binary outcome. All three arms scored 0/32 on strict qualitative completeness.

## D-035 -- Preserve raw eval50 and report an audited-48 sensitivity analysis

- Date: 2026-10-08
- Decision: do not retroactively overwrite the frozen eval50 or its raw scores after
  inspecting model failures. Preserve the intention-to-evaluate 50-item result, but
  separately flag items that are independently shown to lack a unique reliable Gold
  and report a sensitivity analysis that excludes them.
- Audit result: exclude `native-8cdcc31d9a7923` because current XL-MS chemistry does
  not support Tyrosine as the unique least-targeted residue relative to Arginine, and
  exclude `native-ac759e6bfd9e31` because both electrospinning and option C's 3D-
  printing/decellularized approaches are established skeletal-muscle scaffold methods.
- Sensitivity result: on 48 valid items, Direct is 14/48 and Routed/Oracle are 15/48;
  the only paired rescue and zero damages remain unchanged. The exclusion therefore
  corrects evaluation quality but does not alter the go/no-go conclusion.

## D-036 -- Audit qualitative rubrics before changing the agent architecture

- Date: 2026-10-08
- Decision: preserve the frozen eval50, all baseline outputs, and the original strict
  complete-rubric scores. For the 32 qualitative items only, create a separate
  evaluator-side rubric version containing three to five question-entailed core
  knowledge points per item and score partial coverage offline.
- Rubric policy: retain scientifically necessary mechanisms, causal links, conditions,
  and limitations; remove requirements that depend on a particular paper, exact study
  values, or one non-unique explanatory pathway unless the visible question asks for
  them. The audit must be based on the question and reliable Gold, not on what any Qwen
  arm happened to answer.
- Scoring policy: apply the identical fixed rubric to Direct, Routed, and Oracle.
  Prefer deterministic concept-anchor matching, preserve criterion-level evidence, and
  manually adjudicate only ambiguous matches, negations, contradictions, and arm
  differences. Report mean point coverage, explicit scientific-error counts, and
  paired improvements/regressions without replacing the historical all-points result.
- Boundary: this evaluation repair does not authorize prompt changes, Qwen reruns,
  Expert--Synthesizer changes, matched-compute Self-Refine, cold-start, RL, or OPD.
- Outcome: the audited rubric contains 124 core criteria across 32 items. Mean per-item
  coverage is 71.77% Direct, 78.85% Routed, and 77.08% Oracle. Routed improves ten
  items and degrades five relative to Direct; Oracle improves nine and degrades six.
  Responses with a material scientific error are 8, 7, and 7 respectively. This is a
  modest completeness signal, not a decisive correctness or architecture win.
- Consequence: a single pre-registered Expert--Synthesizer interaction ablation is now
  worth comparing against a matched-compute Direct Self-Refine control on unseen data.
  Do not tune or rerun on eval50, and do not start training from this result.

## D-037 -- Run one fixed Expert-Critique versus matched Self-Refine diagnostic

- Date: 2026-10-08
- Decision: at the user's explicit direction, run one fixed comparison on the audited
  48-item native evaluation set. Reuse the existing Direct predictions as identical
  drafts and reuse the existing question-only Router decisions; do not regenerate
  either. Exclude only the two MCQs already removed by D-035.
- Expert arm: each cached Router-selected domain receives the visible question,
  choices, and Direct draft and returns a bounded domain-specific critique. One shared
  reviser then produces the final answer. Critics do not independently solve or route.
- Matched Self-Refine arm: use the same number of generic critique calls as the number
  of selected experts, with the same per-call token cap, followed by the exact same
  reviser and final-answer cap. Generic passes receive no domain names or hidden labels.
  If the cached Router selected no expert, both arms use only the shared reviser, which
  is instructed to self-check the draft.
- Frozen settings: Qwen3-8B, seed 42, greedy non-thinking decoding, 512 tokens per
  critique and 1024 tokens for revision. Prompts are versioned once before the smoke;
  after the smoke only interface/parse defects may stop the run, not answer quality.
- Evaluation: MCQ exact choice on the audited 16 MCQs; the already audited 124-point
  qualitative rubric on all 32 qualitative items; explicit material-error review;
  paired improvement/degradation versus the reused Direct output; and actual calls,
  generated/input tokens, and latency. Preserve complete traces.
- Interpretation boundary: because eval50 has already been inspected, this is a
  bounded architecture diagnostic, not a new independent generalization result. It
  cannot authorize cold-start, RL, OPD, or expert SFT by itself.
- Outcome: both refinement arms completed 48/48 final answers from identical cached
  drafts. Each made 132 new calls. Two of 84 Expert critic calls hit the 512-token cap
  before closing JSON; the failure traces were retained and the shared reviser still
  produced parseable answers. No retry or prompt repair was performed.
- Result: Direct, Expert-Critique, and Self-Refine scored 14/16, 13/16, and 14/16 on
  MCQ. Qualitative mean coverage was 71.77%, 75.52%, and 75.52%; material-error
  responses were 8, 6, and 7. Expert therefore did not beat matched Self-Refine on
  coverage and damaged one previously correct MCQ, although it removed one additional
  qualitative error response.
- Compute: Expert and Self used the same 132-call structure. Expert used 107,245 input
  and 38,296 output tokens versus 104,840 and 34,399 for Self; its output-token use was
  11.33% higher and summed model latency 2.92% higher.
- Consequence: do not continue to expert SFT, Planner cold-start, RL, or OPD from this
  diagnostic. The domain-prompt critics show isolated useful corrections but no
  aggregate advantage beyond matched generic self-refinement. Any later Multi-Agent
  work requires a new held-out set and a mechanism that improves critic reliability,
  not another prompt iteration on eval48.

## D-038 -- Gate real domain-model integration on a single-model astronomy benchmark

- Date: 2026-10-09
- Decision: before inserting a specialist checkpoint into Multi-Agent, compare
  Qwen3-8B and AstroSage-8B directly on a new fixed astronomy set. Freeze all 30
  reviewed astronomy-required pilot records not used by native eval50 and remove them
  from future training eligibility.
- Inference: use the same visible questions, semantic solver instruction, greedy seed,
  and 1024-token budget. Use each released model's supported interface: Qwen chat
  template with thinking disabled, and AstroSage plain completion as documented by its
  model card. Preserve raw prompts, outputs, tokens, latency, and format fallbacks.
- Evaluation: MCQs use exact-choice rules. DeepSeek may perform one blinded
  criterion-level first pass for non-MCQs only after Gold/rubrics are frozen; Codex
  audits Gold first and resolves flagged disagreements and scientific-error claims.
  DeepSeek is not the sole final judge.
- Gate: report complementarity, especially AstroSage-correct/Qwen-wrong records, rather
  than only aggregate accuracy. Run specialist critique against a Qwen prompt expert
  only if Stage 1 demonstrates nontrivial reliable complementarity.
- Boundary: do not alter the current Multi-Agent architecture, rerun native eval48,
  start training, or use this evaluation split for prompt tuning.
- Gold outcome: six non-thinking DeepSeek batches enriched the fixed 30-source pool
  before any model answer was generated. Independent review retained 24 evaluation
  records, edited eight Gold/rubric payloads, and excluded six scientifically
  inconsistent or non-unique questions. The rejected records remain in the training
  exclusion manifest.

## D-039 -- Do not integrate AstroSage after the Stage-1 capability gate

- Date: 2026-10-09
- Decision: stop the planned AstroSage Expert-Critique comparison after the frozen
  single-model astronomy eval24. Do not modify the current Multi-Agent architecture,
  train an AstroSage expert, or treat longer specialist output as additional knowledge.
- Result: Qwen3-8B and AstroSage-8B each scored 10/11 MCQ. On the 13 non-MCQ items,
  audited mean rubric coverage was 45.38% for Qwen and 38.46% for AstroSage. Qwen had
  four responses with material scientific errors; AstroSage had six. Strict paired
  outcomes contained ten both-correct records, one Qwen-only record, thirteen
  both-wrong records, and zero AstroSage-only records.
- Complementarity: AstroSage improved coverage on the Penrose-process and GRB items,
  but both retained material errors, so neither was a reliable specialist rescue. It
  degraded seven non-MCQ items and tied four.
- Compute: AstroSage generated 22,942 tokens in 1,064.34 summed seconds versus Qwen's
  5,216 tokens in 347.65 seconds: 4.40 times the output and 3.06 times the latency.
- Evaluation lesson: the first blind-Judge run omitted the already defined JSON schema
  from the actual system message. Preserve all 13 failed traces. The corrected v2 run
  used 26,227 prompt and 9,278 completion tokens, with one harmless extra top-level
  field recovered offline and all semantic judgments explicitly audited. Never discard
  a paid response or its usage merely because deterministic validation fails.
- Boundary: this is a small astronomy benchmark comparing released systems with
  different base architectures and prompt interfaces. It rejects the present
  integration proposal; it does not claim that all astronomy-specialized models are
  intrinsically inferior. A future domain model needs a new held-out set and clear
  specialist-only wins before any integration experiment.
