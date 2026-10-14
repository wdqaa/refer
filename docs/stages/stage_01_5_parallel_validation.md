# Stage 01.5 -- Data Expansion and Prompt Multi-Agent Gate

## Objective

Run two tracks in parallel before any training:

1. Expand 49 verified QA records toward 150--250, prioritizing astronomy, objective
   questions, and non-mechanism fusion.
2. Build prompt-only Qwen3-8B Direct, routed Multi-Agent, and oracle-domain
   Multi-Agent baselines.

## Baseline contract

```text
Direct: question -> Qwen3-8B -> answer
Routed: question -> Router -> selected Experts -> Synthesizer -> answer
Oracle: question -> evaluation-only required_domains -> Experts -> Synthesizer -> answer
```

The Router sees only the question and choices. Oracle labels stay inside the Python
orchestrator; Experts and Synthesizer receive only the visible question and expert
contributions. Routed and Oracle use the same expert and synthesizer prompts.

## Go/no-go question

Does Oracle-domain Multi-Agent materially improve answer correctness over Direct Qwen
on the same verified questions, and at what call/token cost? Routing quality is reported
separately by exact set match, precision, recall, and over/under-routing.

## Current work

- Data track: the frozen `sources_50.jsonl` wave is complete. Source Analyzer returned
  49 valid analyses; strict selection produced 14 QA candidates. Five machine accepts
  became four curated native-single records, while zero non-mechanism transformations
  survived semantic review. The calibration gate failed, so another wave is paused
  pending a selector revision.
- Agent track: all three paths, trace schema, conservative heterogeneous scoring, and
  the three-item smoke, full v1, and fixed-data completeness v2 are complete. All 147
  latest v2 item-arm predictions parsed, label-leak checks passed, and Router choices
  exactly matched v1.
- Training, RL, and Qwen agent-trajectory generation remain out of scope.

## Execution gates

- Data calibration: 50 Source Analyzer calls, followed by at most 20 non-transform
  QA items and 12 transformations. The total first-pass ceiling is 138 calls
  (`50 + 20*2 + 12*4`). Inspect the Analyzer output and selected manifest before the
  latter calls. At least two non-mechanism transformations must survive semantic review.
- Baseline: run all roles non-thinking and record calls, output tokens, and latency.
  MCQ/numeric can be scored deterministically; non-exact short, symbolic, and
  qualitative answers require blinded rubric review.
- Interpretation: Oracle minus Direct measures the whole oracle-routed multi-call
  pipeline, not pure role specialization, because compute is larger. If positive,
  add a matched-compute Direct/self-refine control before attributing the gain.

## Baseline v1 result

- Direct: 17/49 (34.7%).
- Question-routed Multi-Agent: 16/49 (32.7%).
- Oracle-domain Multi-Agent: 16/49 (32.7%).
- Oracle minus Direct: -1/49 (-2.0 percentage points); paired bootstrap 95% interval
  approximately [-14.3, +10.2] percentage points.
- Router: 65.3% exact domain-set match, 81.1% micro precision, and 84.5% micro recall.
- Calls: Direct 49, Routed 172, Oracle 120. Mean input+output tokens per item were
  about 459, 1712, and 1395 respectively; no hidden-label leakage was detected.
- Review: all 147 final predictions were scored. Deterministic rules covered objective
  exact/tolerance cases; 111 predictions received candidate-code-blinded strict rubric
  review by one reviewer. A second independent review is still desirable.
- Gate outcome: no-go for Planner cold-start/RL under baseline v1. The Oracle path did
  not improve accuracy despite higher compute. Continue Route A and diagnose the
  concise expert/synthesizer protocol on a separate development split.

## Baseline v2 completeness result

- Direct: 16/49 (32.7%); qualitative 1/25, objective 15/24.
- Question-routed Multi-Agent: 20/49 (40.8%); qualitative 3/25, objective 17/24.
- Oracle-domain Multi-Agent: 20/49 (40.8%); qualitative 2/25, objective 18/24.
- Oracle minus Direct: +4/49 (+8.2 percentage points); paired bootstrap 95% interval
  [0.0, +18.4] percentage points. Oracle rescued five Direct failures and damaged one.
- V1 to v2: Direct -1, routed +4, Oracle +4. The unchanged Router selected exactly the
  same domains, isolating the difference to answer-completeness prompts.
- Gate outcome: positive prompt-only trend, but not a training go. Prompts were adapted
  after observing v1 on the same 49 records, and qualitative completeness remains only
  2/25 for Oracle. Freeze v2, test it on unseen verified QA, and use a matched-compute
  Direct/self-refine control before cold-start or RL.

## Data calibration v1 result

- 50 frozen sources; 49 valid Source Analyzer records and one preserved length failure.
- 14 selected QA candidates; 5 machine accepts; 4 semantic accepts.
- Final semantic accepts: two MCQ and two short-answer native-single controls.
- Non-mechanism transformations: 1/4 machine accepted, 0/4 semantic accepted.
- Gate outcome: fail. Do not scale calls or merge these records into the frozen 49.
  Revise source selection to require independently necessary domain pairs and reject
  broad, study-specific, pre-supplied, or subfield-split candidates earlier.

## Native eval48 matched-compute refinement result

The fixed Expert-Critique and Self-Refine diagnostic reused the same Direct drafts and
cached question-only Router. Both arms used 84 critique calls and 48 calls to one shared
reviser. No Gold, rubric, or required-domain label entered model-visible messages.

- MCQ: 14/16 Direct, 13/16 Expert-Critique, 14/16 Self-Refine.
- Qualitative mean coverage: 71.77% Direct and 75.52% for both refinement arms.
- Material-error responses: 8 Direct, 6 Expert-Critique, 7 Self-Refine.
- Expert versus Self compute: identical 132-call count; Expert generated 11.33% more
  output tokens and used 2.92% more summed model latency.
- Reliability: two Expert critic JSON responses were truncated at the fixed cap; all
  final answers parsed. Expert supplied useful unique cosmic-ray corrections but also
  hallucinated a wrong Seurat test and an incorrect stirring condition for polarography.

Gate outcome: current domain-specific critique does not outperform matched generic
self-refinement overall. Do not begin Planner cold-start, expert SFT, RL, or OPD. A
future collaboration experiment would require a new held-out set and an explicit
critic-reliability mechanism; do not tune again on eval48.
