# Stage 01 Native QA Mining

## Goal

Mine unchanged, high-quality native single-domain and native cross-domain QA from the
existing science dataset. This route replaces synthetic cross-domain generation. It
does not rewrite questions or generate agent trajectories or model training data. Only
the subsequently frozen evaluation subset receives evaluator-side solutions and rubrics.

## Source Universe

The scan treats `datasets/science_dataset/mixed_dataset/` as read-only. The known
concatenated duplicate `mixed/mixed_train.jsonl` is excluded. The remaining 19 files
contain 180,029 records across astronomy, biology, chemistry, geography, mathematics,
and physics. Exact normalized question-answer deduplication leaves 178,814 unique pairs;
1,215 source records are exact duplicates.

## Classifier

`native_qa_classifier_v1` receives only the original question, choices, and reference
answer. It does not see the source-directory domain. In batches of five, non-thinking
DeepSeek Flash returns only validity, question type, minimum required domains, domain
type, answer verifiability, and keep priority. Deterministic gates enforce the requested
enums and consistency rules. Invalid records may have no required domain; their
verifiability and priority must be `unreliable` and `low`.

The calibration combines 30 existing human-annotated boundary records with 30 fresh
records sampled across all six directories. The first prompt smoke returned
`single_domain` instead of the required `single` and mislabeled choice-bearing items;
all 30 outputs were preserved as schema failures. A second launch was rejected by the
API before billing because JSON mode requires the literal word `JSON` in the prompt.
Prompt v1.2 corrected both interface issues. It produced 60/60 schema-valid labels in
12 calls without modifying source content.

## Calibration Results

The 60 labels contain 50 valid and 10 invalid records, 49 single-domain and 11 proposed
native cross-domain records. The model marked 26 high, 24 medium, and 10 low priority.

Against the 30-record human gold set:

| Metric | Result |
|---|---:|
| Validity accuracy | 93.33% |
| Required-domain exact match | 76.67% |
| Domain-type accuracy | 76.67% |
| Cross-domain precision | 40.00% |
| Cross-domain recall | 80.00% |

All eight machine `high_priority_cross_domain` candidates came from the human-gold
half. Manual review retained four and rejected four false boundaries. The rejected
items mechanically split astronomy from standard astrophysics or added chemistry to
molecular/quantum physics because the question mentioned an atom or molecule. Therefore
the apparent high-priority cross-domain set has only 50% precision and is not a verified
training artifact.

## Cost

The evaluable 60-item run used 22,091 prompt tokens and 3,668 completion tokens. Under
the official 2026-10-08 DeepSeek-V4.1-Flash rates, it cost approximately $0.0050 off
peak or $0.0100 at peak. The earlier paid schema smoke cost another $0.0053 off peak or
$0.0105 peak; the rejected HTTP 400 requests used zero tokens. Total calibration spend
was therefore about $0.0103 off peak or $0.0205 peak. Linear scaling after exact
deduplication estimates approximately $14.92 off peak or $29.84 at peak for 178,814
unique pairs. This is a token-based projection, not authorization to run the full scan.

## Gate

The full traversal is paused. The pipeline is structurally adequate, validity filtering
is promising, and single-domain controls are retained correctly, but cross-domain
precision is far below the high-precision requirement. Do not spend the projected full
cost until a small, newly held-out boundary set shows materially better precision.

No reasoning generation, cold-start, RL, OPD, Qwen baseline rerun, or source rewriting
was performed.

## Held-out v1.3 Calibration

Prompt v1.3 relaxes retention, not domain necessity. Reliable simple facts, qualitative
questions, and single-domain routing controls may be retained as high or medium rather
than being penalized for low complexity. The required-domain deletion test and mature-
subfield boundary rule remain strict.

The fixed held-out set contains 15 verified native-single QA, nine verified native-
cross-domain QA, and six audited invalid controls, with no overlap with the 60 records
used for EXP-030. Six API calls produced 30/30 schema-valid labels.

| Metric | Result |
|---|---:|
| Valid-item retention | 22/24 (91.67%) |
| Invalid-control rejection | 4/6 (66.67%) |
| Valid required-domain exact match | 19/24 (79.17%) |
| Valid domain-type accuracy | 21/24 (87.50%) |
| Valid cross-domain precision | 6/6 (100%) |
| Valid cross-domain recall | 6/9 (66.67%) |
| Human acceptance, all cross candidates | 6/8 (75.00%) |
| Human acceptance, high cross candidates | 5/6 (83.33%) |

The distinction between the last two precision figures matters: among valid gold QA,
every predicted cross-domain item was a true cross-domain case. The final mined pool
also contained two invalid controls that the classifier failed to reject, reducing
end-to-end candidate precision to 75%. Three true cross-domain items were conservatively
classified as single-domain.

This run cost approximately $0.00317 off peak or $0.00634 peak. The prompt is now
frozen. The result supports only a staged medium pilot with sampled validity and
cross-domain audits; it does not authorize the projected 178,814-pair full scan.

## Staged 500-Item Pilot

The authorized pilot used 500 exact-unique, unchanged source QA with no overlap with
the preceding 90 records. The manifest is approximately balanced by source directory
and MCQ/open surface. These strata were hidden from the classifier.

All 100 API requests succeeded. Strict export retained 496 labels; four records were
preserved as schema failures because the model called them valid while returning no
required domain. The classifier labeled 452 records valid, 395 single-domain, 101
native cross-domain, 208 high priority, 243 medium, and 45 low.

Every one of the 67 high-priority cross-domain candidates received semantic review:

| Disposition | Count |
|---|---:|
| Accepted cross-domain | 17 |
| Reclassified as single-domain | 43 |
| Rejected QA | 7 |

Only 25.37% of machine high-cross candidates remained cross-domain. The main failure
was again mature-field over-splitting: 26 candidates were labeled astronomy+physics,
although most were ordinary astrophysics, relativity, or particle-physics questions.
The review did not discard otherwise useful items; 43 are preserved separately with a
corrected single-domain label. Seven were removed for unreliable Gold, missing
constraints, model-dependent answers, or unstated external-study dependence.

A deterministic audit sampled two machine high-single and two invalid/low decisions
per source directory. All 12 high-single decisions were upheld. Nine of 12 invalid/low
decisions were upheld; three coherent questions had been rejected too aggressively.
Thus the relaxed retention policy works substantially better for single-domain QA than
the cross-domain boundary classifier does on natural source distributions.

The pilot used 242,202 tokens and cost about $0.04276 off peak or $0.08552 peak. The
178,814-pair full traversal remains stopped. Classifier v1.3 may be used as a broad
candidate filter only if all proposed cross-domain records receive semantic review.

### Full high-single review

All 141 machine high-priority single-domain candidates were then reviewed offline. Of
these, 116 remain high priority, 12 were downgraded to medium, and 13 were rejected.
Simple factual and basic calculation questions were retained when correct and useful;
rejections were limited to material faults such as missing figures or detection
thresholds, malformed geometry, no correct option, incorrect numerical Gold, or an
unreliable scientific explanation.

After including the 43 reviewed high-cross false positives under their corrected
single-domain labels, the pilot contains 171 reviewed native-single QA and 17 reviewed
native-cross-domain QA, for 188 reviewed records total. These files remain separate
from the frozen 49-record baseline set. Source-identity and exact question-answer
overlap checks against that set are both zero.

### Medium-cross review and evaluation freeze

The remaining 31 valid medium-priority cross-domain candidates were also reviewed:
18 were accepted as cross-domain, eight were corrected to single-domain, and five were
rejected. The reviewed pilot therefore contains 179 single-domain and 35 cross-domain
QA, 214 records total.

A deterministic seed-131 split freezes 50 previously unseen evaluation records: 25
cross-domain and 25 matched single-domain controls, comprising 32 qualitative questions
and 18 MCQ. The remaining 164 reviewed records are training candidates. Evaluation IDs
must not enter prompt tuning, SFT, cold-start, RL, or OPD.

## Frozen Evaluation Gold

Only the frozen 50-item split was enriched. The versioned Gold prompt asked DeepSeek
Flash for a canonical answer, a short standard solution, and an atomic 10-point rubric
while preserving the keyed MCQ choice. Ten no-retry batches produced 50/50 exportable
records. Usage was 43,884 tokens, costing about $0.01512 off peak or $0.03025 peak.

All 50 outputs then received semantic review. Forty-four were accepted unchanged and
six received explicit, versioned human edits for scientific accuracy or answer
completeness. The raw traces and normalized model-only output remain preserved. The
evaluation-ready artifact is `eval_split_v1/eval_50_gold_curated.jsonl`; the 164
training candidates have not received generated reasoning or rubrics.

The curated Gold and `required_domains` are evaluator-only. Answering models may see
the unchanged question and choices, but not domain labels, reference answers, standard
solutions, rubrics, or review notes.

## Independent Baseline Gate

Frozen baseline v2 ran on the 50-item split without prompt tuning. Direct scored 14/50;
Routed and Oracle each scored 15/50. All 150 outputs parsed and all model-visible fields
passed leakage checks. Objective MCQ accuracy was 14/18 versus 15/18, while all arms
were 0/32 under strict qualitative complete-rubric scoring. The only multi-agent rescue
was a cytochalasin-B MCQ, and Oracle had no outcome advantage over learned routing.

A post-run source-independent audit identified two MCQs with non-unique or unreliable
Gold. The original split and result remain untouched; an audited-48 sensitivity report
excludes them and obtains 14/48 Direct versus 15/48 Routed/Oracle. This does not change
the no-go decision for Planner cold-start or RL.

## Qualitative Rubric Audit

The 32 qualitative items were audited without changing their questions, curated Gold,
baseline prompts, or model traces. A separate evaluator version contains three to five
question-entailed scientific core points per item (124 criteria total). Paper-local
numbers, exact study observations, and a single non-unique solution path were removed
when the visible question did not require them. Examples include replacing the exact
microplastics strain/time/percentage requirement with biodegradation mechanism and
evidence, accepting valid spectral-conversion routes beyond one LSC geometry, and
removing paper-specific permafrost and Qionghai spatiotemporal details.

Deterministic concept-anchor matching handled the primary pass. Fifty-nine ambiguous
criterion decisions and 26 material error claims were retained in a versioned
adjudication file; no review flags remain unresolved. The same fixed rubric was applied
to every arm.

| Arm | Mean item coverage | Covered criteria | Error responses | Error claims | Preserved strict |
|---|---:|---:|---:|---:|---:|
| Direct | 71.77% | 89/124 | 8 | 10 | 0/32 |
| Routed Multi-Agent | 78.85% | 98/124 | 7 | 8 | 0/32 |
| Oracle Multi-Agent | 77.08% | 96/124 | 7 | 8 | 0/32 |

Relative to Direct, Routed improves 10 items, degrades 5, and ties 17, for +7.08 mean
coverage points. Oracle improves 9, degrades 6, and ties 17, for +5.31 points. The main
improvements occur on propene hydration, excited-state calculations, X-ray spectra,
polarography, and cosmic-ray reasoning. Regressions show expert/synthesis dilution on
protein aggregation, PET, tokamak heating, luminescent solar conversion, and selected
environmental questions.

Interpretation: the old all-or-nothing rubric hid useful differences, and Multi-Agent
does modestly improve completeness. It does not yet establish substantially better
scientific correctness: the error-response count falls only from eight to seven, and
Oracle is slightly below Routed. A narrow Expert--Synthesizer interaction experiment is
worth one pre-registered comparison against matched-compute Direct Self-Refine on
unseen data. Eval50 must not be used for prompt tuning or rerun-based optimization, and
this result does not authorize cold-start, RL, or OPD.
