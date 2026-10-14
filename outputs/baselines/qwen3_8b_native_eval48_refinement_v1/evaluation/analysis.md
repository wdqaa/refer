# Expert-Critique vs matched Self-Refine analysis

## Result

| Arm | MCQ | Qualitative mean coverage | Covered criteria | Error responses | Error claims |
|---|---:|---:|---:|---:|---:|
| Direct (cached) | 14/16 | 71.77% | 89/124 | 8 | 10 |
| Expert-Critique | 13/16 | 75.52% | 94/124 | 6 | 7 |
| Self-Refine | 14/16 | 75.52% | 94/124 | 7 | 8 |

Expert and Self both gain 3.75 mean qualitative coverage points over Direct. Expert
improves four qualitative items and degrades none; Self improves five and degrades none.
Neither rescues a Direct MCQ. Expert damages one.

## Domain-critic value

Two Expert corrections were useful where Self remained wrong:

- `native-d216c9ed276999`: Expert distinguishes ion-induced aerosol nucleation/growth
  from treating molecular ions as cloud-condensation nuclei. Self repeats the latter
  error.
- `native-99bb53aab421f9`: Expert gives the approximately 0.3 mSv annual cosmic-ray
  component and appropriately qualifies the small stochastic risk. Self fixes the dose
  but continues to overstate organ-specific cancer, cardiovascular, and cognitive
  effects at ordinary background exposure.

These corrections reduce Expert's qualitative error-response count from Self's seven
to six, but do not increase aggregate rubric coverage.

## Reliability failures

- `native-4a0fb05a96efa5`: a mathematics critic falsely states that Seurat's common
  differential-expression test is a negative-binomial GLM rather than the Wilcoxon
  rank-sum test. Expert changes the correct choice B to C. Self makes the same factual
  criticism but leaves choice B while giving a contradictory negative-binomial answer;
  exact-choice scoring counts it correct.
- `native-f7ea74c01d61ea`: Expert adds that the Ilkovic polarographic diffusion-current
  condition requires a well-stirred solution and that supporting electrolyte provides
  stirring. Classical diffusion-controlled polarography instead uses an unstirred
  solution; supporting electrolyte suppresses migration.
- Both refinement arms preserve confident errors about propene hydration,
  DPC/POLDER, luminescent recovery of non-radiative solar-cell losses, geomagnetic
  field strength, and MRI functional capability.
- Two Expert critic calls were truncated at the 512-token cap. Their incomplete raw
  JSON is preserved; there was no retry. All 96 final answers parsed.

## Compute

| Arm | New calls | Input tokens | Output tokens | Summed model latency |
|---|---:|---:|---:|---:|
| Expert-Critique | 132 | 107,245 | 38,296 | 2,335.08 s |
| Self-Refine | 132 | 104,840 | 34,399 | 2,268.78 s |

Expert uses 2.29% more input tokens, 11.33% more output tokens, and 2.92% more summed
model latency. The call structure is exactly matched item by item; the residual token
difference comes from role instructions and generated length.

## Decision

Current domain-specific critique does not outperform matched generic Self-Refine. Its
small scientific-error advantage is offset by equal coverage, one lost MCQ, higher
token use, and critic hallucinations. Do not start expert SFT, Planner cold-start, RL,
or OPD from this result. Any later collaboration test should use a new held-out set and
an explicit critic-reliability mechanism rather than another prompt iteration on this
eval48.
