# Native eval48 Expert-Critique vs Self-Refine

| Arm | MCQ | Qualitative mean coverage | Error responses | Error claims | Calls | Output tokens |
|---|---:|---:|---:|---:|---:|---:|
| direct | 14/16 | 71.77% | 8 | 10 | 48 | 12116 |
| expert_critique | 13/16 | 75.52% | 6 | 7 | 132 | 38296 |
| self_refine | 14/16 | 75.52% | 7 | 8 | 132 | 34399 |

## Paired changes

- `expert_critique_minus_direct`: MCQ rescued 0, damaged 1; qualitative coverage +3.75%, improved 4, degraded 0, tied 28.
- `self_refine_minus_direct`: MCQ rescued 0, damaged 0; qualitative coverage +3.75%, improved 5, degraded 0, tied 27.

Unresolved review flags: 0.

The same audited rubric is used for all arms. This eval48 was used to design the fixed refinement prompts, so results are diagnostic rather than an independent generalization estimate.
