# Native eval50 qualitative coverage v2

Rubric: `native_eval50_qualitative_rubric_v2.0`; items: 32; criteria: 124.

| Arm | Mean item coverage | Micro criterion coverage | Error responses | Error claims | Historical strict |
|---|---:|---:|---:|---:|---:|
| direct | 71.77% | 71.77% | 8 | 10 | 0/32 |
| routed_multi_agent | 78.85% | 79.03% | 7 | 8 | 0/32 |
| oracle_multi_agent | 77.08% | 77.42% | 7 | 8 | 0/32 |

## Paired coverage changes

- `routed_multi_agent_minus_direct`: mean delta +7.08%; improved 10; degraded 5; tied 17.
- `oracle_multi_agent_minus_direct`: mean delta +5.31%; improved 9; degraded 6; tied 17.

Unresolved review flags: 0.

Coverage is evaluator-side partial credit under the audited fixed rubric; it does not replace the preserved historical strict complete-rubric result.
