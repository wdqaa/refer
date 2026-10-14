# Source Dataset Analysis

Generated: `2026-09-27T15:56:04+08:00`

## Scope

- Files: 20
- Non-mixed raw records: 180,029
- Mixed aggregate records (reported separately): 158,985
- All records including mixed: 339,014

## Domain counts

| Domain | Raw records | Question words (median / P90) | Answer words (median / P90) |
|---|---:|---:|---:|
| astronomy | 4,585 | 23.0 / 46.0 | 282.0 / 484.0 |
| biology | 27,540 | 21.0 / 33.0 | 378.0 / 480.0 |
| chemistry | 24,391 | 27.0 / 62.0 | 228.0 / 446.0 |
| geography | 36,426 | 19.0 / 30.0 | 78.0 / 149.0 |
| math | 66,858 | 44.0 / 72.0 | 81.0 / 157.0 |
| mixed | 158,985 | 30.0 / 62.0 | 110.0 / 430.0 |
| physics | 20,229 | 32.0 / 58.0 | 262.0 / 473.0 |

## Coarse distributions (including mixed)

- Answer format: `{'multiple_choice': 35954, 'open_ended': 303060}`
- Reasoning proxy: `{'conceptual': 160359, 'quantitative': 178655}`
- Difficulty proxy: `{'easy': 144979, 'hard': 23411, 'medium': 170624}`

## Interpretation limits

- Counts are raw records; cross-file deduplication has not been applied.
- The mixed domain is reported separately and excluded from non_mixed_total.
- Lengths are approximate word/CJK-character counts of normalized stems.
- Difficulty and reasoning labels are rule-based proxies, not gold labels.
- Directory domain is source provenance, not a verified required-domain label.
- The JSON report contains per-file schemas, distributions, and bounded examples.
