# Stage 01 -- Bridge-Guided Cross-Domain Transformation

## Objective

Test a small human-authored bridge library as a generation prior without source-source
retrieval, embeddings, Top-k matching, knowledge graphs, or matching scores. For each
single-domain source, DeepSeek must select one allowed bridge or return `none`.

## Bridge library

`configs/data/interdisciplinary_bridge_library.yaml` contains 49 bridge names across
the five source domains and target domains from the six callable experts. Mathematics
is allowed only for nontrivial dynamics, probability/statistics, optimization,
estimation, or differential-equation analysis. The library is a prompt prior, not a
scientific ontology.

## Calibration set

The fixed calibration contains 25 sources from the unused 150-source pool: chemistry
8, biology 7, astronomy 4, physics 4, and geography 2. Primary domains were manually
corrected before selection. The set intentionally mixes plausible bridges with mature-
field false boundaries, weak reference answers, and source-directory noise.

## Results

- Selector v1: 25 requests, 24 schema-valid, one invalid, 0 `select`, 24 `none`.
- The first diagnosis showed that v1 incorrectly required the source to already contain
  the future quantitative model or measurements.
- Selector v1.1 made one bounded clarification: a Transformer may later add minimal
  hypothetical values, initial conditions, and boundary conditions while preserving
  the source mechanism. It was tested on four human-positive and four boundary-negative
  records rather than rerun on all 25.
- Selector v1.1 smoke: eight requests, seven schema-valid, one invalid, 0 `select`,
  seven `none`. All four negative controls were rejected appropriately, but all four
  human-positive bridges were also rejected.
- No source entered the Transformer, Blind Solver, or Judge. Accepted QA: zero.

## Interpretation

The bridge library constrained false-boundary generation, but the Selector achieved
that precision by adopting a degenerate all-NONE policy. The v1.1 clarification did not
restore recall. This does not test the downstream Transformer quality because the
pipeline correctly stopped at its first gate.

Compared qualitatively with the prior free-form wave, bridge guidance eliminated
machine-proposed false-boundary transforms only by eliminating all transforms. Neither
method produced a semantically accepted cross-domain QA in its calibration.

## Decision

Do not add a third Selector prompt, force human-positive bridges into the Transformer,
expand API calls, or start two-source retrieval. Reassess whether bridge choice should
be a small human-authored generation prior rather than another model decision. Frozen
QA, baselines, cold-start, and RL remain unchanged.
