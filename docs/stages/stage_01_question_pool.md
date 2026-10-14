# Stage 01 -- Question Pool Overview

## Objective

Validate whether real science QA can anchor natural questions whose solutions have
genuine, hidden cross-domain dependencies, and whether prompted agents collaborate
usefully on them.

## Research Questions

- Are prompt-based experts meaningfully differentiated?
- Which source questions are already cross-domain?
- Can minimal transformations add a necessary reasoning dependency without synthetic
  wording or abnormal length?
- Does each `required_domains` label survive domain ablation?
- Do multi-agent baselines help on accepted medium/hard items?

## Stage Split

1. **01A -- Source Domain Dependency Analysis:** understand the original question.
2. **01B -- Natural Cross-domain Transformation:** transform verified candidates.
3. **01C -- Question Pool Quality Audit:** combine hard rules and judgment.
4. **01D -- Prompt Multi-Agent Baseline:** compare direct, fixed, and routed answering.

01A has provisionally passed. A bounded 01B DeepSeek pilot is active; repair and
answering baselines remain paused.

## Implementation

The normalized loader, 30-item source-analysis gold set, Source Analyzer v3,
no-repair evaluator, and 24 tests are implemented. Earlier generation code remains
available for reproducibility but is not the active pipeline.

## Experiments and Results

The first 30-item run revealed that three nominal cross-domain gold examples were
themselves ambiguous. After replacing them with stronger dependency cases, v3
achieved 90% exact domain match, 100% cross-domain precision, 80% recall, and zero
math/subfield false positives. See `EXP-007` and `EXP-008`.

## Problems

- Schemas differ substantially but are covered by the normalized loader.
- Source directory labels are noisy proxies for actual required knowledge.
- Validity and a few chemistry/physics boundaries remain less stable than the main
  domain-routing labels.
- `transform_candidate` is subjective and should not dominate the 01A exit decision.

## Decisions

See D-001 through D-009 in `docs/decisions.md`.

## Next Step

Freeze the 30-item protocol after a final spot review, then specify 01B so the
generator consumes verified analysis without reclassifying the source.
