# Agent Contracts

Prompts live in `src/prompts/` and use stable IDs. Long prompt text must not be
duplicated in Python code. All structured outputs are JSON.

## Source Analyzer

- Role: analyze only the original question before generation.
- Input: question and choices; not the source answer or directory label.
- Output: validity, primary/required/background domains with evidence, existing
  cross-domain status, ordinary-math status, and a coarse keep/transform/reject decision.
- Forbidden: rewriting, generating, repairing, or letting a future transformation
  affect the source-domain judgment.
- Controlled domains: physics, chemistry, mathematics, biology, and astronomy.
- Prompt: `src/prompts/source_analyzer_v3.json`.

## Planner

- Role: select the next useful action or finalize.
- Input: question and visible call trace; never hidden labels.
- Output: one action with rationale and typed arguments.
- Allowed: call one domain expert, call critic, continue, or finalize.
- Forbidden: reading `required_domains`, fabricating expert results, or calling an
  expert merely to appear interdisciplinary.
- Schema: `{action, domain?, query?, rationale, final_answer?}`.

## Domain Experts

Applies to Physics, Chemistry, Mathematics, Biology, and Astronomy.

- Role: provide only domain-relevant facts, equations, constraints, and checks.
- Input: question plus a focused request from the planner.
- Output: relevance judgment, contribution, assumptions, and confidence.
- Allowed: state that the domain is not useful.
- Forbidden: unrelated exposition, pretending expertise from another role, routing,
  or silently inventing missing data.
- Schema: `{relevant, contribution, assumptions, confidence}`.

## Critic

- Role: audit scientific correctness, consistency, missing constraints, and agent use.
- Input: question, draft answer, and visible trace.
- Output: issues and a recommendation; not a wholesale rewritten answer.
- Forbidden: adding unsupported requirements or rewarding extra calls by default.
- Schema: `{valid, issues, unnecessary_calls, missing_checks, recommendation}`.

## Question Generator

- Status: paused until Stage 01A is frozen.
- Role: minimally transform only a verified `transform_candidate`.
- Input: normalized source record plus verified source analysis, treated as fixed.
- Output: candidate question, answer, derivation, and proposed hidden metadata.
- Forbidden: explicit discipline naming, formulaic fusion wording, irrelevant details,
  or loss of source provenance.
- Schema: the candidate subset of the record in `docs/data_protocol.md`.

## Question Verifier

- Role: score validity, answer correctness, naturalness, difficulty, fusion necessity,
  and synthetic smell.
- Input: complete evaluation record, including hidden metadata.
- Output: per-axis scores, per-domain necessity judgments, issues, and accept/reject.
- Forbidden: approving a domain solely because it appears in the scenario.
- Schema: the `verification` object in `docs/data_protocol.md`.
