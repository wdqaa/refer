# Repository Working Rules

## Scope

The active scope is Stage 00 (bootstrap) and Stage 01 (question-pool validation).
Do not start LoRA training, GRPO, reward-model training, OPD, or large-scale jobs.

## Priorities

1. Build a small, natural, scientifically valid question pool.
2. Verify that each declared domain is actually needed.
3. Compare prompt-based single-model and multi-agent baselines only after pool validation.

## Data Safety

- Treat `datasets/science_dataset/mixed_dataset/` as read-only.
- Write normalized data to `datasets/processed/` and candidate questions to
  `datasets/question_pool/`.
- Never expose `required_domains` to the planner or answering model.
- Preserve source identity and the original question in generated records.

## Implementation

- Keep prompts in `src/prompts/`; do not embed long prompts in Python modules.
- Give prompts stable version identifiers.
- Every executable Python script must document purpose, input, output, and an
  example command, and must use explicit CLI arguments.
- Use deterministic seeds where supported and record generation parameters.
- Start with 20--50 samples; do not scale until diagnostics and human review pass.
- Add lightweight tests for loaders, schemas, prompts, parsers, and serialization.

## Documentation

- `docs/PROJECT.md` is the current-status source of truth.
- Record all experiments, including failures, in `docs/experiments.md`.
- Record consequential design changes in `docs/decisions.md` before implementing them.
- Keep documentation concise and update the relevant stage document after each module.

