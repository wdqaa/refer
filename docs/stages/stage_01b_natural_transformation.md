# Stage 01B -- Natural Cross-domain Transformation

## Active workflow

1. Select only verified `transform_candidate` sources.
2. DeepSeek Flash proposes a minimal transformation and candidate solution.
3. A separate blind call sees only the transformed question and solves it.
4. Hard gates require the original domains and exactly one new domain to appear in
   the blind solution, with matching multiple-choice answers.
5. Protocol review rejects parallel-subquestion conjunctions and other semantic failures;
   a person must still confirm records before they are called final gold.

No repair is used. Generator answers are candidates, not gold labels.

## First pilot

Prompt v1 exposed the new-domain fact directly. Prompt v2 preserved the new domain
but replaced the original-domain reasoning with supplied premises. Prompt v3 added
ablation tests for both original and new domains.

With frozen v3, five questions passed machine checks. Codex protocol review retained
four candidates and rejected one collider-plus-rotation-curve item because it merely
concatenated two independent observations. These are not final gold until user/human
confirmation. Run usage was 42,768 tokens, estimated at $0.0221 at published off-peak
prices, excluding prompt-development calls.

## Next gate

Before scaling, the prompt or semantic judge must explicitly reject transformations
whose domain-specific conclusions can be solved independently and then concatenated.
