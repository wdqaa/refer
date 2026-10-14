# Stage 01 Human-Specified Bridge Transformation Smoke

## Purpose

This bounded go/no-go experiment isolates the Transformer from the failed automatic
Bridge Selector. Eight existing reviewed sources receive a fixed human-authored target
domain and bridge. The Transformer may only transform the source or reject the fixed
pair. No bridge selection, retrieval, repair, source expansion, or training is involved.

## Frozen Pairs

The four positive pairs are polymer--mathematics/optimization, insulin
feedback--mathematics/nonlinear dynamics, nephron--physics/membrane transport, and
cosmic-ray dose--biology/radiation health effect. Four negative controls cover mature
astrophysics, physical chemistry, decorative biology, and a source whose directory
label does not match its projectile-motion content. The exact manifest is
`configs/data/human_bridge_transform_smoke_v1.json`.

## Pipeline

Each pair runs directly through the fixed-bridge Transformer. Only valid transformations
continue to the Blind Solver, Judge, deterministic gate, and semantic review. There is
no repair loop. All raw and failed traces are retained under
`datasets/processed/human_bridge_transform_smoke_v1/`.

The initial thinking-enabled execution produced eight technical failures: completions
spent the 6000-token budget in reasoning or ended as truncated JSON. Those traces are
preserved in `trace.jsonl` but do not count as scientific outcomes. The bounded rerun
disabled thinking and changed no scientific prompt or pair; its trace is
`trace_nonthinking.jsonl`.

## Results

In the non-thinking run, all four positive pairs were transformed, but none passed the
downstream machine gates and none survived semantic review. The failures were substantive:

- polymer became a self-contained optimization exercise with decorative chemistry and
  inconsistent provisional arithmetic;
- insulin became an arbitrary ODE simulation whose biological mechanism was supplied in
  the equations rather than required from the solver;
- nephron omitted the compartment volumes needed for a diffusion half-time and invented
  them only inside the derivation;
- cosmic-ray dose supplied the complete dose-to-risk rule and reduced both domains to
  generic conversion and arithmetic.

Among the four negative controls, one was correctly rejected, two were transformed into
pseudo-fusions, and one returned malformed JSON. The pseudo-fusions were the accretion-
disk radiation case, which mechanically split normal astrophysics, and the organic-
reaction/prodrug case, whose appended enzyme and binding parameters did not participate
coherently in the solution.

| Measure | Result |
|---|---:|
| Positive Transformer transforms | 4/4 |
| Positive final human accepts | 0/4 |
| Negative correct Transformer rejects | 1/4 |
| Negative pseudo-fusions | 2/4 |
| Negative technical errors | 1/4 |
| Final accepted cross-domain QA | 0 |

## Decision

This is a no-go result. Manually supplying a plausible bridge makes the Transformer
attempt every positive, but it does not make both domains indispensable or preserve a
self-contained, correct solution. Because zero positives meet the preregistered semantic
gate, no further Transformer prompt tuning, sample expansion, automatic bridge selection,
or synthetic single-source cross-domain production is warranted.

The data route should instead use native single-domain QA, native cross-domain QA,
already verified transformed QA, and externally grounded real cross-domain benchmarks or
data sources. This experiment does not authorize cold-start, RL, OPD, or any Qwen baseline
change.
