# Stage 01 -- Transformability Source Selector Calibration

## Objective

Replace keyword-based fusion-affordance sampling with an explicit judgment:

> Can this source question be naturally extended so that a new, independently
> necessary domain contribution changes the reasoning needed for one integrated answer?

The Selector does not transform questions, write answers, or assign final
`required_domains` labels.

## Target relations

1. `quantitative_modeling`: nontrivial statistics, probability, optimization,
   dynamical systems, or another mathematical model; routine arithmetic is excluded.
2. `constraint_coupling`: the source mechanism and an independent new-domain
   constraint must jointly determine one conclusion.
3. `measurement_inference`: an observable is connected to a latent property or
   scientific conclusion through a genuine measurement or interpretation model.

## Candidate and selection protocol

1. Draw 150 unused sources deterministically across the five source domains and broad
   surface forms. Do not use relation keywords or affordance hints.
2. Run `source_transformability_selector_v1` with `deepseek-flash`, thinking disabled,
   strict JSON, at most one request per source, and no repair loop.
3. Shortlist only records with `self_contained=true`, `answer_grounded=true`,
   `transformability=high`, and no risk flags. Cap the human shortlist at 30.
4. Human-review the proposed relation, new-domain prior, and boundary legitimacy.
5. Only after a separate report and approval may 10--15 sources enter the existing QA
   pipeline. At least five must become semantically valid cross-domain QA to pass.

## Required risks

- `underspecified`
- `paper_specific`
- `time_sensitive`
- `external_context_required`
- `answer_already_supplied`
- `false_boundary_split`
- `too_broad`
- `fact_lookup_only`
- `no_natural_extension`

## Current status

- Design recorded in D-023.
- Prompt, deterministic candidate-pool builder, Selector runner, strict shortlist, and
  lightweight tests are implemented.
- The local candidate pool contains 150 unused sources: astronomy 40, biology 28,
  chemistry 28, physics 27, and geography 27; 44 are MCQ and 106 are open response.
  It excludes the previous 150-source protocol and 50-source calibration batches.
- The pool uses no relation keywords and stores no affordance hints. The Selector is
  solely responsible for judging actual transformability.
- The 150-source Selector run is complete: 137 schema-valid, 13 preserved invalid,
  and no API errors. Valid results comprise 4 high, 55 medium, and 78 low.
- The strict human shortlist contains only four records. One polymer-chemistry source
  was judged worth transforming; three were rejected as false boundary/source-label
  effects. This is insufficient to form the planned 10--15 source QA calibration.
- A recall-oriented secondary audit reviewed all 15 clean-medium records
  (`self_contained`, `answer_grounded`, and no risk flags). Three survived: nephron
  membrane transport, insulin-feedback dynamics, and cosmic-ray dose/health risk.
  Combined with the one strict-high survivor, only four of 19 reviewed sources remain.
- No Transformer calls or QA generation were made. The gate failed with zero QA added;
  baseline evaluation, cold-start, and RL remain untouched.
- Per D-024, the next research option is a separately designed two-source grounded
  synthesis protocol rather than another random single-source wave.
