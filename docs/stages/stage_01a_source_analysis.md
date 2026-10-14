# Stage 01A -- Source Domain Dependency Analysis

## Question

Can the model identify which of five top-level domains are genuinely indispensable
to solving the original question, without being distracted by topic words, source
folders, subfields, or routine calculation?

## Protocol

- 30 source questions: six boundary categories, five items each.
- Model sees only the question and choices.
- Analyzer performs classification plus evidence; it never generates a new question.
- Outputs are parsed strictly, with no repair or silent domain canonicalization.
- Gold annotations may be revised during protocol audit, but are frozen before any
  prompt comparison.

## Primary Metrics

- exact required-domain match;
- cross-domain precision and recall;
- mathematics false-positive rate;
- subfield false-positive rate.

Validity, primary-domain accuracy, and decision accuracy are reported. The
`transform_candidate` decision is secondary because natural extensibility is more
subjective than domain necessity.

## Current Result

The audited v3 run parsed 30/30 records, reached 90% exact domain match, 100%
cross-domain precision, 80% recall, and 0% math/subfield false-positive rates.
This provisionally passes the precision-first gate. The three domain-boundary errors
and two validity disagreements remain recorded for review rather than repaired.

## Handoff to 01B

Only records marked `transform_candidate` after verification may enter 01B. The
generator must treat source analysis as fixed and may either add one indispensable
domain through a minimal natural change or reject the transformation.
