# Source Analyzer v3 -- Error Analysis

## Core domain errors

| ID | Gold | Prediction | Diagnosis |
|---|---|---|---|
| `sa_cross_05` | biology + chemistry | biology | Molecular shape, receptor fit, and volatility were absorbed into biology; this is the single false negative among five audited cross-domain items. |
| `sa_surface_02` | mathematics | biology | A pure percentage word problem was classified by its frog-mutation story despite a near-matching math-only few-shot. This is a genuine setting-versus-dependency failure. |
| `sa_mismatch_04` | chemistry | physics | Molecular-orbital formation sits near a chemistry/quantum-physics boundary. Under this protocol it is chemistry because the item tests bonding/MO terminology. |

## Validity disagreements

`sa_mismatch_03` omits the combustion heat needed to determine the calorimeter
constant. `sa_mismatch_05` omits the standard/reference potential needed to infer
hydrogen-ion activity. Gold marks both `reject`; the model treated the missing values
as standard background data and accepted them. These affect validity accuracy, not
the cross-domain routing result.

## Decision disagreements

Most remaining mismatches are `transform_candidate` versus `keep_control`. They
reflect conservative judgments about whether a natural extension exists, not a
failure to identify existing required domains. Decision accuracy is therefore a
secondary diagnostic for Stage 01A.

## Interpretation

The audited result supports moving forward under a precision-first policy:
cross-domain precision is 100%, recall is 80%, and both math and subfield false
positive rates are 0%. The next generator pilot should consume verified analysis as
fixed input and reject uncertain transformations rather than repair them.
