# Additional Baseline Expansion Design

## Completion Notes (2026-04-05)
- The design is implemented: `RBPModel`, `GDDPFisherModel`, and `BOQAModel` were added to the dynamic ensemble registry.
- Subsequent follow-up work enabled BOQA against the repository's `2022` HPO assets and quieted BOQA's external Java logs, so the current default script includes all three targeted baselines in the available-model set.

## Goal
Expand the dynamic diagnosis ensemble with the next most practical locally available baselines beyond the current set.

## Recommended next models
Add these in order:
1. `RBPModel`
2. `GDDPFisherModel`
3. `BOQAModel`

## Why this order
- `RBPModel` is a straightforward ranking/similarity-style baseline and usually has the smallest integration risk.
- `GDDPFisherModel` appears to have local model assets already present and adds methodological diversity.
- `BOQAModel` is valuable but may carry additional runtime/resource assumptions, so it is best integrated last among this batch.

## Integration strategy
For each model:
1. Add a new candidate entry to the registry.
2. Add availability detection based on local files or local constructibility.
3. Add a builder function.
4. Include it in the dynamic model list and quiet-output path.
5. Add focused tests for candidate detection and builder wiring.

## Non-goals
- Do not refactor the overall script architecture again.
- Do not add CSV export or model-filter CLI in this phase.
- Do not attempt to force incompatible models that fail due to unrelated legacy dependencies.

## Verification
After each added model:
- targeted tests should pass
- the script should still run end-to-end
- the available-models line should show the new model when applicable
