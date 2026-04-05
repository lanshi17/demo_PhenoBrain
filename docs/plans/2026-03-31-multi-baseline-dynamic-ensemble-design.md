# Multi-Baseline Dynamic Ensemble Design

## Completion Notes (2026-04-05)
- The metadata-driven registry design is implemented in `example_predict_ensemble.py` and later extended beyond the initial expansion set.
- The current candidate registry includes `ICTODQAcross-Ave-Random`, `HPOProbMNB-Random`, `CNB-Random`, `NN-Mixup-Random-1`, `MICAModel`, `MICALinModel`, `MICAJCModel`, `MinICModel`, `RBPModel`, `GDDPFisherModel`, and `BOQAModel`, while still preserving the subset and single-model degrade paths.

## Goal
Expand the offline diagnosis script from a 4-model subset Ensemble into a larger dynamic baseline ensemble that automatically includes whichever diagnosis models are locally available or locally constructible.

## Motivation
The downloaded `codes/core/model/` directory contains more than just the four original Ensemble components. It also includes multiple baseline-model artifacts such as MICA/Lin/JC/MinIC/GDDP/RBP/BOQA, and some of those models can also be built directly from local ontology/data without relying on pre-downloaded weights. A dynamic candidate pool will use the available resources better than a hardcoded 4-model bundle.

## Model categories
### Core diagnosis models
These remain the highest-priority models because they are already part of the project’s public Ensemble story:
- `ICTO(A)` -> `ICTODQAcross-Ave-Random`
- `HPOProb` -> `HPOProbMNB-Random`
- `CNB` -> `CNB-Random`
- `MLP` -> `NN-Mixup-Random-1`

### Baseline expansion pool
Candidate baseline models to include when available:
- `MICAModel`
- `MICALinModel`
- `MICAJCModel`
- `MinICModel`
- `GDDPFisherModel`
- `RBPModel`
- `BOQAModel`
- `SimTOModel`
- optionally later: `CosineModel`, `JaccardModel`, `SimGICModel`

## Availability strategy
Each candidate model should be assigned one of these activation modes:

1. **Always construct locally**
   Models that can reasonably be initialized from ontology/data without requiring downloaded saved-model files.

2. **Use downloaded assets if present, otherwise construct locally**
   Good for models with both a fast path (load) and a fallback (train/init).

3. **Require downloaded assets**
   Only include when exact files exist.

## Recommended default candidate set
To avoid over-expanding too quickly, the first multi-baseline version should prioritize these eight models:
- `ICTODQAcross-Ave-Random`
- `HPOProbMNB-Random`
- `CNB-Random`
- `NN-Mixup-Random-1`
- `MICAModel`
- `MICALinModel`
- `MICAJCModel`
- `MinICModel`

This keeps the design focused on strong ontology/similarity baselines plus the existing probabilistic and ML components.

## Runtime behavior
The script should:
1. Build a candidate pool from configuration metadata rather than hardcoded `if` blocks.
2. Detect available assets in both the original reader-scoped directory layout and the newly downloaded flat layout when applicable.
3. Construct all eligible models.
4. Report which models were included and why.
5. Build an `OrderedMultiModel` over the included set when more than one model is present.
6. Degrade to a single model when only one candidate is usable.
7. Raise a clear error when no diagnosis models can be built.

## Design direction
Refactor the current script toward a metadata-driven registry like:
- model public name
- wrapper name
- constructor function
- required assets
- fallback mode
- preferred reader type

That will make adding more baselines incremental rather than duplicating conditionals.

## Testing focus
Tests should cover:
- core four-model detection under multiple path layouts
- extra baseline inclusion when corresponding assets exist
- default inclusion of locally constructible baselines
- stable reporting of participating model names
- ensemble vs single-model vs no-model behavior
