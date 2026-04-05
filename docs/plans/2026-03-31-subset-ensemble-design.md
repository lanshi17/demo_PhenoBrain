# Subset Ensemble Offline Design

## Completion Notes (2026-04-05)
- The availability-driven subset behavior is implemented and remains part of the current dynamic ensemble path.
- Optional models can be absent without breaking the script: the current implementation still degrades to a subset or single-model prediction path instead of failing an all-or-nothing preflight.

## Goal
Make the offline diagnosis example usable even when the cloud drive is incomplete by dynamically building an Ensemble from whichever models are actually available.

## Context
The original offline example assumed all four public Ensemble components were present:
- `ICTO(A)` -> `ICTODQAcross-Ave-Random`
- `HPOProb` -> `HPOProbMNB-Random`
- `CNB` -> `CNB-Random`
- `MLP` -> `NN-Mixup-Random-1`

In the current workspace, `timgroup_disease_diagnosis/codes/core/model/` is empty except for `README.md`, so the strict 4-model preflight makes the script unusable.

## Design
### Availability model
Treat candidate models in two classes:

1. **Weight-optional / locally constructible**
   - `ICTODQAcrossModel`
   - `HPOProbMNBModel`
   These can be initialized directly from local ontology/data structures when model files are absent.

2. **Weight-required**
   - `CNBModel`
   - `LRNeuronModel`
   These should only be included when their saved model files exist locally.

### Runtime behavior
The example script should:
1. Inspect which candidate models are available.
2. Always include locally constructible models.
3. Include `CNB` and `MLP` only when their assets exist.
4. Build an `OrderedMultiModel` from the resulting subset.
5. Print or expose the actual participating model names.
6. If only one model is available, explicitly degrade to single-model prediction rather than pretending it is a full Ensemble.
7. If zero models are available, raise a clear error.

### Public API shape
Keep the simple user-facing API:
- `build_ensemble_model()`
- `predict_ensemble(hpo_list, topk=10)`

Internally add helpers like:
- `get_available_model_names()`
- `build_available_models()`
- `build_ensemble_or_single_model()`

## Recommendation
Implement a dynamic subset ensemble instead of hard-failing on a missing full 4-model bundle. This preserves correctness while making offline use practical with partial cloud-drive contents.

## Testing
Add tests for:
- no optional weights -> uses `ICTO(A)` + `HPOProb`
- `CNB` weight present -> it is included
- `MLP` weight present -> it is included
- one-model / multi-model behavior is reported correctly
