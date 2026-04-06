# Order Statistic Outer Ensemble Design

## Completion Notes (2026-04-06)
- The design is implemented through a dedicated `OrderStatisticMultiModel` in `core/predict/ensemble/order_statistic_multi_model.py`.
- Only the outer `Ensemble` now uses Stuart-style order-statistics fusion; the inner wrappers still use the existing `OrderedMultiModel` rank-sum behavior.
- Verification completed with 43 passing focused tests and a successful real-script run.

## Goal
Replace the current outer-ensemble rank-sum fusion with an order-statistics-based significance fusion while keeping all inner per-model wrappers unchanged.

## Context
- The current offline ensemble uses `OrderedMultiModel` for both the inner wrappers and the outer `Ensemble`.
- `OrderedMultiModel` converts each model score vector into rank scores and sums them, which is a Borda-style rank aggregation.
- The requested change is to use an order-statistics fusion such as Stuart's method only for the outermost ensemble layer.

## Design
### Scope
- Keep these inner wrappers unchanged:
  - `ICTODQAcross-Ave-Random`
  - `HPOProbMNB-Random`
  - `CNB-Random`
  - `NN-Mixup-Random-1`
- Replace only the outer `Ensemble` returned by `_build_outer_ensemble(...)`.

### New outer fusion class
Add a new ensemble class:
- `core/predict/ensemble/order_statistic_multi_model.py`

The class will:
1. collect one score vector from each child model,
2. convert each score vector into descending rank ratios `rank / disease_count`,
3. sort the per-disease rank ratios ascending,
4. compute Stuart's `Z` statistic through the recursive dynamic-programming formula, and
5. rank diseases by `-log(Z)` so "more significant consensus" maps to "larger score is better".

### Why not modify OrderedMultiModel directly
- Inner wrappers still want the existing rank-sum behavior.
- A separate class limits regression risk and makes the outer-fusion change explicit.
- Tests can verify that only `_build_outer_ensemble(...)` changes behavior.

### Output behavior
- Keep the same public APIs:
  - `build_ensemble_model()`
  - `predict_ensemble(...)`
  - `query(...)`
- Preserve the current `keep_raw_score` behavior so the script output shape stays consistent with the existing interface.

## Testing Focus
- Rank-ratio conversion uses descending ranks and stays deterministic.
- Stuart significance prefers stronger cross-model consensus than weaker rankings.
- `_build_outer_ensemble(...)` returns the new class instead of `OrderedMultiModel`.
- The focused offline regression suite still passes after the outer-fusion swap.
