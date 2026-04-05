# Offline Ensemble Python Invocation Design

## Completion Notes (2026-04-05)
- The direct-import offline example is implemented in `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py` and exposes `build_ensemble_model()` plus `predict_ensemble()`.
- The original fixed 4-model assumption was later generalized to dynamic availability, but the design goal of a local Python invocation path is now satisfied and exercised by the current offline script.

## Goal
Provide a minimal offline Python example that directly imports project classes and runs one Ensemble prediction from a list of HPO terms.

## Constraints
- No local HTTP service.
- Prefer direct class import over evaluation/test harnesses.
- Keep the example as a single runnable script.
- The repo’s Ensemble implementation is `OrderedMultiModel`, which combines child model score vectors by rank aggregation.

## Confirmed code facts
- `OrderedMultiModel` accepts `model_list` or `model_inits` and combines model scores in `query_score_vec()` and `combine_score_vecs()`: `timgroup_disease_diagnosis/codes/core/core/predict/ensemble/ordered_multi_model.py:8-52`
- Single-sample prediction can use `model.query(hpo_list, topk)`: `timgroup_disease_diagnosis/codes/core/core/predict/model.py:57-68`
- The public API exposes `Ensemble` as a supported diagnosis model: `timgroup_disease_diagnosis/PhenoBrain_Web_API/README.md:904`
- The project README states Ensemble aggregates the four methods `ICTO`, `PPO`, `CNB`, `MLP`: `timgroup_disease_diagnosis/README.md:121-125`

## Recommended approach
1. Add the repo’s `codes/core` directory to `sys.path`.
2. Import `HPOReader`, `OrderedMultiModel`, and the four underlying model classes.
3. Instantiate a shared `HPOReader`.
4. Instantiate the four base models with the same `hpo_reader`.
5. Construct `OrderedMultiModel(model_list=[...], hpo_reader=hpo_reader, model_name="Ensemble")`.
6. Call `ensemble_model.query(hpo_list, topk)`.
7. Print ranked disease codes and scores.

## Open point before coding
The exact internal class mapping / constructor defaults for the four public names `ICTO`, `PPO`, `CNB`, and `MLP` still needs to be verified in code before writing the final script.

## Why this approach
This is the smallest example that matches the repo’s actual Ensemble implementation and avoids pulling the user into the bulk evaluation pipeline in `test_optimal_model.py`.
