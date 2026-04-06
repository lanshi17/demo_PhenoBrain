# Order Statistic Outer Ensemble Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace only the outer offline `Ensemble` fusion with an order-statistics-based significance aggregator while preserving the current inner model wrappers.

**Architecture:** Introduce a dedicated `OrderStatisticMultiModel` class beside `OrderedMultiModel`, export it through the ensemble package, and change `_build_outer_ensemble(...)` to use it. Keep the inner wrappers and the rest of the offline script unchanged so only the top-level fusion semantics change.

**Tech Stack:** Python, NumPy, existing `core.predict.ensemble` package, pytest.

---

## Completion Notes (2026-04-06)
- Added `OrderStatisticMultiModel` plus helper functions for descending rank ratios, Stuart p-value recursion, and `-log(Z)` fusion scores.
- `_build_outer_ensemble(...)` now uses `OrderStatisticMultiModel`, so only the outer `Ensemble` changed behavior.
- Verification completed with:
  - `PYTHONPATH=timgroup_disease_diagnosis/codes/core .venv/bin/python -m pytest timgroup_disease_diagnosis/codes/core/tests/test_order_statistic_multi_model.py timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py -q`
  - `PYTHONPATH=timgroup_disease_diagnosis/codes/core .venv/bin/python -m pytest timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py timgroup_disease_diagnosis/codes/core/tests/test_cycommon_fallback.py timgroup_disease_diagnosis/codes/core/tests/test_python312_compat.py timgroup_disease_diagnosis/codes/core/tests/test_boqa_model.py timgroup_disease_diagnosis/codes/core/tests/test_hpo_reader_paths.py tests/test_jupyter_setup.py -q`
  - `PYTHONPATH=timgroup_disease_diagnosis/codes/core .venv/bin/python timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py --topk 5`

### Task 1: Add failing algorithm tests for rank ratios and Stuart significance

**Files:**
- Create: `timgroup_disease_diagnosis/codes/core/tests/test_order_statistic_multi_model.py`
- Create: `timgroup_disease_diagnosis/codes/core/core/predict/ensemble/order_statistic_multi_model.py`

**Step 1: Write the failing tests**

Create focused tests such as:

```python
import numpy as np

from core.predict.ensemble.order_statistic_multi_model import (
    score_vec_to_rank_ratios,
    stuart_order_statistic_score,
)


def test_score_vec_to_rank_ratios_uses_descending_ranks():
    score_vec = np.array([0.9, 0.1, 0.5], dtype=np.float64)
    rank_ratios = score_vec_to_rank_ratios(score_vec)
    assert np.allclose(rank_ratios, np.array([1/3, 1.0, 2/3], dtype=np.float64))


def test_stuart_order_statistic_score_prefers_consensus_of_excellence():
    better = stuart_order_statistic_score(np.array([0.01, 0.02, 0.05], dtype=np.float64))
    worse = stuart_order_statistic_score(np.array([0.2, 0.2, 0.2], dtype=np.float64))
    assert better > worse
```

**Step 2: Run tests to verify they fail**

Run:
```bash
PYTHONPATH=timgroup_disease_diagnosis/codes/core .venv/bin/python -m pytest \
  timgroup_disease_diagnosis/codes/core/tests/test_order_statistic_multi_model.py -q
```

Expected:
- FAIL because the new module and helpers do not exist yet.

**Step 3: Write minimal implementation**

Add the new module with:
- `score_vec_to_rank_ratios(score_vec)`
- `stuart_order_statistic_pvalue(sorted_rank_ratios)`
- `stuart_order_statistic_score(sorted_rank_ratios)`

Implement the recursive dynamic-programming formula with `V_0 = 1` and `Z = N! * V_N`, then convert to `-log(Z)` for ranking.

**Step 4: Run tests to verify they pass**

Run the same test file.
Expected:
- PASS

**Step 5: Commit**

Do not commit unless the user explicitly asks.

### Task 2: Add a failing outer-ensemble integration test

**Files:**
- Modify: `timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py`
- Modify: `timgroup_disease_diagnosis/codes/core/core/predict/ensemble/__init__.py`
- Modify: `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`

**Step 1: Write the failing test**

Add a helper-level test asserting the outer builder uses the new class:

```python
def test_build_outer_ensemble_uses_order_statistic_multi_model(monkeypatch):
    module = load_module()

    class DummyOrderStatisticMultiModel:
        def __init__(self, model_list=None, hpo_reader=None, model_name=None, **kwargs):
            self.model_list = model_list
            self.hpo_reader = hpo_reader
            self.model_name = model_name

    fake_ensemble_module = types.ModuleType('core.predict.ensemble')
    fake_ensemble_module.OrderStatisticMultiModel = DummyOrderStatisticMultiModel
    monkeypatch.setitem(sys.modules, 'core.predict.ensemble', fake_ensemble_module)

    class DummyModel:
        def __init__(self):
            self.hpo_reader = object()

    model_list = [DummyModel(), DummyModel()]
    ensemble = module._build_outer_ensemble(model_list)
    assert isinstance(ensemble, DummyOrderStatisticMultiModel)
    assert ensemble.model_name == 'Ensemble'
```

**Step 2: Run test to verify it fails**

Run the single test.
Expected:
- FAIL because `_build_outer_ensemble(...)` still imports `OrderedMultiModel`.

**Step 3: Write minimal implementation**

- Export `OrderStatisticMultiModel` from `core.predict.ensemble.__init__`
- Update `_build_outer_ensemble(...)` to import and use `OrderStatisticMultiModel`

**Step 4: Run test to verify it passes**

Run the single test again.
Expected:
- PASS

**Step 5: Commit**

Do not commit unless the user explicitly asks.

### Task 3: Implement the outer fusion class behavior

**Files:**
- Modify: `timgroup_disease_diagnosis/codes/core/core/predict/ensemble/order_statistic_multi_model.py`
- Test: `timgroup_disease_diagnosis/codes/core/tests/test_order_statistic_multi_model.py`

**Step 1: Add a failing class-level fusion test**

Add a focused test such as:

```python
def test_combine_score_vecs_uses_order_statistics():
    score_vecs = [
        np.array([0.99, 0.60, 0.10], dtype=np.float64),
        np.array([0.98, 0.20, 0.60], dtype=np.float64),
        np.array([0.97, 0.30, 0.50], dtype=np.float64),
    ]
    fused = combine_score_vecs_with_order_statistics(score_vecs)
    assert fused[0] > fused[1]
    assert fused[0] > fused[2]
```

**Step 2: Run tests to verify they fail**

Run the test file again.
Expected:
- FAIL until the class/helper logic is wired.

**Step 3: Write minimal implementation**

Implement:
- rank-ratio conversion per child score vector
- per-disease sorted ratio extraction
- `combine_score_vecs(...)` that returns the fused significance scores
- `query_score_vec(...)` and `query_score_mat(...)` analogous to `OrderedMultiModel`

Preserve `keep_raw_score` behavior for result rendering.

**Step 4: Run tests to verify they pass**

Run:
```bash
PYTHONPATH=timgroup_disease_diagnosis/codes/core .venv/bin/python -m pytest \
  timgroup_disease_diagnosis/codes/core/tests/test_order_statistic_multi_model.py -q
```

Expected:
- PASS

**Step 5: Commit**

Do not commit unless the user explicitly asks.

### Task 4: Run regression and real-script verification

**Files:**
- Review: `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`
- Review: `timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py`

**Step 1: Run focused ensemble tests**

Run:
```bash
PYTHONPATH=timgroup_disease_diagnosis/codes/core .venv/bin/python -m pytest \
  timgroup_disease_diagnosis/codes/core/tests/test_order_statistic_multi_model.py \
  timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py -q
```

Expected:
- PASS

**Step 2: Run the broader focused suite**

Run:
```bash
PYTHONPATH=timgroup_disease_diagnosis/codes/core .venv/bin/python -m pytest \
  timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py \
  timgroup_disease_diagnosis/codes/core/tests/test_cycommon_fallback.py \
  timgroup_disease_diagnosis/codes/core/tests/test_python312_compat.py \
  timgroup_disease_diagnosis/codes/core/tests/test_boqa_model.py \
  timgroup_disease_diagnosis/codes/core/tests/test_hpo_reader_paths.py \
  tests/test_jupyter_setup.py -q
```

Expected:
- PASS

**Step 3: Run the real script**

Run:
```bash
PYTHONPATH=timgroup_disease_diagnosis/codes/core .venv/bin/python \
  timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py \
  --topk 5
```

Expected:
- Script still runs successfully
- `Available models:` line is unchanged
- output remains quiet and table-formatted

**Step 4: Record completion**

Add `Completion Notes` to this plan and its paired design doc summarizing the new outer-fusion behavior and verification evidence.

**Step 5: Commit**

Do not commit unless the user explicitly asks.
