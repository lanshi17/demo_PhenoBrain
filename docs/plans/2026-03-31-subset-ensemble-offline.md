# Subset Ensemble Offline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the offline diagnosis example build and run with a dynamically selected subset of available models instead of requiring the full 4-model Ensemble bundle.

**Architecture:** Keep the existing offline example entrypoints, but replace the strict all-or-nothing asset gate with dynamic availability detection. Always include the two models that can be constructed locally from ontology/data (`ICTO(A)` and `HPOProb`), and include `CNB` and `MLP` only when their saved assets exist; then build an `OrderedMultiModel` over the resulting subset or degrade explicitly to single-model prediction.

**Tech Stack:** Python, pytest, internal `core.predict.*` model classes, `OrderedMultiModel`, `HPOIntegratedDatasetReader`.

## Completion Notes (2026-04-05)
- The strict preflight was replaced by the current `build_available_models()` and `build_ensemble_model()` flow in `example_predict_ensemble.py`.
- Current regression coverage includes no-model, single-model, and subset-ensemble behaviors, and the focused offline suite passed with 40 tests on 2026-04-05.

---

### Task 1: Add a failing test for the default subset behavior

**Files:**
- Modify: `timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py`
- Modify: `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`

**Step 1: Write the failing test**

Add a test that documents the intended baseline behavior when no optional saved-model assets exist:

```python
def test_build_available_models_defaults_to_icto_and_hpoprob(monkeypatch):
    module = load_module()
    fake_types = install_fake_core_modules(monkeypatch)
    monkeypatch.setattr(module, '_asset_exists', lambda _path: False)

    models = module.build_available_models()

    assert [model.model_name for model in models] == [
        'ICTODQAcross-Ave-Random',
        'HPOProbMNB-Random',
    ]
```

**Step 2: Run test to verify it fails**

Run:
```bash
PYTHONPATH=/mnt/data/Projects/02_Research/demo_PhenoBrain/timgroup_disease_diagnosis/codes/core \
/mnt/data/Projects/02_Research/demo_PhenoBrain/.venv/bin/python -m pytest \
/mnt/data/Projects/02_Research/demo_PhenoBrain/timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py::test_build_available_models_defaults_to_icto_and_hpoprob -v
```

Expected:
- FAIL because `build_available_models()` does not exist yet.

**Step 3: Write minimal implementation**

In `example_predict_ensemble.py`, add helper structure for:
- `_asset_exists(path)`
- `build_available_models()`

Initially implement just enough so the test can pass by returning the two always-available wrapped models.

**Step 4: Run test to verify it passes**

Run the same command.
Expected: PASS

**Step 5: Commit**

Do not commit unless the user explicitly asks.

### Task 2: Add failing tests for optional CNB and MLP inclusion

**Files:**
- Modify: `timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py`
- Modify: `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`

**Step 1: Write the failing tests**

Add two focused tests:

```python
def test_build_available_models_includes_cnb_when_asset_exists(monkeypatch):
    module = load_module()
    install_fake_core_modules(monkeypatch)

    def fake_exists(path):
        return str(path).endswith('CNB.joblib')

    monkeypatch.setattr(module, '_asset_exists', fake_exists)
    models = module.build_available_models()
    assert 'CNB-Random' in [model.model_name for model in models]


def test_build_available_models_includes_mlp_when_all_checkpoint_assets_exist(monkeypatch):
    module = load_module()
    install_fake_core_modules(monkeypatch)

    def fake_exists(path):
        path = str(path)
        return path.endswith('model.ckpt.index') or path.endswith('model.ckpt.data-00000-of-00001')

    monkeypatch.setattr(module, '_asset_exists', fake_exists)
    models = module.build_available_models()
    assert 'NN-Mixup-Random-1' in [model.model_name for model in models]
```

**Step 2: Run tests to verify they fail**

Run:
```bash
PYTHONPATH=/mnt/data/Projects/02_Research/demo_PhenoBrain/timgroup_disease_diagnosis/codes/core \
/mnt/data/Projects/02_Research/demo_PhenoBrain/.venv/bin/python -m pytest \
/mnt/data/Projects/02_Research/demo_PhenoBrain/timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py -v
```

Expected:
- FAIL because optional-model inclusion logic does not exist yet.

**Step 3: Write minimal implementation**

Add helpers like:

```python
def _has_cnb_assets():
    return _asset_exists(_MODEL_DIR / 'INTEGRATE_CCRD_OMIM_ORPHA/CNBModel/CNB.joblib')


def _has_mlp_assets():
    return (
        _asset_exists(_MODEL_DIR / 'INTEGRATE_CCRD_OMIM_ORPHA/LRNeuronModel/NN-Mixup-1/model.ckpt.index')
        and _asset_exists(_MODEL_DIR / 'INTEGRATE_CCRD_OMIM_ORPHA/LRNeuronModel/NN-Mixup-1/model.ckpt.data-00000-of-00001')
    )
```

and append the corresponding wrapped models only when those checks pass.

**Step 4: Run tests to verify they pass**

Run the same full test command.
Expected: PASS

**Step 5: Commit**

Do not commit unless the user explicitly asks.

### Task 3: Add failing tests for subset assembly behavior

**Files:**
- Modify: `timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py`
- Modify: `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`

**Step 1: Write the failing tests**

Add tests for public behavior:

```python
def test_build_ensemble_model_returns_ordered_multi_model_for_multiple_models(monkeypatch):
    module = load_module()
    dummy_a = object()
    dummy_b = object()
    monkeypatch.setattr(module, 'build_available_models', lambda: [dummy_a, dummy_b])
    monkeypatch.setattr(module, '_build_outer_ensemble', lambda models: ('ensemble', models))

    result = module.build_ensemble_model()
    assert result == ('ensemble', [dummy_a, dummy_b])


def test_build_ensemble_model_returns_single_model_when_only_one_available(monkeypatch):
    module = load_module()
    dummy = object()
    monkeypatch.setattr(module, 'build_available_models', lambda: [dummy])

    result = module.build_ensemble_model()
    assert result is dummy
```

Also add a zero-model guard:

```python
def test_build_ensemble_model_raises_when_no_models_available(monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, 'build_available_models', lambda: [])
    with pytest.raises(RuntimeError):
        module.build_ensemble_model()
```

**Step 2: Run tests to verify they fail**

Run the full test file.
Expected: FAIL because `build_ensemble_model()` still assumes the old all-or-nothing flow.

**Step 3: Write minimal implementation**

Refactor to:

```python
def _build_outer_ensemble(model_list):
    return OrderedMultiModel(model_list=model_list, hpo_reader=model_list[0].hpo_reader, model_name='Ensemble')


def build_ensemble_model():
    model_list = build_available_models()
    if not model_list:
        raise RuntimeError('No diagnosis models are available for offline prediction.')
    if len(model_list) == 1:
        return model_list[0]
    return _build_outer_ensemble(model_list)
```

**Step 4: Run tests to verify they pass**

Run the full test file.
Expected: PASS

**Step 5: Commit**

Do not commit unless the user explicitly asks.

### Task 4: Add a failing test for reporting participating model names

**Files:**
- Modify: `timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py`
- Modify: `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`

**Step 1: Write the failing test**

```python
def test_get_available_model_names_matches_built_models(monkeypatch):
    module = load_module()

    class DummyModel:
        def __init__(self, name):
            self.model_name = name

    monkeypatch.setattr(module, 'build_available_models', lambda: [DummyModel('A'), DummyModel('B')])
    assert module.get_available_model_names() == ['A', 'B']
```

**Step 2: Run test to verify it fails**

Run the single test.
Expected: FAIL because `get_available_model_names()` does not exist yet.

**Step 3: Write minimal implementation**

```python
def get_available_model_names():
    return [model.model_name for model in build_available_models()]
```

**Step 4: Run test to verify it passes**

Run the single test.
Expected: PASS

**Step 5: Commit**

Do not commit unless the user explicitly asks.

### Task 5: Update the script entrypoint and usage text

**Files:**
- Modify: `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`
- Test: `timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py`

**Step 1: Write the failing test**

Add a lightweight test for the new message helper or output-oriented function rather than asserting on `__main__` directly. Example:

```python
def test_describe_available_models_reports_subset(monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, 'get_available_model_names', lambda: ['ICTODQAcross-Ave-Random', 'HPOProbMNB-Random'])
    assert 'ICTODQAcross-Ave-Random' in module.describe_available_models()
```

**Step 2: Run test to verify it fails**

Run the single test.
Expected: FAIL because the helper does not exist yet.

**Step 3: Write minimal implementation**

Add:

```python
def describe_available_models():
    names = get_available_model_names()
    if not names:
        return 'Available models: none'
    return 'Available models: ' + ', '.join(names)
```

Update the top docstring to explain that:
- the script now uses the subset of available models
- `ICTO(A)` and `HPOProb` are expected to work without downloaded weights
- `CNB` and `MLP` are included only if local assets exist

In `__main__`, print the available model description before running prediction.

**Step 4: Run tests to verify they pass**

Run the full test file.
Expected: PASS

**Step 5: Commit**

Do not commit unless the user explicitly asks.

### Task 6: Final verification and behavior check

**Files:**
- Review: `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`
- Review: `timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py`

**Step 1: Run the full test file**

Run:
```bash
PYTHONPATH=/mnt/data/Projects/02_Research/demo_PhenoBrain/timgroup_disease_diagnosis/codes/core \
/mnt/data/Projects/02_Research/demo_PhenoBrain/.venv/bin/python -m pytest \
/mnt/data/Projects/02_Research/demo_PhenoBrain/timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py -v
```

Expected: all tests pass.

**Step 2: Run the script smoke test**

Run:
```bash
PYTHONPATH=/mnt/data/Projects/02_Research/demo_PhenoBrain/timgroup_disease_diagnosis/codes/core \
/mnt/data/Projects/02_Research/demo_PhenoBrain/.venv/bin/python \
/mnt/data/Projects/02_Research/demo_PhenoBrain/timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py
```

Expected:
- It prints the actual participating model names.
- In the current repo state, it should not insist on `CNB` or `MLP` assets.
- It should build from the baseline subset or fail only if even the baseline subset cannot be constructed.

**Step 3: Summarize caveats for the user**

Mention explicitly:
- `CNB` and `MLP` remain optional because they depend on downloaded saved models.
- The script now builds a subset ensemble rather than the paper’s strict 4-model bundle.
- If only one model is available, prediction is intentionally degraded to a single-model path.

**Step 4: Commit**

Do not commit unless the user explicitly asks.
