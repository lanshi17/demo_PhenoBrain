# Multi-Baseline Dynamic Ensemble Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade the offline diagnosis script so it can build a larger dynamic ensemble from multiple baseline models based on locally available assets and locally constructible model classes.

**Architecture:** Replace the current ad-hoc subset logic with a metadata-driven candidate registry. Each candidate model describes its wrapper name, construction function, reader strategy, and asset-detection behavior, allowing the script to assemble a larger multi-baseline `OrderedMultiModel` when multiple models are available while still degrading cleanly to a single-model path.

**Tech Stack:** Python, pytest, internal `core.predict.*` model classes, `OrderedMultiModel`, `HPOIntegratedDatasetReader`, local file-path inspection.

## Completion Notes (2026-04-05)
- The registry-driven dynamic ensemble is implemented in `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py` through `MODEL_CANDIDATES`, `build_available_models()`, and `build_ensemble_model()`.
- Later plans expanded the registry with `RBPModel`, `GDDPFisherModel`, and `BOQAModel`; verification on 2026-04-05 showed the focused offline suite at 40 passing tests and a successful default script run.

---

### Task 1: Add failing tests for a metadata-driven candidate registry

**Files:**
- Modify: `timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py`
- Modify: `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`

**Step 1: Write the failing test**

Add a test that asserts the script exposes a candidate registry containing the initial expanded baseline set:

```python
def test_model_candidate_registry_contains_initial_multi_baseline_set():
    module = load_module()
    candidate_names = [candidate['name'] for candidate in module.MODEL_CANDIDATES]
    assert candidate_names == [
        'ICTODQAcross-Ave-Random',
        'HPOProbMNB-Random',
        'CNB-Random',
        'NN-Mixup-Random-1',
        'MICAModel',
        'MICALinModel',
        'MICAJCModel',
        'MinICModel',
    ]
```

**Step 2: Run test to verify it fails**

Run:
```bash
PYTHONPATH=/mnt/data/Projects/02_Research/demo_PhenoBrain/timgroup_disease_diagnosis/codes/core \
/mnt/data/Projects/02_Research/demo_PhenoBrain/.venv/bin/python -m pytest \
/mnt/data/Projects/02_Research/demo_PhenoBrain/timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py::test_model_candidate_registry_contains_initial_multi_baseline_set -v
```

Expected: FAIL because `MODEL_CANDIDATES` does not exist yet.

**Step 3: Write minimal implementation**

Add a top-level registry structure in `example_predict_ensemble.py` such as:

```python
MODEL_CANDIDATES = [
    {'name': 'ICTODQAcross-Ave-Random', ...},
    {'name': 'HPOProbMNB-Random', ...},
    {'name': 'CNB-Random', ...},
    {'name': 'NN-Mixup-Random-1', ...},
    {'name': 'MICAModel', ...},
    {'name': 'MICALinModel', ...},
    {'name': 'MICAJCModel', ...},
    {'name': 'MinICModel', ...},
]
```

Only add enough metadata fields to support future tasks.

**Step 4: Run test to verify it passes**

Run the same single test.
Expected: PASS

**Step 5: Commit**

Do not commit unless the user explicitly asks.

### Task 2: Add failing tests for expanded baseline inclusion

**Files:**
- Modify: `timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py`
- Modify: `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`

**Step 1: Write the failing tests**

Add focused tests for the four new similarity baselines being included by default:

```python
def test_build_available_models_includes_mica_family_and_minic(monkeypatch):
    module = load_module()
    fake_types = install_fake_core_modules(monkeypatch)
    monkeypatch.setattr(module, '_candidate_is_available', lambda candidate: candidate['name'] in {
        'ICTODQAcross-Ave-Random',
        'HPOProbMNB-Random',
        'MICAModel',
        'MICALinModel',
        'MICAJCModel',
        'MinICModel',
    })

    models = module.build_available_models()
    assert [model.model_name for model in models] == [
        'ICTODQAcross-Ave-Random',
        'HPOProbMNB-Random',
        'MICAModel',
        'MICALinModel',
        'MICAJCModel',
        'MinICModel',
    ]
```

**Step 2: Run tests to verify they fail**

Run the targeted test.
Expected: FAIL because registry-driven availability logic does not exist yet.

**Step 3: Write minimal implementation**

Refactor `build_available_models()` to iterate over `MODEL_CANDIDATES`, call `_candidate_is_available(candidate)`, and append `candidate['builder'](...)` results for the approved candidates.

**Step 4: Run tests to verify they pass**

Run the targeted test again.
Expected: PASS

**Step 5: Commit**

Do not commit unless the user explicitly asks.

### Task 3: Add failing tests for alternate asset-path compatibility

**Files:**
- Modify: `timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py`
- Modify: `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`

**Step 1: Write the failing tests**

Add tests that document support for the downloaded flat layout under `/home/lanshi/.../codes/core/model` in addition to the original reader-scoped layout:

```python
def test_cnb_candidate_accepts_flat_download_layout(tmp_path, monkeypatch):
    module = load_module()
    model_dir = tmp_path / 'model'
    monkeypatch.setattr(module, '_MODEL_DIR', model_dir)
    touch(model_dir / 'CNBModel/CNB.joblib')

    cnb_candidate = next(candidate for candidate in module.MODEL_CANDIDATES if candidate['name'] == 'CNB-Random')
    assert module._candidate_is_available(cnb_candidate) is True


def test_mlp_candidate_accepts_flat_download_layout(tmp_path, monkeypatch):
    module = load_module()
    model_dir = tmp_path / 'model'
    monkeypatch.setattr(module, '_MODEL_DIR', model_dir)
    touch(model_dir / 'NN-Mixup-1/model.ckpt.index')
    touch(model_dir / 'NN-Mixup-1/model.ckpt.data-00000-of-00001')

    mlp_candidate = next(candidate for candidate in module.MODEL_CANDIDATES if candidate['name'] == 'NN-Mixup-Random-1')
    assert module._candidate_is_available(mlp_candidate) is True
```

**Step 2: Run tests to verify they fail**

Run the two tests.
Expected: FAIL because compatibility checks still assume one layout.

**Step 3: Write minimal implementation**

Add a helper like:

```python
def _any_asset_layout_exists(relative_paths):
    return any((_MODEL_DIR / rel).exists() for rel in relative_paths)
```

and let candidate metadata list multiple acceptable paths.

**Step 4: Run tests to verify they pass**

Run the same two tests.
Expected: PASS

**Step 5: Commit**

Do not commit unless the user explicitly asks.

### Task 4: Add failing tests for reporting expanded model names

**Files:**
- Modify: `timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py`
- Modify: `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`

**Step 1: Write the failing test**

Add a test that the reporting helper reflects the expanded dynamic set:

```python
def test_describe_available_models_reports_expanded_multi_baseline_set(monkeypatch):
    module = load_module()
    monkeypatch.setattr(
        module,
        'get_available_model_names',
        lambda: ['ICTODQAcross-Ave-Random', 'HPOProbMNB-Random', 'MICAModel', 'MICALinModel'],
    )
    assert module.describe_available_models() == (
        'Available models: ICTODQAcross-Ave-Random, HPOProbMNB-Random, MICAModel, MICALinModel'
    )
```

**Step 2: Run test to verify it fails**

Only if it fails with the expected reason. If it already passes, skip implementation and keep the test.

**Step 3: Write minimal implementation**

Only if needed; otherwise leave implementation unchanged.

**Step 4: Run test to verify it passes**

Run the targeted test.
Expected: PASS

**Step 5: Commit**

Do not commit unless the user explicitly asks.

### Task 5: Add failing tests for outer-ensemble behavior with expanded set

**Files:**
- Modify: `timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py`
- Modify: `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`

**Step 1: Write the failing test**

Add a test that multiple expanded baselines still produce an outer `OrderedMultiModel`:

```python
def test_build_ensemble_model_wraps_expanded_model_list(monkeypatch):
    module = load_module()

    class DummyModel:
        def __init__(self, name, reader):
            self.model_name = name
            self.hpo_reader = reader

    class DummyReader:
        pass

    shared_reader = DummyReader()
    models = [
        DummyModel('ICTODQAcross-Ave-Random', shared_reader),
        DummyModel('HPOProbMNB-Random', shared_reader),
        DummyModel('MICAModel', shared_reader),
        DummyModel('MICALinModel', shared_reader),
    ]
    monkeypatch.setattr(module, 'build_available_models', lambda: models)
    monkeypatch.setattr(module, '_build_outer_ensemble', lambda model_list: ('ensemble', model_list))

    assert module.build_ensemble_model() == ('ensemble', models)
```

**Step 2: Run test to verify it fails**

Only if it fails with the expected reason. If it already passes under current outer-ensemble logic, keep the test and move on.

**Step 3: Write minimal implementation**

Only if needed; otherwise leave implementation unchanged.

**Step 4: Run test to verify it passes**

Run the targeted test.
Expected: PASS

**Step 5: Commit**

Do not commit unless the user explicitly asks.

### Task 6: Final verification and smoke check

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
- It prints the expanded participating model list.
- It recognizes assets under the downloaded flat layout where supported.
- If runtime still fails, the failure should now be due to missing Python dependencies or model-specific runtime requirements, not the old rigid asset-path assumptions.

**Step 3: Summarize caveats**

Mention explicitly:
- some baseline models may still need additional Python dependencies at runtime
- some models may be constructible but slow because they initialize from ontology/data instead of loading saved matrices
- the dynamic ensemble now prefers breadth over matching the original paper’s exact 4-model bundle

**Step 4: Commit**

Do not commit unless the user explicitly asks.
