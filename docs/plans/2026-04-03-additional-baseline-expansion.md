# Additional Baseline Expansion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expand the dynamic ensemble with the next practical locally available baselines: `RBPModel`, `GDDPFisherModel`, and `BOQAModel`.

**Architecture:** Continue using the existing registry-driven candidate model system. Add each new baseline one at a time with focused availability detection, builder wiring, and tests, verifying after each addition that the script still runs end-to-end and reports the updated available-model set.

**Tech Stack:** Python, pytest, existing diagnosis script, local model registry, internal `core.predict.*` model classes.

## Completion Notes (2026-04-05)

- `RBPModel`, `GDDPFisherModel`, and `BOQAModel` are now wired into `example_predict_ensemble.py` through the candidate registry and builder map.
- `BOQAModel` is intentionally gated by runtime assets. The original `2019` raw HPO files were absent, and a follow-up fallback now resolves BOQA against the repository's existing `2022` `hp.obo` and `phenotype.hpoa`, so `BOQAModel` is currently available.
- Regression verification completed with:
  - `PYTHONPATH=timgroup_disease_diagnosis/codes/core .venv/bin/python -m pytest timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py timgroup_disease_diagnosis/codes/core/tests/test_cycommon_fallback.py timgroup_disease_diagnosis/codes/core/tests/test_python312_compat.py -q`
  - default script execution
  - CLI override execution with `--topk 3 --hpo-list ...`
- Current end-to-end output is quiet and table-formatted, which also closes the pending verification items from the `quiet-output`, `result-table-formatting`, and `cli-parameters` plans.
- Subsequent follow-up work enabled BOQA against the repository's `2022` HPO assets and moved BOQA Java execution to a quiet subprocess path, so the old trailing Java-log boundary is now closed.

---

### Task 1: Add failing tests and wiring for `RBPModel`

**Files:**
- Modify: `timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py`
- Modify: `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`

**Step 1: Write the failing test**

Add focused tests like:

```python
def test_model_candidates_includes_rbp_model():
    module = load_module()
    assert any(candidate['name'] == 'RBPModel' for candidate in module.MODEL_CANDIDATES)


def test_build_available_models_includes_rbp_when_candidate_available(monkeypatch):
    module = load_module()
    fake_types = install_fake_core_modules(monkeypatch)
    monkeypatch.setattr(module, '_is_candidate_available', lambda candidate: candidate['name'] in {
        'ICTODQAcross-Ave-Random', 'HPOProbMNB-Random', 'RBPModel'
    })

    models = module.build_available_models()
    assert 'RBPModel' in [model.name for model in models]
```

**Step 2: Run test to verify it fails**

Run the targeted `RBPModel` tests.
Expected: FAIL because `RBPModel` is not yet in the registry/builders.

**Step 3: Write minimal implementation**

- Add `{'name': 'RBPModel', 'kind': 'baseline'}` to `MODEL_CANDIDATES`
- Import `RBPModel`
- Add a builder:

```python
'RBPModel': lambda: RBPModel(
    hpo_reader=hpo_reader_with_all_hpo,
    model_name='RBPModel',
)
```

**Step 4: Run test to verify it passes**

Run the targeted tests again.
Expected: PASS

**Step 5: Commit**

Do not commit unless the user explicitly asks.

### Task 2: Add failing tests and wiring for `GDDPFisherModel`

**Files:**
- Modify: `timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py`
- Modify: `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`

**Step 1: Write the failing test**

Add focused tests like:

```python
def test_model_candidates_includes_gddp_fisher_model():
    module = load_module()
    assert any(candidate['name'] == 'GDDPFisherModel' for candidate in module.MODEL_CANDIDATES)


def test_build_available_models_includes_gddp_when_candidate_available(monkeypatch):
    module = load_module()
    install_fake_core_modules(monkeypatch)
    monkeypatch.setattr(module, '_is_candidate_available', lambda candidate: candidate['name'] in {
        'ICTODQAcross-Ave-Random', 'HPOProbMNB-Random', 'GDDPFisherModel'
    })
    models = module.build_available_models()
    assert 'GDDPFisherModel' in [model.name for model in models]
```

**Step 2: Run tests to verify they fail**

Expected: FAIL because `GDDPFisherModel` is not yet wired.

**Step 3: Write minimal implementation**

Add the candidate entry and builder for `GDDPFisherModel` using the existing integrated reader.

**Step 4: Run tests to verify they pass**

Expected: PASS

**Step 5: Commit**

Do not commit unless the user explicitly asks.

### Task 3: Add failing tests and wiring for `BOQAModel`

**Files:**
- Modify: `timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py`
- Modify: `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`

**Step 1: Write the failing test**

Add focused tests like:

```python
def test_model_candidates_includes_boqa_model():
    module = load_module()
    assert any(candidate['name'] == 'BOQAModel' for candidate in module.MODEL_CANDIDATES)


def test_build_available_models_includes_boqa_when_candidate_available(monkeypatch):
    module = load_module()
    install_fake_core_modules(monkeypatch)
    monkeypatch.setattr(module, '_is_candidate_available', lambda candidate: candidate['name'] in {
        'ICTODQAcross-Ave-Random', 'HPOProbMNB-Random', 'BOQAModel'
    })
    models = module.build_available_models()
    assert 'BOQAModel' in [model.name for model in models]
```

**Step 2: Run tests to verify they fail**

Expected: FAIL because `BOQAModel` is not yet wired.

**Step 3: Write minimal implementation**

Add the candidate entry and builder for `BOQAModel`.

**Step 4: Run tests to verify they pass**

Expected: PASS

**Step 5: Commit**

Do not commit unless the user explicitly asks.

### Task 4: Expand final quiet output verification

**Files:**
- Modify: `timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py`
- Review: `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`

**Step 1: Write the failing test**

If practical, add a helper-level test that `describe_available_models()` can include one of the new names when the build list contains it:

```python
def test_describe_available_models_reports_new_baselines(monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, 'get_available_model_names', lambda: ['RBPModel', 'GDDPFisherModel'])
    assert 'RBPModel' in module.describe_available_models()
    assert 'GDDPFisherModel' in module.describe_available_models()
```

**Step 2: Run test to verify it passes or fails**

If it already passes, keep it and move on.

**Step 3: Minimal implementation only if needed**

Prefer no change if existing reporting already works.

**Step 4: Commit**

Do not commit unless the user explicitly asks.

### Task 5: Final verification with the real script

**Files:**
- Review: `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`
- Review: `timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py`

**Step 1: Run the targeted tests**

Run the newly added tests for `RBPModel`, `GDDPFisherModel`, and `BOQAModel`.

**Step 2: Run the full regression test set**

Run the same regression suite that previously passed.
Expected: PASS

**Step 3: Run the real script**

Run:
```bash
PYTHONPATH=/mnt/data/Projects/02_Research/demo_PhenoBrain/timgroup_disease_diagnosis/codes/core \
/mnt/data/Projects/02_Research/demo_PhenoBrain/.venv/bin/python \
/mnt/data/Projects/02_Research/demo_PhenoBrain/timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py
```

Expected:
- The available-models line includes the new baselines when they are usable.
- The script still prints a quiet result table.
- If one of the new baselines fails for a legacy/runtime reason, capture that exact model-specific boundary rather than masking it.

**Step 4: Summarize**

Report:
- which new baselines were successfully added
- whether they appear in the available-models line
- whether any one of them still has a model-specific blocker

**Step 5: Commit**

Do not commit unless the user explicitly asks.
