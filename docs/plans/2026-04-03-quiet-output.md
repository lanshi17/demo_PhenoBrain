# Quiet Output Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the offline diagnosis script print only the available model list and final prediction results, suppressing non-essential warnings and noisy intermediate output.

**Architecture:** Add a script-local quiet mode inside `example_predict_ensemble.py` using warning filters and temporary stdout/stderr redirection. Keep the suppression narrowly scoped around model construction and prediction, while leaving the final user-facing prints outside the quiet block.

**Tech Stack:** Python standard library (`warnings`, `contextlib`, `io`), existing diagnosis script, pytest.

## Completion Notes (2026-04-05)
- The script-local quiet helper and warning configuration are in place, and BOQA's external Java invocation now also runs through a quiet subprocess helper.
- Verification on 2026-04-05: the focused offline suite passed with 40 tests, and the default script output now ends at the aligned result table with no trailing runtime logs.

---

### Task 1: Add a failing test for a quiet helper

**Files:**
- Modify: `timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py`
- Modify: `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`

**Step 1: Write the failing test**

Add a minimal helper-oriented test such as:

```python
def test_run_quietly_returns_result_without_leaking_stdout(capsys):
    module = load_module()

    def noisy():
        print('noise')
        return 'ok'

    assert module.run_quietly(noisy) == 'ok'
    captured = capsys.readouterr()
    assert captured.out == ''
```

**Step 2: Run test to verify it fails**

Run:
```bash
PYTHONPATH=/mnt/data/Projects/02_Research/demo_PhenoBrain/timgroup_disease_diagnosis/codes/core \
/mnt/data/Projects/02_Research/demo_PhenoBrain/.venv/bin/python -m pytest \
/mnt/data/Projects/02_Research/demo_PhenoBrain/timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py::test_run_quietly_returns_result_without_leaking_stdout -v
```

Expected: FAIL because `run_quietly` does not exist yet.

**Step 3: Write minimal implementation**

Add a helper in `example_predict_ensemble.py`:

```python
import io
from contextlib import redirect_stdout, redirect_stderr


def run_quietly(fn, *args, **kwargs):
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        return fn(*args, **kwargs)
```

**Step 4: Run test to verify it passes**

Run the same test again.
Expected: PASS

**Step 5: Commit**

Do not commit unless the user explicitly asks.

### Task 2: Add a failing test for warning filter setup

**Files:**
- Modify: `timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py`
- Modify: `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`

**Step 1: Write the failing test**

Add a focused test that the configuration helper exists and can be called:

```python
def test_configure_quiet_warnings_is_callable():
    module = load_module()
    module.configure_quiet_warnings()
```

Then refine to check at least one known filter target if practical.

**Step 2: Run test to verify it fails**

Run the single test.
Expected: FAIL because the helper does not exist yet.

**Step 3: Write minimal implementation**

Add:

```python
import warnings


def configure_quiet_warnings():
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
```

Then refine as needed to target the observed warnings more specifically.

**Step 4: Run test to verify it passes**

Run the test again.
Expected: PASS

**Step 5: Commit**

Do not commit unless the user explicitly asks.

### Task 3: Add a failing test for quiet model-name retrieval

**Files:**
- Modify: `timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py`
- Modify: `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`

**Step 1: Write the failing test**

Add a test that the user-facing helper wraps noisy internals quietly:

```python
def test_get_available_model_names_suppresses_internal_prints(monkeypatch, capsys):
    module = load_module()

    class DummyModel:
        def __init__(self, name):
            self.name = name

    def noisy_build():
        print('training...')
        return [DummyModel('A')]

    monkeypatch.setattr(module, 'build_available_models', noisy_build)

    assert module.get_available_model_names() == ['A']
    captured = capsys.readouterr()
    assert captured.out == ''
```

**Step 2: Run test to verify it fails**

Run the single test.
Expected: FAIL because current implementation leaks stdout.

**Step 3: Write minimal implementation**

Change:

```python
def get_available_model_names():
    return [model.name for model in run_quietly(build_available_models)]
```

**Step 4: Run test to verify it passes**

Run the test again.
Expected: PASS

**Step 5: Commit**

Do not commit unless the user explicitly asks.

### Task 4: Add a failing test for quiet prediction execution

**Files:**
- Modify: `timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py`
- Modify: `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`

**Step 1: Write the failing test**

Add a focused test around prediction:

```python
def test_predict_ensemble_suppresses_internal_prints(monkeypatch, capsys):
    module = load_module()

    class DummyModel:
        def query(self, hpo_list, topk):
            print('training...')
            return [('RD:1', 1.0)]

    monkeypatch.setattr(module, 'build_ensemble_model', lambda: DummyModel())

    assert module.predict_ensemble(['HP:0000118'], topk=1) == [('RD:1', 1.0)]
    captured = capsys.readouterr()
    assert captured.out == ''
```

**Step 2: Run test to verify it fails**

Run the single test.
Expected: FAIL because current prediction path leaks stdout.

**Step 3: Write minimal implementation**

Change:

```python
def predict_ensemble(hpo_list, topk=10):
    return run_quietly(build_ensemble_model().query, hpo_list, topk)
```

Or equivalently wrap the whole `build_ensemble_model().query(...)` call in `run_quietly`.

**Step 4: Run test to verify it passes**

Run the test again.
Expected: PASS

**Step 5: Commit**

Do not commit unless the user explicitly asks.

### Task 5: Add warning filtering to script entrypoint and verify final output shape

**Files:**
- Modify: `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`
- Test: `timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py`

**Step 1: Write the failing test**

If practical, add a helper-level test that calling the wrapper sequence does not emit warning text through stdout/stderr. If too brittle, keep verification at the script command layer in Task 6.

**Step 2: Run test to verify it fails**

Only if a reliable helper-level failure can be demonstrated.

**Step 3: Write minimal implementation**

At script execution time:

```python
if __name__ == '__main__':
    configure_quiet_warnings()
    print(describe_available_models())
    print(predict_ensemble(sample_hpo_list, topk=5))
```

Ensure warning filtering happens before model construction.

**Step 4: Run helper tests to verify they pass**

Run any tests added in this task.

**Step 5: Commit**

Do not commit unless the user explicitly asks.

### Task 6: Final verification with the real script

**Files:**
- Review: `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`
- Review: `timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py`

**Step 1: Run targeted tests**

Run the updated test file or targeted quiet-mode tests.

**Step 2: Run the real script**

Run:
```bash
PYTHONPATH=/mnt/data/Projects/02_Research/demo_PhenoBrain/timgroup_disease_diagnosis/codes/core \
/mnt/data/Projects/02_Research/demo_PhenoBrain/.venv/bin/python \
/mnt/data/Projects/02_Research/demo_PhenoBrain/timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py
```

Expected:
- Output should contain only:
  - one `Available models: ...` line
  - one final prediction-result line
- It should not print training progress bars, conflict logs, or warnings during the successful path.

**Step 3: Summarize**

Report:
- which noises were suppressed
- whether the script still produced predictions correctly
- any non-suppressed output that remains and why

**Step 4: Commit**

Do not commit unless the user explicitly asks.
