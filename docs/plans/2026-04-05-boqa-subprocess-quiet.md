# BOQA Subprocess Quiet Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Silence BOQA's external Java logs in the offline example output without masking BOQA execution failures.

**Architecture:** Introduce a tiny subprocess helper inside `boqa_model.py` and route the current Java execution through it. Use `subprocess.run(...)` with redirected output and explicit error handling so the Python quiet wrapper can remain effective.

**Tech Stack:** Python standard library `subprocess`, pytest, existing `BOQAModel`.

---

## Completion Notes (2026-04-05)

- `BOQAModel` no longer invokes Java through `os.system(...)`; it now uses a small `subprocess.run(...)` helper with `stdout` and `stderr` redirected to `DEVNULL`.
- A non-zero Java exit now raises a clear `RuntimeError` that includes the exit code and command summary.
- Verification completed with:
  - `PYTHONPATH=timgroup_disease_diagnosis/codes/core .venv/bin/python -m pytest timgroup_disease_diagnosis/codes/core/tests/test_boqa_model.py timgroup_disease_diagnosis/codes/core/tests/test_hpo_reader_paths.py timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py -q`
  - `PYTHONPATH=timgroup_disease_diagnosis/codes/core .venv/bin/python -m pytest timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py timgroup_disease_diagnosis/codes/core/tests/test_cycommon_fallback.py timgroup_disease_diagnosis/codes/core/tests/test_python312_compat.py timgroup_disease_diagnosis/codes/core/tests/test_boqa_model.py -q`
  - default script execution
- Current real-script output ends at the aligned result table with no trailing SLF4J / Ontologizer logs.

### Task 1: Add a failing subprocess-invocation test

**Files:**
- Create: `timgroup_disease_diagnosis/codes/core/tests/test_boqa_model.py`
- Modify: `timgroup_disease_diagnosis/codes/core/core/predict/prob_model/boqa_model.py`

**Step 1: Write the failing test**

Create a focused test that documents the intended subprocess behavior:

```python
from subprocess import DEVNULL

import core.predict.prob_model.boqa_model as boqa_model_module


def test_run_boqa_command_uses_quiet_subprocess(monkeypatch):
    calls = {}

    def fake_run(cmd, stdout=None, stderr=None, check=None):
        calls['cmd'] = cmd
        calls['stdout'] = stdout
        calls['stderr'] = stderr
        calls['check'] = check

    monkeypatch.setattr(boqa_model_module.subprocess, 'run', fake_run)

    boqa_model_module.run_boqa_command(['java', '-jar', 'boqa.jar'])

    assert calls['cmd'] == ['java', '-jar', 'boqa.jar']
    assert calls['stdout'] is DEVNULL
    assert calls['stderr'] is DEVNULL
    assert calls['check'] is True
```

**Step 2: Run test to verify it fails**

Run:
```bash
PYTHONPATH=timgroup_disease_diagnosis/codes/core .venv/bin/python -m pytest \
  timgroup_disease_diagnosis/codes/core/tests/test_boqa_model.py::test_run_boqa_command_uses_quiet_subprocess -q
```

Expected:
- FAIL because `run_boqa_command` does not exist yet.

**Step 3: Write minimal implementation**

Add `run_boqa_command(command)` in `boqa_model.py` using `subprocess.run(..., stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)`.

**Step 4: Run test to verify it passes**

Run the same single test.
Expected:
- PASS

**Step 5: Commit**

Do not commit unless the user explicitly asks.

### Task 2: Add a failing error-handling test

**Files:**
- Modify: `timgroup_disease_diagnosis/codes/core/tests/test_boqa_model.py`
- Modify: `timgroup_disease_diagnosis/codes/core/core/predict/prob_model/boqa_model.py`

**Step 1: Write the failing test**

Add:

```python
import subprocess

import pytest

import core.predict.prob_model.boqa_model as boqa_model_module


def test_run_boqa_command_raises_runtime_error_on_nonzero_exit(monkeypatch):
    def fake_run(cmd, stdout=None, stderr=None, check=None):
        raise subprocess.CalledProcessError(returncode=7, cmd=cmd)

    monkeypatch.setattr(boqa_model_module.subprocess, 'run', fake_run)

    with pytest.raises(RuntimeError) as exc_info:
        boqa_model_module.run_boqa_command(['java', '-jar', 'boqa.jar'])

    assert 'boqa.jar' in str(exc_info.value)
    assert '7' in str(exc_info.value)
```

**Step 2: Run test to verify it fails**

Run the two tests in `test_boqa_model.py`.
Expected:
- FAIL because the helper does not yet translate subprocess failure into a clear runtime error.

**Step 3: Write minimal implementation**

Wrap `subprocess.run(...)` in `try/except subprocess.CalledProcessError` and raise `RuntimeError(...)` with the return code and command summary.

**Step 4: Run test to verify it passes**

Run the file again.
Expected:
- PASS

**Step 5: Commit**

Do not commit unless the user explicitly asks.

### Task 3: Route BOQA execution through the subprocess helper

**Files:**
- Modify: `timgroup_disease_diagnosis/codes/core/core/predict/prob_model/boqa_model.py`
- Test: `timgroup_disease_diagnosis/codes/core/tests/test_boqa_model.py`

**Step 1: Write the failing integration-oriented test**

If practical, add a small test for command construction, for example by monkeypatching `run_boqa_command` and asserting the generated argument list starts with `['java', '-jar', ...]`.

**Step 2: Run targeted tests to verify it fails**

Only if a focused failing test is added. Otherwise proceed to implementation.

**Step 3: Write minimal implementation**

In `query_many_multi_wrap(...)`:
- remove the `print(...)`
- replace the shell string + `os.system(...)` call with an argument list
- call `run_boqa_command(...)`

**Step 4: Run targeted tests to verify they pass**

Run:
```bash
PYTHONPATH=timgroup_disease_diagnosis/codes/core .venv/bin/python -m pytest \
  timgroup_disease_diagnosis/codes/core/tests/test_boqa_model.py -q
```

Expected:
- PASS

**Step 5: Commit**

Do not commit unless the user explicitly asks.

### Task 4: Verify quiet real-script output

**Files:**
- Review: `timgroup_disease_diagnosis/codes/core/core/predict/prob_model/boqa_model.py`
- Review: `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`
- Review: `docs/plans/2026-04-05-boqa-subprocess-quiet.md`

**Step 1: Run the BOQA-specific tests**

Run:
```bash
PYTHONPATH=timgroup_disease_diagnosis/codes/core .venv/bin/python -m pytest \
  timgroup_disease_diagnosis/codes/core/tests/test_boqa_model.py \
  timgroup_disease_diagnosis/codes/core/tests/test_hpo_reader_paths.py \
  timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py -q
```

Expected:
- PASS

**Step 2: Run the broader regression suite**

Run:
```bash
PYTHONPATH=timgroup_disease_diagnosis/codes/core .venv/bin/python -m pytest \
  timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py \
  timgroup_disease_diagnosis/codes/core/tests/test_cycommon_fallback.py \
  timgroup_disease_diagnosis/codes/core/tests/test_python312_compat.py \
  timgroup_disease_diagnosis/codes/core/tests/test_boqa_model.py -q
```

Expected:
- PASS

**Step 3: Run the real script**

Run:
```bash
PYTHONPATH=timgroup_disease_diagnosis/codes/core .venv/bin/python \
  timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py
```

Expected:
- `Available models:` includes `BOQAModel`
- the aligned result table prints
- no trailing SLF4J / Ontologizer logs appear after the table

**Step 4: Record completion**

Add `Completion Notes` to this plan documenting the verification command results.

**Step 5: Commit**

Do not commit unless the user explicitly asks.
