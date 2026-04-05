# cycommon Python Fallback Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the diagnosis code importable and runnable on Python 3.12 by providing a pure Python fallback for `core.utils.cycommon`.

**Architecture:** Add a new `cycommon.py` module beside the legacy Cython artifacts so `from core.utils.cycommon import to_rank_score` continues to work without changing callers. Implement only the single required function, `to_rank_score`, matching the Cython logic closely enough for current ensemble use.

**Tech Stack:** Python, NumPy, pytest, existing `core.utils` package structure.

## Completion Notes (2026-04-05)
- `core.utils.cycommon` now resolves to a pure Python fallback module that implements `to_rank_score(...)` with the required in-place ranking behavior.
- Verification on 2026-04-05: `test_cycommon_fallback.py` and the focused offline suite passed, and the previous `ModuleNotFoundError` boundary is gone.

---

### Task 1: Add a failing import-and-behavior test for cycommon

**Files:**
- Create: `timgroup_disease_diagnosis/codes/core/tests/test_cycommon_fallback.py`
- Create: `timgroup_disease_diagnosis/codes/core/core/utils/cycommon.py`

**Step 1: Write the failing test**

Create a focused test file with one import/behavior test:

```python
import numpy as np

from core.utils.cycommon import to_rank_score


def test_to_rank_score_rewrites_matrix_in_place():
    score_mat = np.array([[0.1, 0.4, 0.4, 0.9]], dtype=np.float64)
    arg_mat = np.argsort(score_mat).astype(np.int32)

    to_rank_score(score_mat, arg_mat)

    assert score_mat.shape == (1, 4)
    assert score_mat[0, arg_mat[0, 0]] == 0.0
```
```

**Step 2: Run test to verify it fails**

Run:
```bash
PYTHONPATH=/mnt/data/Projects/02_Research/demo_PhenoBrain/timgroup_disease_diagnosis/codes/core \
/mnt/data/Projects/02_Research/demo_PhenoBrain/.venv/bin/python -m pytest \
/mnt/data/Projects/02_Research/demo_PhenoBrain/timgroup_disease_diagnosis/codes/core/tests/test_cycommon_fallback.py -v
```

Expected:
- FAIL because `core.utils.cycommon` cannot currently be imported on Python 3.12.

**Step 3: Write minimal implementation**

Create:
- `timgroup_disease_diagnosis/codes/core/core/utils/cycommon.py`

Start with:

```python
import numpy as np


def to_rank_score(score_mat, arg_mat):
    raise NotImplementedError
```

**Step 4: Run test to verify it still fails correctly**

Expected:
- FAIL now because `NotImplementedError` is raised, proving the import path is fixed and behavior is next.

**Step 5: Commit**

Do not commit unless the user explicitly asks.

### Task 2: Implement the minimal pure Python to_rank_score logic

**Files:**
- Modify: `timgroup_disease_diagnosis/codes/core/core/utils/cycommon.py`
- Test: `timgroup_disease_diagnosis/codes/core/tests/test_cycommon_fallback.py`

**Step 1: Write the failing test refinement**

Extend the test to assert expected numeric behavior, matching the Cython algorithm for one row with a tie:

```python
def test_to_rank_score_rewrites_matrix_in_place():
    score_mat = np.array([[0.1, 0.4, 0.4, 0.9]], dtype=np.float64)
    arg_mat = np.argsort(score_mat).astype(np.int32)

    to_rank_score(score_mat, arg_mat)

    assert np.allclose(score_mat, np.array([[0.0, 0.25, 0.25, 0.5]], dtype=np.float64))
```

**Step 2: Run test to verify it fails**

Run the same single test file.
Expected: FAIL because the stub does not implement ranking yet.

**Step 3: Write minimal implementation**

Implement the pure Python fallback mirroring the Cython loop:

```python
import numpy as np


def to_rank_score(score_mat, arg_mat):
    eps = 1e-12
    row_num, col_num = score_mat.shape
    score_step = 1.0
    for i in range(row_num):
        score = 0.0
        last_raw_score = score_mat[i, arg_mat[i, 0]]
        for j in range(1, col_num):
            col = arg_mat[i, j]
            diff = score_mat[i, col] - last_raw_score
            last_raw_score = score_mat[i, col]
            if not (-eps < diff < eps):
                score += score_step
            score_mat[i, col] = score
        score_mat[i, arg_mat[i, 0]] = 0.0
        score_step /= col_num
```

Then immediately correct the per-row `score_step` placement so it resets correctly each row (it should be recomputed per row, not accumulated across rows).

Final minimal correct version should behave like the Cython code for each row independently.

**Step 4: Run test to verify it passes**

Run the test file again.
Expected: PASS

**Step 5: Commit**

Do not commit unless the user explicitly asks.

### Task 3: Add one multi-row regression test

**Files:**
- Modify: `timgroup_disease_diagnosis/codes/core/tests/test_cycommon_fallback.py`
- Modify: `timgroup_disease_diagnosis/codes/core/core/utils/cycommon.py`

**Step 1: Write the failing test**

Add one more test to ensure row independence:

```python
def test_to_rank_score_handles_multiple_rows_independently():
    score_mat = np.array([
        [0.1, 0.4, 0.4, 0.9],
        [0.2, 0.3, 0.8, 0.8],
    ], dtype=np.float64)
    arg_mat = np.argsort(score_mat).astype(np.int32)

    to_rank_score(score_mat, arg_mat)

    assert np.allclose(score_mat[0], np.array([0.0, 0.25, 0.25, 0.5]))
    assert np.allclose(score_mat[1], np.array([0.0, 0.25, 0.5, 0.5]))
```

**Step 2: Run test to verify it fails if row handling is wrong**

Run only this test or the file.
Expected: FAIL if the implementation leaked row state.

**Step 3: Write minimal implementation**

Adjust `to_rank_score` so `score` and `score_step` are handled per row exactly like the Cython version.

**Step 4: Run tests to verify they pass**

Run the full `test_cycommon_fallback.py` file.
Expected: PASS

**Step 5: Commit**

Do not commit unless the user explicitly asks.

### Task 4: Re-run the original Python 3.12 compatibility test

**Files:**
- Test: `timgroup_disease_diagnosis/codes/core/tests/test_python312_compat.py`
- Review: `timgroup_disease_diagnosis/codes/core/core/utils/utils.py`

**Step 1: Run the compatibility test**

Run:
```bash
PYTHONPATH=/mnt/data/Projects/02_Research/demo_PhenoBrain/timgroup_disease_diagnosis/codes/core \
/mnt/data/Projects/02_Research/demo_PhenoBrain/.venv/bin/python -m pytest \
/mnt/data/Projects/02_Research/demo_PhenoBrain/timgroup_disease_diagnosis/codes/core/tests/test_python312_compat.py -v
```

Expected:
- It should progress beyond the old `Iterable` and `cycommon` import failures.
- If it now fails on another missing package, that is acceptable evidence of progress.

**Step 2: Verify no cycommon import error remains**

The important success criterion is that the traceback no longer reports:
- `ModuleNotFoundError: No module named 'core.utils.cycommon'`

**Step 3: Commit**

Do not commit unless the user explicitly asks.

### Task 5: Re-run the offline script and capture the next boundary

**Files:**
- Review: `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`
- Review: `timgroup_disease_diagnosis/codes/core/core/utils/cycommon.py`

**Step 1: Run the script smoke test**

Run:
```bash
PYTHONPATH=/mnt/data/Projects/02_Research/demo_PhenoBrain/timgroup_disease_diagnosis/codes/core \
/mnt/data/Projects/02_Research/demo_PhenoBrain/.venv/bin/python \
/mnt/data/Projects/02_Research/demo_PhenoBrain/timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py
```

**Step 2: Record the next blocker**

Expected:
- The script should move past the old `cycommon` failure.
- If it now fails later, capture that exact next boundary for the user.

**Step 3: Summarize**

Report:
- the new fallback file added
- the tests that now pass
- whether the script progressed beyond the compiled-extension boundary
- the next blocker, if any

**Step 4: Commit**

Do not commit unless the user explicitly asks.
