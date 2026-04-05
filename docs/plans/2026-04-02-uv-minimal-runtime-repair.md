# UV Minimal Runtime Repair Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the offline diagnosis script progress past the current missing-package failure using the smallest persistent `uv`-managed dependency changes.

**Architecture:** Treat the current failure as an environment/runtime defect, not a model-selection defect. Add the smallest likely runtime dependency set to `pyproject.toml` via `uv add`, re-run the script after each change boundary, and stop before forcing legacy TensorFlow packages into a Python 3.12 environment.

**Tech Stack:** `uv`, Python 3.12 virtual environment, `pyproject.toml`, legacy ML/scientific Python packages.

## Completion Notes (2026-04-05)
- `pyproject.toml` now contains the minimal scientific runtime packages that were originally missing, and the current `.venv` supports the offline diagnosis stack used in the focused suite.
- The original import-time boundary is fully cleared: verification on 2026-04-05 showed 40 passing focused tests and a successful default offline script run.

---

### Task 1: Reproduce the current missing-package failure

**Files:**
- Check: `pyproject.toml`
- Check: `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`

**Step 1: Write the failing test**

For this environment task, the failing reproduction is a direct command rather than a Python unit test:

```bash
PYTHONPATH=/mnt/data/Projects/02_Research/demo_PhenoBrain/timgroup_disease_diagnosis/codes/core \
/mnt/data/Projects/02_Research/demo_PhenoBrain/.venv/bin/python \
/mnt/data/Projects/02_Research/demo_PhenoBrain/timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py
```

**Step 2: Run reproduction to verify it fails**

Expected:
- FAIL with `ModuleNotFoundError: No module named 'scipy'`

**Step 3: Record the root cause**

Note that the failure occurs during import of `scipy.sparse` through `core.predict.model`, before model selection logic runs.

**Step 4: Commit**

Do not commit unless the user explicitly asks.

### Task 2: Add the minimum scientific-runtime packages with uv

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Step 1: Write the failing test**

The failing reproduction remains the script command from Task 1.

**Step 2: Run command to verify current failure still occurs**

Run the script once more if needed to confirm the baseline before changing dependencies.

**Step 3: Write minimal implementation**

Run:

```bash
uv add scipy scikit-learn joblib pyemd
```

This should update `pyproject.toml` and `uv.lock` persistently.

**Step 4: Run script to verify progress**

Run:

```bash
PYTHONPATH=/mnt/data/Projects/02_Research/demo_PhenoBrain/timgroup_disease_diagnosis/codes/core \
/mnt/data/Projects/02_Research/demo_PhenoBrain/.venv/bin/python \
/mnt/data/Projects/02_Research/demo_PhenoBrain/timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py
```

Expected:
- It should fail later than the current `scipy` import boundary, or succeed further into model construction.
- It should no longer fail with `No module named 'scipy'`.

**Step 5: Commit**

Do not commit unless the user explicitly asks.

### Task 3: Inspect the next failure and stop at the correct boundary

**Files:**
- Check: runtime stack trace output only
- Possibly modify: `pyproject.toml` / `uv.lock` only if the next missing package is clearly compatible and still part of the minimal runtime set

**Step 1: Write the failing test**

Use the next observed script failure as the new reproduction.

**Step 2: Run command to verify the next failure**

Re-run the script if needed.

**Step 3: Form one hypothesis**

Examples:
- If the next error is another missing scientific package compatible with Python 3.12, it belongs in the minimal runtime set.
- If the next error is `tensorflow==1.x` incompatibility, stop and report that the environment has reached a legacy-version boundary.

**Step 4: Write minimal implementation only if justified**

Only if the next failure is a clearly missing, compatible runtime package, add it with `uv add <package>`.

Do **not** immediately add legacy TensorFlow packages.

**Step 5: Run script to verify**

Expected:
- Either progress continues, or the script now fails at a legacy framework compatibility boundary that needs a separate design decision.

**Step 6: Commit**

Do not commit unless the user explicitly asks.

### Task 4: Verify the dependency state explicitly

**Files:**
- Check: `pyproject.toml`
- Check: `uv.lock`

**Step 1: Write the failing test**

Use package inspection commands as verification:

```bash
/mnt/data/Projects/02_Research/demo_PhenoBrain/.venv/bin/python -m pip show scipy scikit-learn joblib pyemd
```

**Step 2: Run verification command**

Expected:
- Installed package metadata is shown for each package that was added.

**Step 3: Verify dependency declarations**

Check that `pyproject.toml` contains the newly added packages.

**Step 4: Verify lockfile updated**

Confirm `uv.lock` changed as part of the `uv add` action.

**Step 5: Commit**

Do not commit unless the user explicitly asks.

### Task 5: Final verification and handoff

**Files:**
- Review: `pyproject.toml`
- Review: `uv.lock`

**Step 1: Run the script one final time**

Run:

```bash
PYTHONPATH=/mnt/data/Projects/02_Research/demo_PhenoBrain/timgroup_disease_diagnosis/codes/core \
/mnt/data/Projects/02_Research/demo_PhenoBrain/.venv/bin/python \
/mnt/data/Projects/02_Research/demo_PhenoBrain/timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py
```

**Step 2: Capture the exact current boundary**

Possible outcomes:
- The script now prints available models and continues into prediction
- The script reaches a model-specific runtime error
- The script stops at a legacy TensorFlow compatibility boundary

**Step 3: Summarize for the user**

Report:
- which packages were added with `uv`
- whether the script moved past the `scipy` failure
- the exact next blocker, if any
- whether that blocker is a normal missing package or a deeper Python 3.12 vs TensorFlow 1.x compatibility issue

**Step 4: Commit**

Do not commit unless the user explicitly asks.
