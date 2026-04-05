# Result Table Formatting Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Render the final prediction results as a clean aligned text table without changing the existing prediction API.

**Architecture:** Keep `predict_ensemble()` returning the same list of `(disease_code, score)` tuples. Add a small pure-formatting helper in the script and use it only in the `__main__` printing path.

**Tech Stack:** Python standard library string formatting, existing diagnosis script, pytest.

## Completion Notes (2026-04-05)
- The formatting helper is implemented and the `__main__` path prints the aligned table instead of a raw tuple list.
- Verification on 2026-04-05: formatting tests passed within the focused offline suite, and the real script output shows the expected `Rank`, `Disease`, and `Score` columns.

---

### Task 1: Add a failing formatting-helper test

**Files:**
- Modify: `timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py`
- Modify: `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`

**Step 1: Write the failing test**

Add a focused helper test:

```python
def test_format_results_table_renders_rank_disease_score_columns():
    module = load_module()
    results = [
        ('RD:7367', 1.0107991360691828),
        ('RD:6963', 1.0016198704104353),
    ]

    table = module.format_results_table(results)

    assert 'Rank' in table
    assert 'Disease' in table
    assert 'Score' in table
    assert 'RD:7367' in table
    assert '1.010799' in table
```

**Step 2: Run test to verify it fails**

Run:
```bash
PYTHONPATH=/mnt/data/Projects/02_Research/demo_PhenoBrain/timgroup_disease_diagnosis/codes/core \
/mnt/data/Projects/02_Research/demo_PhenoBrain/.venv/bin/python -m pytest \
/mnt/data/Projects/02_Research/demo_PhenoBrain/timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py::test_format_results_table_renders_rank_disease_score_columns -v
```

Expected: FAIL because `format_results_table` does not exist yet.

**Step 3: Write minimal implementation**

Add:

```python
def format_results_table(results):
    ...
```

with just enough logic to produce a readable aligned table.

**Step 4: Run test to verify it passes**

Run the same single test.
Expected: PASS

**Step 5: Commit**

Do not commit unless the user explicitly asks.

### Task 2: Add a failing test for empty-result formatting

**Files:**
- Modify: `timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py`
- Modify: `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`

**Step 1: Write the failing test**

Add one edge-case test:

```python
def test_format_results_table_handles_empty_results():
    module = load_module()
    assert module.format_results_table([]) == 'No results.'
```

**Step 2: Run test to verify it fails**

Run the single test.
Expected: FAIL because empty handling is missing.

**Step 3: Write minimal implementation**

Add a small early return:

```python
if not results:
    return 'No results.'
```

**Step 4: Run test to verify it passes**

Run the single test.
Expected: PASS

**Step 5: Commit**

Do not commit unless the user explicitly asks.

### Task 3: Update the script entrypoint to print the table

**Files:**
- Modify: `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`
- Test: `timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py`

**Step 1: Write the failing test**

If practical, add a helper-level test that `format_results_table` output is used rather than raw tuple repr. If that is too coupled to `__main__`, skip direct `__main__` testing and rely on the final script verification.

**Step 2: Run test to verify it fails**

Only if a focused helper-level failure is added.

**Step 3: Write minimal implementation**

Change the script footer from:

```python
print(predict_ensemble(sample_hpo_list, topk=5))
```

to:

```python
results = predict_ensemble(sample_hpo_list, topk=5)
print(format_results_table(results))
```

**Step 4: Run targeted tests to verify they pass**

Run the formatting tests.
Expected: PASS

**Step 5: Commit**

Do not commit unless the user explicitly asks.

### Task 4: Final verification with the real script

**Files:**
- Review: `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`
- Review: `timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py`

**Step 1: Run the formatting tests**

Run the new formatting-related tests.

**Step 2: Run the real script**

Run:
```bash
PYTHONPATH=/mnt/data/Projects/02_Research/demo_PhenoBrain/timgroup_disease_diagnosis/codes/core \
/mnt/data/Projects/02_Research/demo_PhenoBrain/.venv/bin/python \
/mnt/data/Projects/02_Research/demo_PhenoBrain/timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py
```

Expected output shape:
- one `Available models: ...` line
- one aligned table with columns `Rank`, `Disease`, `Score`
- no raw Python tuple list repr

**Step 3: Summarize**

Report:
- the helper added
- whether formatting tests pass
- whether the script now prints a readable table

**Step 4: Commit**

Do not commit unless the user explicitly asks.
