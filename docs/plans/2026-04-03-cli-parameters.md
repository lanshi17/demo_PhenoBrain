# CLI Parameters Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `--topk` and `--hpo-list` command-line parameters to the offline diagnosis script while preserving current defaults and formatted output.

**Architecture:** Keep the current script behavior intact for zero-argument execution, then add a small `argparse` layer that overrides the sample HPO list and default top-k only when the user provides CLI values. Reuse the existing `predict_ensemble()` and `format_results_table()` logic so the CLI layer stays thin.

**Tech Stack:** Python standard library (`argparse`), existing diagnosis script, pytest.

## Completion Notes (2026-04-05)
- `--topk` and `--hpo-list` are implemented in `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`.
- Verification on 2026-04-05 included both helper-test coverage and successful real-script override execution; the current default and CLI modes both work.

---

### Task 1: Add a failing parser/helper test for comma-separated HPO input

**Files:**
- Modify: `timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py`
- Modify: `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`

**Step 1: Write the failing test**

Add a focused helper test:

```python
def test_parse_hpo_list_splits_comma_separated_values():
    module = load_module()
    assert module.parse_hpo_list('HP:0001913,HP:0008513,HP:0001123') == [
        'HP:0001913',
        'HP:0008513',
        'HP:0001123',
    ]
```

**Step 2: Run test to verify it fails**

Run:
```bash
PYTHONPATH=/mnt/data/Projects/02_Research/demo_PhenoBrain/timgroup_disease_diagnosis/codes/core \
/mnt/data/Projects/02_Research/demo_PhenoBrain/.venv/bin/python -m pytest \
/mnt/data/Projects/02_Research/demo_PhenoBrain/timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py::test_parse_hpo_list_splits_comma_separated_values -v
```

Expected: FAIL because `parse_hpo_list` does not exist yet.

**Step 3: Write minimal implementation**

Add:

```python
def parse_hpo_list(value):
    return [item.strip() for item in value.split(',') if item.strip()]
```

**Step 4: Run test to verify it passes**

Run the same test again.
Expected: PASS

**Step 5: Commit**

Do not commit unless the user explicitly asks.

### Task 2: Add a failing test for argument parsing defaults

**Files:**
- Modify: `timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py`
- Modify: `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`

**Step 1: Write the failing test**

Add a helper-oriented test for CLI defaults:

```python
def test_parse_args_uses_default_topk_and_no_hpo_list():
    module = load_module()
    args = module.parse_args([])
    assert args.topk == 5
    assert args.hpo_list is None
```

**Step 2: Run test to verify it fails**

Run the single test.
Expected: FAIL because `parse_args` does not exist yet.

**Step 3: Write minimal implementation**

Add a thin parser helper:

```python
import argparse


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--hpo-list')
    return parser.parse_args(argv)
```

**Step 4: Run test to verify it passes**

Run the test again.
Expected: PASS

**Step 5: Commit**

Do not commit unless the user explicitly asks.

### Task 3: Add a failing test for CLI override behavior

**Files:**
- Modify: `timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py`
- Modify: `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`

**Step 1: Write the failing test**

Add a focused helper test:

```python
def test_parse_args_accepts_topk_and_hpo_list_overrides():
    module = load_module()
    args = module.parse_args(['--topk', '10', '--hpo-list', 'HP:1,HP:2'])
    assert args.topk == 10
    assert args.hpo_list == 'HP:1,HP:2'
```

**Step 2: Run test to verify it fails**

Run the single test.
Expected: FAIL until the parser is wired correctly.

**Step 3: Write minimal implementation**

Refine `parse_args()` if needed so the override test passes.

**Step 4: Run test to verify it passes**

Run the test again.
Expected: PASS

**Step 5: Commit**

Do not commit unless the user explicitly asks.

### Task 4: Update the script entrypoint to use parsed arguments

**Files:**
- Modify: `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`
- Test: `timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py`

**Step 1: Write the failing test**

If practical, add a helper-level test around a new resolver function instead of `__main__` directly:

```python
def test_resolve_cli_inputs_prefers_user_hpo_list():
    module = load_module()
    args = module.parse_args(['--hpo-list', 'HP:1,HP:2', '--topk', '3'])
    hpo_list, topk = module.resolve_cli_inputs(args)
    assert hpo_list == ['HP:1', 'HP:2']
    assert topk == 3
```

**Step 2: Run test to verify it fails**

Run the single test.
Expected: FAIL because `resolve_cli_inputs` does not exist yet.

**Step 3: Write minimal implementation**

Add:

```python
DEFAULT_SAMPLE_HPO_LIST = [...]


def resolve_cli_inputs(args):
    hpo_list = parse_hpo_list(args.hpo_list) if args.hpo_list else DEFAULT_SAMPLE_HPO_LIST
    return hpo_list, args.topk
```

Then change `__main__` to:

```python
args = parse_args()
hpo_list, topk = resolve_cli_inputs(args)
print(describe_available_models())
results = predict_ensemble(hpo_list, topk=topk)
print(format_results_table(results))
```

**Step 4: Run test to verify it passes**

Run the targeted tests.
Expected: PASS

**Step 5: Commit**

Do not commit unless the user explicitly asks.

### Task 5: Final verification with real CLI execution

**Files:**
- Review: `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`
- Review: `timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py`

**Step 1: Run the new CLI helper tests**

Run the newly added tests.

**Step 2: Run the script with defaults**

Run:
```bash
PYTHONPATH=/mnt/data/Projects/02_Research/demo_PhenoBrain/timgroup_disease_diagnosis/codes/core \
/mnt/data/Projects/02_Research/demo_PhenoBrain/.venv/bin/python \
/mnt/data/Projects/02_Research/demo_PhenoBrain/timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py
```

Expected:
- Existing default sample behavior still works.

**Step 3: Run the script with explicit CLI parameters**

Run:
```bash
PYTHONPATH=/mnt/data/Projects/02_Research/demo_PhenoBrain/timgroup_disease_diagnosis/codes/core \
/mnt/data/Projects/02_Research/demo_PhenoBrain/.venv/bin/python \
/mnt/data/Projects/02_Research/demo_PhenoBrain/timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py \
  --topk 3 \
  --hpo-list HP:0001913,HP:0008513,HP:0001123
```

Expected:
- Output contains the same quiet model list line
- Result table contains exactly 3 rows
- No raw tuple repr is printed

**Step 4: Summarize**

Report:
- helper functions added
- whether default mode still works
- whether CLI override mode works

**Step 5: Commit**

Do not commit unless the user explicitly asks.
