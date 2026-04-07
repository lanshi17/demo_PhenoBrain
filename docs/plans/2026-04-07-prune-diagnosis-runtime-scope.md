# Prune Diagnosis Runtime Scope Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce `timgroup_disease_diagnosis/` to the files needed for the current offline diagnosis workflow documented in the root README while preserving the supported CLI path and focused verification suite.

**Architecture:** Treat the root `README.md` as the product contract and `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py` as the runtime entrypoint. Keep only the transitive code/data/model/test surface required by that script and the README-listed tests, then remove unrelated phenotype-extraction, paper-reproduction, training, Docker, and API materials in small verified batches.

**Tech Stack:** Python 3.12, pytest, Jupyter config, shell utilities, existing `codes/core` package structure.

---

### Task 1: Freeze the keep-scope in docs before deleting anything

**Files:**
- Modify: `README.md:37-105`
- Create: `docs/plans/2026-04-07-prune-diagnosis-runtime-scope.md`

**Step 1: Write the keep/delete checklist into this plan**

```md
Keep:
- `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`
- `timgroup_disease_diagnosis/codes/core/core/predict/**`
- `timgroup_disease_diagnosis/codes/core/core/reader/**`
- `timgroup_disease_diagnosis/codes/core/core/utils/**`
- `timgroup_disease_diagnosis/codes/core/core/helper/**` (needed by retained predict models)
- `timgroup_disease_diagnosis/codes/core/core/patient/**` (needed by `core.helper.data.data_helper`)
- `timgroup_disease_diagnosis/codes/core/core/explainer/**` (runtime imports exist in retained modules)
- `timgroup_disease_diagnosis/codes/core/data/**`
- `timgroup_disease_diagnosis/codes/core/model/**`
- `timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py`
- `timgroup_disease_diagnosis/codes/core/tests/test_cycommon_fallback.py`
- `timgroup_disease_diagnosis/codes/core/tests/test_python312_compat.py`
- `timgroup_disease_diagnosis/codes/core/tests/test_boqa_model.py`
- `timgroup_disease_diagnosis/codes/core/tests/test_hpo_reader_paths.py`
- `timgroup_disease_diagnosis/codes/core/tests/test_order_statistic_multi_model.py`
```

**Step 2: Record explicit first-pass deletion targets**

```md
Delete first:
- `timgroup_disease_diagnosis/codes/bert_syn_project/`
- `timgroup_disease_diagnosis/Docker/`
- `timgroup_disease_diagnosis/PhenoBrain_Web_API/`
- `timgroup_disease_diagnosis/example_result/`
- `timgroup_disease_diagnosis/codes/core/core/script/test/test_optimal_model.py`
- `timgroup_disease_diagnosis/codes/core/core/script/train/`
- `timgroup_disease_diagnosis/codes/core/tests/test_prob_model_imports.py`
```

**Step 3: Add a brief note to `README.md` if needed**

```md
This embedded README still documents the original upstream research project; the repository root README is the authoritative contract for the pruned offline runtime.
```

**Step 4: Run a quick grep to confirm the root README verification targets still match the plan**

Run: `python - <<'PY'
from pathlib import Path
text = Path('README.md').read_text()
for needle in ['example_predict_ensemble.py', 'test_example_predict_ensemble.py', 'test_order_statistic_multi_model.py']:
    assert needle in text, needle
print('ok')
PY`
Expected: `ok`

**Step 5: Commit**

```bash
git add README.md docs/plans/2026-04-07-prune-diagnosis-runtime-scope.md
git commit -m "docs: define runtime pruning scope"
```

### Task 2: Add a failing structure test for removed top-level runtime exclusions

**Files:**
- Create: `tests/test_pruned_runtime_layout.py`

**Step 1: Write the failing test**

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / 'timgroup_disease_diagnosis'


def test_pruned_runtime_keeps_only_supported_top_level_extras():
    assert not (DIAG / 'Docker').exists()
    assert not (DIAG / 'PhenoBrain_Web_API').exists()
    assert not (DIAG / 'codes' / 'bert_syn_project').exists()
    assert not (DIAG / 'example_result').exists()


def test_offline_runtime_entrypoint_still_exists():
    assert (DIAG / 'codes' / 'core' / 'core' / 'script' / 'example_predict_ensemble.py').exists()
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=timgroup_disease_diagnosis/codes/core .venv/bin/python -m pytest tests/test_pruned_runtime_layout.py -q`
Expected: FAIL because the to-be-removed directories still exist.

**Step 3: Write minimal implementation**

Implementation for this task is deletion only; do not change code yet.

**Step 4: Re-run after deletions later**

Run: `PYTHONPATH=timgroup_disease_diagnosis/codes/core .venv/bin/python -m pytest tests/test_pruned_runtime_layout.py -q`
Expected: PASS after Task 3.

**Step 5: Commit**

```bash
git add tests/test_pruned_runtime_layout.py
git commit -m "test: codify pruned runtime layout"
```

### Task 3: Remove obvious top-level non-runtime directories

**Files:**
- Delete: `timgroup_disease_diagnosis/codes/bert_syn_project/**`
- Delete: `timgroup_disease_diagnosis/Docker/**`
- Delete: `timgroup_disease_diagnosis/PhenoBrain_Web_API/**`
- Delete: `timgroup_disease_diagnosis/example_result/**`

**Step 1: Verify targets exist before deleting**

Run: `ls timgroup_disease_diagnosis && ls timgroup_disease_diagnosis/codes`
Expected: output includes `Docker`, `PhenoBrain_Web_API`, and `bert_syn_project`.

**Step 2: Delete only the approved directories**

Run: `rm -rf timgroup_disease_diagnosis/codes/bert_syn_project timgroup_disease_diagnosis/Docker timgroup_disease_diagnosis/PhenoBrain_Web_API timgroup_disease_diagnosis/example_result`
Expected: command succeeds with no output.

**Step 3: Run the new structure test**

Run: `PYTHONPATH=timgroup_disease_diagnosis/codes/core .venv/bin/python -m pytest tests/test_pruned_runtime_layout.py -q`
Expected: PASS.

**Step 4: Run the existing Jupyter sanity test to ensure unrelated repo tooling still works**

Run: `PYTHONPATH=timgroup_disease_diagnosis/codes/core .venv/bin/python -m pytest tests/test_jupyter_setup.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_pruned_runtime_layout.py timgroup_disease_diagnosis
git commit -m "refactor: drop non-runtime diagnosis directories"
```

### Task 4: Remove paper-reproduction and training script surfaces not covered by the runtime contract

**Files:**
- Delete: `timgroup_disease_diagnosis/codes/core/core/script/test/test_optimal_model.py`
- Delete: `timgroup_disease_diagnosis/codes/core/core/script/train/**`
- Review: `timgroup_disease_diagnosis/codes/core/core/script/__init__.py`

**Step 1: Add a failing test that the old reproduction entrypoints are gone**

```python
def test_old_research_entrypoints_are_removed():
    script_root = DIAG / 'codes' / 'core' / 'core' / 'script'
    assert not (script_root / 'test' / 'test_optimal_model.py').exists()
    assert not (script_root / 'train').exists()
```

Add this to `tests/test_pruned_runtime_layout.py`.

**Step 2: Run the focused test to verify it fails**

Run: `PYTHONPATH=timgroup_disease_diagnosis/codes/core .venv/bin/python -m pytest tests/test_pruned_runtime_layout.py::test_old_research_entrypoints_are_removed -q`
Expected: FAIL.

**Step 3: Delete the obsolete script tree**

Run: `rm -rf timgroup_disease_diagnosis/codes/core/core/script/train timgroup_disease_diagnosis/codes/core/core/script/test/test_optimal_model.py`
Expected: command succeeds.

**Step 4: Run the structure test and the ensemble tests**

Run: `PYTHONPATH=timgroup_disease_diagnosis/codes/core .venv/bin/python -m pytest tests/test_pruned_runtime_layout.py timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py timgroup_disease_diagnosis/codes/core/tests/test_order_statistic_multi_model.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_pruned_runtime_layout.py timgroup_disease_diagnosis/codes/core/core/script
git commit -m "refactor: remove research-only diagnosis scripts"
```

### Task 5: Remove tests outside the supported verification contract

**Files:**
- Delete: `timgroup_disease_diagnosis/codes/core/tests/test_prob_model_imports.py`
- Review: `README.md:92-105`

**Step 1: Verify the file is outside the root README test contract**

Run: `python - <<'PY'
from pathlib import Path
text = Path('README.md').read_text()
assert 'test_prob_model_imports.py' not in text
print('ok')
PY`
Expected: `ok`

**Step 2: Delete the extra test**

Run: `rm timgroup_disease_diagnosis/codes/core/tests/test_prob_model_imports.py`
Expected: command succeeds.

**Step 3: Run the README-listed test suite exactly as documented**

Run: `PYTHONPATH=timgroup_disease_diagnosis/codes/core .venv/bin/python -m pytest timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py timgroup_disease_diagnosis/codes/core/tests/test_cycommon_fallback.py timgroup_disease_diagnosis/codes/core/tests/test_python312_compat.py timgroup_disease_diagnosis/codes/core/tests/test_boqa_model.py timgroup_disease_diagnosis/codes/core/tests/test_hpo_reader_paths.py timgroup_disease_diagnosis/codes/core/tests/test_order_statistic_multi_model.py tests/test_jupyter_setup.py tests/test_pruned_runtime_layout.py -q`
Expected: PASS.

**Step 4: Check the CLI still starts and prints the quiet summary line**

Run: `PYTHONPATH=timgroup_disease_diagnosis/codes/core .venv/bin/python timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py --topk 3 --hpo-list HP:0001913,HP:0008513,HP:0001123`
Expected: prints `Available models:` followed by a 3-row table. Some optional models may be skipped if local assets are absent.

**Step 5: Commit**

```bash
git add tests/test_pruned_runtime_layout.py README.md timgroup_disease_diagnosis/codes/core/tests
git commit -m "test: align diagnosis verification with runtime scope"
```

### Task 6: Do a final repo sweep for stale references to deleted areas

**Files:**
- Modify: `README.md` if any runtime-facing instructions still mention removed paths
- Review: `timgroup_disease_diagnosis/README.md`
- Review: `docs/plans/*.md` only if runtime-facing docs point users at deleted paths

**Step 1: Search for stale references**

Run: `python - <<'PY'
from pathlib import Path
needles = ['bert_syn_project', 'PhenoBrain_Web_API', 'Docker/', 'core/script/test/test_optimal_model.py']
for needle in needles:
    print(f'## {needle}')
    for path in Path('.').rglob('*'):
        if path.is_file() and '.git/' not in str(path):
            try:
                text = path.read_text()
            except Exception:
                continue
            if needle in text:
                print(path)
PY`
Expected: runtime-facing docs are either updated or intentionally historical only.

**Step 2: Make the smallest necessary doc edits**

```md
- Root README remains authoritative for current runtime usage.
- Embedded upstream README can remain historical if clearly marked as such.
```

**Step 3: Re-run the exact supported suite**

Run: `PYTHONPATH=timgroup_disease_diagnosis/codes/core .venv/bin/python -m pytest timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py timgroup_disease_diagnosis/codes/core/tests/test_cycommon_fallback.py timgroup_disease_diagnosis/codes/core/tests/test_python312_compat.py timgroup_disease_diagnosis/codes/core/tests/test_boqa_model.py timgroup_disease_diagnosis/codes/core/tests/test_hpo_reader_paths.py timgroup_disease_diagnosis/codes/core/tests/test_order_statistic_multi_model.py tests/test_jupyter_setup.py tests/test_pruned_runtime_layout.py -q`
Expected: PASS.

**Step 4: Inspect git diff for scope discipline**

Run: `git diff --stat`
Expected: deletions are concentrated in explicitly approved paths plus the new pruning test/doc updates.

**Step 5: Commit**

```bash
git add README.md timgroup_disease_diagnosis/README.md tests/test_pruned_runtime_layout.py
git commit -m "docs: finalize diagnosis runtime pruning"
```
