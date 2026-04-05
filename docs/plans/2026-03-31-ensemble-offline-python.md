# Ensemble Offline Python Example Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a minimal offline Python example that directly imports the repo’s diagnosis classes and runs one `Ensemble` prediction from a supplied HPO list.

**Architecture:** Place a single runnable script under the existing diagnosis script tree so it can reuse the repo’s path assumptions and imports. Build the public `Ensemble` behavior by instantiating the four internal component models used by the project (`ICTO`, `HPOProb`/`PPO`, `CNB`, `MLP`) and wrapping them in `OrderedMultiModel`, then call the standard single-sample `query(hpo_list, topk)` interface.

**Tech Stack:** Python 3.6-style project code, NumPy, TensorFlow 1.x / scikit-learn-backed project models, internal `core.*` packages.

## Completion Notes (2026-04-05)
- The offline example script exists at `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py` and has since been extended with dynamic candidate selection, CLI parameters, quiet output, result-table formatting, BOQA fallback, and BOQA subprocess quieting.
- Verification on 2026-04-05: the focused offline suite passed with 40 tests, and the default script execution succeeded end-to-end.

---

### Task 1: Verify model assets and import assumptions

**Files:**
- Check: `timgroup_disease_diagnosis/codes/core/model/`
- Check: `timgroup_disease_diagnosis/codes/core/core/utils/constant.py:6-13`
- Check: `timgroup_disease_diagnosis/README.md:64-91`
- Check: `timgroup_disease_diagnosis/codes/core/core/script/test/test_optimal_model.py:500-506`
- Check: `timgroup_disease_diagnosis/codes/core/core/script/test/test_optimal_model.py:558-566`
- Check: `timgroup_disease_diagnosis/codes/core/core/script/test/test_optimal_model.py:714-720`
- Check: `timgroup_disease_diagnosis/codes/core/core/script/test/test_optimal_model.py:730-736`

**Step 1: Check that required saved model assets exist**

Run:
```bash
ls -R timgroup_disease_diagnosis/codes/core/model
```

Expected:
- Folders/files for the reader-specific saved models exist.
- If the folder only contains a placeholder README, the script can still be written but runtime should fail fast with a clear message.

**Step 2: Confirm run location assumptions**

Run:
```bash
python - <<'PY'
from pathlib import Path
p = Path('timgroup_disease_diagnosis/codes/core/core/utils/constant.py').resolve()
print(p)
PY
```

Expected:
- Confirms the example should run with `PYTHONPATH` pointing at `timgroup_disease_diagnosis/codes/core`.

**Step 3: Record the exact internal Ensemble composition**

Use the repo’s existing definitions:
- `ICTO(A)` -> `ICTODQAcrossModel + RandomModel`
- `PPO` public concept -> `HPOProbMNBModel + RandomModel`
- `CNB` -> `CNBModel + RandomModel`
- `MLP` -> `LRNeuronModel + RandomModel`

**Step 4: Commit**

Do not commit yet unless the user explicitly asks.

### Task 2: Write a failing smoke test for the example script

**Files:**
- Create: `timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py`
- Test target: `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`

**Step 1: Write the failing test**

```python
import importlib.util
from pathlib import Path


def test_example_script_exists_and_exports_builder():
    script_path = Path(__file__).resolve().parents[1] / 'core' / 'script' / 'example_predict_ensemble.py'
    assert script_path.exists()

    spec = importlib.util.spec_from_file_location('example_predict_ensemble', str(script_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, 'build_ensemble_model')
    assert hasattr(module, 'predict_ensemble')
```

**Step 2: Run test to verify it fails**

Run:
```bash
PYTHONPATH=timgroup_disease_diagnosis/codes/core pytest timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py -v
```

Expected:
- FAIL because `example_predict_ensemble.py` does not exist yet.

**Step 3: Commit**

Do not commit yet unless the user explicitly asks.

### Task 3: Add the minimal offline example script

**Files:**
- Create: `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`
- Reference: `timgroup_disease_diagnosis/codes/core/core/predict/ensemble/ordered_multi_model.py:8-52`
- Reference: `timgroup_disease_diagnosis/codes/core/core/predict/model.py:57-68`
- Reference: `timgroup_disease_diagnosis/codes/core/core/reader/__init__.py`

**Step 1: Write minimal implementation skeleton**

Start with this structure:

```python
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

from core.reader import HPOIntegratedDatasetReader
from core.predict.ensemble import OrderedMultiModel, RandomModel
from core.predict.sim_model import ICTODQAcrossModel
from core.predict.prob_model import HPOProbMNBModel, CNBModel
from core.predict.ml_model import LRNeuronModel
from core.utils.constant import PHELIST_REDUCE, VEC_TYPE_0_1


DEFAULT_KEEP_DNAMES = ['OMIM', 'ORPHA', 'CCRD']
DEFAULT_TEST_SEED = 777
```

**Step 2: Add a builder for the shared reader**

```python
def build_hpo_reader():
    return HPOIntegratedDatasetReader(keep_dnames=DEFAULT_KEEP_DNAMES)
```

**Step 3: Add the Ensemble builder**

Build the exact internal public Ensemble composition in a readable way:

```python
def build_ensemble_model(hpo_reader=None):
    hpo_reader = hpo_reader or build_hpo_reader()

    component_models = [
        OrderedMultiModel(
            [
                (ICTODQAcrossModel, (hpo_reader,), {'model_name': 'ICTODQAcross-Ave', 'sym_mode': 'ave'}),
                (RandomModel, (hpo_reader,), {'seed': DEFAULT_TEST_SEED}),
            ],
            model_name='ICTODQAcross-Ave-Random',
            hpo_reader=hpo_reader,
        ),
        OrderedMultiModel(
            [
                (HPOProbMNBModel, (hpo_reader,), {'phe_list_mode': PHELIST_REDUCE, 'model_name': 'HPOProbMNB'}),
                (RandomModel, (hpo_reader,), {'seed': DEFAULT_TEST_SEED}),
            ],
            model_name='HPOProbMNB-Random',
            hpo_reader=hpo_reader,
        ),
        OrderedMultiModel(
            [
                (CNBModel, (hpo_reader, VEC_TYPE_0_1), {},),
                (RandomModel, (hpo_reader,), {'seed': DEFAULT_TEST_SEED}),
            ],
            model_name='CNB-Random',
            hpo_reader=hpo_reader,
        ),
        OrderedMultiModel(
            [
                (LRNeuronModel, (hpo_reader, VEC_TYPE_0_1), {'model_name': 'NN-Mixup-1'}),
                (RandomModel, (hpo_reader,), {'seed': DEFAULT_TEST_SEED}),
            ],
            model_name='NN-Mixup-Random-1',
            hpo_reader=hpo_reader,
        ),
    ]

    return OrderedMultiModel(model_list=component_models, hpo_reader=hpo_reader, model_name='Ensemble')
```

**Step 4: Add the single-call helper**

```python
def predict_ensemble(hpo_list, topk=10):
    model = build_ensemble_model()
    return model.query(hpo_list, topk)
```

**Step 5: Add a runnable CLI example**

```python
if __name__ == '__main__':
    hpo_list = [
        'HP:0001913',
        'HP:0008513',
        'HP:0001123',
        'HP:0000365',
        'HP:0002857',
        'HP:0001744',
    ]
    topk = 5
    results = predict_ensemble(hpo_list, topk=topk)
    for disease_code, score in results:
        print(f'{disease_code}\t{score}')
```

**Step 6: Fail fast when model weights are missing**

Add a small check before model construction that verifies the required subfolders exist under `codes/core/model`. Raise `FileNotFoundError` with a short actionable message if missing.

**Step 7: Commit**

Do not commit yet unless the user explicitly asks.

### Task 4: Make the smoke test pass

**Files:**
- Modify: `timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py`
- Modify: `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`

**Step 1: Extend the test to avoid loading heavyweight assets**

Patch the test to validate importability and API shape only:

```python
def test_predict_ensemble_accepts_inputs(monkeypatch):
    script_path = Path(__file__).resolve().parents[1] / 'core' / 'script' / 'example_predict_ensemble.py'
    spec = importlib.util.spec_from_file_location('example_predict_ensemble', str(script_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class DummyModel:
        def query(self, hpo_list, topk):
            return [('RD:1', 1.0)]

    monkeypatch.setattr(module, 'build_ensemble_model', lambda: DummyModel())
    assert module.predict_ensemble(['HP:0000118'], topk=1) == [('RD:1', 1.0)]
```

**Step 2: Run the test to verify it passes**

Run:
```bash
PYTHONPATH=timgroup_disease_diagnosis/codes/core pytest timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py -v
```

Expected:
- PASS

**Step 3: Run a direct script smoke check**

Run:
```bash
PYTHONPATH=timgroup_disease_diagnosis/codes/core python timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py
```

Expected:
- Either prints Top-K disease results, or exits with the explicit missing-model-assets error from Task 3 Step 6.
- It should not fail with `ImportError` or path errors.

**Step 4: Commit**

Do not commit yet unless the user explicitly asks.

### Task 5: Add concise usage notes

**Files:**
- Modify: `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`
- Optional modify: `timgroup_disease_diagnosis/README.md`

**Step 1: Keep usage notes in the script header**

Add a short top-of-file docstring covering:
- how to set `PYTHONPATH`
- where model assets must exist
- how to replace the sample `hpo_list`

**Step 2: Only update the main README if needed**

If the script is self-explanatory, skip README edits. Do not add extra docs unless necessary.

**Step 3: Re-run verification**

Run:
```bash
PYTHONPATH=timgroup_disease_diagnosis/codes/core pytest timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py -v
PYTHONPATH=timgroup_disease_diagnosis/codes/core python timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py
```

Expected:
- Test stays green.
- Script still imports and either runs or fails fast with the intended message.

**Step 4: Commit**

Do not commit yet unless the user explicitly asks.

### Task 6: Final verification and handoff

**Files:**
- Review: `timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py`
- Review: `timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py`

**Step 1: Verify final behavior**

Check that the script:
- uses the repo’s real `OrderedMultiModel`
- exposes `build_ensemble_model()` and `predict_ensemble()`
- accepts plain `hpo_list`
- prints stable, readable output
- explains missing model assets clearly

**Step 2: Provide user-facing invocation snippet**

```bash
PYTHONPATH=timgroup_disease_diagnosis/codes/core \
python timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py
```

**Step 3: Summarize caveats**

Mention explicitly:
- docs say `PPO`, but the code-level model used in Ensemble is `HPOProbMNBModel`
- runtime depends on local saved model assets under `timgroup_disease_diagnosis/codes/core/model/`

**Step 4: Commit**

Do not commit yet unless the user explicitly asks.
