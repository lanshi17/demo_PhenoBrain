# GA4GH Full Benchmark Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a reproducible full GA4GH benchmark flow that mirrors the existing MME benchmark, runs all 384 converted GA4GH cases, saves a summary JSON, and documents the final metrics in README.

**Architecture:** Keep the flow script-based and lightweight. `scripts/convert_ga4gh_benchmark.py` owns source-to-PhenoBrain dataset conversion and validation, while `scripts/benchmark_ga4gh.py` owns model/testor setup, metric execution, summary formatting, and summary JSON output. Tests cover conversion and benchmark helper behavior without loading full model assets.

**Tech Stack:** Python 3.12, pytest, `ModelTestor`, `HPOIntegratedDatasetReader`, `example_predict_ensemble.py`, JSON files under `data/GA4GH` and `data/inputs`.

---

### Task 1: Make GA4GH conversion reusable and tested

**Files:**
- Create: `tests/test_ga4gh_benchmark_conversion.py`
- Modify: `scripts/convert_ga4gh_benchmark.py`

**Step 1: Write the failing conversion tests**

Create `tests/test_ga4gh_benchmark_conversion.py`:

```python
import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'convert_ga4gh_benchmark.py'


def load_module():
    spec = importlib.util.spec_from_file_location('convert_ga4gh_benchmark', str(SCRIPT_PATH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def test_convert_ga4gh_to_dataset_normalizes_mim_and_preserves_omim(tmp_path):
    module = load_module()
    questions_path = tmp_path / 'questions.json'
    answers_path = tmp_path / 'answers.json'
    output_path = tmp_path / 'converted.json'
    write_json(
        questions_path,
        [
            {
                'patient_id': 'p1',
                'hpo_terms': [
                    {'hpo_id': 'HP:0000118', 'hpo_name': 'Phenotypic abnormality'},
                    {'hpo_id': 'HP:0001250', 'hpo_name': 'Seizure'},
                ],
            },
            {
                'patient_id': 'p2',
                'hpo_terms': [{'hpo_id': 'HP:0004322', 'hpo_name': 'Short stature'}],
            },
        ],
    )
    write_json(
        answers_path,
        [
            {'patient_id': 'p1', 'answers': [{'omim_id': 'MIM:123456', 'disease_name': 'A'}]},
            {'patient_id': 'p2', 'answers': [{'omim_id': 'OMIM:654321', 'disease_name': 'B'}]},
        ],
    )

    dataset = module.convert_ga4gh_to_dataset(questions_path, answers_path, output_path)

    assert dataset == [
        [['HP:0000118', 'HP:0001250'], ['OMIM:123456']],
        [['HP:0004322'], ['OMIM:654321']],
    ]
    assert json.loads(output_path.read_text(encoding='utf-8')) == dataset


def test_convert_ga4gh_to_dataset_skips_missing_or_empty_answers(tmp_path):
    module = load_module()
    questions_path = tmp_path / 'questions.json'
    answers_path = tmp_path / 'answers.json'
    output_path = tmp_path / 'converted.json'
    write_json(
        questions_path,
        [
            {'patient_id': 'p1', 'hpo_terms': [{'hpo_id': 'HP:0000118'}]},
            {'patient_id': 'p2', 'hpo_terms': [{'hpo_id': 'HP:0001250'}]},
            {'patient_id': 'p3', 'hpo_terms': [{'hpo_id': 'HP:0004322'}]},
        ],
    )
    write_json(
        answers_path,
        [
            {'patient_id': 'p1', 'answers': [{'omim_id': 'MIM:123456'}]},
            {'patient_id': 'p2', 'answers': []},
        ],
    )

    dataset = module.convert_ga4gh_to_dataset(questions_path, answers_path, output_path)

    assert dataset == [[['HP:0000118'], ['OMIM:123456']]]
```

**Step 2: Run the conversion tests to verify they fail**

Run:

```bash
uv run pytest tests/test_ga4gh_benchmark_conversion.py -q
```

Expected: FAIL with `TypeError` because `convert_ga4gh_to_dataset()` currently accepts no path arguments.

**Step 3: Implement the minimal reusable converter**

Modify `scripts/convert_ga4gh_benchmark.py`:

```python
#!/usr/bin/env python3
"""Convert GA4GH benchmark questions/answers to the standard PhenoBrain dataset format."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GA4GH_DIR = PROJECT_ROOT / 'data' / 'GA4GH'
QUESTIONS_PATH = GA4GH_DIR / 'GA4GH.benchmark_patients.questions.json'
ANSWERS_PATH = GA4GH_DIR / 'GA4GH.benchmark_patients.answers.json'
OUTPUT_PATH = PROJECT_ROOT / 'data' / 'inputs' / 'GA4GH.benchmark_patients.json'


def load_json(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def normalize_omim_id(omim_id):
    if omim_id.startswith('MIM:'):
        return 'OMIM:' + omim_id[4:]
    return omim_id


def convert_ga4gh_to_dataset(questions_path=QUESTIONS_PATH, answers_path=ANSWERS_PATH, output_path=OUTPUT_PATH):
    questions = {question['patient_id']: question for question in load_json(questions_path)}
    answers = {answer['patient_id']: answer for answer in load_json(answers_path)}
    dataset = []

    for patient_id, question in questions.items():
        answer = answers.get(patient_id)
        if answer is None:
            print(f'Warning: No answer for patient {patient_id}, skipping')
            continue

        hpo_list = [hpo['hpo_id'] for hpo in question['hpo_terms']]
        dis_list = [normalize_omim_id(item['omim_id']) for item in answer['answers']]
        if not dis_list:
            print(f'Warning: No disease answers for patient {patient_id}, skipping')
            continue

        dataset.append([hpo_list, dis_list])

    print(f'Conversion complete: {len(dataset)} patients out of {len(questions)} total')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(dataset, handle, indent=2)
    print(f'Saved to {output_path}')
    return dataset


if __name__ == '__main__':
    convert_ga4gh_to_dataset()
```

**Step 4: Run the conversion tests to verify they pass**

Run:

```bash
uv run pytest tests/test_ga4gh_benchmark_conversion.py -q
```

Expected: PASS.

**Step 5: Commit**

Only commit if commits are authorized in the implementation session.

```bash
git add scripts/convert_ga4gh_benchmark.py tests/test_ga4gh_benchmark_conversion.py
git commit -m "test: cover GA4GH benchmark conversion"
```

---

### Task 2: Add GA4GH full-data integrity tests

**Files:**
- Modify: `tests/test_ga4gh_benchmark_conversion.py`
- Data inputs used: `data/GA4GH/GA4GH.benchmark_patients.questions.json`, `data/GA4GH/GA4GH.benchmark_patients.answers.json`, `data/inputs/GA4GH.benchmark_patients.json`

**Step 1: Write the failing/current-data integrity test**

Append to `tests/test_ga4gh_benchmark_conversion.py`:

```python
def test_checked_in_ga4gh_full_dataset_matches_source_files():
    root = Path(__file__).resolve().parents[1]
    questions = json.loads(
        (root / 'data' / 'GA4GH' / 'GA4GH.benchmark_patients.questions.json').read_text(encoding='utf-8')
    )
    answers = json.loads(
        (root / 'data' / 'GA4GH' / 'GA4GH.benchmark_patients.answers.json').read_text(encoding='utf-8')
    )
    dataset = json.loads(
        (root / 'data' / 'inputs' / 'GA4GH.benchmark_patients.json').read_text(encoding='utf-8')
    )

    assert len(questions) == 384
    assert len(answers) == 384
    assert len(dataset) == 384
    assert all(hpo_list for hpo_list, _ in dataset)
    assert all(dis_list for _, dis_list in dataset)
    assert all(dis_code.startswith('OMIM:') for _, dis_list in dataset for dis_code in dis_list)
```

**Step 2: Run the integrity test**

Run:

```bash
uv run pytest tests/test_ga4gh_benchmark_conversion.py::test_checked_in_ga4gh_full_dataset_matches_source_files -q
```

Expected: PASS if the current converted file is already correct. If it fails because `data/inputs/GA4GH.benchmark_patients.json` is stale, continue to Step 3.

**Step 3: Regenerate the converted GA4GH dataset if needed**

Run:

```bash
uv run python scripts/convert_ga4gh_benchmark.py
```

Expected output includes:

```text
Conversion complete: 384 patients out of 384 total
Saved to .../data/inputs/GA4GH.benchmark_patients.json
```

**Step 4: Re-run the integrity test**

Run:

```bash
uv run pytest tests/test_ga4gh_benchmark_conversion.py::test_checked_in_ga4gh_full_dataset_matches_source_files -q
```

Expected: PASS.

**Step 5: Commit**

Only commit if commits are authorized in the implementation session.

```bash
git add data/GA4GH data/inputs/GA4GH.benchmark_patients.json tests/test_ga4gh_benchmark_conversion.py
git commit -m "test: verify GA4GH benchmark dataset integrity"
```

---

### Task 3: Refactor GA4GH benchmark script around testable helpers

**Files:**
- Create: `tests/test_benchmark_ga4gh.py`
- Modify: `scripts/benchmark_ga4gh.py`

**Step 1: Write failing tests for paths and summary formatting**

Create `tests/test_benchmark_ga4gh.py`:

```python
import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'benchmark_ga4gh.py'


def load_module():
    spec = importlib.util.spec_from_file_location('benchmark_ga4gh', str(SCRIPT_PATH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DummyModel:
    name = 'Ensemble'


class DummyComponent:
    def __init__(self, name):
        self.name = name


def test_benchmark_paths_are_project_relative():
    module = load_module()
    root = Path(__file__).resolve().parents[1]

    assert module.PROJECT_ROOT == root
    assert module.GA4GH_DATASET_PATH == root / 'data' / 'inputs' / 'GA4GH.benchmark_patients.json'
    assert module.SUMMARY_PATH == root / 'results' / 'ga4gh_benchmark_summary.json'


def test_build_summary_rounds_top_k_counts():
    module = load_module()
    metrics = {
        'Mic.Recall.1': 0.5,
        'Mic.Recall.3': 0.7209,
        'Mic.Recall.5': 0.7674,
        'Mic.RankMedian': 4.0,
    }

    summary = module.build_summary(
        model=DummyModel(),
        component_models=[DummyComponent('ICTODQAcross-Ave-Random'), DummyComponent('HPOProbMNB-Random')],
        dataset_size=43,
        metrics=metrics,
    )

    assert summary['model'] == 'Ensemble'
    assert summary['dataset'] == 'GA4GH'
    assert summary['num_patients'] == 43
    assert summary['component_models'] == ['ICTODQAcross-Ave-Random', 'HPOProbMNB-Random']
    assert summary['top_k_summary']['top3'] == {'count': 31, 'total': 43, 'recall': 0.7209}
    assert summary['metrics'] == metrics
```

**Step 2: Run the benchmark helper tests to verify they fail**

Run:

```bash
uv run pytest tests/test_benchmark_ga4gh.py -q
```

Expected: FAIL because `SUMMARY_PATH` and `build_summary()` do not exist yet, and importing the script may still trigger heavy top-level imports.

**Step 3: Refactor `scripts/benchmark_ga4gh.py` minimally**

Replace `scripts/benchmark_ga4gh.py` with:

```python
#!/usr/bin/env python3
"""Run the full GA4GH benchmark with the same metrics used for the MME benchmark."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = PROJECT_ROOT / 'timgroup_disease_diagnosis' / 'codes' / 'core'
SCRIPT_DIR = CORE_DIR / 'core' / 'script'
DATASET_NAME = 'GA4GH'
GA4GH_DATASET_PATH = PROJECT_ROOT / 'data' / 'inputs' / 'GA4GH.benchmark_patients.json'
SUMMARY_PATH = PROJECT_ROOT / 'results' / 'ga4gh_benchmark_summary.json'
TOP_K_LIST = (1, 3, 5, 10, 30)


def configure_import_paths():
    for path in (CORE_DIR, SCRIPT_DIR):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def build_benchmark_model():
    configure_import_paths()
    from example_predict_ensemble import _build_outer_ensemble, build_available_models, describe_available_models

    print('Available models:')
    print(describe_available_models())
    print()

    component_models = build_available_models()
    if not component_models:
        raise RuntimeError('No diagnosis models are available for GA4GH benchmark.')
    if len(component_models) == 1:
        return component_models[0], component_models
    return _build_outer_ensemble(component_models), component_models


def build_testor(dataset_path=GA4GH_DATASET_PATH):
    configure_import_paths()
    from core.predict.model_testor import ModelTestor
    from core.reader import HPOIntegratedDatasetReader
    from core.utils.constant import CUSTOM_DATA

    hpo_reader = HPOIntegratedDatasetReader(
        keep_dnames=['OMIM', 'ORPHA', 'CCRD'],
        rm_no_use_hpo=False,
    )
    testor = ModelTestor(eval_data=CUSTOM_DATA, hpo_reader=hpo_reader)
    testor.set_custom_data_set(name_to_path={DATASET_NAME: str(dataset_path)}, data_names=[DATASET_NAME])
    testor.load_test_data(data_names=[DATASET_NAME])
    return testor


def top_k_count(recall, total):
    return int(round(recall * total))


def build_summary(model, component_models, dataset_size, metrics):
    return {
        'model': model.name,
        'dataset': DATASET_NAME,
        'num_patients': dataset_size,
        'component_models': [model.name for model in component_models],
        'metrics': metrics,
        'top_k_summary': {
            f'top{k}': {
                'count': top_k_count(metrics[f'Mic.Recall.{k}'], dataset_size),
                'total': dataset_size,
                'recall': metrics[f'Mic.Recall.{k}'],
            }
            for k in TOP_K_LIST
            if f'Mic.Recall.{k}' in metrics
        },
    }


def print_summary(summary, results_path):
    print('\n' + '=' * 60)
    print(f"Benchmark Results for {summary['dataset']} ({summary['model']})")
    print('=' * 60)
    for k in TOP_K_LIST:
        item = summary['top_k_summary'].get(f'top{k}')
        if item:
            print(f"top{k}: {item['count']}/{item['total']} ({item['recall']:.4f})")
    rank_median = summary['metrics'].get('Mic.RankMedian')
    if rank_median is not None:
        print(f'\nMedian Rank: {rank_median:.2f}')
    print('\nFull results saved to:')
    print(f'  {results_path}')


def write_summary(summary, summary_path=SUMMARY_PATH):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
    return summary_path


def run_benchmark():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    model, component_models = build_benchmark_model()
    print(f'Running benchmark with model: {model.name}')
    print(f'Number of component models: {len(component_models)}')
    print()

    testor = build_testor()
    logger.info('Starting benchmark calculation...')
    metric_dict = testor.cal_metric_and_save(
        model,
        data_names=[DATASET_NAME],
        cpu_use=min(8, os.cpu_count() or 4),
        use_query_many=False,
        save_raw_results=True,
        logger=logger,
    )
    metrics = metric_dict[DATASET_NAME]
    summary = build_summary(model, component_models, testor.get_dataset_size(DATASET_NAME), metrics)
    print_summary(summary, testor.RESULT_PATH)
    summary_path = write_summary(summary)
    print(f'\nSummary saved to: {summary_path}')
    return summary


def main():
    run_benchmark()


if __name__ == '__main__':
    main()
```

**Step 4: Run the benchmark helper tests**

Run:

```bash
uv run pytest tests/test_benchmark_ga4gh.py -q
```

Expected: PASS.

**Step 5: Commit**

Only commit if commits are authorized in the implementation session.

```bash
git add scripts/benchmark_ga4gh.py tests/test_benchmark_ga4gh.py
git commit -m "test: cover GA4GH benchmark summary helpers"
```

---

### Task 4: Make the benchmark shell wrapper portable

**Files:**
- Modify: `scripts/run_benchmark.sh`

**Step 1: Update the shell wrapper**

Replace `scripts/run_benchmark.sh` with:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

cd "$PROJECT_ROOT"
uv run python scripts/benchmark_ga4gh.py 2>&1 | tee scripts/benchmark_output.log
```

**Step 2: Validate the wrapper syntax**

Run:

```bash
bash -n scripts/run_benchmark.sh
```

Expected: no output and exit code 0.

**Step 3: Commit**

Only commit if commits are authorized in the implementation session.

```bash
git add scripts/run_benchmark.sh
git commit -m "chore: make GA4GH benchmark wrapper portable"
```

---

### Task 5: Run unit tests and regenerate GA4GH converted input

**Files:**
- May modify: `data/inputs/GA4GH.benchmark_patients.json`

**Step 1: Run focused tests**

Run:

```bash
uv run pytest tests/test_ga4gh_benchmark_conversion.py tests/test_benchmark_ga4gh.py -q
```

Expected: PASS.

**Step 2: Regenerate converted GA4GH data**

Run:

```bash
uv run python scripts/convert_ga4gh_benchmark.py
```

Expected output includes:

```text
Conversion complete: 384 patients out of 384 total
Saved to .../data/inputs/GA4GH.benchmark_patients.json
```

**Step 3: Re-run focused tests after regeneration**

Run:

```bash
uv run pytest tests/test_ga4gh_benchmark_conversion.py tests/test_benchmark_ga4gh.py -q
```

Expected: PASS.

**Step 4: Commit**

Only commit if commits are authorized in the implementation session.

```bash
git add data/inputs/GA4GH.benchmark_patients.json
git commit -m "data: refresh GA4GH benchmark input"
```

If the regenerated file is byte-for-byte unchanged, do not create this commit.

---

### Task 6: Run the full GA4GH benchmark

**Files:**
- Generated/modified: `results/ga4gh_benchmark_summary.json`
- Generated/modified: `scripts/benchmark_output.log`
- Generated under result root: raw `ModelTestor` metric/raw-result files

**Step 1: Run the full benchmark**

Run:

```bash
bash scripts/run_benchmark.sh
```

Expected:

- The script prints available models.
- The script prints `Running benchmark with model: Ensemble` when more than one component model is available.
- The script prints top1/top3/top5/top10/top30 results.
- `results/ga4gh_benchmark_summary.json` exists.

**Step 2: Inspect the summary JSON**

Run:

```bash
uv run python - <<'PY'
import json
from pathlib import Path
summary = json.loads(Path('results/ga4gh_benchmark_summary.json').read_text(encoding='utf-8'))
print(summary['model'])
print(summary['num_patients'])
print(summary['component_models'])
for key in ['top1', 'top3', 'top5', 'top10', 'top30']:
    print(key, summary['top_k_summary'][key])
print('Mic.RankMedian', summary['metrics'].get('Mic.RankMedian'))
PY
```

Expected:

- `num_patients` is `384`.
- All five top-k keys are present.
- Component model list matches the models actually available in this environment.

**Step 3: Commit generated benchmark artifacts if desired**

Only commit if benchmark result artifacts are meant to be versioned and commits are authorized. If results are large or noisy, leave them uncommitted and use only the summary values for README.

```bash
git add results/ga4gh_benchmark_summary.json scripts/benchmark_output.log
git commit -m "data: add GA4GH benchmark results"
```

---

### Task 7: Document GA4GH benchmark results in README

**Files:**
- Modify: `README.md`

**Step 1: Read the benchmark summary values**

Use the command from Task 6 Step 2 and copy the exact counts/recalls.

**Step 2: Add README section near `### MME 回归基准`**

Insert after the MME table:

```markdown
### GA4GH 全量基准

当前 GA4GH 全量评估集包含 `384` 个有答案病例。运行命令：

```bash
bash scripts/run_benchmark.sh
```

| Model | top1 | top3 | top5 | top10 | top30 |
|---|---:|---:|---:|---:|---:|
| Ensemble(<component models from summary>) | <top1 count>/384 (<top1 recall>) | <top3 count>/384 (<top3 recall>) | <top5 count>/384 (<top5 recall>) | <top10 count>/384 (<top10 recall>) | <top30 count>/384 (<top30 recall>) |
```

Replace placeholders with values from `results/ga4gh_benchmark_summary.json`. Use four decimal places for recall values to match the MME table.

**Step 3: Verify README metrics match summary JSON**

Run:

```bash
uv run python - <<'PY'
import json
import re
from pathlib import Path
summary = json.loads(Path('results/ga4gh_benchmark_summary.json').read_text(encoding='utf-8'))
readme = Path('README.md').read_text(encoding='utf-8')
for key in ['top1', 'top3', 'top5', 'top10', 'top30']:
    item = summary['top_k_summary'][key]
    expected = f"{item['count']}/{item['total']} ({item['recall']:.4f})"
    assert expected in readme, expected
assert '`384`' in readme
print('README GA4GH metrics match summary')
PY
```

Expected output:

```text
README GA4GH metrics match summary
```

**Step 4: Commit**

Only commit if commits are authorized in the implementation session.

```bash
git add README.md
git commit -m "docs: add GA4GH full benchmark results"
```

---

### Task 8: Final verification

**Files:**
- No new files expected beyond previous tasks.

**Step 1: Run all focused project tests**

Run:

```bash
PYTHONPATH=timgroup_disease_diagnosis/codes/core uv run pytest \
  timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py \
  timgroup_disease_diagnosis/codes/core/tests/test_cycommon_fallback.py \
  timgroup_disease_diagnosis/codes/core/tests/test_python312_compat.py \
  timgroup_disease_diagnosis/codes/core/tests/test_boqa_model.py \
  timgroup_disease_diagnosis/codes/core/tests/test_hpo_reader_paths.py \
  timgroup_disease_diagnosis/codes/core/tests/test_order_statistic_multi_model.py \
  tests/test_pruned_runtime_layout.py \
  tests/test_jupyter_setup.py \
  tests/test_ga4gh_benchmark_conversion.py \
  tests/test_benchmark_ga4gh.py \
  -q
```

Expected: PASS.

**Step 2: Check git status**

Run:

```bash
git status --short
```

Expected: only intended files are modified/untracked.

**Step 3: Use verification skill before claiming completion**

Before reporting the work as complete, use @verification-before-completion and include:

- focused pytest result
- full GA4GH benchmark command result
- path to `results/ga4gh_benchmark_summary.json`
- README metric consistency check result
