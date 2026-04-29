# Benchmark CLI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `scripts/benchmark.py`, a general benchmark CLI for selecting models, ensemble presets, datasets, and top-k accuracy metrics.

**Architecture:** Keep the CLI script-based and self-contained. It will reuse `example_predict_ensemble.py` for available model construction and outer ensemble construction, convert MME/GA4GH question+answer files into benchmark patients in memory, run `ModelTestor.cal_metric_and_save()`, print compact top-k summaries, and write `results/benchmark_summary.json`.

**Tech Stack:** Python 3.12, argparse, JSON, pytest, `ModelTestor`, `HPOIntegratedDatasetReader`, `OrderStatisticMultiModel` via `example_predict_ensemble.py`.

---

### Task 1: Add CLI parsing and selection constants

**Files:**
- Create: `tests/test_benchmark_cli.py`
- Create: `scripts/benchmark.py`

**Step 1: Write failing parser tests**

Create `tests/test_benchmark_cli.py` with parser coverage for default datasets, metrics, and ensemble; comma-separated overrides; metric key conversion; and unsupported metrics.

**Step 2: Run parser tests to verify RED**

Run `uv run pytest tests/test_benchmark_cli.py -q`.

Expected: FAIL because `scripts/benchmark.py` does not exist.

**Step 3: Implement minimal parser**

Create `scripts/benchmark.py` with repository-relative constants, `HPOP-ICT-CNB-NN` ensemble preset, MME/GA4GH dataset specs, CSV parsing, metric key conversion, and argparse handling for `--model`, `--ensemble`, `--dataset`, `--metrics`, and `--list-models`.

**Step 4: Run parser tests to verify GREEN**

Run `uv run pytest tests/test_benchmark_cli.py -q`.

Expected: PASS.

---

### Task 2: Add model and ensemble selection

**Files:**
- Modify: `tests/test_benchmark_cli.py`
- Modify: `scripts/benchmark.py`

**Step 1: Write failing selection tests**

Add tests for default single-model selection, requested model ordering, missing model errors, and `HPOP-ICT-CNB-NN` component ordering.

**Step 2: Run selection tests to verify RED**

Run `uv run pytest tests/test_benchmark_cli.py -q`.

Expected: FAIL because selection helpers are missing.

**Step 3: Implement selection helpers**

Add import-path configuration, available model construction via `example_predict_ensemble.py`, outer ensemble construction, single-model selection, component selection, and ensemble preset construction.

**Step 4: Run selection tests to verify GREEN**

Run `uv run pytest tests/test_benchmark_cli.py -q`.

Expected: PASS.

---

### Task 3: Add benchmark dataset loading

**Files:**
- Modify: `tests/test_benchmark_cli.py`
- Modify: `scripts/benchmark.py`

**Step 1: Write failing dataset tests**

Add tests that convert questions plus answers to `[[hpo_list, disease_list], ...]`, skip empty-answer cases, normalize `MIM:*` to `OMIM:*`, and preserve requested dataset order.

**Step 2: Run dataset tests to verify RED**

Run `uv run pytest tests/test_benchmark_cli.py -q`.

Expected: FAIL because dataset helpers are missing.

**Step 3: Implement dataset helpers**

Add JSON loading, OMIM normalization, benchmark dataset conversion, and dataset spec resolution.

**Step 4: Run dataset tests to verify GREEN**

Run `uv run pytest tests/test_benchmark_cli.py -q`.

Expected: PASS.

---

### Task 4: Add testor construction and benchmark summary helpers

**Files:**
- Modify: `tests/test_benchmark_cli.py`
- Modify: `scripts/benchmark.py`

**Step 1: Write failing summary tests**

Add tests for top-k count summaries and printed compact output.

**Step 2: Run summary tests to verify RED**

Run `uv run pytest tests/test_benchmark_cli.py -q`.

Expected: FAIL because summary helpers are missing.

**Step 3: Implement testor and summary helpers**

Add RD-code conversion, `ModelTestor` construction, top-k count calculation, per-run summary construction, and compact summary printing.

**Step 4: Run summary tests to verify GREEN**

Run `uv run pytest tests/test_benchmark_cli.py -q`.

Expected: PASS.

---

### Task 5: Add benchmark orchestration and CLI main

**Files:**
- Modify: `tests/test_benchmark_cli.py`
- Modify: `scripts/benchmark.py`

**Step 1: Write failing orchestration tests**

Add tests that monkeypatch model/testor construction and verify each selected model/dataset pair runs with `cpu_use=1`, `use_query_many=False`, `save_raw_results=True`, plus a JSON summary writer test.

**Step 2: Run orchestration tests to verify RED**

Run `uv run pytest tests/test_benchmark_cli.py -q`.

Expected: FAIL because orchestration helpers are missing.

**Step 3: Implement orchestration and main**

Add `write_summary`, `run_benchmark`, `print_models`, and `main`.

**Step 4: Run orchestration tests to verify GREEN**

Run `uv run pytest tests/test_benchmark_cli.py -q`.

Expected: PASS.

---

### Task 6: Add documentation and compatibility check

**Files:**
- Modify: `README.md`

**Step 1: Update docs after CLI tests pass**

Add a `通用 benchmark CLI` section with default and filtered command examples.

**Step 2: Run focused tests**

Run `uv run pytest tests/test_benchmark_cli.py tests/test_benchmark_ga4gh.py timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py -q`.

Expected: PASS.

**Step 3: Smoke-test CLI listing**

Run `uv run python scripts/benchmark.py --list-models`.

Expected: prints available single models and `HPOP-ICT-CNB-NN` without running benchmarks.

**Step 4: Optional narrow runtime smoke test**

If model assets are available locally and runtime is acceptable, run `uv run python scripts/benchmark.py --dataset MME --model MICAModel --ensemble none --metrics top1`.

Expected: completes one MME model/dataset run and writes `results/benchmark_summary.json`.
