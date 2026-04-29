# GA4GH Full Benchmark Design

## Goal

Add a reproducible full GA4GH benchmark flow that mirrors the existing MME regression benchmark. The flow should convert GA4GH questions and answers into the dataset format consumed by `ModelTestor`, run the same ensemble evaluation protocol, save machine-readable results, and document the final top-k metrics in the README.

## Source data

- Questions: `data/GA4GH/GA4GH.benchmark_patients.questions.json`
- Answers: `data/GA4GH/GA4GH.benchmark_patients.answers.json`
- Converted dataset: `data/inputs/GA4GH.benchmark_patients.json`

The converted dataset format is `[[hpo_list, disease_list], ...]`, matching `DataHelper.get_dataset_with_path()` and `ModelTestor(CUSTOM_DATA)`. Disease identifiers are normalized to `OMIM:*` so they align with the integrated OMIM/ORPHA/CCRD reader vocabulary.

## Benchmark execution

`scripts/benchmark_ga4gh.py` should use repository-relative paths instead of machine-specific absolute paths. It should build the same outer ensemble used for the MME benchmark via `example_predict_ensemble.py`, then run `ModelTestor.cal_metric_and_save()` with `save_raw_results=True` on the custom `GA4GH` dataset.

The benchmark should report these metrics, matching the MME README table:

- `Mic.Recall.1`
- `Mic.Recall.3`
- `Mic.Recall.5`
- `Mic.Recall.10`
- `Mic.Recall.30`
- `Mic.RankMedian` when present

The script should also persist a summary JSON under `results/ga4gh_benchmark_summary.json` containing the model name, component models, dataset size, full metric dictionary, and top-k count/recall pairs.

## README update

After the full run completes, README should gain a `GA4GH 全量基准` section near the MME benchmark section. The section should include:

- dataset size (`384` converted cases if validation confirms all source cases have answers)
- the ensemble model list actually used in the run
- a top-k table for top1/top3/top5/top10/top30
- the command used to reproduce the benchmark

## Validation

Before running the full benchmark, validate that:

- GA4GH questions count equals answers count
- converted dataset count matches source count
- no converted case has an empty HPO list
- no converted case has an empty disease answer list
- converted disease codes use `OMIM:*`

After running the benchmark, verify that README values match `results/ga4gh_benchmark_summary.json`.

## Chosen approach

Use the MME-aligned path rather than a minimal patch or a generic benchmark runner. This keeps the implementation focused on the requested GA4GH full benchmark while preserving reproducibility and comparability with the existing MME baseline.
