# Benchmark CLI Design

## Goal

Add a general benchmark CLI that can run PhenoBrain evaluation across selectable models, ensemble presets, datasets, and top-k metrics. With no arguments it should run all available single models plus the default `HPOP-ICT-CNB-NN` ensemble across all benchmark datasets and report all requested top-k accuracy metrics.

## Recommended approach

Create a new `scripts/benchmark.py` entrypoint instead of renaming or overloading `scripts/benchmark_ga4gh.py`. The existing GA4GH script remains as a narrow reproducibility shortcut, while the new CLI becomes the reusable interface for MME, GA4GH, model selection, ensemble selection, and metric selection.

## CLI behavior

- `--model`: comma-separated single-model names. Omitted means all available single models from `build_available_models()`.
- `--ensemble`: comma-separated ensemble presets. Omitted means `HPOP-ICT-CNB-NN`.
- `--dataset`: comma-separated datasets. Omitted means all supported datasets: `MME,GA4GH`.
- `--metrics`: comma-separated top-k metrics such as `top1,top3,top5,top10,top30`. Omitted means all supported top-k metrics.
- `--list-models`: print available single models and ensemble presets without running benchmarks.

The default ensemble `HPOP-ICT-CNB-NN` maps to these component models in this order: `HPOProbMNB-Random`, `ICTODQAcross-Ave-Random`, `CNB-Random`, `NN-Mixup-Random-1`. It uses the same `OrderStatisticMultiModel` outer fusion already used by the current benchmark flow.

## Data flow

The CLI configures import paths, builds all available models once, resolves requested single models and ensembles, then builds one `ModelTestor` per dataset using `CUSTOM_DATA`. Benchmark datasets are loaded from repository-relative questions and answers files, converted in memory to `[[hpo_list, disease_list], ...]`, and assigned directly to `ModelTestor.data`:

- `MME`: `data/inputs/MME.benchmark_patients.questions.json` + `data/inputs/MME.benchmark_patients.answers.json`
- `GA4GH`: `data/GA4GH/GA4GH.benchmark_patients.questions.json` + `data/GA4GH/GA4GH.benchmark_patients.answers.json`

Empty-answer cases are skipped. Disease answers are normalized from `MIM:*` to `OMIM:*`, then converted to RD codes with `source_codes_to_rd_codes`, matching the existing GA4GH benchmark behavior.

## Output

Each model/dataset run prints a compact top-k summary. The CLI writes one JSON summary to `results/benchmark_summary.json` containing selected models, selected datasets, requested metrics, dataset sizes, full metric dictionaries, and top-k count/recall values.

## Testing

Use TDD. Add tests for argument parsing, dataset/model/metric selection, ensemble preset construction, summary formatting, and benchmark orchestration with monkeypatched dummy model/testor objects. Avoid loading real model assets in unit tests.
