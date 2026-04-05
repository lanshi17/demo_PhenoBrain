# CLI Parameters Design

## Completion Notes (2026-04-05)
- The CLI layer is implemented in `example_predict_ensemble.py` through `parse_hpo_list()`, `parse_args()`, and `resolve_cli_inputs()`.
- The current script supports both zero-argument default execution and explicit `--topk` / `--hpo-list` overrides, and those behaviors are covered by the focused offline test suite.

## Goal
Add simple command-line parameters for `topk` and HPO input while preserving the current default behavior and formatted output.

## Parameters
- `--topk <int>`
- `--hpo-list <comma-separated HPO codes>`

## Default behavior
If no CLI parameters are provided, the script should continue to use:
- the existing sample HPO list
- `topk=5`

## Example usage
```bash
PYTHONPATH="timgroup_disease_diagnosis/codes/core" \
./.venv/bin/python \
timgroup_disease_diagnosis/codes/core/core/script/example_predict_ensemble.py \
  --topk 10 \
  --hpo-list HP:0001913,HP:0008513,HP:0001123
```

## Implementation direction
Use a minimal `argparse` setup in `example_predict_ensemble.py`:
- parse optional `--topk`
- parse optional `--hpo-list`
- split comma-separated HPO values into a list
- fall back to the existing sample list when absent

## Output
Keep the current output shape:
1. `Available models: ...`
2. aligned results table

## Why this approach
- smallest CLI surface
- easy to copy/paste in shell
- no change to model logic or API return types
