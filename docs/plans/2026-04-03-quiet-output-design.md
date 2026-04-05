# Quiet Output Design

## Completion Notes (2026-04-05)
- The Python-side quiet wrapper and warning filtering are implemented in `example_predict_ensemble.py`.
- A later BOQA subprocess follow-up removed the remaining external Java log leak, so the current end-to-end output is quiet and ends at the result table.

## Goal
Make the offline diagnosis script print only essential user-facing output: the available model list and the final prediction results.

## Context
The script now works end-to-end, but runtime is noisy due to:
- warnings from NumPy / scikit-learn / internal math
- `training...` and `training end.` prints
- `tqdm` progress bars
- data/reader conflict logs like `Level Confilict`, `Dup mapping`, `into ...`

These messages obscure the result even though prediction succeeds.

## Design
### Keep
- `Available models: ...`
- final prediction results

### Suppress
- runtime warnings during script execution
- stdout/stderr emitted while constructing models and running prediction
- progress bars and low-level conflict logs that currently go to stdout/stderr

### Implementation approach
Use a script-local quiet mode in `example_predict_ensemble.py`:
1. Add a helper/context manager that temporarily redirects `stdout` and `stderr` to an in-memory buffer.
2. Add a helper that configures targeted `warnings.filterwarnings(...)` for the known non-fatal warning classes/messages.
3. Wrap model construction and prediction calls in the quiet context.
4. Keep the final user-facing `print(...)` calls outside the quiet block.

## Why this approach
- Minimal blast radius: no need to patch many library files
- Preserves model behavior
- Keeps the script usable immediately
- Easy to remove or disable later if verbose debugging is needed

## Testing
Add a focused test around a helper function rather than asserting on full script stdout. For example, verify that a helper returns model names / prediction results without exposing noisy intermediate output.
