# UV Minimal Runtime Repair Design

## Completion Notes (2026-04-05)
- The planned minimal runtime dependency set is now present in `pyproject.toml`, including `scipy`, `scikit-learn`, `joblib`, and `pyemd`.
- Later runtime work carried the script far beyond the original `scipy` import boundary: the current offline example executes end-to-end with BOQA enabled.

## Goal
Use `uv` to make the offline diagnosis script runnable with the smallest persistent dependency change possible.

## Context
The dynamic ensemble script now detects downloaded model assets correctly, but runtime currently fails before model construction because the environment is missing `scipy`. The current project already uses `uv` and a `.venv`, and `pyproject.toml` is intentionally lightweight for notebook/data work rather than the full legacy diagnosis stack.

## Recommended strategy
Use a staged, minimal persistent repair:

1. Add the smallest likely runtime dependency set with `uv add`:
   - `scipy`
   - `scikit-learn`
   - `joblib`
   - `pyemd`
2. Re-run the offline script.
3. Only if a new import/runtime error appears, investigate that next dependency specifically.
4. Do **not** immediately add legacy TensorFlow packages, because the current environment is Python 3.12 and the old TensorFlow 1.x stack is likely incompatible.

## Why this strategy
- It keeps changes small and durable by updating `pyproject.toml` and `.venv` through `uv`.
- It addresses the currently observed failure (`ModuleNotFoundError: scipy`) and the most likely next failures for the ML / similarity models.
- It avoids a high-risk bulk dependency install for an old codebase with known version-era mismatch.

## Expected outcome
After this repair, the next script failure—if any—should be more informative and closer to the actual model runtime boundary, such as a TensorFlow incompatibility or a model-specific missing package.
