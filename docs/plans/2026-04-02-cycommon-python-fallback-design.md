# cycommon Python Fallback Design

## Completion Notes (2026-04-05)
- The pure Python fallback module is implemented at `timgroup_disease_diagnosis/codes/core/core/utils/cycommon.py`.
- The import-compatible fallback now supports Python 3.12 and is covered by `test_cycommon_fallback.py` plus the focused offline compatibility suite.

## Goal
Make the diagnosis code importable on Python 3.12 by replacing the Python-3.6-only `core.utils.cycommon` extension module with a pure Python fallback implementation.

## Context
The current runtime no longer fails on third-party scientific packages first. It now fails because `core.utils.cycommon` is only available as a compiled extension built for Python 3.6:
- `cycommon.cpython-36m-x86_64-linux-gnu.so`

Python 3.12 cannot load that binary, so `from core.utils.cycommon import to_rank_score` fails. Inspection shows that `cycommon.pyx` currently exposes only one function actually needed here:
- `to_rank_score(score_mat, arg_mat)`

## Design
### Compatibility strategy
Add a pure Python module at:
- `timgroup_disease_diagnosis/codes/core/core/utils/cycommon.py`

This file should implement:
- `to_rank_score(score_mat, arg_mat)`

The import sites can remain unchanged, because Python will import `cycommon.py` when the incompatible `.so` cannot be loaded.

### Implementation scope
Keep the fallback minimal and behaviorally equivalent to the Cython version:
- mutate `score_mat` in place
- use the same row-wise ranking logic
- preserve tie handling via epsilon comparison

### Testing
Add a focused compatibility test that:
1. imports `core.utils.cycommon`
2. calls `to_rank_score` on a small `score_mat` + `arg_mat`
3. verifies the matrix is rewritten as expected

## Recommendation
Use the pure Python fallback first. It is the smallest safe change, avoids fighting Python/Cython binary compatibility, and preserves the existing import path for the rest of the codebase.
