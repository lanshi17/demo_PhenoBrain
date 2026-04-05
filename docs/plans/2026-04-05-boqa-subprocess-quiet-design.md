# BOQA Subprocess Quiet Design

## Completion Notes (2026-04-05)
- The design is implemented through `run_boqa_command(...)` and `subprocess.run(...)` in `core/predict/prob_model/boqa_model.py`.
- The current default script still includes `BOQAModel`, but no longer prints trailing Java-side logs after the result table.

## Goal
Stop BOQA's external Java process from leaking SLF4J and Ontologizer logs into the offline example output while preserving failure visibility.

## Context
- The offline example already wraps Python-side model construction and prediction with a quiet helper.
- `BOQAModel` still uses `os.system(...)` to invoke `boqa.jar`.
- Because `os.system(...)` attaches directly to the terminal, Java logs bypass the Python quiet wrapper and appear after the result table.
- BOQA now runs successfully with the `2022` HPO fallback, so the remaining issue is output hygiene, not functional correctness.

## Recommended Approach
Replace `os.system(...)` in `core/predict/prob_model/boqa_model.py` with a small subprocess helper that:
1. builds the Java command as an argument list,
2. executes it via `subprocess.run(...)`,
3. discards `stdout` and `stderr` by default, and
4. raises a clear Python exception when the Java command exits non-zero.

## Why This Approach
- Minimal behavioral change: only the external process execution path changes.
- Keeps BOQA quiet in the common success path.
- Still surfaces failures explicitly instead of silently swallowing them.
- Avoids shell quoting issues by switching from a formatted shell string to argument lists.

## Non-Goals
- Do not change BOQA ranking behavior.
- Do not redesign multiprocessing or chunking.
- Do not change example-script formatting beyond suppressing the external Java logs.

## Testing Focus
- The Java subprocess runner is invoked with `stdout` and `stderr` redirected away from the terminal.
- A non-zero Java exit raises a meaningful error.
- Existing offline example tests still pass.
- Real script output ends at the result table with no trailing Java logs.
