# Result Table Formatting Design

## Completion Notes (2026-04-05)
- `format_results_table(results)` is implemented in the current offline script and preserves the existing prediction API while changing only the printed rendering.
- The default script now prints an aligned `Rank / Disease / Score` table, and that formatting is covered by the focused offline suite.

## Goal
Format the final prediction output as a readable aligned table while keeping the prediction function’s return value unchanged.

## Context
The script currently prints:
- one `Available models: ...` line
- one raw Python list of tuples for results

That raw tuple representation works technically but is hard to read at a glance.

## Design
### Keep
- `predict_ensemble(hpo_list, topk=10)` should keep returning the existing list of `(disease_code, score)` tuples.
- `Available models: ...` output stays unchanged.

### Change
Add a small formatting helper such as:
- `format_results_table(results)`

This helper should render a plain-text aligned table with columns:
- `Rank`
- `Disease`
- `Score`

### Example output
```text
Rank  Disease   Score
1     RD:7367   1.010799
2     RD:6963   1.001620
3     RD:6003   0.989525
4     RD:2684   1.008747
5     RD:7969   1.000540
```

### Implementation scope
- No change to model inference
- No change to ranking semantics
- Only change how `__main__` prints the returned results

## Why this approach
- Minimal blast radius
- Preserves programmatic API compatibility
- Makes CLI/script usage much easier to read
