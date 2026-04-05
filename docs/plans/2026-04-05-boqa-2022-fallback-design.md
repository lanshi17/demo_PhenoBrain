# BOQA 2022 Fallback Design

## Completion Notes (2026-04-05)
- The design is implemented through `resolve_hpo_raw_path(...)` in `hpo_reader.py` and the fallback-aware BOQA availability gate in `example_predict_ensemble.py`.
- `BOQAModel` now becomes available against the repository's existing `2022` HPO raw assets and participates in the default script run.

## Goal
Make `BOQAModel` usable with the HPO raw assets that already exist in this repository by falling back from the legacy `2019` paths to the available `2022` paths.

## Context
- `BOQAModel` already has a working `boqa.jar` in `codes/core/core/predict/prob_model/boqa-master/out/artifacts/boqa_jar/boqa.jar`.
- The repository currently contains:
  - `codes/core/data/raw/HPO/2022/Ontology/hp.obo`
  - `codes/core/data/raw/HPO/2022/Annotations/phenotype.hpoa`
- `HPOReader` still hardcodes `2019` for both `hp.obo` and `phenotype.hpoa`.
- The offline example currently avoids BOQA by gating the candidate when the legacy `2019` assets are missing.

## Recommended Approach
Introduce a tiny HPO raw-path resolver that prefers the legacy `2019` files when they exist, but falls back to the repository's `2022` files when they do not.

Apply that resolver in two places:
1. `core/reader/hpo_reader.py`
   - Resolve `self.HPO_OBO_PATH`
   - Resolve `self.ANNOTATION_HPOA_PATH`
2. `core/script/example_predict_ensemble.py`
   - Update BOQA candidate availability to accept the same fallback paths so the candidate becomes available when `2022` assets are present.

## Why This Approach
- Minimal change surface: two file-path entry points, no BOQA protocol changes.
- Keeps backwards compatibility for environments that still have the original `2019` files.
- Avoids copying or duplicating raw data directories.
- Preserves the current BOQA runtime gate so script startup stays safe if required assets are absent.

## Non-Goals
- Do not rewrite BOQA Java invocation.
- Do not move or duplicate raw data files.
- Do not refactor unrelated HPO reader logic.
- Do not broaden the fallback mechanism beyond `hp.obo` and `phenotype.hpoa`.

## Testing Focus
- `HPOReader` selects `2022` assets when `2019` assets are absent.
- BOQA candidate availability returns true with `2022` assets plus `boqa.jar` and `java`.
- Existing example script regression tests still pass.
- Real script verification shows whether BOQA is now merely available or fully executable; if a new runtime blocker appears, capture that boundary explicitly.
