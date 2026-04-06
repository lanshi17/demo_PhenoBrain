# LFS Flatten Diagnosis Directory Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Convert `timgroup_disease_diagnosis` from an embedded git repository into a normal parent-tracked directory with Git LFS applied to every file larger than 10MB.

**Architecture:** Replace the current gitlink with a standard directory entry in the parent repository. Use an explicit >10MB file inventory to configure `.gitattributes` before re-adding the directory, so large assets are stored as LFS pointers while normal files remain standard Git objects.

**Tech Stack:** Git, Git LFS, shell utilities, existing focused pytest suite.

---

## Completion Notes (2026-04-06)

- Inventory result: 64 files under `timgroup_disease_diagnosis` exceeded 10MB and were configured for Git LFS tracking.
- The parent repository gitlink was removed and replaced with a normal directory entry.
- Spot checks confirmed staged pointer content for large files such as:
  - `timgroup_disease_diagnosis/Docker/openjdk.deb`
  - `timgroup_disease_diagnosis/codes/core/model/CNBModel/CNB.joblib`
- Verification completed with:
  - `git lfs ls-files`
  - staged pointer spot checks using `git show :<path>`
  - `PYTHONPATH=timgroup_disease_diagnosis/codes/core .venv/bin/python -m pytest timgroup_disease_diagnosis/codes/core/tests/test_example_predict_ensemble.py timgroup_disease_diagnosis/codes/core/tests/test_cycommon_fallback.py timgroup_disease_diagnosis/codes/core/tests/test_python312_compat.py timgroup_disease_diagnosis/codes/core/tests/test_boqa_model.py timgroup_disease_diagnosis/codes/core/tests/test_hpo_reader_paths.py timgroup_disease_diagnosis/codes/core/tests/test_order_statistic_multi_model.py tests/test_jupyter_setup.py -q`
- Verification result: 46 tests passed.
