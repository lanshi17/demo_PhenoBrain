# LFS Flatten Diagnosis Directory Design

## Goal
Replace the embedded `timgroup_disease_diagnosis` git repository with a normal directory tracked directly by the parent repository, while storing all files larger than 10MB through Git LFS.

## Context
- The parent repository currently tracks `timgroup_disease_diagnosis` as a gitlink (`160000`), not a normal directory.
- The nested repository had its own `.git` directory and previously used its own history and remotes.
- The directory is large, so directly checking all contents into the parent repository without LFS would produce an impractically large normal Git history.

## Design
### Conversion strategy
1. Build a deterministic list of all files under `timgroup_disease_diagnosis` larger than 10MB.
2. Configure Git LFS tracking for those exact paths in the parent repository.
3. Remove the parent repository's gitlink entry for `timgroup_disease_diagnosis`.
4. Remove nested git metadata from `timgroup_disease_diagnosis`:
   - `.git`
   - `.worktrees`
5. Re-add `timgroup_disease_diagnosis` as a normal directory in the parent repository so large files become LFS pointers.

### Tracking rule choice
Use exact current file paths for the >10MB file set rather than broad extension globs. This keeps the conversion faithful to the "all files over 10MB" rule without unexpectedly forcing small unrelated files into LFS.

### Safety boundary
- Do not rewrite existing parent history.
- Do not mutate file contents other than replacing large files with LFS pointers in the Git index.
- Keep the working tree content intact on disk.

## Completion Notes (2026-04-06)
- The embedded `timgroup_disease_diagnosis` git repository was converted into a normal directory tracked by the parent repository.
- All current files over 10MB under `timgroup_disease_diagnosis` are now tracked through Git LFS using explicit path rules in `.gitattributes`.
- Focused verification passed with 46 tests, and spot checks confirmed staged large files are stored as LFS pointers.
