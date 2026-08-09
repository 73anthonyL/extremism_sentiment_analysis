# Splits folder

This folder contains the fixed train/validation/test split assignments for the dataset.

## Expected files

| File | Purpose |
|---|---|
| `split_assignments.csv` | Canonical split assignment file used by all model notebooks. |
| `split_assignments.PRE-REPAIR.csv` | The stale pre-2026-07-29 file, kept for provenance. Do not use it. |

## Current split version

```text
split_v1_stratified_70_15_15_seed30
```

## Split policy

All model notebooks should use the same split assignments. This makes model comparisons more reliable because each technique is evaluated on the same train, validation, and test examples.

Do not regenerate splits unless the dataset is intentionally revised or a new experimental split version is being created.

`00_create_dataset_and_splits.ipynb` sets `overwrite_existing_split: True`. Set
it to `False` before running that notebook against this directory.

## This file is a mirror, and it has been wrong before

`split_assignments.csv` mirrors the frozen split assignment; it is not the
authority for it. Until 2026-07-29 the committed mirror was a stale
pre-correction artifact that disagreed with the frozen assignment on **728
labels and 1420 split assignments**. Anyone replicating against it would have
gotten results that could not match the reported ones, with nothing obviously
wrong to point at.

Verify the mirror before trusting anything downstream:

```bash
python3 tools/protocol_check.py --all
```

The first invariant reported is `Split mirror matches the frozen assignment`. If
it fails:

```bash
python3 tools/repair_split_mirror.py
```

That tool refuses to write unless its reconstruction reproduces
`results_summary/foundation/` exactly. **Repairing the mirror is not
regenerating the split** — the assignment itself is unchanged; only the
committed copy of it is corrected. Regenerating the assignment would invalidate
every result in the repository at once.

## Test-set rule

The test split should be used only for final locked evaluation. Hyperparameter selection and threshold selection should use training and validation data only.
