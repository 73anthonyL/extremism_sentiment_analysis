# Splits folder

This folder contains the fixed train/validation/test split assignments for the dataset.

## Expected file

| File | Purpose |
|---|---|
| `split_assignments.csv` | Canonical split assignment file used by all model notebooks. |

## Current split version

```text
split_v1_stratified_70_15_15_seed30
```

## Split policy

All model notebooks should use the same split assignments. This makes model comparisons more reliable because each technique is evaluated on the same train, validation, and test examples.

Do not regenerate splits unless the dataset is intentionally revised or a new experimental split version is being created.

## Test-set rule

The test split should be used only for final locked evaluation. Hyperparameter selection and threshold selection should use training and validation data only.
