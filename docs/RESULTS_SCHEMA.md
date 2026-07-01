# Results schema

This document defines the expected output structure for each experiment in `results_summary/`.

## Folder layout

Each model family should write results to:

```text
results_summary/<TECHNIQUE>/
```

Example:

```text
results_summary/LOG-REG_TF-IDF/
├── ablation_results.csv
├── best_config.json
├── classification_report_test.json
├── confusion_matrix_test.csv
├── metrics_validation.json
└── metrics_test.json
```

Some model families may include additional metadata files, such as `metadata.json`, when useful.

## Required files

### `best_config.json`

Stores the selected model configuration.

Recommended fields:

| Field | Description |
|---|---|
| `technique` | Technique name, such as `LOG-REG_TF-IDF`. |
| `model_family` | General model type. |
| `feature_family` | Feature representation. |
| `random_seed` | Random seed used where applicable. |
| `split_version` | Split version used for the run. |
| `hyperparameters` | Selected hyperparameters. |
| `threshold_strategy` | Method used to choose the decision threshold. |
| `selected_threshold` | Final threshold used for test evaluation. |

### `metrics_validation.json`

Stores metrics on the validation split used during selection.

Required or recommended fields:

- `technique`
- `split`
- `threshold`
- `support`
- `positive_support`
- `negative_support`
- `accuracy`
- `balanced_accuracy`
- `positive_precision`
- `positive_recall`
- `positive_f1`
- `macro_f1`
- `weighted_f1`
- `roc_auc`
- `pr_auc`
- `brier_score`, if probabilities are available
- `tn`, `fp`, `fn`, `tp`
- `false_positive_rate`
- `false_negative_rate`

### `metrics_test.json`

Stores final locked metrics on the held-out test split.

This file should not be generated until the model configuration and threshold have been selected.

### `confusion_matrix_test.csv`

Recommended format:

```csv
actual,predicted,count
NON_EXTREMIST,NON_EXTREMIST,250
NON_EXTREMIST,EXTREMIST,30
EXTREMIST,NON_EXTREMIST,36
EXTREMIST,EXTREMIST,134
```

Alternate matrix-style formats are acceptable if clearly documented.

### `classification_report_test.json`

Stores class-level precision, recall, F1, and support from the final test evaluation.

### `ablation_results.csv`

Stores the validation results of compared configurations.

Recommended columns:

- `technique`
- `config_id`
- `model_family`
- `feature_family`
- `hyperparameter_summary`
- `threshold`
- `validation_accuracy`
- `validation_positive_f1`
- `validation_roc_auc`
- `validation_pr_auc`
- `notes`

## Foundation folder

Dataset and split artifacts should be stored under:

```text
results_summary/foundation/
```

Expected files:

- `dataset_manifest.json`
- `label_distribution.csv`
- `split_label_distribution.csv`
- `text_length_summary.csv`
- `duplicate_text_report.csv`
- `rows_removed_summary.json`

## Reproducibility expectations

Every saved result should make clear:

- Which dataset version was used.
- Which split version was used.
- Which model family and feature representation were used.
- Whether the threshold was selected on validation data.
- Whether test data was excluded from model selection.
