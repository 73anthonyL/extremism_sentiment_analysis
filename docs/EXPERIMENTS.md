# Experiment protocol

This document defines the standard protocol for experiments in this repository.

The goal is to make model comparisons meaningful by controlling the dataset, splits, threshold-selection process, metrics, and saved outputs.

## Core rule

Every model family should be evaluated under the same fixed train/validation/test split assignments unless the repository explicitly introduces a new split version.

The canonical split file is:

```text
splits/split_assignments.csv
```

## Standard workflow

1. Load `data/dataset.csv`.
2. Run or verify the processed dataset created by `00_create_dataset_and_splits.ipynb`.
3. Merge the fixed split assignments from `splits/split_assignments.csv`.
4. Train only on the training split.
5. Tune hyperparameters only using the training and validation splits.
6. Select any probability or decision threshold using validation data only.
7. Evaluate on the test split once the model configuration and threshold are locked.
8. Save all required output artifacts under `results_summary/<TECHNIQUE>/`.

## Required metrics

Each model should report at least:

- Accuracy
- Balanced accuracy
- Positive-class precision
- Positive-class recall
- Positive-class F1
- Macro F1
- Weighted F1
- ROC-AUC
- PR-AUC
- Brier score when calibrated probabilities are available
- Confusion-matrix counts: `tn`, `fp`, `fn`, `tp`
- False-positive rate
- False-negative rate

The positive class is `EXTREMIST`.

## Threshold policy

Threshold selection must be based on validation data, not test data.

Recommended threshold strategies:

- Default threshold, usually `0.50`, when appropriate.
- Validation F1 maximization.
- Validation PR-AUC-informed operating point.
- Fixed recall or precision target, if motivated by the research question.

The selected strategy and selected threshold must be saved in `metrics_validation.json`, `metrics_test.json`, or `best_config.json`.

## Naming convention

Use descriptive technique names that identify both the model and the feature representation.

Examples:

```text
LOG-REG_TF-IDF
LIN-SVM_TF-IDF
SLP_TF-IDF
LOG-REG_WORD-CHAR-TF-IDF
XGBOOST_TF-IDF
SENT-EMB_LOG-REG
HATEBERT-FEATS_LOG-REG
```

## Required output folder

Each technique should write outputs to:

```text
results_summary/<TECHNIQUE>/
```

Required files are defined in `docs/RESULTS_SCHEMA.md`.

## Current baseline techniques

Completed baseline model families:

| Technique | Model family | Feature family |
|---|---|---|
| `LOG-REG_TF-IDF` | Logistic Regression | Word-level TF-IDF |
| `LIN-SVM_TF-IDF` | Calibrated Linear SVM | Word-level TF-IDF |
| `SLP_TF-IDF` | Single-Layer Perceptron | Word-level TF-IDF |

## Planned comparison techniques

Planned internal experiments include:

- `LOG-REG_WORD-CHAR-TF-IDF`
- `XGBOOST_TF-IDF`
- `XGBOOST_WORD-CHAR-TF-IDF`
- `RAND-FOREST_TF-IDF`
- `SENT-EMB_LOG-REG`
- `SENT-EMB_XGBOOST`
- `HATEBERT-FEATS_LOG-REG`

## Reporting rule

Do not report a technique as directly comparable unless it follows the same dataset, split, and evaluation protocol.

If a model uses a different preprocessing pipeline, external training data, different split, or competition-only evaluation setup, clearly label it as a separate condition.
