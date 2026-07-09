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
8. Save compact output artifacts under `results_summary/<TECHNIQUE>/`.

## Required metrics

Each model should report at least:

* Accuracy
* Balanced accuracy
* Positive-class precision
* Positive-class recall
* Positive-class F1
* Macro F1
* Weighted F1
* ROC-AUC
* PR-AUC
* Brier score when calibrated probabilities are available
* Confusion-matrix counts: `tn`, `fp`, `fn`, `tp`
* False-positive rate
* False-negative rate

The positive class is `EXTREMIST`.

## Threshold policy

Threshold selection must be based on validation data, not test data.

Recommended threshold strategies:

* Default threshold, usually `0.50`, when appropriate.
* Validation F1 maximization.
* Validation PR-AUC-informed operating point.
* Fixed recall or precision target, if motivated by the research question.

The selected strategy and selected threshold must be saved in `metrics_validation.json`, `metrics_test.json`, or `best_config.json`.

## Naming convention

Use descriptive technique names that identify the representation and model family.

The current repository contains a few legacy names where the model appears first, but newer experiments follow this clearer pattern:

```text
<NUMBER>_<REPRESENTATION>_<MODEL>
```

Examples:

```text
04_CHAR-TF-IDF_LIN-SVM
05_WORD-CHAR-TF-IDF_LIN-SVM
06_FASTTEXT-EMB_LOG-REG
07_TWITTER-ROBERTA_FINE-TUNE
```

For transformer fine-tuning, the representation and classifier are integrated into the same pretrained model, so the technique is named by the transformer family and fine-tuning method.

## Required output folder

Each technique should write outputs to:

```text
results_summary/<TECHNIQUE>/
```

Required and optional files are defined in `docs/RESULTS_SCHEMA.md`.

## Current controlled techniques

| Technique | Model family | Feature / representation family | Status |
|---|---|---|---|
| `01_LOG-REG_TF-IDF` | Logistic Regression | word-level TF-IDF | completed |
| `02_LIN-SVM_TF-IDF` | calibrated Linear SVM | word-level TF-IDF | completed |
| `03_SLP_TF-IDF` | Single-Layer Perceptron | word-level TF-IDF | completed |
| `04_CHAR-TF-IDF_LIN-SVM` | calibrated Linear SVM | character-level TF-IDF | completed |
| `05_WORD-CHAR-TF-IDF_LIN-SVM` | calibrated Linear SVM | combined word + character TF-IDF | completed |
| `06_FASTTEXT-EMB_LOG-REG` | Logistic Regression | FastText document embeddings | completed |
| `07_TWITTER-ROBERTA_FINE-TUNE` | fine-tuned transformer classifier | Twitter-RoBERTa contextual representations | completed |

## Current held-out test results

| Technique | Representation / model family | Accuracy | Positive F1 | Positive precision | Positive recall | ROC-AUC | PR-AUC | Threshold |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `01_LOG-REG_TF-IDF` | word TF-IDF + logistic regression | 0.8533 | 0.8024 | 0.8171 | 0.7882 | 0.9111 | 0.8881 | 0.45 |
| `02_LIN-SVM_TF-IDF` | word TF-IDF + calibrated linear SVM | 0.8556 | 0.7962 | 0.8523 | 0.7471 | 0.9037 | 0.8825 | 0.47 |
| `03_SLP_TF-IDF` | word TF-IDF + single-layer perceptron | 0.8378 | 0.7768 | 0.8089 | 0.7471 | 0.9022 | 0.8777 | 0.50 |
| `04_CHAR-TF-IDF_LIN-SVM` | character TF-IDF + calibrated linear SVM | 0.8333 | 0.7875 | 0.7596 | 0.8176 | 0.9028 | 0.8745 | 0.40 |
| `05_WORD-CHAR-TF-IDF_LIN-SVM` | combined word + character TF-IDF + calibrated linear SVM | 0.8356 | 0.7886 | 0.7667 | 0.8118 | 0.9032 | 0.8818 | 0.42 |
| `06_FASTTEXT-EMB_LOG-REG` | FastText document embeddings + logistic regression | 0.8178 | 0.7722 | 0.7316 | 0.8176 | 0.9033 | 0.8763 | 0.47 |
| `07_TWITTER-ROBERTA_FINE-TUNE` | Twitter-RoBERTa transformer fine-tuning | 0.8889 | 0.8555 | 0.8409 | 0.8706 | 0.9496 | 0.9233 | 0.72 |


## Reporting rule

Do not report a technique as directly comparable unless it follows the same dataset, split, and evaluation protocol.

If a model uses a different preprocessing pipeline, external pretraining, different split, or competition-only evaluation setup, clearly label it as a separate condition. The Twitter-RoBERTa notebook uses transfer learning from a pretrained transformer, so it is comparable under the same split protocol but should be described as a contextual pretrained-model condition rather than a from-scratch classical baseline.
