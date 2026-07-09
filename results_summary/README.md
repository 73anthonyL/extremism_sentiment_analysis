# Results summary folder

This folder stores compact result artifacts for the research notebooks.

The goal is to make the repository reviewable without requiring readers to re-run every notebook before understanding the current findings.

## Expected structure

```text
results_summary/
├── foundation/
├── 01_LOG-REG_TF-IDF/
├── 02_LIN-SVM_TF-IDF/
├── 03_SLP_TF-IDF/
├── 04_CHAR-TF-IDF_LIN-SVM/
├── 05_WORD-CHAR-TF-IDF_LIN-SVM/
├── 06_FASTTEXT-EMB_LOG-REG/
└── 07_TWITTER-ROBERTA_FINE-TUNE/
```

## Foundation artifacts

The `foundation/` folder stores dataset-level artifacts such as label counts, split counts, duplicate checks, removed-row summaries, and the dataset manifest.

## Model result artifacts

Most classical and embedding model folders should contain:

```text
ablation_results.csv
best_config.json
classification_report_test.json
confusion_matrix_test.csv
metrics_validation.json
metrics_test.json
```

The RoBERTa folder currently uses a compact transformer-specific summary:

```text
ablation_results.csv
best_config.json
confusion_matrix_test.png
metrics_validation.json
metrics_test.json
threshold_sweep_validation.csv
```

Additional transformer artifacts, raw predictions, local attribution files, and trained model weights should not be committed to normal Git unless they are intentionally sanitized or stored through Git LFS/releases/external storage.

## Current held-out test results

The positive class is `EXTREMIST`.

| Technique | Representation / model family | Accuracy | Positive F1 | Positive precision | Positive recall | ROC-AUC | PR-AUC | Threshold |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `01_LOG-REG_TF-IDF` | word TF-IDF + logistic regression | 0.8533 | 0.8024 | 0.8171 | 0.7882 | 0.9111 | 0.8881 | 0.45 |
| `02_LIN-SVM_TF-IDF` | word TF-IDF + calibrated linear SVM | 0.8556 | 0.7962 | 0.8523 | 0.7471 | 0.9037 | 0.8825 | 0.47 |
| `03_SLP_TF-IDF` | word TF-IDF + single-layer perceptron | 0.8378 | 0.7768 | 0.8089 | 0.7471 | 0.9022 | 0.8777 | 0.50 |
| `04_CHAR-TF-IDF_LIN-SVM` | character TF-IDF + calibrated linear SVM | 0.8333 | 0.7875 | 0.7596 | 0.8176 | 0.9028 | 0.8745 | 0.40 |
| `05_WORD-CHAR-TF-IDF_LIN-SVM` | combined word + character TF-IDF + calibrated linear SVM | 0.8356 | 0.7886 | 0.7667 | 0.8118 | 0.9032 | 0.8818 | 0.42 |
| `06_FASTTEXT-EMB_LOG-REG` | FastText document embeddings + logistic regression | 0.8178 | 0.7722 | 0.7316 | 0.8176 | 0.9033 | 0.8763 | 0.47 |
| `07_TWITTER-ROBERTA_FINE-TUNE` | Twitter-RoBERTa transformer fine-tuning | 0.8889 | 0.8555 | 0.8409 | 0.8706 | 0.9496 | 0.9233 | 0.72 |


## Interpretation rule

Results in this folder are baseline research metrics. They should not be interpreted as deployment readiness or as evidence that the models can safely make automated moderation decisions.

## Update rule

When a notebook is rerun and results change, update the corresponding result folder and verify that README tables are still accurate.
