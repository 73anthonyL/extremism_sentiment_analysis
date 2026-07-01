# Results summary folder

This folder stores compact result artifacts for the research notebooks.

The goal is to make the repository reviewable without requiring readers to re-run every notebook before understanding the current findings.

## Expected structure

```text
results_summary/
├── foundation/
├── LOG-REG_TF-IDF/
├── LIN-SVM_TF-IDF/
└── SLP_TF-IDF/
```

## Foundation artifacts

The `foundation/` folder stores dataset-level artifacts such as label counts, split counts, duplicate checks, removed-row summaries, and the dataset manifest.

## Model result artifacts

Each model folder should contain:

```text
ablation_results.csv
best_config.json
classification_report_test.json
confusion_matrix_test.csv
metrics_validation.json
metrics_test.json
```

See `docs/RESULTS_SCHEMA.md` for field-level expectations.

## Interpretation rule

Results in this folder are baseline research metrics. They should not be interpreted as deployment readiness or as evidence that the models can safely make automated moderation decisions.

## Update rule

When a notebook is rerun and results change, update the corresponding result folder and verify that README tables are still accurate.
