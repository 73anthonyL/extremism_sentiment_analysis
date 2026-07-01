# Replication guide

This guide describes how to reproduce the core research artifacts in this repository.

## 1. Prepare the environment

Clone the repository and install dependencies.

```bash
git clone https://github.com/asuresh952/extremism_sentiment_analysis.git
cd extremism_sentiment_analysis
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For stricter environment replication, use:

```bash
pip install -r requirements-lock.txt
```

## 2. Verify source files

Confirm that the repository contains:

```text
data/dataset.csv
data/extremism_lexicon.txt
splits/split_assignments.csv
notebooks/00_create_dataset_and_splits.ipynb
notebooks/01_LOG-REG_TF-IDF.ipynb
notebooks/02_LIN-SVM_TF-IDF.ipynb
notebooks/03_SLP_TF-IDF.ipynb
```

## 3. Recreate the dataset foundation

Run:

```text
notebooks/00_create_dataset_and_splits.ipynb
```

This notebook should validate the source dataset, create or verify processed rows, assign canonical IDs, and write foundation artifacts.

Expected foundation artifacts:

```text
results_summary/foundation/dataset_manifest.json
results_summary/foundation/label_distribution.csv
results_summary/foundation/split_label_distribution.csv
results_summary/foundation/text_length_summary.csv
results_summary/foundation/duplicate_text_report.csv
results_summary/foundation/rows_removed_summary.json
```

## 4. Run model notebooks

Run the model notebooks in order:

```text
notebooks/01_LOG-REG_TF-IDF.ipynb
notebooks/02_LIN-SVM_TF-IDF.ipynb
notebooks/03_SLP_TF-IDF.ipynb
```

Each notebook should read the fixed split assignments and write a result folder under `results_summary/`.

## 5. Check comparability

A replicated result is comparable only if:

- The same dataset version is used.
- The same split assignments are used.
- Hyperparameters are selected without using test labels.
- The same metric definitions are used.
- The reported test metrics come from the held-out test split.

## 6. Expected baseline ranking

Current ranking by held-out test PR-AUC:

1. `LOG-REG_TF-IDF`
2. `LIN-SVM_TF-IDF`
3. `SLP_TF-IDF`

Small numeric differences may occur across environments, especially for neural models or version-sensitive libraries. Any material difference should be documented.

## 7. When results do not match

Check the following first:

- Python version and dependency versions.
- Whether `requirements-lock.txt` was used.
- Whether the split file was regenerated accidentally.
- Whether threshold selection used validation data only.
- Whether labels were mapped consistently.
- Whether the model was run with the same random seed.
