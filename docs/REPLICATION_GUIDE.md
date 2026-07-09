# Replication guide

This guide describes how to reproduce the core research artifacts in this repository.

## 1. Prepare the environment

Clone the repository and install dependencies.

```bash
git clone https://github.com/73anthonyL/extremism_sentiment_analysis.git
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

Transformer runs are easiest on Kaggle or another GPU environment. CPU-only runs may be slow.

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
notebooks/04_CHAR-TF-IDF_LIN-SVM.ipynb
notebooks/05_WORD-CHAR-TF-IDF_LIN-SVM.ipynb
notebooks/06_FASTTEXT-EMB_LOG-REG.ipynb
notebooks/07_TWITTER-ROBERTA_FINE-TUNE.ipynb
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
notebooks/04_CHAR-TF-IDF_LIN-SVM.ipynb
notebooks/05_WORD-CHAR-TF-IDF_LIN-SVM.ipynb
notebooks/06_FASTTEXT-EMB_LOG-REG.ipynb
notebooks/07_TWITTER-ROBERTA_FINE-TUNE.ipynb
```

Each notebook should read the fixed split assignments and write a result folder under `results_summary/`.

For `07_TWITTER-ROBERTA_FINE-TUNE.ipynb`, use a GPU runtime when possible and avoid committing trained model weights to normal Git.

## 5. Check comparability

A replicated result is comparable only if:

* The same dataset version is used.
* The same split assignments are used.
* Hyperparameters are selected without using test labels.
* Thresholds are selected using validation data only.
* The same metric definitions are used.
* The reported test metrics come from the held-out test split.
* External pretraining or transfer learning is disclosed as part of the experimental condition.

## 6. Current ranking by held-out test PR-AUC

1. `07_TWITTER-ROBERTA_FINE-TUNE`
2. `01_LOG-REG_TF-IDF`
3. `02_LIN-SVM_TF-IDF`
4. `05_WORD-CHAR-TF-IDF_LIN-SVM`
5. `03_SLP_TF-IDF`
6. `06_FASTTEXT-EMB_LOG-REG`
7. `04_CHAR-TF-IDF_LIN-SVM`

Small numeric differences may occur across environments, especially for neural or transformer models. Any material difference should be documented.

## 7. When results do not match

Check the following first:

* Python version and dependency versions.
* Whether `requirements-lock.txt` was used.
* Whether the split file was regenerated accidentally.
* Whether threshold selection used validation data only.
* Whether labels were mapped consistently.
* Whether the model was run with the same random seed.
* Whether a GPU/non-GPU environment changed transformer reproducibility.
* Whether the pretrained transformer checkpoint changed or was unavailable.
