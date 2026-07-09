# Notebooks folder

This folder contains the research notebooks for dataset preparation, model training, evaluation, and analysis.

## Current notebooks

| Notebook | Purpose |
|---|---|
| `00_create_dataset_and_splits.ipynb` | Validates the dataset, creates processed artifacts, and writes fixed split assignments and foundation summaries. |
| `01_LOG-REG_TF-IDF.ipynb` | Logistic Regression baseline using word-level TF-IDF features. |
| `02_LIN-SVM_TF-IDF.ipynb` | Calibrated Linear SVM baseline using word-level TF-IDF features. |
| `03_SLP_TF-IDF.ipynb` | Single-Layer Perceptron baseline using word-level TF-IDF features. |
| `04_CHAR-TF-IDF_LIN-SVM.ipynb` | Calibrated Linear SVM using character-level TF-IDF features. |
| `05_WORD-CHAR-TF-IDF_LIN-SVM.ipynb` | Calibrated Linear SVM using combined word + character TF-IDF features. |
| `06_FASTTEXT-EMB_LOG-REG.ipynb` | Logistic Regression using FastText document embeddings trained from the training split. |
| `07_TWITTER-ROBERTA_FINE-TUNE.ipynb` | Fine-tuned Twitter-RoBERTa transformer with token-attribution outputs for explainability. |

## Notebook conventions

Each model notebook should:

* State the technique name near the top.
* Load the fixed split assignments from `splits/split_assignments.csv`.
* Avoid using the test split for model or threshold selection.
* Select thresholds using validation data only.
* Save validation and test metrics under `results_summary/<TECHNIQUE>/`.
* Save the best configuration used for the final test run.
* Include a short interpretation of false positives, false negatives, and limitations.
* Avoid committing raw prediction files or large model artifacts unless they are intentionally tracked outside normal Git.

## Numbering convention

Notebook numbering should reflect the intended execution order.

```text
00_...  dataset and split preparation
01_...  first baseline model
02_...  second baseline model
03_...  third baseline model
04_...  character-level robustness baseline
05_...  word + character hybrid baseline
06_...  dense static embedding baseline
07_...  contextual transformer fine-tuning experiment
```

Newer notebooks use the pattern:

```text
<NUMBER>_<REPRESENTATION>_<MODEL>.ipynb
```

For transformer fine-tuning, the representation and classifier are bundled into the transformer architecture, so the file is named by the model family and training method:

```text
07_TWITTER-ROBERTA_FINE-TUNE.ipynb
```
