# Notebooks folder

This folder contains the research notebooks for dataset preparation, model training, evaluation, and analysis.

## Current notebooks

| Notebook | Purpose |
|---|---|
| `00_create_dataset_and_splits.ipynb` | Validates the dataset, creates processed artifacts, and writes fixed split assignments and foundation summaries. |
| `01_LOG-REG_TF-IDF.ipynb` | Logistic Regression baseline using TF-IDF features. |
| `02_LIN-SVM_TF-IDF.ipynb` | Calibrated Linear SVM baseline using TF-IDF features. |
| `03_SLP_TF-IDF.ipynb` | Single-Layer Perceptron baseline using TF-IDF features. |

## Notebook conventions

Each model notebook should:

- State the technique name near the top.
- Load the fixed split assignments from `splits/split_assignments.csv`.
- Avoid using the test split for model or threshold selection.
- Save validation and test metrics under `results_summary/<TECHNIQUE>/`.
- Save the best configuration used for the final test run.
- Include a short interpretation of false positives, false negatives, and limitations.

## Numbering convention

Notebook numbering should reflect the intended execution order.

Suggested pattern:

```text
00_...  dataset and split preparation
01_...  first baseline model
02_...  second baseline model
03_...  third baseline model
```

Additional experiments can continue the numbering sequence or use a model-family prefix if the repository grows.
