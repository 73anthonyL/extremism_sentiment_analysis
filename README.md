# Social Media Extremism Detection

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-20BEFF.svg)](https://www.kaggle.com/datasets/adityasureshgithub/digital-extremism-detection-curated-dataset)
[![Kaggle Challenge](https://img.shields.io/badge/Kaggle-Competition-20BEFF.svg)](https://www.kaggle.com/competitions/social-media-extremism-detection-challenge)
[![Status](https://img.shields.io/badge/status-active%20research-orange.svg)](#project-status)

A reproducible NLP research repository for **binary classification of social media text as extremist or non-extremist**. The project includes a curated hand-labeled dataset, fixed train/validation/test splits, baseline machine-learning experiments, standardized model outputs, and interpretable feature-attribution artifacts for error analysis.

> **Important:** This repository is a research project, not a production moderation system. Model outputs should not be used to make automated decisions about people, accounts, or communities without human review, additional validation, and bias/safety auditing.

## Table of contents

- [Social Media Extremism Detection](#social-media-extremism-detection)
  - [Table of contents](#table-of-contents)
  - [Project status](#project-status)
  - [Project overview](#project-overview)
  - [Repository structure](#repository-structure)
  - [Dataset and splits](#dataset-and-splits)
  - [Current results](#current-results)
  - [Reproducible workflow](#reproducible-workflow)
  - [Installation](#installation)
  - [How to run](#how-to-run)
    - [Option A: Kaggle notebooks](#option-a-kaggle-notebooks)
    - [Option B: Local execution](#option-b-local-execution)
  - [Experiment protocol](#experiment-protocol)
  - [Interpretability](#interpretability)
  - [Roadmap](#roadmap)
  - [Responsible use](#responsible-use)
  - [Citation](#citation)
  - [Maintainers](#maintainers)

## Project status

This repository is under active research development. The current version contains the dataset foundation plus three baseline model families. Results, paper language, and documentation should be treated as evolving.

## Project overview

Violent extremist text can be difficult to distinguish from general toxicity or offensive language because it may rely on coded phrasing, ideological framing, or calls for violence that are context-dependent. This repository treats extremist-content detection as a distinct NLP classification problem and focuses on building a transparent, reproducible baseline suite.

This project currently emphasizes:

- **Dataset curation:** a manually labeled extremist / non-extremist social-media text dataset.
- **Reproducible evaluation:** fixed row-level split assignments shared by all model notebooks.
- **Validation discipline:** hyperparameters and thresholds are selected on validation data only.
- **Held-out testing:** final metrics are reported once on the locked test split.
- **Interpretability:** model-level and local feature-attribution outputs for inspecting errors.

Related project pages:

- Kaggle dataset: <https://www.kaggle.com/datasets/adityasureshgithub/digital-extremism-detection-curated-dataset>
- Kaggle competition: <https://www.kaggle.com/competitions/social-media-extremism-detection-challenge>

## Repository structure

```text
extremism_sentiment_analysis/
├── data/
│   ├── dataset.csv
│   └── extremism_lexicon.txt
├── notebooks/
│   ├── 00_create_dataset_and_splits.ipynb
│   ├── 01_LOG-REG_TF-IDF.ipynb
│   ├── 02_LIN-SVM_TF-IDF.ipynb
│   └── 03_SLP_TF-IDF.ipynb
├── results_summary/
│   ├── foundation/
│   ├── LOG-REG_TF-IDF/
│   ├── LIN-SVM_TF-IDF/
│   └── SLP_TF-IDF/
├── splits/
│   └── split_assignments.csv
├── README.md
└── requirements.txt
```

Recommended supplemental files for a more professional research repository are included in `docs/` and `.github/`:

```text
docs/
├── DATA_CARD.md
├── MODEL_CARD.md
├── EXPERIMENTS.md
├── RESPONSIBLE_USE.md
├── REPOSITORY_CLEANUP_CHECKLIST.md
└── RESULTS_SCHEMA.md
.github/
├── pull_request_template.md
└── ISSUE_TEMPLATE/
    ├── bug_report.md
    └── experiment_proposal.md
```

## Dataset and splits

The current processed dataset is created from `data/dataset.csv` using `notebooks/00_create_dataset_and_splits.ipynb`.

| Item | Value |
|---|---:|
| Raw rows | 3000 |
| Processed rows | 2999 |
| Rows removed | 1 |
| Duplicate text hashes | 0 |
| Non-extremist rows | 1870 (62.35%) |
| Extremist rows | 1129 (37.65%) |
| Random seed | 30 |
| Split version | `split_v1_stratified_70_15_15_seed30` |

Split counts are stratified at approximately 70/15/15:

| Split | Rows | Non-extremist | Extremist |
|---|---:|---:|---:|
| Train | 2099 | 1309 | 790 |
| Validation | 450 | 281 | 169 |
| Test | 450 | 280 | 170 |

All downstream notebooks should reuse `splits/split_assignments.csv`. Do **not** regenerate splits unless the dataset itself is being corrected and the split version is intentionally updated.

## Current results

The table below reports the locked held-out test metrics from `results_summary/`. The positive class is `EXTREMIST`.

| Technique | Accuracy | Positive F1 | Positive precision | Positive recall | ROC-AUC | PR-AUC | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| `LOG-REG_TF-IDF` | 0.8533 | 0.8024 | 0.8171 | 0.7882 | 0.9111 | 0.8881 | 0.45 |
| `LIN-SVM_TF-IDF` | 0.8556 | 0.7962 | 0.8523 | 0.7471 | 0.9037 | 0.8825 | 0.47000000000000003 |
| `SLP_TF-IDF` | 0.8378 | 0.7768 | 0.8089 | 0.7471 | 0.9022 | 0.8777 | 0.5 |

Confusion-matrix summary:

| Technique | TN | FP | FN | TP | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|
| `LOG-REG_TF-IDF` | 250 | 30 | 36 | 134 | 0.1071 | 0.2118 |
| `LIN-SVM_TF-IDF` | 258 | 22 | 43 | 127 | 0.0786 | 0.2529 |
| `SLP_TF-IDF` | 250 | 30 | 43 | 127 | 0.1071 | 0.2529 |

Current ranking by held-out test PR-AUC:

1. `LOG-REG_TF-IDF` — PR-AUC 0.8881, ROC-AUC 0.9111, positive F1 0.8024
2. `LIN-SVM_TF-IDF` — PR-AUC 0.8825, ROC-AUC 0.9037, positive F1 0.7962
3. `SLP_TF-IDF` — PR-AUC 0.8777, ROC-AUC 0.9022, positive F1 0.7768

These results should be interpreted as **baseline research metrics**, not deployment evidence. Additional validation is needed before any real-world use.

## Reproducible workflow

The project is organized as a sequence of notebooks:

| Notebook | Purpose |
|---|---|
| `00_create_dataset_and_splits.ipynb` | Cleans/validates the raw dataset, creates canonical row IDs, creates stratified train/validation/test splits, and writes the dataset manifest. |
| `01_LOG-REG_TF-IDF.ipynb` | Trains and evaluates Logistic Regression with TF-IDF features. |
| `02_LIN-SVM_TF-IDF.ipynb` | Trains and evaluates calibrated Linear SVM with TF-IDF features. |
| `03_SLP_TF-IDF.ipynb` | Trains and evaluates a single-layer perceptron over TF-IDF features. |

Each model notebook should output the same core artifact types:

```text
results_summary/<TECHNIQUE>/
├── ablation_results.csv
├── best_config.json
├── classification_report_test.json
├── confusion_matrix_test.csv
├── metrics_validation.json
└── metrics_test.json
```

Full experiment outputs may also include model files, prediction CSVs, plots, interpretability artifacts, and error-analysis files.

## Installation

Clone the repository:

```bash
git clone https://github.com/asuresh952/extremism_sentiment_analysis.git
cd extremism_sentiment_analysis
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional Jupyter kernel setup:

```bash
python -m ipykernel install --user --name extremism-research --display-name "Python (extremism research)"
```

## How to run

### Option A: Kaggle notebooks

1. Upload or attach the Kaggle dataset to the notebook environment.
2. Run `00_create_dataset_and_splits.ipynb` first.
3. Save the resulting `processed_dataset.csv`, `split_assignments.csv`, and `dataset_manifest.json`.
4. Run each model notebook using those fixed artifacts.
5. Commit only the summarized outputs needed for review, not large generated model artifacts.

### Option B: Local execution

1. Place `dataset.csv` under `data/`.
2. Run the foundation notebook to regenerate the canonical processed dataset and split files.
3. Confirm that `dataset_manifest.json` matches the intended dataset version.
4. Run model notebooks in numeric order.

## Experiment protocol

To keep the comparison research-grade:

- Use the same processed dataset for every technique.
- Use the same `split_assignments.csv` for every technique.
- Tune hyperparameters on train/validation only.
- Select decision thresholds on validation only.
- Evaluate on the test split only after the model and threshold are locked.
- Record every experiment in a standardized folder named after the technique.
- Save `best_config.json`, `metrics_validation.json`, `metrics_test.json`, and `predictions_test.csv` for every model.
- Do not compare models using different preprocessing unless the preprocessing itself is the experimental variable.

For additional detail, see [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## Interpretability

The current repository focuses on interpretable baselines:

- Logistic Regression: coefficient-based global and local explanations.
- Linear SVM: margin-coefficient global and local explanations after probability calibration.
- Single-layer perceptron: exact linear logit contributions and background-adjusted linear attributions over TF-IDF features.

Interpretability outputs should be used for error analysis and hypothesis generation, not as proof that the model understands intent or ideology.

## Roadmap

Planned comparison extensions:

- `LOG-REG_WORD-CHAR-TF-IDF`
- `XGBOOST_TF-IDF`
- `XGBOOST_WORD-CHAR-TF-IDF`
- `RAND-FOREST_TF-IDF`
- `SENT-EMB_LOG-REG`
- `SENT-EMB_XGBOOST`
- `HATEBERT-FEATS_LOG-REG`

Repository-quality improvements:

- Add a license file.
- Add a formal citation file.
- Add dataset/model cards.
- Add a results table that is regenerated automatically from JSON files.
- Move reusable code from notebooks into a small `src/` package.
- Add unit tests for split integrity and metric calculation.
- Add a GitHub Actions check for notebook execution or at least notebook linting.

## Responsible use

This project involves sensitive social-media content and high-stakes classification labels. False positives can unfairly characterize benign speech, while false negatives can miss genuinely harmful rhetoric. Any real-world use would require domain-expert review, dataset-bias evaluation, subgroup analysis, privacy review, and a human decision process.

See [`docs/RESPONSIBLE_USE.md`](docs/RESPONSIBLE_USE.md) before using or extending this repository.

## Citation

A starter `CITATION.cff` is included in this documentation package. After publication details are finalized, update the title, author list, version, DOI, and release date.

## Maintainers

- Aditya Suresh
- Anthony Lu
