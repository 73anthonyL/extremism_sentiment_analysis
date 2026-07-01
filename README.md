# Social Media Extremism Detection

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-20BEFF.svg)](https://www.kaggle.com/datasets/adityasureshgithub/digital-extremism-detection-curated-dataset)
[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Competition-20BEFF.svg)](https://www.kaggle.com/competitions/social-media-extremism-detection-challenge)
[![Status](https://img.shields.io/badge/status-research%20repo-orange.svg)](#project-status)

This repository contains a reproducible NLP research project for **binary classification of social-media text as extremist or non-extremist**. It includes a curated dataset, fixed train/validation/test splits, baseline machine-learning experiments, standardized result summaries, and interpretability-oriented analysis artifacts.

The repository is public so that the research process can be reviewed, replicated, and evaluated. It is **not** intended to be a production moderation system, a general-purpose software package, or a community-maintained open-source project.

> **Responsible-use note:** The models in this repository should not be used to make automated decisions about people, accounts, posts, or communities. Extremism detection is a high-stakes task with serious false-positive and false-negative risks. Any applied use would require domain-expert review, bias evaluation, privacy review, and human oversight.

## Table of contents

- [Social Media Extremism Detection](#social-media-extremism-detection)
  - [Table of contents](#table-of-contents)
  - [Project status](#project-status)
  - [Research motivation](#research-motivation)
  - [Project links](#project-links)
  - [Kaggle competition](#kaggle-competition)
  - [Repository structure](#repository-structure)
  - [Dataset and fixed splits](#dataset-and-fixed-splits)
  - [Current baseline results](#current-baseline-results)
  - [Reproducibility workflow](#reproducibility-workflow)
  - [Installation](#installation)
  - [Running the experiments](#running-the-experiments)
    - [Option A: Kaggle](#option-a-kaggle)
    - [Option B: Local environment](#option-b-local-environment)
  - [Experiment protocol](#experiment-protocol)
  - [Interpretability](#interpretability)
  - [Planned internal experiments](#planned-internal-experiments)
  - [Limitations](#limitations)
  - [Repository maintenance](#repository-maintenance)
  - [Citation](#citation)
  - [Authors](#authors)

## Project status

This is an active research repository maintained by the project authors. The current version contains the dataset foundation, fixed split assignments, and three completed baseline model families:

- Logistic Regression with TF-IDF features
- Calibrated Linear SVM with TF-IDF features
- Single-Layer Perceptron with TF-IDF features

Additional model families and analysis notebooks are being developed internally. Results and paper language should be treated as evolving until the final manuscript is complete.

## Research motivation

Most online safety NLP work focuses on toxicity, hate speech, or general harmful-content classification. Violent extremist text can overlap with those categories, but it is not identical to them. It may involve ideological framing, support for violence, coded language, or contextual references that are not captured well by surface-level offensiveness alone.

This project treats extremist-content detection as a distinct text-classification problem. The research goals are to:

1. Build a transparent baseline suite for extremist versus non-extremist classification.
2. Use fixed splits and standardized metrics so model comparisons are meaningful.
3. Compare classical machine-learning, neural, embedding-based, and transformer-informed approaches under the same evaluation protocol.
4. Use interpretable feature-attribution methods to inspect what models learn and where they fail.
5. Document the dataset, methodology, and limitations clearly enough for replication.

## Project links

- **GitHub repository:** <https://github.com/asuresh952/extremism_sentiment_analysis>
- **Kaggle dataset:** <https://www.kaggle.com/datasets/adityasureshgithub/digital-extremism-detection-curated-dataset>
- **Kaggle competition:** <https://www.kaggle.com/competitions/social-media-extremism-detection-challenge>

## Kaggle competition

We also hosted the **Social Media Extremism Detection Challenge** on Kaggle as a separate public benchmark for the same binary classification task. The competition invited participants to build models that classify social-media text as `EXTREMIST` or `NON_EXTREMIST`.

In this repository, the competition is treated as an external research reference rather than as the main experimental protocol. The leaderboard and participant approaches are useful for understanding what other modeling strategies may perform well, but the results reported in this repository are based on our controlled train/validation/test split workflow.

The distinction is important:

| Source | Purpose | How it is used here |
|---|---|---|
| Kaggle dataset | Public release of the curated dataset | Source dataset for replication and external review |
| Kaggle competition | Public benchmark and community exploration | Reference point for future comparison and model ideas |
| This repository | Controlled research pipeline | Main source for reproducible experiments, metrics, and analysis |

Future versions of this repository may include a short competition-analysis document summarizing high-level lessons from the Kaggle challenge, but competition leaderboard results should not be mixed directly with the controlled baseline results unless they are rerun under the same split and metric protocol.

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

Recommended documentation files as the repository develops:

```text
docs/
├── DATA_CARD.md          # Dataset construction, labels, caveats, and intended use
├── EXPERIMENTS.md        # Standard experiment protocol and comparison rules
├── MODEL_CARD.md         # Model families, metrics, risks, and evaluation notes
├── RESPONSIBLE_USE.md    # Safety, misuse, and deployment limitations
├── RESULTS_SCHEMA.md     # Expected result files for each model folder
└── COMPETITION.md        # Optional summary of Kaggle competition context
```

These files are for research transparency and replication. They are not meant to turn the repository into a general community-contribution project.

## Dataset and fixed splits

The processed dataset is created from `data/dataset.csv` using `notebooks/00_create_dataset_and_splits.ipynb`.

| Item | Value |
|---|---:|
| Raw rows | 3000 |
| Processed rows | 2999 |
| Rows removed | 1 |
| Duplicate text hashes | 0 |
| Non-extremist rows | 1870 |
| Extremist rows | 1129 |
| Non-extremist proportion | 62.35% |
| Extremist proportion | 37.65% |
| Random seed | 30 |
| Split version | `split_v1_stratified_70_15_15_seed30` |

The dataset uses fixed stratified train/validation/test splits:

| Split | Rows | Non-extremist | Extremist |
|---|---:|---:|---:|
| Train | 2099 | 1309 | 790 |
| Validation | 450 | 281 | 169 |
| Test | 450 | 280 | 170 |

All model notebooks should reuse `splits/split_assignments.csv`. Do not regenerate splits unless the dataset is intentionally revised and a new split version is created.

## Current baseline results

The table below reports held-out test metrics from `results_summary/`. The positive class is `EXTREMIST`.

| Technique | Accuracy | Positive F1 | Positive precision | Positive recall | ROC-AUC | PR-AUC | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| `LOG-REG_TF-IDF` | 0.8533 | 0.8024 | 0.8171 | 0.7882 | 0.9111 | 0.8881 | 0.45 |
| `LIN-SVM_TF-IDF` | 0.8556 | 0.7962 | 0.8523 | 0.7471 | 0.9037 | 0.8825 | 0.47 |
| `SLP_TF-IDF` | 0.8378 | 0.7768 | 0.8089 | 0.7471 | 0.9022 | 0.8777 | 0.50 |

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

These are baseline research metrics. They should not be interpreted as deployment readiness.

## Reproducibility workflow

The project is organized around a small set of notebooks that can be run in order.

| Notebook | Purpose |
|---|---|
| `00_create_dataset_and_splits.ipynb` | Creates the processed dataset, validates labels, assigns canonical row IDs, creates fixed stratified splits, and writes the dataset manifest. |
| `01_LOG-REG_TF-IDF.ipynb` | Trains and evaluates Logistic Regression with TF-IDF features. |
| `02_LIN-SVM_TF-IDF.ipynb` | Trains and evaluates calibrated Linear SVM with TF-IDF features. |
| `03_SLP_TF-IDF.ipynb` | Trains and evaluates a single-layer perceptron over TF-IDF features. |

Expected output structure for each model family:

```text
results_summary/<TECHNIQUE>/
├── ablation_results.csv
├── best_config.json
├── classification_report_test.json
├── confusion_matrix_test.csv
├── metrics_validation.json
└── metrics_test.json
```

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

## Running the experiments

### Option A: Kaggle

1. Attach the Kaggle dataset to the notebook environment.
2. Run `00_create_dataset_and_splits.ipynb` first.
3. Save the generated processed dataset, split assignments, and dataset manifest.
4. Run each model notebook using the fixed split assignments.
5. Commit the summarized results needed for review and replication.

### Option B: Local environment

1. Place `dataset.csv` under `data/`.
2. Run `00_create_dataset_and_splits.ipynb`.
3. Confirm that the generated split files match the expected split version.
4. Run model notebooks in numeric order.
5. Compare results using the JSON and CSV artifacts in `results_summary/`.

## Experiment protocol

To keep the model comparison research-grade:

- Use the same processed dataset for every technique.
- Use the same `split_assignments.csv` for every technique.
- Tune hyperparameters on the training and validation splits only.
- Select thresholds using validation data only.
- Evaluate on the test split only after model configuration and threshold selection are complete.
- Report accuracy, positive-class precision, positive-class recall, positive-class F1, ROC-AUC, PR-AUC, and confusion-matrix counts.
- Save the best configuration, validation metrics, test metrics, and confusion matrix for each technique.
- Treat preprocessing changes as part of the experimental condition, not as an uncontrolled implementation detail.

## Interpretability

The current experiments emphasize interpretable baseline models and feature-level analysis.

Current interpretability directions include:

- Coefficient-based analysis for Logistic Regression.
- Margin/coefficient-based analysis for Linear SVM.
- Linear logit contribution analysis for the Single-Layer Perceptron.
- Local error analysis for false positives and false negatives.
- Future SHAP-based or SHAP-compatible comparisons where appropriate.

Interpretability artifacts should be used to inspect model behavior and guide error analysis. They should not be treated as proof that the model understands ideology, intent, or real-world risk.

## Planned internal experiments

The next stage of the project is a controlled comparison across additional feature/model combinations. Planned internal techniques include:

- `LOG-REG_WORD-CHAR-TF-IDF`
- `XGBOOST_TF-IDF`
- `XGBOOST_WORD-CHAR-TF-IDF`
- `RAND-FOREST_TF-IDF`
- `SENT-EMB_LOG-REG`
- `SENT-EMB_XGBOOST`
- `HATEBERT-FEATS_LOG-REG`

Each technique should follow the same split protocol and write results into a standardized `results_summary/<TECHNIQUE>/` folder.

## Limitations

This project has several important limitations:

- The dataset is relatively small for a high-stakes NLP task.
- Labels depend on human interpretation and may contain subjective judgment calls.
- Social-media language changes over time, so model performance may degrade on future data.
- The binary label structure simplifies a complex phenomenon.
- Models may learn surface-level lexical cues rather than robust contextual understanding.
- Strong held-out metrics do not guarantee fairness, safety, or reliability in applied moderation settings.

These limitations are central to the research and should be discussed alongside any reported results.

## Repository maintenance

This repository is maintained by the research authors for transparency, replication, and project review. It is not currently structured as an open community-development project.

External readers are welcome to inspect the code, reproduce the notebooks, and cite or discuss the work, but the primary development workflow is internal to the research team.

## Citation

If you use this dataset, repository, or related competition materials, please cite the project authors and link to the repository and Kaggle dataset. A formal citation entry can be added after the manuscript, version, or DOI is finalized.

## Authors

- Aditya Suresh
- Anthony Lu