# Social Media Extremism Detection

This repository contains a reproducible NLP research project for binary classification of social-media text as extremist or non-extremist. It includes a curated dataset, fixed train/validation/test splits, classical machine-learning baselines, embedding-based experiments, a transformer fine-tuning experiment, standardized result summaries, and interpretability-oriented analysis artifacts.

The repository is public so that the research process can be reviewed, replicated, and evaluated. It is not intended to be a production moderation system, a general-purpose software package, or a community-maintained open-source project.

> Responsible-use note: The models in this repository should not be used to make automated decisions about people, accounts, posts, or communities. Extremism detection is a high-stakes task with serious false-positive and false-negative risks. Any applied use would require domain-expert review, bias evaluation, privacy review, and human oversight.

## Project status

This is an active research repository maintained by the project authors. The current version contains the dataset foundation, fixed split assignments, and seven completed model families:

* Logistic Regression with word-level TF-IDF features.
* Calibrated Linear SVM with word-level TF-IDF features.
* Single-Layer Perceptron with word-level TF-IDF features.
* Calibrated Linear SVM with character-level TF-IDF features.
* Calibrated Linear SVM with combined word + character TF-IDF features.
* Logistic Regression with FastText document embeddings.
* Twitter-RoBERTa transformer fine-tuning.

The main research finding so far is that the classical TF-IDF and static-embedding approaches cluster around a similar performance range, while the contextual Twitter-RoBERTa model provides the strongest held-out test result. This supports the hypothesis that extremist-text classification benefits from context-aware representations that preserve word order, stance, negation, and social-media phrasing.

## Research motivation

Most online safety NLP work focuses on toxicity, hate speech, or general harmful-content classification. Violent extremist text can overlap with those categories, but it is not identical to them. It may involve ideological framing, support for violence, coded language, or contextual references that are not captured well by surface-level offensiveness alone.

This project treats extremist-content detection as a distinct text-classification problem. The research goals are to:

1. Build a transparent baseline suite for extremist versus non-extremist classification.
2. Use fixed splits and standardized metrics so model comparisons are meaningful.
3. Compare classical machine-learning, neural, embedding-based, and transformer-informed approaches under the same evaluation protocol.
4. Use interpretable feature-attribution methods to inspect what models learn and where they fail.
5. Document the dataset, methodology, and limitations clearly enough for replication.

## Project links

* GitHub repository: <https://github.com/73anthonyL/extremism_sentiment_analysis>
* Kaggle dataset: <https://www.kaggle.com/datasets/adityasureshgithub/digital-extremism-detection-curated-dataset>
* Kaggle competition: <https://www.kaggle.com/competitions/social-media-extremism-detection-challenge>

## Kaggle competition

The Social Media Extremism Detection Challenge on Kaggle is treated as an external research reference for the same binary classification task. Competition results should not be mixed directly with the controlled results in this repository unless they are rerun under the same fixed split and metric protocol.

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
│   ├── 03_SLP_TF-IDF.ipynb
│   ├── 04_CHAR-TF-IDF_LIN-SVM.ipynb
│   ├── 05_WORD-CHAR-TF-IDF_LIN-SVM.ipynb
│   ├── 06_FASTTEXT-EMB_LOG-REG.ipynb
│   └── 07_TWITTER-ROBERTA_FINE-TUNE.ipynb
├── results_summary/
│   ├── foundation/
│   ├── 01_LOG-REG_TF-IDF/
│   ├── 02_LIN-SVM_TF-IDF/
│   ├── 03_SLP_TF-IDF/
│   ├── 04_CHAR-TF-IDF_LIN-SVM/
│   ├── 05_WORD-CHAR-TF-IDF_LIN-SVM/
│   ├── 06_FASTTEXT-EMB_LOG-REG/
│   └── 07_TWITTER-ROBERTA_FINE-TUNE/
├── splits/
│   └── split_assignments.csv
├── docs/
│   ├── DATA_CARD.md
│   ├── EXPERIMENTS.md
│   ├── MODEL_CARD.md
│   ├── RESPONSIBLE_USE.md
│   ├── RESULTS_SCHEMA.md
│   ├── REPLICATION_GUIDE.md
│   ├── COMPETITION.md
│   └── RELEASE_CHECKLIST.md
├── CITATION.cff
├── LICENSE
├── README.md
├── requirements.txt
└── requirements-lock.txt
```

The `docs/` files support research transparency and replication:

| File | Purpose |
|---|---|
| `docs/DATA_CARD.md` | Dataset construction, labels, intended use, and caveats. |
| `docs/EXPERIMENTS.md` | Standard experiment protocol and comparison rules. |
| `docs/MODEL_CARD.md` | Model families, metrics, risks, and evaluation notes. |
| `docs/RESPONSIBLE_USE.md` | Safety, misuse, and deployment limitations. |
| `docs/RESULTS_SCHEMA.md` | Expected result files and metric fields for each experiment folder. |
| `docs/REPLICATION_GUIDE.md` | Step-by-step workflow for reproducing the experiments. |
| `docs/COMPETITION.md` | Kaggle competition context and how competition results relate to this repository. |
| `docs/RELEASE_CHECKLIST.md` | Pre-release checklist before public result updates or manuscript-aligned releases. |

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

## Current controlled results

The table below reports held-out test metrics from `results_summary/`. The positive class is `EXTREMIST`.

| Technique | Representation / model family | Accuracy | Positive F1 | Positive precision | Positive recall | ROC-AUC | PR-AUC | Threshold |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `01_LOG-REG_TF-IDF` | word TF-IDF + logistic regression | 0.8533 | 0.8024 | 0.8171 | 0.7882 | 0.9111 | 0.8881 | 0.45 |
| `02_LIN-SVM_TF-IDF` | word TF-IDF + calibrated linear SVM | 0.8556 | 0.7962 | 0.8523 | 0.7471 | 0.9037 | 0.8825 | 0.47 |
| `03_SLP_TF-IDF` | word TF-IDF + single-layer perceptron | 0.8378 | 0.7768 | 0.8089 | 0.7471 | 0.9022 | 0.8777 | 0.50 |
| `04_CHAR-TF-IDF_LIN-SVM` | character TF-IDF + calibrated linear SVM | 0.8333 | 0.7875 | 0.7596 | 0.8176 | 0.9028 | 0.8745 | 0.40 |
| `05_WORD-CHAR-TF-IDF_LIN-SVM` | combined word + character TF-IDF + calibrated linear SVM | 0.8356 | 0.7886 | 0.7667 | 0.8118 | 0.9032 | 0.8818 | 0.42 |
| `06_FASTTEXT-EMB_LOG-REG` | FastText document embeddings + logistic regression | 0.8178 | 0.7722 | 0.7316 | 0.8176 | 0.9033 | 0.8763 | 0.47 |
| `07_TWITTER-ROBERTA_FINE-TUNE` | Twitter-RoBERTa transformer fine-tuning | 0.8889 | 0.8555 | 0.8409 | 0.8706 | 0.9496 | 0.9233 | 0.72 |


Confusion-matrix summary:

| Technique | TN | FP | FN | TP | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|
| `01_LOG-REG_TF-IDF` | 250 | 30 | 36 | 134 | 0.1071 | 0.2118 |
| `02_LIN-SVM_TF-IDF` | 258 | 22 | 43 | 127 | 0.0786 | 0.2529 |
| `03_SLP_TF-IDF` | 250 | 30 | 43 | 127 | 0.1071 | 0.2529 |
| `04_CHAR-TF-IDF_LIN-SVM` | 236 | 44 | 31 | 139 | 0.1571 | 0.1824 |
| `05_WORD-CHAR-TF-IDF_LIN-SVM` | 238 | 42 | 32 | 138 | 0.1500 | 0.1882 |
| `06_FASTTEXT-EMB_LOG-REG` | 229 | 51 | 31 | 139 | 0.1821 | 0.1824 |
| `07_TWITTER-ROBERTA_FINE-TUNE` | 252 | 28 | 22 | 148 | 0.1000 | 0.1294 |


Current ranking by held-out test PR-AUC:

1. `07_TWITTER-ROBERTA_FINE-TUNE` — PR-AUC 0.9233, ROC-AUC 0.9496, positive F1 0.8555.
2. `01_LOG-REG_TF-IDF` — PR-AUC 0.8881, ROC-AUC 0.9111, positive F1 0.8024.
3. `02_LIN-SVM_TF-IDF` — PR-AUC 0.8825, ROC-AUC 0.9037, positive F1 0.7962.
4. `05_WORD-CHAR-TF-IDF_LIN-SVM` — PR-AUC 0.8818, ROC-AUC 0.9032, positive F1 0.7886.
5. `03_SLP_TF-IDF` — PR-AUC 0.8777, ROC-AUC 0.9022, positive F1 0.7768.
6. `06_FASTTEXT-EMB_LOG-REG` — PR-AUC 0.8763, ROC-AUC 0.9033, positive F1 0.7722.
7. `04_CHAR-TF-IDF_LIN-SVM` — PR-AUC 0.8745, ROC-AUC 0.9028, positive F1 0.7875.

These are baseline research metrics. They should not be interpreted as deployment readiness.

## Reproducibility workflow

Run notebooks in numeric order:

| Notebook | Purpose |
|---|---|
| `00_create_dataset_and_splits.ipynb` | Creates the processed dataset, validates labels, assigns canonical row IDs, creates fixed stratified splits, and writes the dataset manifest. |
| `01_LOG-REG_TF-IDF.ipynb` | Trains and evaluates Logistic Regression with word-level TF-IDF features. |
| `02_LIN-SVM_TF-IDF.ipynb` | Trains and evaluates calibrated Linear SVM with word-level TF-IDF features. |
| `03_SLP_TF-IDF.ipynb` | Trains and evaluates a single-layer perceptron over TF-IDF features. |
| `04_CHAR-TF-IDF_LIN-SVM.ipynb` | Tests whether character n-grams improve robustness to misspellings, obfuscation, hashtags, and noisy social-media phrasing. |
| `05_WORD-CHAR-TF-IDF_LIN-SVM.ipynb` | Combines word-level semantic lexical cues with character-level robustness. |
| `06_FASTTEXT-EMB_LOG-REG.ipynb` | Tests dense FastText document embeddings with logistic regression. |
| `07_TWITTER-ROBERTA_FINE-TUNE.ipynb` | Fine-tunes a contextual Twitter-RoBERTa transformer and saves token-attribution interpretability outputs. |

Expected compact output structure for most classical and embedding model families:

```text
results_summary/<TECHNIQUE>/
├── ablation_results.csv
├── best_config.json
├── classification_report_test.json
├── confusion_matrix_test.csv
├── metrics_validation.json
└── metrics_test.json
```

The RoBERTa folder may instead save a compact image confusion matrix and threshold sweep:

```text
results_summary/07_TWITTER-ROBERTA_FINE-TUNE/
├── ablation_results.csv
├── best_config.json
├── confusion_matrix_test.png
├── metrics_validation.json
├── metrics_test.json
└── threshold_sweep_validation.csv
```

## Installation and environment

Clone the repository:

```bash
git clone https://github.com/73anthonyL/extremism_sentiment_analysis.git
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

For stricter replication of the original working environment, use the lock file instead:

```bash
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
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
5. Commit only compact, reviewable result summaries. Avoid committing raw prediction files or large model artifacts.

### Option B: Local environment

1. Place `dataset.csv` under `data/`.
2. Run `00_create_dataset_and_splits.ipynb`.
3. Confirm that the generated split files match the expected split version.
4. Run model notebooks in numeric order.
5. Compare results using the JSON and CSV artifacts in `results_summary/`.

## Experiment protocol

To keep the model comparison research-grade:

* Use the same processed dataset for every technique.
* Use the same `split_assignments.csv` for every technique.
* Tune hyperparameters on the training and validation splits only.
* Select thresholds using validation data only.
* Evaluate on the test split only after model configuration and threshold selection are complete.
* Report accuracy, positive-class precision, positive-class recall, positive-class F1, ROC-AUC, PR-AUC, and confusion-matrix counts.
* Save the best configuration, validation metrics, test metrics, and confusion matrix for each technique.
* Treat preprocessing changes, external pretraining, and task-specific transfer learning as part of the experimental condition.

## Interpretability

The project emphasizes explainable AI and model auditing. Interpretability methods differ by model family:

* Linear TF-IDF models: coefficient- or margin-based feature analysis.
* SLP over TF-IDF: linear logit contribution analysis.
* FastText embeddings: embedding-direction and token-contribution approximations.
* Twitter-RoBERTa: token-level gradient attribution and local error review.

Interpretability artifacts should be used to inspect model behavior and guide error analysis. They should not be treated as proof that a model understands ideology, intent, or real-world risk.

## Limitations

* The task is high-stakes and cannot be reduced safely to a single automated label.
* The dataset is small enough that external validation is necessary before broad claims.
* The binary label simplifies a complex social and political phenomenon.
* Models may learn lexical, ideological, demographic, or topical shortcuts.
* The transformer result demonstrates performance gains under this split, not deployment readiness.

## Repository maintenance

When adding or rerunning a model notebook:

1. Keep the fixed split protocol unchanged unless a new split version is explicitly introduced.
2. Save only compact result artifacts to `results_summary/`.
3. Avoid committing large model artifacts, raw-text predictions, or local explanation files containing raw text.
4. Update `README.md`, `docs/EXPERIMENTS.md`, `docs/MODEL_CARD.md`, `docs/RESULTS_SCHEMA.md`, `docs/REPLICATION_GUIDE.md`, `notebooks/README.md`, and `results_summary/README.md` when a new technique becomes part of the controlled comparison.

## Citation

See `CITATION.cff` for citation metadata.

## Authors

Maintained by the project authors as a research and replication artifact.
