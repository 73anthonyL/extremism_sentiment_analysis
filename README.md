# Social Media Extremism Detection

This repository contains a reproducible NLP research project for binary classification of social-media text as extremist or non-extremist. It includes a curated dataset, fixed train/validation/test splits, classical machine-learning baselines, embedding-based experiments, a transformer fine-tuning experiment, standardized result summaries, and interpretability-oriented analysis artifacts.

The repository is public so that the research process can be reviewed, replicated, and evaluated. It is not intended to be a production moderation system, a general-purpose software package, or a community-maintained open-source project.

> Responsible-use note: The models in this repository should not be used to make automated decisions about people, accounts, posts, or communities. Extremism detection is a high-stakes task with serious false-positive and false-negative risks. Any applied use would require domain-expert review, bias evaluation, privacy review, and human oversight.

## Project status

This is an active research repository maintained by the project authors. The current version contains the dataset foundation, fixed split assignments, a verification toolkit, and eight model families whose results are registered in the controlled comparison:

* Logistic Regression with word-level TF-IDF features.
* Calibrated Linear SVM with word-level TF-IDF features.
* Single-Layer Perceptron with word-level TF-IDF features.
* Calibrated Linear SVM with character-level TF-IDF features.
* Calibrated Linear SVM with combined word + character TF-IDF features.
* Logistic Regression with FastText document embeddings.
* Twitter-RoBERTa transformer fine-tuning.
* Heterogeneous multi-checkpoint logit-pooled transformer ensemble.

Three further transformer-ensemble notebooks (`08`, `09`, `10`) exist but are **not** part of the controlled comparison. Two ran and reported test numbers without leaving a derivable result folder; one was never run. The section [Transformer ensemble notebooks](#transformer-ensemble-notebooks) records where each one stands.

**All twelve notebooks are now primed for rerun.** Each has been rewritten on one template, loads one shared helper (`tools/notebook_kit.py`), selects its threshold by one rule, explains its model with one attribution method, and writes its outputs in one layout. Their cell outputs are empty until each is rerun on Kaggle. The numbers in this README come from the runs made before priming and will be replaced, technique by technique, as the reruns land; a rerun's numbers will differ somewhat because the threshold rule and the early-stopping monitors were unified. `notebooks/README.md` describes the template and how to run a notebook.

The main research finding so far is that the classical TF-IDF and static-embedding approaches cluster around a similar performance range, while contextual transformer models provide the strongest registered held-out test results — the single fine-tuned Twitter-RoBERTa run, and above it a logit-pooled ensemble of heterogeneous checkpoints. This supports the hypothesis that extremist-text classification benefits from context-aware representations that preserve word order, stance, negation, and social-media phrasing.

A second finding is methodological, and it constrains how the first one may be stated: the test split is 450 rows, so detecting a difference at McNemar exact significance requires roughly +3 accuracy points, about 14 rows. The transformer family's margin over the classical baselines clears that floor comfortably. The gaps *among* the classical baselines do not come close, and the gaps *among* the transformer variants tried so far sit at or below it — and no amount of decimal places changes that. `INCONCLUSIVE` is therefore a first-class, publishable outcome here, and a higher accuracy number is not by itself evidence of a better model.

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
│   ├── 07_TWITTER-ROBERTA_FINE-TUNE.ipynb
│   ├── 08_TWITTER-ROBERTA_SEED-ENSEMBLE.ipynb       # closed line
│   ├── 09_MULTI-CHECKPOINT_LOGIT-STACK.ipynb        # closed line; superseded by 11
│   ├── 10_TWITTER-ROBERTA_LOGIT-POOL-STABLE.ipynb   # closed line
│   └── 11_MULTI-CHECKPOINT_LOGIT-POOL.ipynb         # result registered
├── results_summary/
│   ├── foundation/
│   ├── 01_LOG-REG_TF-IDF/
│   ├── 02_LIN-SVM_TF-IDF/
│   ├── 03_SLP_TF-IDF/
│   ├── 04_CHAR-TF-IDF_LIN-SVM/
│   ├── 05_WORD-CHAR-TF-IDF_LIN-SVM/
│   ├── 06_FASTTEXT-EMB_LOG-REG/
│   ├── 07_TWITTER-ROBERTA_FINE-TUNE/
│   └── 11_MULTI-CHECKPOINT_LOGIT-POOL/
├── research_loop/            # experiment records: probability artifacts, registry, preregistrations
│   └── probs/                # sanitized probability artifacts (row_id, split, y_true, y_prob)
├── tools/
│   ├── notebook_kit.py           # the helper every notebook loads: split, metrics, exports
│   ├── check_notebook.py         # static check of a notebook against the template
│   ├── attributions.py           # word-level SHAP artifact contract
│   ├── validate_shap.py          # pipeline check against logistic-regression coefficients
│   ├── categorize_attributions.py  # RQ2: slur / framing / identity / topical shares
│   ├── compare_reliance.py       # RQ3: pretraining and ensembling vs reliance; stability
│   ├── identity_fpr.py           # RQ4: false positives on posts with identity terms
│   ├── run_manifest.py           # reconciles committed files with external assets
│   ├── validate_results_folder.py
│   ├── render_tables.py          # regenerates every doc results table
│   ├── scan_text_leakage.py      # dataset text in committed files
│   └── tests/                    # the toolkit's own pytest suite
├── legacy_tools/             # archived adjudication toolkit (ledger, judge); reference only
├── splits/
│   ├── split_assignments.csv
│   └── split_assignments.PRE-REPAIR.csv
├── docs/
│   ├── RESEARCH_PLAN.md
│   └── RESULTS_SCHEMA.md
├── CLAUDE.md
├── CITATION.cff
├── LICENSE
├── README.md
├── requirements.txt
├── requirements-lock.txt
└── requirements-dev.txt
```

`docs/RESEARCH_PLAN.md` is the working plan for the paper: the four research
questions, what each technique has and lacks today, the work packages in
dependency order, and the decision rules that fix what the prose may claim.
`docs/RESULTS_SCHEMA.md` defines every committed artifact: result folders,
probability artifacts, word-level attribution artifacts, the RQ outputs, and
the run manifest. `data/lexicons/README.md` describes the word lists the RQ2
and RQ4 tools use.

## The verification layer

Results in this repository are derived and checked by tooling rather than
transcribed by hand. Two rules follow from that, and they are enforced:

* **Results are derived, not transcribed.** Every metric is computed from a
  committed probability artifact with the same function the notebooks use.
  `tools/validate_results_folder.py` recomputes each threshold-dependent
  metric from the stored confusion counts and fails on any disagreement.
* **Doc tables are rendered, not edited.** `tools/render_tables.py --write`
  regenerates every results table in this README and in
  `results_summary/README.md` from the committed artifacts. The tables live
  inside paired HTML-comment markers and are rewritten wholesale; hand edits
  inside a region are clobbered.

```bash
python3 tools/check_notebook.py --all           # notebooks against the template (--primed before a run)
python3 tools/validate_results_folder.py --all  # schema + internal consistency
python3 tools/render_tables.py --check          # documentation drift
python3 tools/scan_text_leakage.py              # dataset text, hashes, row ids in committed files
python3 tools/run_manifest.py check --all       # committed vs external assets
python3 -m pytest tools/tests/ -q               # the toolkit's own tests
```

Notebooks run on Kaggle and are never executed locally, so
`tools/check_notebook.py` verifies them statically: the section structure,
the `CONFIG` literals, that the threshold is selected once on validation and
the test split evaluated once, that the probability artifact and the
attribution runs are exported, that no cell would print a post or a row id,
that every cell parses, and that every name it uses is defined.
`tools/tests/test_notebook_pipeline.py` runs the kit end to end on synthetic
text and hands the outputs to the real tools, including the coefficient check.

The explainability tools run on committed word-level attribution artifacts
(see `docs/RESULTS_SCHEMA.md`):

```bash
python3 tools/validate_shap.py --technique 01_LOG-REG_TF-IDF   # pipeline check first
python3 tools/categorize_attributions.py --all                  # RQ2
python3 tools/compare_reliance.py                               # RQ3
python3 tools/identity_fpr.py --all                              # RQ4
```

Notebooks are committed with their cell outputs on purpose, so a reader can
follow them. `scan_text_leakage.py` is what keeps a post, a text hash, or a
row id out of those outputs. The primed notebooks have no outputs yet and no
cell that would print one; the earlier runs' outputs, which did contain
dataset text, are kept in git history at tag `notebooks-pre-priming`.

The earlier champion-promotion toolkit (hash-chained test ledger, McNemar
judge, protocol audit) is archived in `legacy_tools/` for reference. Its
records in `research_loop/` are kept as data.

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

It is rendered from the committed result artifacts by `tools/render_tables.py`. A technique appears here only once it has a complete, schema-valid result folder derived from a committed probability artifact — which is why notebooks `08`, `09`, and `10` are absent.

<!-- RENDERED-TABLE:BEGIN id=main-comparison -->
| Technique | Validation accuracy | Test accuracy | Test balanced accuracy | Test macro F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|
| `01_LOG-REG_TF-IDF` | 0.8267 | 0.8556 | 0.8423 | 0.8451 | 0.9137 |
| `02_LIN-SVM_TF-IDF` | 0.8244 | 0.8533 | 0.8348 | 0.8409 | 0.9065 |
| `03_SLP_TF-IDF` | 0.8222 | 0.8400 | 0.8067 | 0.8200 | 0.9042 |
| `04_CHAR-TF-IDF_LIN-SVM` | 0.8267 | 0.8311 | 0.7996 | 0.8112 | 0.9147 |
| `05_WORD-CHAR-TF-IDF_LIN-SVM` | 0.8244 | 0.8556 | 0.8192 | 0.8355 | 0.9198 |
| `06_FASTTEXT-EMB_LOG-REG` | 0.7956 | 0.8333 | 0.8002 | 0.8128 | 0.8986 |
| `07_TWITTER-ROBERTA_FINE-TUNE` | 0.8644 | 0.8889 | 0.8807 | 0.8816 | 0.9510 |
| `11_MULTI-CHECKPOINT_LOGIT-POOL` | 0.8733 | 0.9089 | 0.8956 | 0.9015 | 0.9682 |

Rendered by tools/render_tables.py from results_summary/ — do not edit by hand.
<!-- RENDERED-TABLE:END id=main-comparison -->


Confusion-matrix summary:

<!-- RENDERED-TABLE:BEGIN id=confusion-test -->
| Technique | TN | FP | FN | TP | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|
| `01_LOG-REG_TF-IDF` | 251 | 29 | 36 | 134 | 0.1036 | 0.2118 |
| `02_LIN-SVM_TF-IDF` | 255 | 25 | 41 | 129 | 0.0893 | 0.2412 |
| `03_SLP_TF-IDF` | 264 | 16 | 56 | 114 | 0.0571 | 0.3294 |
| `04_CHAR-TF-IDF_LIN-SVM` | 260 | 20 | 56 | 114 | 0.0714 | 0.3294 |
| `05_WORD-CHAR-TF-IDF_LIN-SVM` | 271 | 9 | 56 | 114 | 0.0321 | 0.3294 |
| `06_FASTTEXT-EMB_LOG-REG` | 262 | 18 | 57 | 113 | 0.0643 | 0.3353 |
| `07_TWITTER-ROBERTA_FINE-TUNE` | 256 | 24 | 26 | 144 | 0.0857 | 0.1529 |
| `11_MULTI-CHECKPOINT_LOGIT-POOL` | 266 | 14 | 27 | 143 | 0.0500 | 0.1588 |

Rendered by tools/render_tables.py from results_summary/ — do not edit by hand.
<!-- RENDERED-TABLE:END id=confusion-test -->


Current ranking by held-out test PR-AUC:

1. `11_MULTI-CHECKPOINT_LOGIT-POOL` — PR-AUC 0.9553, ROC-AUC 0.9682, positive F1 0.8746.
2. `07_TWITTER-ROBERTA_FINE-TUNE` — PR-AUC 0.9233, ROC-AUC 0.9496, positive F1 0.8555.
3. `01_LOG-REG_TF-IDF` — PR-AUC 0.8881, ROC-AUC 0.9111, positive F1 0.8024.
4. `02_LIN-SVM_TF-IDF` — PR-AUC 0.8825, ROC-AUC 0.9037, positive F1 0.7962.
5. `05_WORD-CHAR-TF-IDF_LIN-SVM` — PR-AUC 0.8818, ROC-AUC 0.9032, positive F1 0.7886.
6. `03_SLP_TF-IDF` — PR-AUC 0.8777, ROC-AUC 0.9022, positive F1 0.7768.
7. `06_FASTTEXT-EMB_LOG-REG` — PR-AUC 0.8763, ROC-AUC 0.9033, positive F1 0.7722.
8. `04_CHAR-TF-IDF_LIN-SVM` — PR-AUC 0.8745, ROC-AUC 0.9028, positive F1 0.7875.

A ranking is not a significance test. On a 450-row test split, roughly 14 rows of difference are needed for McNemar significance, and most gaps above are smaller. Accuracy is RQ1 and is kept brief; the contribution is the attribution comparison below.

These are baseline research metrics. They should not be interpreted as deployment readiness.

## Explainability results

The paper's contribution is an attribution comparison across the model
spectrum. The tables below are rendered from committed artifacts by
`tools/render_tables.py`; a technique appears once it has the corresponding
artifact.

RQ2, share of positive attribution mass by lexicon category (top-50 words,
whole-model runs averaged):

<!-- RENDERED-TABLE:BEGIN id=rq2-category-shares -->
| Technique | Runs | Slur | Extremist framing | Identity term | Topical |
|---|---:|---:|---:|---:|---:|
| _no attribution runs categorized yet_ | | | | | |

Rendered by tools/render_tables.py from results_summary/ — do not edit by hand.
<!-- RENDERED-TABLE:END id=rq2-category-shares -->

RQ4, false-positive rate on non-extremist test posts with vs without identity
terms:

<!-- RENDERED-TABLE:BEGIN id=rq4-identity-fpr -->
| Technique | Non-extremist posts with identity terms | FPR with identity terms | FPR without | Ratio | Fisher p |
|---|---:|---:|---:|---:|---:|
| `01_LOG-REG_TF-IDF` | 67 | 0.1642 | 0.0845 | 1.9428 | 0.0691 |
| `02_LIN-SVM_TF-IDF` | 67 | 0.1493 | 0.0704 | 2.1194 | 0.0817 |
| `03_SLP_TF-IDF` | 67 | 0.0896 | 0.0469 | 1.9075 | 0.2260 |
| `04_CHAR-TF-IDF_LIN-SVM` | 67 | 0.1343 | 0.0516 | 2.6011 | 0.0298 |
| `05_WORD-CHAR-TF-IDF_LIN-SVM` | 67 | 0.0746 | 0.0188 | 3.9739 | 0.0385 |
| `06_FASTTEXT-EMB_LOG-REG` | 67 | 0.1045 | 0.0516 | 2.0231 | 0.1517 |
| `07_TWITTER-ROBERTA_FINE-TUNE` | 67 | 0.1343 | 0.0704 | 1.9075 | 0.1312 |
| `11_MULTI-CHECKPOINT_LOGIT-POOL` | 67 | 0.1194 | 0.0282 | 4.2388 | 0.0064 |

Rendered by tools/render_tables.py from results_summary/ — do not edit by hand.
<!-- RENDERED-TABLE:END id=rq4-identity-fpr -->

RQ3 comparisons live in `results_summary/rq/rq3_reliance_comparison.csv` and
`rq3_stability.csv` once at least two attribution runs per technique exist.

## Transformer ensemble notebooks

Notebooks `08`–`11` ensemble contextual transformers. Only `11` has a registered result and takes part in the controlled comparison; `08`, `09` and `10` are closed lines (`docs/RESEARCH_PLAN.md`, section 6) and do not appear in the tables above. All four are primed to the same template.

| Notebook | Earlier run | Status |
|---|---|---|
| `08_TWITTER-ROBERTA_SEED-ENSEMBLE` | ran on Kaggle | Closed line. Not registered: the earlier run left no result folder and no probability export. Renamed from `08_BEST-ROBERTA_SEED-ENSEMBLE`; a notebook name never encodes a claim, and its own earlier test accuracy (0.8778) sat below notebook `07`'s. The primed notebook exports the probability artifact and per-member attribution runs. |
| `09_MULTI-CHECKPOINT_LOGIT-STACK` | built, never run | Closed line. Superseded by notebook `11`, which keeps the multi-checkpoint idea but replaces the learned logit stacker with notebook `10`'s equal-weight mean-log-odds pool. |
| `10_TWITTER-ROBERTA_LOGIT-POOL-STABLE` | ran on Kaggle | Closed line. The earlier run evaluated the test split but exported no probabilities, so no folder could be derived. The primed notebook exports them. |
| `11_MULTI-CHECKPOINT_LOGIT-POOL` | ran on Kaggle | **Has a result folder**, derived from its committed probability artifacts, so it appears in the tables above. The primed notebook adds whole-ensemble and per-component attribution runs. |

### About the highest accuracy figures

Notebook `11` reports the strongest held-out test result in the repository: **409 of 450 rows, accuracy 0.9089**, ROC-AUC 0.9682, PR-AUC 0.9553. It pools mean log-odds across four admitted checkpoints — a fifth, `microsoft/deberta-v3-base`, was excluded by the prespecified 0.84 validation admission floor — at a fixed 0.50 cutoff.

The way it got there is worth as much as the number. Its validation gate passed at 393/450 against a prespecified bar of 392, and the prespecified primary decision rule was **retained**, because the strongest of four challengers gained only 2 correct validation examples against a required 3. The result was not tuned into existence after the fact.

| Technique | Correct / 450 | Accuracy | Status |
|---|---:|---:|---|
| `11_MULTI-CHECKPOINT_LOGIT-POOL` | 409 | 0.9089 | registered, not adjudicated |
| `10_TWITTER-ROBERTA_LOGIT-POOL-STABLE` | 403 | 0.8956 | unregistered, no probability export |
| `07_TWITTER-ROBERTA_FINE-TUNE` | 400 | 0.8889 | **current champion** |

Two caveats still apply, and they are the difference between a strong measurement and an established improvement:

1. **The margin is below the detection floor.** Notebook `11` leads the champion by 9 test rows. Roughly 14 are needed for a McNemar-detectable difference on a 450-row split, before Holm correction over a family that has now consumed ten test unlocks. It may well come back `INCONCLUSIVE`.
2. **No adjudication was run.** The champion pointer in `research_loop/registry.json` still names `07_TWITTER-ROBERTA_FINE-TUNE`; the judge that would move it is archived in `legacy_tools/`.

So notebook `11` is a real, derived, schema-valid result that leads on every metric. What matters for the paper is what it relies on, which the explainability tools measure.

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
| `07_TWITTER-ROBERTA_FINE-TUNE.ipynb` | Fine-tunes a contextual Twitter-RoBERTa transformer. |
| `11_MULTI-CHECKPOINT_LOGIT-POOL.ipynb` | Pools mean log-odds across fine-tuned checkpoints spanning pretraining corpus, architecture and scale, three seeds each. |

The remaining notebooks are closed lines, not part of the controlled comparison:

| Notebook | Purpose |
|---|---|
| `08_TWITTER-ROBERTA_SEED-ENSEMBLE.ipynb` | Averages the probabilities of three seeds of the fine-tuned transformer. |
| `09_MULTI-CHECKPOINT_LOGIT-STACK.ipynb` | Learned stacker over several transformer checkpoints with train-side cross-validation. Never run; superseded by `11`. |
| `10_TWITTER-ROBERTA_LOGIT-POOL-STABLE.ipynb` | Mean log-odds pooling across seeds. A guarded auxiliary blend is implemented and switched off. |

Every primed model notebook writes the same output structure:

```text
results_summary/<TECHNIQUE>/
├── ablation_results.csv
├── best_config.json
├── classification_report_test.json
├── confusion_matrix_test.csv
├── metrics_validation.json
├── metrics_test.json
├── threshold_sweep_validation.csv
└── attributions/
    ├── shap-partition_test_seed30.csv, .json
    ├── shap-partition_validation_seed30.csv, .json
    ├── shap-partition_test_seed30_member-<id>.csv, .json   (ensembles)
    └── coefficients.csv                                    (01, 06)
research_loop/probs/<TECHNIQUE>__validation.csv, __test.csv, __meta.json
```

Folders committed before priming use older layouts (the transformer folder,
for example, has a confusion-matrix image instead of the CSV); each is
replaced when its notebook is rerun.

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

The notebooks are written for Kaggle. `notebooks/README.md` has the full
procedure; in short:

1. Attach this repository to the kernel as an input dataset. It provides
   `tools/notebook_kit.py` and the committed split to verify against.
2. Run `00_create_dataset_and_splits.ipynb` first, with the Kaggle dataset
   attached, and save the version. It reproduces the frozen split and stops if
   the result differs from the committed assignment.
3. Attach the output of notebook 00 to each model notebook and run it. Use a
   GPU with internet on for notebooks `07` to `11`.
4. Each notebook writes `<TECHNIQUE>_repo_files.zip`, which unpacks at the
   repository root, and an `external/` folder (weights and per-post files)
   that is uploaded to a Kaggle Dataset and recorded with
   `tools/run_manifest.py`. Per-post files and weights are never committed.
5. Run the tools on the copied artifacts, regenerate the tables, and check
   the notebook and the repository for leaked text before committing.

The tools (`tools/*.py`) run locally on committed files and need only the
packages in `requirements.txt`.

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
* Evaluate the test split once per technique. Do not describe a technique as better than another on a gap below the McNemar detection floor.
* Export a word-level attribution artifact for every technique so it can take part in RQ2-RQ4.

## Interpretability

The paper's contribution is an attribution comparison, so every technique is
explained by one method: SHAP's partition explainer over a word-level text
masker, applied to the model as a function from text to P(EXTREMIST). The
explainer masks words and measures how the probability moves. It needs
nothing from inside the model, so a TF-IDF pipeline, an embedding model, a
fine-tuned transformer and a pooled ensemble are explained by the same call,
and a word's attribution means the same thing in every row of a table.

Earlier versions of the notebooks used a different method per family
(coefficients, margin contributions, embedding-direction scores, gradient
times embedding). Those quantities are not comparable with one another and
are no longer committed artifacts. Two survive in supporting roles:
logistic-regression coefficients (notebook 01, and a word-level projection in
notebook 06) are what the SHAP pipeline is validated against, and
gradient-times-embedding is kept in notebook 07 as an appendix comparison.

Each run is reduced to the committed word-level table defined in
`docs/RESULTS_SCHEMA.md`. Per-post attributions carry dataset text and go to
external storage.

Interpretability artifacts should be used to inspect model behavior and guide error analysis. They should not be treated as proof that a model understands ideology, intent, or real-world risk. Masked text is unlike real posts, and an attribution is not a causal claim about language.

## Limitations

* The task is high-stakes and cannot be reduced safely to a single automated label.
* The dataset is small enough that external validation is necessary before broad claims.
* The binary label simplifies a complex social and political phenomenon.
* Models may learn lexical, ideological, demographic, or topical shortcuts.
* The transformer result demonstrates performance gains under this split, not deployment readiness.

## Repository maintenance

When adding or rerunning a model notebook:

1. Keep the fixed split protocol unchanged unless a new split version is explicitly introduced. Never regenerate the split *assignment*.
2. Start from the template (`python3 tools/check_notebook.py --print-template <role>`) and use `tools/notebook_kit.py` for loading, metrics, thresholding and every export. Check the notebook with `tools/check_notebook.py --primed` before uploading it, and without `--primed` before committing its outputs.
3. Export a sanitized probability artifact (`row_id`, `split`, `y_true`, `y_prob`) and a word-level attribution artifact (`word`, `mean_abs_attribution`, `mean_attribution`, `support`). Do not hand-copy metrics into JSON.
4. Save only compact result artifacts to `results_summary/`. Weights, logs, and per-post files go to external storage and are recorded with `tools/run_manifest.py`.
5. Never print dataset text, row ids, or text hashes in a notebook cell: outputs are committed so readers can follow the notebook, and `tools/scan_text_leakage.py` fails the commit otherwise.
6. Regenerate the documentation tables with `tools/render_tables.py --write` rather than editing them, then confirm `--check` exits 0.
7. Update the prose in `README.md`, `docs/RESULTS_SCHEMA.md`, `notebooks/README.md`, and `results_summary/README.md` when a new technique joins the comparison.

## Citation

See `CITATION.cff` for citation metadata.

## Authors

Maintained by the project authors as a research and replication artifact.
