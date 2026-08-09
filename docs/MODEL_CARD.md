# Model card: extremist-text classifiers

## Model task

The models in this repository perform binary text classification:

```text
input:  social-media text
output: EXTREMIST or NON_EXTREMIST
```

The positive class is `EXTREMIST`.

## Status

These are research models. They are not production moderation systems and should not be used for automated decisions about users, accounts, posts, communities, or safety interventions.

## Current model families

| Technique | Description |
|---|---|
| `01_LOG-REG_TF-IDF` | Logistic Regression trained on word-level TF-IDF features. |
| `02_LIN-SVM_TF-IDF` | Calibrated Linear SVM trained on word-level TF-IDF features. |
| `03_SLP_TF-IDF` | Single-Layer Perceptron trained on word-level TF-IDF features. |
| `04_CHAR-TF-IDF_LIN-SVM` | Calibrated Linear SVM trained on character-level TF-IDF features. |
| `05_WORD-CHAR-TF-IDF_LIN-SVM` | Calibrated Linear SVM trained on combined word + character TF-IDF features. |
| `06_FASTTEXT-EMB_LOG-REG` | Logistic Regression trained on FastText document embeddings. |
| `07_TWITTER-ROBERTA_FINE-TUNE` | Fine-tuned Twitter-RoBERTa transformer classifier. Current champion. |

## Model families without registered results

Four transformer-ensemble notebooks exist whose results are **not** part of this
card's metrics. Their numbers are not reported here as model metrics because
none of them is backed by a derivable, schema-valid result folder.

| Technique | Description | Status |
|---|---|---|
| `08_BEST-ROBERTA_SEED-ENSEMBLE` | Seed ensemble of the fine-tuned transformer, probability-averaged. | Ran; regressed against `07`. Unregistered. |
| `09_MULTI-CHECKPOINT_LOGIT-STACK` | Learned stacker over multiple transformer checkpoints. | Built; never run. Superseded by `11`. |
| `10_TWITTER-ROBERTA_LOGIT-POOL-STABLE` | Mean-log-odds pooling across three seeds of `cardiffnlp/twitter-roberta-base-hate-latest` at a fixed 0.50 cutoff. | Ran; unregistered. |
| `11_MULTI-CHECKPOINT_LOGIT-POOL` | Mean-log-odds pooling across the admitted subset of five checkpoints spanning fine-tuning lineage, pretraining corpus, architecture/tokenizer, and scale. | Ran; unregistered pending artifact retrieval. |

Notebook `10` reports a higher raw test accuracy than the champion — 403 of 450
rows against 400 of 450. That 3-row gap is well inside the noise floor of a
450-row test split, which needs roughly 14 rows for a McNemar-detectable
difference, and no verdict has been issued by `tools/compare_techniques.py`. It
is not treated as an improvement, and `07_TWITTER-ROBERTA_FINE-TUNE` remains the
champion in `research_loop/registry.json`.

## Current held-out test metrics

Rendered from `results_summary/` by `tools/render_tables.py`; do not edit by
hand.

<!-- RENDERED-TABLE:BEGIN id=test-detail -->
| Technique | Accuracy | Positive F1 | Positive precision | Positive recall | ROC-AUC | PR-AUC | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| `01_LOG-REG_TF-IDF` | 0.8533 | 0.8024 | 0.8171 | 0.7882 | 0.9111 | 0.8881 | 0.45 |
| `02_LIN-SVM_TF-IDF` | 0.8556 | 0.7962 | 0.8523 | 0.7471 | 0.9037 | 0.8825 | 0.47 |
| `03_SLP_TF-IDF` | 0.8378 | 0.7768 | 0.8089 | 0.7471 | 0.9022 | 0.8777 | 0.5 |
| `04_CHAR-TF-IDF_LIN-SVM` | 0.8333 | 0.7875 | 0.7596 | 0.8176 | 0.9028 | 0.8745 | 0.4 |
| `05_WORD-CHAR-TF-IDF_LIN-SVM` | 0.8356 | 0.7886 | 0.7667 | 0.8118 | 0.9032 | 0.8818 | 0.42 |
| `06_FASTTEXT-EMB_LOG-REG` | 0.8178 | 0.7722 | 0.7316 | 0.8176 | 0.9033 | 0.8763 | 0.47 |
| `07_TWITTER-ROBERTA_FINE-TUNE` | 0.8889 | 0.8555 | 0.8409 | 0.8706 | 0.9496 | 0.9233 | 0.72 |

Rendered by tools/render_tables.py from results_summary/ — do not edit by hand.
<!-- RENDERED-TABLE:END id=test-detail -->


## Confusion-matrix summary

<!-- RENDERED-TABLE:BEGIN id=confusion-test -->
| Technique | TN | FP | FN | TP | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|
| `01_LOG-REG_TF-IDF` | 250 | 30 | 36 | 134 | 0.1071 | 0.2118 |
| `02_LIN-SVM_TF-IDF` | 258 | 22 | 43 | 127 | 0.0786 | 0.2529 |
| `03_SLP_TF-IDF` | 250 | 30 | 43 | 127 | 0.1071 | 0.2529 |
| `04_CHAR-TF-IDF_LIN-SVM` | 236 | 44 | 31 | 139 | 0.1571 | 0.1824 |
| `05_WORD-CHAR-TF-IDF_LIN-SVM` | 238 | 42 | 32 | 138 | 0.1500 | 0.1882 |
| `06_FASTTEXT-EMB_LOG-REG` | 229 | 51 | 31 | 139 | 0.1821 | 0.1824 |
| `07_TWITTER-ROBERTA_FINE-TUNE` | 252 | 28 | 22 | 148 | 0.1000 | 0.1294 |

Rendered by tools/render_tables.py from results_summary/ — do not edit by hand.
<!-- RENDERED-TABLE:END id=confusion-test -->


## Evaluation data

Models are evaluated on a fixed held-out test split of 450 rows:

* 280 non-extremist examples
* 170 extremist examples

Model configuration and threshold selection should be completed before test evaluation.

## Intended use

Appropriate uses:

* Research replication.
* Model-comparison experiments.
* Error analysis.
* Feature-attribution and interpretability research.
* Educational discussion of limitations in high-stakes NLP classification.

Unsupported uses:

* Automated moderation decisions.
* User-level risk scoring.
* Real-time enforcement or escalation systems.
* Claims about an individual's ideology, belief, or intent.
* Safety-critical deployment without extensive external validation.

## Key risks

* False positives may incorrectly flag benign, quoted, journalistic, academic, counterspeech, or contextual content.
* False negatives may miss implicit, coded, or emerging extremist language.
* Classical models may overfit lexical cues instead of understanding context.
* Transformer models may improve contextual performance while still learning spurious correlations.
* The dataset size limits robustness claims.
* Binary labels simplify a much more complex social and political phenomenon.

## Interpretability notes

Interpretability methods differ across model families:

* Linear TF-IDF models support direct feature-weight inspection.
* The SLP model supports linear logit contribution analysis over TF-IDF features.
* FastText embeddings support approximate token and embedding-direction analysis, but embedding dimensions are not directly human-readable.
* Twitter-RoBERTa supports token-level gradient attribution, but attributions should be treated as model-behavior signals rather than proof of true reasoning.

Any attribution analysis should be paired with manual error review.

## Result interpretation

The classical TF-IDF and FastText-based approaches remain useful interpretable baselines. The Twitter-RoBERTa fine-tuning run is the strongest current registered model by PR-AUC, ROC-AUC, positive F1, and accuracy, but the result should be understood as a controlled research result on this split rather than as deployment evidence.

### Statistical power

The held-out test split is 450 rows, so the resolution of any comparison made on
it is coarse. Detecting a difference at McNemar exact significance needs roughly
+3 accuracy points, about 14 rows.

Read the metrics table accordingly:

* The classical and embedding baselines sit within a handful of rows of each
  other — `01` and `02` differ by a single test row. Their ordering carries no
  statistical weight.
* The transformer's margin over them is the one gap in this table wide enough to
  take seriously as a direction.
* The transformer *variants* explored so far — `07`, `08`, `10`, and `11` —
  span 395 to 409 correct rows. Only the widest pair reaches the detection floor
  at all, and Holm correction over ten consumed test unlocks raises the bar
  further, so none of them should be treated as separated without a verdict.

A ranking is not a significance test, and `tools/compare_techniques.py` is the
only thing in this repository that issues a verdict.
