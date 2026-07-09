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
| `07_TWITTER-ROBERTA_FINE-TUNE` | Fine-tuned Twitter-RoBERTa transformer classifier. |

## Current held-out test metrics

| Technique | Representation / model family | Accuracy | Positive F1 | Positive precision | Positive recall | ROC-AUC | PR-AUC | Threshold |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `01_LOG-REG_TF-IDF` | word TF-IDF + logistic regression | 0.8533 | 0.8024 | 0.8171 | 0.7882 | 0.9111 | 0.8881 | 0.45 |
| `02_LIN-SVM_TF-IDF` | word TF-IDF + calibrated linear SVM | 0.8556 | 0.7962 | 0.8523 | 0.7471 | 0.9037 | 0.8825 | 0.47 |
| `03_SLP_TF-IDF` | word TF-IDF + single-layer perceptron | 0.8378 | 0.7768 | 0.8089 | 0.7471 | 0.9022 | 0.8777 | 0.50 |
| `04_CHAR-TF-IDF_LIN-SVM` | character TF-IDF + calibrated linear SVM | 0.8333 | 0.7875 | 0.7596 | 0.8176 | 0.9028 | 0.8745 | 0.40 |
| `05_WORD-CHAR-TF-IDF_LIN-SVM` | combined word + character TF-IDF + calibrated linear SVM | 0.8356 | 0.7886 | 0.7667 | 0.8118 | 0.9032 | 0.8818 | 0.42 |
| `06_FASTTEXT-EMB_LOG-REG` | FastText document embeddings + logistic regression | 0.8178 | 0.7722 | 0.7316 | 0.8176 | 0.9033 | 0.8763 | 0.47 |
| `07_TWITTER-ROBERTA_FINE-TUNE` | Twitter-RoBERTa transformer fine-tuning | 0.8889 | 0.8555 | 0.8409 | 0.8706 | 0.9496 | 0.9233 | 0.72 |


## Confusion-matrix summary

| Technique | TN | FP | FN | TP | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|
| `01_LOG-REG_TF-IDF` | 250 | 30 | 36 | 134 | 0.1071 | 0.2118 |
| `02_LIN-SVM_TF-IDF` | 258 | 22 | 43 | 127 | 0.0786 | 0.2529 |
| `03_SLP_TF-IDF` | 250 | 30 | 43 | 127 | 0.1071 | 0.2529 |
| `04_CHAR-TF-IDF_LIN-SVM` | 236 | 44 | 31 | 139 | 0.1571 | 0.1824 |
| `05_WORD-CHAR-TF-IDF_LIN-SVM` | 238 | 42 | 32 | 138 | 0.1500 | 0.1882 |
| `06_FASTTEXT-EMB_LOG-REG` | 229 | 51 | 31 | 139 | 0.1821 | 0.1824 |
| `07_TWITTER-ROBERTA_FINE-TUNE` | 252 | 28 | 22 | 148 | 0.1000 | 0.1294 |


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

The classical TF-IDF and FastText-based approaches remain useful interpretable baselines. The Twitter-RoBERTa fine-tuning run is the strongest current model by PR-AUC, ROC-AUC, positive F1, and accuracy, but the result should be understood as a controlled research result on this split rather than as deployment evidence.
