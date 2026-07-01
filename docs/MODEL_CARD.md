# Model card: baseline extremist-text classifiers

## Model task

The models in this repository perform binary text classification:

```text
input:  social-media text
output: EXTREMIST or NON_EXTREMIST
```

The positive class is `EXTREMIST`.

## Status

These are research baseline models. They are not production moderation systems and should not be used for automated decisions about users, accounts, posts, communities, or safety interventions.

## Current baseline models

| Technique | Description |
|---|---|
| `LOG-REG_TF-IDF` | Logistic Regression trained on TF-IDF features. |
| `LIN-SVM_TF-IDF` | Calibrated Linear SVM trained on TF-IDF features. |
| `SLP_TF-IDF` | Single-Layer Perceptron trained on TF-IDF features. |

## Current held-out test metrics

| Technique | Accuracy | Positive F1 | Positive precision | Positive recall | ROC-AUC | PR-AUC | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| `LOG-REG_TF-IDF` | 0.8533 | 0.8024 | 0.8171 | 0.7882 | 0.9111 | 0.8881 | 0.45 |
| `LIN-SVM_TF-IDF` | 0.8556 | 0.7962 | 0.8523 | 0.7471 | 0.9037 | 0.8825 | 0.47 |
| `SLP_TF-IDF` | 0.8378 | 0.7768 | 0.8089 | 0.7471 | 0.9022 | 0.8777 | 0.50 |

## Confusion-matrix summary

| Technique | TN | FP | FN | TP | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|
| `LOG-REG_TF-IDF` | 250 | 30 | 36 | 134 | 0.1071 | 0.2118 |
| `LIN-SVM_TF-IDF` | 258 | 22 | 43 | 127 | 0.0786 | 0.2529 |
| `SLP_TF-IDF` | 250 | 30 | 43 | 127 | 0.1071 | 0.2529 |

## Evaluation data

Models are evaluated on a fixed held-out test split of 450 rows:

- 280 non-extremist examples
- 170 extremist examples

Model configuration and threshold selection should be completed before test evaluation.

## Intended use

Appropriate uses:

- Research replication.
- Model-comparison experiments.
- Error analysis.
- Feature-attribution and interpretability research.
- Educational discussion of limitations in high-stakes NLP classification.

Unsupported uses:

- Automated moderation decisions.
- User-level risk scoring.
- Real-time enforcement or escalation systems.
- Claims about an individual's ideology, belief, or intent.
- Safety-critical deployment without extensive external validation.

## Key risks

- False positives may incorrectly flag benign, quoted, journalistic, academic, counterspeech, or contextual content.
- False negatives may miss implicit, coded, or emerging extremist language.
- Models may overfit lexical cues instead of understanding context.
- The dataset size limits robustness claims.
- Binary labels simplify a much more complex social and political phenomenon.

## Interpretability notes

Interpretable feature weights or attribution scores can help identify model behavior, but they should not be interpreted as proof that a model understands violent intent or ideology.

Any attribution analysis should be paired with manual error review.
