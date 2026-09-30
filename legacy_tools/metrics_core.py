"""Metric computation shared by the toolkit and the experiment notebooks.

`compute_binary_metrics` is copied VERBATIM from notebook 07 cell 6
(`notebooks/07_TWITTER-ROBERTA_FINE-TUNE.ipynb`), with the single change that
`positive_label` is a plain default argument instead of reading a notebook-global
`CONFIG` dict. Keeping it byte-identical is deliberate: locally derived metrics
must equal the numbers the notebooks produced on Kaggle, or the whole premise
that results_summary/ can be regenerated from probability artifacts collapses.

Do not "improve" this function. If it must change, change it in the notebooks
first and re-derive every affected result folder.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from repo_paths import POSITIVE_LABEL

# The metric fields every metrics_validation.json / metrics_test.json must
# carry (docs/RESULTS_SCHEMA.md). Kept as a tuple so validators can iterate
# it without mutating it.
REQUIRED_METRIC_FIELDS = (
    "threshold",
    "support",
    "positive_support",
    "negative_support",
    "accuracy",
    "balanced_accuracy",
    "positive_precision",
    "positive_recall",
    "positive_f1",
    "f1_macro",
    "f1_weighted",
    "roc_auc",
    "pr_auc",
    "tn",
    "fp",
    "fn",
    "tp",
    "false_positive_rate",
    "false_negative_rate",
)

# Older artifacts used different spellings for a few fields; readers accept
# these aliases so historical folders still validate, but writers must emit
# the canonical names.
METRIC_ALIASES = {
    "f1_macro": ("macro_f1",),
    "f1_weighted": ("weighted_f1",),
    "false_positive_rate": ("fpr",),
    "false_negative_rate": ("fnr",),
}


def safe_roc_auc(y_true, y_prob):
    """ROC-AUC that returns NaN rather than raising on a single-class input."""
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, y_prob)
    except Exception:
        return np.nan


def safe_pr_auc(y_true, y_prob):
    """Average precision that returns NaN rather than raising on a single-class input."""
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return average_precision_score(y_true, y_prob)
    except Exception:
        return np.nan


def compute_binary_metrics(y_true, y_prob, threshold, positive_label=POSITIVE_LABEL):
    """Compute the full standard metric set for one split at one threshold.

    Verbatim from notebook 07 cell 6. Returns the superset of fields required by
    docs/RESULTS_SCHEMA.md; predictions are `y_prob >= threshold`.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()

    metrics = {
        "threshold": float(threshold),
        "support": int(len(y_true)),
        "positive_support": int((y_true == positive_label).sum()),
        "negative_support": int((y_true != positive_label).sum()),
        "positive_rate": float((y_true == positive_label).mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "positive_precision": float(
            precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
        ),
        "positive_recall": float(
            recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
        ),
        "positive_f1": float(f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
        "roc_auc": float(safe_roc_auc(y_true, y_prob)),
        "pr_auc": float(safe_pr_auc(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else np.nan,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else np.nan,
    }
    return metrics


def select_threshold(y_true, y_prob, thresholds, metric="accuracy"):
    """Pick the threshold maximizing `metric`, with deterministic tie-breaking.

    Ties break on positive_recall, then positive_precision, then
    balanced_accuracy -- matching notebook 07's `select_threshold` so that a
    locally re-derived threshold equals the notebook's.

    The default metric is "accuracy" rather than notebook 07's "positive_f1":
    the research loop's promotion metric is test accuracy, so thresholds are
    selected for the metric the technique is judged on.

    Returns (selected_threshold, sweep_dataframe).
    """
    rows = [compute_binary_metrics(y_true, y_prob, t) for t in thresholds]
    sweep = pd.DataFrame(rows)
    sweep = sweep.sort_values(
        by=[metric, "positive_recall", "positive_precision", "balanced_accuracy"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return float(sweep.iloc[0]["threshold"]), sweep


def read_metric(metrics, field):
    """Read a metric field, tolerating the alias spellings in older artifacts.

    Raises KeyError naming the canonical field if neither it nor any alias is
    present, so schema failures identify the field the writer should have used.
    """
    if field in metrics:
        return metrics[field]
    for alias in METRIC_ALIASES.get(field, ()):
        if alias in metrics:
            return metrics[alias]
    raise KeyError(field)


def recompute_from_confusion(tn, fp, fn, tp):
    """Derive threshold-dependent metrics from confusion counts alone.

    Used to cross-check stored values: any disagreement means the stored metrics
    and the stored confusion matrix describe different predictions, which is the
    signature of a hand-edited or mis-transcribed result file.
    """
    total = tn + fp + fn + tp
    predicted_positive = tp + fp
    actual_positive = tp + fn
    actual_negative = tn + fp

    precision = tp / predicted_positive if predicted_positive else 0.0
    recall = tp / actual_positive if actual_positive else 0.0
    specificity = tn / actual_negative if actual_negative else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "support": total,
        "positive_support": actual_positive,
        "negative_support": actual_negative,
        "accuracy": (tp + tn) / total if total else 0.0,
        "balanced_accuracy": (recall + specificity) / 2,
        "positive_precision": precision,
        "positive_recall": recall,
        "positive_f1": f1,
        "false_positive_rate": fp / actual_negative if actual_negative else 0.0,
        "false_negative_rate": fn / actual_positive if actual_positive else 0.0,
    }
