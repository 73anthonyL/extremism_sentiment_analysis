"""Derive a complete results_summary/<TECHNIQUE>/ folder from probability artifacts.

This is what replaces hand-copying numbers out of Kaggle output. Given committed
probability artifacts, it produces every metric file the schema requires, using
the same `compute_binary_metrics` the notebooks use, so derived numbers equal
notebook numbers.

Two refusals are load-bearing:

* Threshold selection over the test split is a hard error. The threshold is a
  fitted parameter; fitting it on test is the single most consequential way to
  invalidate a held-out result, so it is blocked in code rather than in review.
* Any probability artifact carrying a text column is rejected upstream in
  probs_artifact.load_probs.

USAGE
-----
    python3 tools/eval_from_probs.py --technique 08_TWITTER-ROBERTA_SEED-ENSEMBLE
    python3 tools/eval_from_probs.py --technique X --threshold 0.585   # use a fixed threshold
    python3 tools/eval_from_probs.py --technique X --dry-run           # print, write nothing
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics_core import compute_binary_metrics, select_threshold
from probs_artifact import load_technique_probs, read_meta
from repo_paths import DATASET_VERSION, NEGATIVE_CLASS_NAME, POSITIVE_CLASS_NAME, RANDOM_SEED, RESULTS_DIR, SPLIT_VERSION, VAL_LOG_JSONL


# Threshold policy.
#
# The sweep grid is shared by every technique so sweep CSVs are comparable
# row-for-row. 0.05..0.95 in 0.005 steps is fine enough that neighbouring
# candidates on the 450-row validation split differ by at most one row's
# worth of any counting metric, so the sweep cannot skip past the discrete
# optimum, and coarse enough that the committed sweep CSV stays small and
# reviewable.
# The round() keeps the selected threshold a stable three-decimal value in
# best_config.json instead of a float repr artifact of linspace.
THRESHOLD_GRID = [round(x, 3) for x in np.linspace(0.05, 0.95, 181)]

# The metric maximized when selecting a threshold. Accuracy is the repo's
# headline comparison metric (see docs/EXPERIMENTS.md); --threshold-metric
# overrides it per run.
DEFAULT_THRESHOLD_METRIC = "accuracy"

SELECTION_SPLIT = "validation"
EVALUATION_SPLIT = "test"


class TestSplitSelectionError(RuntimeError):
    """Raised on any attempt to fit a parameter using the held-out test split."""


def derive_threshold(validation_probs, metric=DEFAULT_THRESHOLD_METRIC):
    """Select a decision threshold on the validation split only.

    Returns (threshold, sweep_frame). Refuses non-validation input: the guard is
    here, at the only place a threshold is ever chosen.
    """
    split_values = set(validation_probs["split"].unique())
    if split_values != {SELECTION_SPLIT}:
        raise TestSplitSelectionError(
            f"threshold selection attempted on split(s) {sorted(split_values)}; only "
            f"'{SELECTION_SPLIT}' may be used to fit a threshold"
        )
    return select_threshold(
        validation_probs["y_true"].values,
        validation_probs["y_prob"].values,
        THRESHOLD_GRID,
        metric=metric,
    )


def confusion_matrix_frame(metrics):
    """Build the long-form confusion matrix CSV documented in RESULTS_SCHEMA.md.

    docs/RESULTS_SCHEMA.md specifies `actual,predicted,count`; committed folders
    currently use three different dialects. New folders use the documented one.
    """
    return pd.DataFrame(
        [
            {"actual": NEGATIVE_CLASS_NAME, "predicted": NEGATIVE_CLASS_NAME, "count": metrics["tn"]},
            {"actual": NEGATIVE_CLASS_NAME, "predicted": POSITIVE_CLASS_NAME, "count": metrics["fp"]},
            {"actual": POSITIVE_CLASS_NAME, "predicted": NEGATIVE_CLASS_NAME, "count": metrics["fn"]},
            {"actual": POSITIVE_CLASS_NAME, "predicted": POSITIVE_CLASS_NAME, "count": metrics["tp"]},
        ]
    )


def classification_report_payload(metrics):
    """Per-class precision/recall/F1/support, derived from confusion counts."""
    tn, fp, fn = metrics["tn"], metrics["fp"], metrics["fn"]
    negative_precision = tn / (tn + fn) if (tn + fn) else 0.0
    negative_recall = tn / (tn + fp) if (tn + fp) else 0.0
    negative_f1 = (
        2 * negative_precision * negative_recall / (negative_precision + negative_recall)
        if (negative_precision + negative_recall)
        else 0.0
    )
    return {
        NEGATIVE_CLASS_NAME: {
            "precision": negative_precision,
            "recall": negative_recall,
            "f1-score": negative_f1,
            "support": metrics["negative_support"],
        },
        POSITIVE_CLASS_NAME: {
            "precision": metrics["positive_precision"],
            "recall": metrics["positive_recall"],
            "f1-score": metrics["positive_f1"],
            "support": metrics["positive_support"],
        },
        "accuracy": metrics["accuracy"],
        "macro avg": {
            "precision": metrics["precision_macro"],
            "recall": metrics["recall_macro"],
            "f1-score": metrics["f1_macro"],
            "support": metrics["support"],
        },
        "weighted avg": {
            "precision": metrics["precision_weighted"],
            "recall": metrics["recall_weighted"],
            "f1-score": metrics["f1_weighted"],
            "support": metrics["support"],
        },
    }


def log_validation_evaluation(technique, threshold, metrics, metric_name):
    """Append to the validation evaluation log for multiplicity accounting.

    Every validation evaluation is recorded so the number of configurations
    actually screened can be compared against what a preregistration declared.
    Selection effects on a 450-row validation split are large; this makes the
    screening count auditable instead of remembered.
    """
    VAL_LOG_JSONL.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "technique": technique,
        "threshold": threshold,
        "threshold_metric": metric_name,
        "validation_accuracy": metrics["accuracy"],
        "validation_positive_f1": metrics["positive_f1"],
        "validation_pr_auc": metrics["pr_auc"],
    }
    with open(VAL_LOG_JSONL, "a") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def build_results_folder(
    technique,
    threshold=None,
    threshold_metric=DEFAULT_THRESHOLD_METRIC,
    dry_run=False,
):
    """Produce the full results_summary/<technique>/ artifact set.

    When `threshold` is None it is selected on validation. When supplied (e.g. to
    reproduce a historical run's published threshold) it is used as given and
    recorded as externally supplied.
    """
    validation_probs = load_technique_probs(technique, SELECTION_SPLIT)
    test_probs = load_technique_probs(technique, EVALUATION_SPLIT)

    if threshold is None:
        threshold, sweep = derive_threshold(validation_probs, metric=threshold_metric)
        threshold_strategy = f"maximize_validation_{threshold_metric}"
    else:
        _, sweep = select_threshold(
            validation_probs["y_true"].values,
            validation_probs["y_prob"].values,
            THRESHOLD_GRID,
            metric=threshold_metric,
        )
        threshold_strategy = "externally_supplied"

    validation_metrics = compute_binary_metrics(
        validation_probs["y_true"].values, validation_probs["y_prob"].values, threshold
    )
    test_metrics = compute_binary_metrics(
        test_probs["y_true"].values, test_probs["y_prob"].values, threshold
    )

    for metrics, split_name in ((validation_metrics, SELECTION_SPLIT), (test_metrics, EVALUATION_SPLIT)):
        metrics["technique"] = technique
        metrics["split"] = split_name

    meta = read_meta(technique)
    best_config = {
        "technique": technique,
        "model_family": meta.get("model_family", "unknown"),
        "feature_family": meta.get("feature_family", "unknown"),
        "random_seed": meta.get("random_seed", RANDOM_SEED),
        "dataset_version": meta.get("dataset_version", DATASET_VERSION),
        "split_version": meta.get("split_version", SPLIT_VERSION),
        "hyperparameters": meta.get("hyperparameters", {}),
        "threshold_strategy": threshold_strategy,
        "threshold_metric": threshold_metric,
        "selected_threshold": threshold,
        "threshold_selected_on_split": SELECTION_SPLIT,
        "test_set_used_for_selection": False,
        "derived_by": "tools/eval_from_probs.py",
    }

    if dry_run:
        print(f"[dry-run] {technique}: threshold={threshold} ({threshold_strategy})")
        print(f"[dry-run]   validation accuracy={validation_metrics['accuracy']:.4f}")
        print(f"[dry-run]   test accuracy={test_metrics['accuracy']:.4f}")
        return best_config, validation_metrics, test_metrics

    output_dir = RESULTS_DIR / technique
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_json(output_dir / "best_config.json", best_config)
    _write_json(output_dir / "metrics_validation.json", validation_metrics)
    _write_json(output_dir / "metrics_test.json", test_metrics)
    _write_json(
        output_dir / "classification_report_test.json",
        classification_report_payload(test_metrics),
    )
    confusion_matrix_frame(test_metrics).to_csv(
        output_dir / "confusion_matrix_test.csv", index=False
    )
    sweep.sort_values("threshold").to_csv(
        output_dir / "threshold_sweep_validation.csv", index=False
    )

    log_validation_evaluation(technique, threshold, validation_metrics, threshold_metric)

    print(f"Wrote {output_dir}")
    print(f"  threshold {threshold} ({threshold_strategy})")
    print(f"  validation accuracy {validation_metrics['accuracy']:.4f}")
    print(f"  test accuracy       {test_metrics['accuracy']:.4f}")
    return best_config, validation_metrics, test_metrics


def _write_json(path, payload):
    """Write JSON with NaN rendered as null so the file stays strictly valid."""
    cleaned = {
        key: None if isinstance(value, float) and np.isnan(value) else value
        for key, value in payload.items()
    } if isinstance(payload, dict) else payload
    with open(path, "w") as handle:
        json.dump(cleaned, handle, indent=2, sort_keys=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--technique", required=True, help="technique id, e.g. 08_TWITTER-ROBERTA_SEED-ENSEMBLE")
    parser.add_argument("--threshold", type=float, default=None, help="use this threshold instead of selecting one")
    parser.add_argument("--threshold-metric", default=DEFAULT_THRESHOLD_METRIC)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        build_results_folder(
            args.technique,
            threshold=args.threshold,
            threshold_metric=args.threshold_metric,
            dry_run=args.dry_run,
        )
    except (TestSplitSelectionError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
