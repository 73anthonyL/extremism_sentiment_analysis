"""Schema and internal-consistency gate for results_summary/<TECHNIQUE>/ folders.

The decisive check here is not "are the required fields present" but "do the
stored metrics agree with the stored confusion matrix". A hand-edited or
mis-transcribed number will pass a presence check and fail a recomputation
check, because accuracy, precision, recall, F1, FPR and FNR are all determined
by (tn, fp, fn, tp). Any disagreement means the metrics and the confusion matrix
describe different predictions.

USAGE
-----
    python3 tools/validate_results_folder.py --all
    python3 tools/validate_results_folder.py --technique 01_LOG-REG_TF-IDF
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics_core import (
    REQUIRED_METRIC_FIELDS,
    read_metric,
    recompute_from_confusion,
)
from repo_paths import (
    EXPECTED_SPLIT_COUNTS,
    RESULTS_DIR,
    technique_dirs,
)

# Stored metrics are rounded to six decimal places, so any disagreement past
# this tolerance is a real inconsistency between the metrics and the confusion
# matrix, not floating-point noise.
RECOMPUTE_TOLERANCE = 1e-6

REQUIRED_FILES = ("best_config.json", "metrics_validation.json", "metrics_test.json")

# Either file satisfies the confusion-matrix requirement; the CSV is preferred
# because it is the one the consistency checks can read.
CONFUSION_FILES = ("confusion_matrix_test.csv", "confusion_matrix_test.png")

EXIT_OK = 0
EXIT_FAILED = 1


def _load_json(path):
    with open(path) as handle:
        return json.load(handle)


def check_metric_fields(metrics, label, problems):
    """Verify every required field is readable, tolerating documented aliases."""
    for field in REQUIRED_METRIC_FIELDS:
        try:
            read_metric(metrics, field)
        except KeyError:
            problems.append(f"{label}: missing required metric field '{field}'")


def check_internal_consistency(metrics, label, problems):
    """Recompute threshold-dependent metrics from confusion counts and compare."""
    try:
        counts = [int(metrics[k]) for k in ("tn", "fp", "fn", "tp")]
    except KeyError:
        problems.append(f"{label}: confusion counts tn/fp/fn/tp missing; cannot verify")
        return

    derived = recompute_from_confusion(*counts)
    for field, expected in derived.items():
        try:
            stored = float(read_metric(metrics, field))
        except (KeyError, TypeError):
            continue
        if abs(stored - expected) > RECOMPUTE_TOLERANCE:
            problems.append(
                f"{label}: '{field}' stored as {stored:.6f} but the confusion matrix ("
                f"{counts[0]},{counts[1]},{counts[2]},{counts[3]}) implies {expected:.6f}"
            )


def check_split_shape(metrics, split, label, problems):
    """Assert the split's row and class counts match the frozen protocol."""
    if split not in EXPECTED_SPLIT_COUNTS:
        return
    expected_total, expected_neg, expected_pos = EXPECTED_SPLIT_COUNTS[split]

    total = sum(int(metrics[k]) for k in ("tn", "fp", "fn", "tp") if k in metrics)
    if total and total != expected_total:
        problems.append(
            f"{label}: confusion counts sum to {total}, but the {split} split has "
            f"{expected_total} rows"
        )

    try:
        positive_support = int(read_metric(metrics, "positive_support"))
        if positive_support != expected_pos:
            problems.append(
                f"{label}: positive_support {positive_support} != frozen {expected_pos} for the "
                f"{split} split"
            )
    except (KeyError, TypeError):
        pass

    try:
        negative_support = int(read_metric(metrics, "negative_support"))
        if negative_support != expected_neg:
            problems.append(
                f"{label}: negative_support {negative_support} != frozen {expected_neg} for the "
                f"{split} split"
            )
    except (KeyError, TypeError):
        pass


def check_confusion_csv(folder, metrics_test, problems):
    """Cross-check the confusion CSV against metrics_test.json, if a CSV exists."""
    csv_path = folder / "confusion_matrix_test.csv"
    if not csv_path.exists():
        if not (folder / "confusion_matrix_test.png").exists():
            problems.append(
                f"{folder.name}: no confusion matrix in any of {CONFUSION_FILES}"
            )
        return

    frame = pd.read_csv(csv_path)
    total_csv = int(frame["count"].sum()) if "count" in frame.columns else None
    if total_csv is None:
        # No tidy 'count' column: fall back to summing every numeric cell.
        numeric = frame.select_dtypes("number")
        total_csv = int(numeric.values.sum())

    total_json = sum(int(metrics_test[k]) for k in ("tn", "fp", "fn", "tp") if k in metrics_test)
    if total_json and total_csv != total_json:
        problems.append(
            f"{folder.name}: confusion_matrix_test.csv totals {total_csv} but metrics_test.json totals "
            f"{total_json}"
        )


def check_threshold_agreement(folder, best_config, metrics_test, problems):
    """The locked threshold must be the one the test metrics were computed at."""
    configured = best_config.get("selected_threshold")
    evaluated = metrics_test.get("threshold")
    if configured is None or evaluated is None:
        return
    if abs(float(configured) - float(evaluated)) > RECOMPUTE_TOLERANCE:
        problems.append(
            f"{folder.name}: best_config selected_threshold {configured} != metrics_test threshold "
            f"{evaluated}"
        )


def check_selection_hygiene(best_config, folder, problems):
    """Flag any config that records the test split as having informed selection."""
    if best_config.get("test_set_used_for_selection") is True:
        problems.append(
            f"{folder.name}: best_config records test_set_used_for_selection=true"
        )
    selected_on = best_config.get("threshold_selected_on_split")
    if selected_on is not None and selected_on != "validation":
        problems.append(
            f"{folder.name}: threshold selected on '{selected_on}', not validation"
        )


def validate_folder(folder):
    """Run every check against one result folder. Returns a problems list."""
    problems = []

    for filename in REQUIRED_FILES:
        if not (folder / filename).exists():
            problems.append(f"{folder.name}: missing required file {filename}")

    metrics_test_path = folder / "metrics_test.json"
    metrics_validation_path = folder / "metrics_validation.json"
    best_config_path = folder / "best_config.json"

    metrics_test = _load_json(metrics_test_path) if metrics_test_path.exists() else None
    metrics_validation = (
        _load_json(metrics_validation_path) if metrics_validation_path.exists() else None
    )
    best_config = _load_json(best_config_path) if best_config_path.exists() else None

    if metrics_test is not None:
        label = f"{folder.name}/metrics_test.json"
        check_metric_fields(metrics_test, label, problems)
        check_internal_consistency(metrics_test, label, problems)
        check_split_shape(metrics_test, "test", label, problems)
        check_confusion_csv(folder, metrics_test, problems)

    if metrics_validation is not None:
        label = f"{folder.name}/metrics_validation.json"
        check_metric_fields(metrics_validation, label, problems)
        check_internal_consistency(metrics_validation, label, problems)
        check_split_shape(metrics_validation, "validation", label, problems)

    if best_config is not None:
        check_selection_hygiene(best_config, folder, problems)
        if metrics_test is not None:
            check_threshold_agreement(folder, best_config, metrics_test, problems)

    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--technique", default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.technique:
        folders = [RESULTS_DIR / args.technique]
    elif args.all:
        folders = technique_dirs()
    else:
        parser.error("pass --technique <id> or --all")
        return EXIT_FAILED

    total_problems = 0
    for folder in folders:
        if not folder.exists():
            print(f"FAIL  {folder.name}: folder does not exist")
            total_problems += 1
            continue
        problems = validate_folder(folder)
        status = "PASS" if not problems else "FAIL"
        print(f"{status}  {folder.name}  ({len(problems)} problem(s))")
        for problem in problems:
            print(f"        {problem}")
        total_problems += len(problems)

    print(f"\n{total_problems} problem(s) across {len(folders)} folder(s)")
    return EXIT_OK if total_problems == 0 else EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
