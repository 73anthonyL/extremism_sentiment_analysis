"""RQ1: which accuracy gaps are real, and do the models miss the same posts?

Reads every technique's committed probability artifact for one split
(research_loop/probs/<TECHNIQUE>__<split>.csv) and its locked threshold
(results_summary/<TECHNIQUE>/best_config.json), marks each post right or
wrong per technique at that threshold, and writes three text-free tables to
results_summary/rq/:

rq1_mcnemar_pairs.csv
    One row per technique pair: both-correct, both-wrong and discordant
    counts, the exact two-sided McNemar p-value over the discordant pairs,
    the Holm-adjusted p-value over every pair in the table, and the Jaccard
    overlap of the two error sets. `accuracy_gap_rows` is the number of test
    rows by which technique_a leads technique_b (negative when it trails).

rq1_error_overlap.csv
    One row per technique: errors, false positives, false negatives, how
    many of its errors no other technique makes, and how many at least one
    other technique also makes.

rq1_shared_errors.csv
    One row per k: how many posts are wrong for exactly k techniques, split
    by true label (a shared error on a negative post is a shared false
    positive).

Per-post rows are never committed, because results_summary/ may not carry
row ids. --per-post-out writes the correctness matrix to a path of your
choosing; keep it under external/.

A paired test is only valid when every technique was scored on the same
rows, so the tool refuses artifacts whose row-id sets or labels differ.
It reads the locked probabilities; it is not a new test evaluation.

USAGE
-----
    python3 tools/mcnemar_pairs.py
    python3 tools/mcnemar_pairs.py --techniques 01_LOG-REG_TF-IDF 07_TWITTER-ROBERTA_FINE-TUNE
    python3 tools/mcnemar_pairs.py --per-post-out external/rq1/correctness_test.csv
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probs_artifact import ProbsArtifactError, available_techniques, load_technique_probs
from repo_paths import (
    POSITIVE_LABEL,
    RESULTS_DIR,
    RQ1_ERROR_OVERLAP_CSV,
    RQ1_MCNEMAR_PAIRS_CSV,
    RQ1_SHARED_ERRORS_CSV,
)

DEFAULT_SPLIT = "test"
ALPHA = 0.05

PAIR_COLUMNS = (
    "technique_a",
    "technique_b",
    "split",
    "n_rows",
    "threshold_a",
    "threshold_b",
    "accuracy_a",
    "accuracy_b",
    "accuracy_gap_rows",
    "n_both_correct",
    "n_both_wrong",
    "n_only_a_wrong",
    "n_only_b_wrong",
    "n_discordant",
    "error_jaccard",
    "mcnemar_p",
    "holm_p",
    "significant_holm",
)

OVERLAP_COLUMNS = (
    "technique",
    "split",
    "threshold",
    "n_rows",
    "n_errors",
    "n_false_positives",
    "n_false_negatives",
    "n_errors_unique",
    "n_errors_shared",
    "share_errors_unique",
)

SHARED_COLUMNS = (
    "split",
    "n_techniques",
    "n_techniques_wrong",
    "n_posts",
    "n_negative_posts",
    "n_positive_posts",
    "share_of_posts",
)


class McNemarError(RuntimeError):
    pass


def load_threshold(technique, results_dir=None):
    base = RESULTS_DIR if results_dir is None else Path(results_dir)
    path = base / technique / "best_config.json"
    if not path.exists():
        raise McNemarError(f"{technique}: no best_config.json with a locked threshold under {base}")
    with open(path) as handle:
        config = json.load(handle)
    if "selected_threshold" not in config:
        raise McNemarError(f"{technique}: best_config.json has no selected_threshold")
    return float(config["selected_threshold"])


def correctness_matrix(techniques, split=DEFAULT_SPLIT, results_dir=None, thresholds=None):
    """Return (matrix, y_true, thresholds).

    `matrix` is a boolean DataFrame indexed by row_id with one column per
    technique (True where the technique classifies the row correctly at its
    locked threshold); `y_true` is the shared label Series. Refuses artifacts
    that do not cover exactly the same rows with the same labels.
    """
    if len(techniques) < 2:
        raise McNemarError("need at least two techniques for a paired comparison")
    thresholds = {} if thresholds is None else dict(thresholds)
    frames = {}
    for technique in techniques:
        frames[technique] = load_technique_probs(technique, split)
        if technique not in thresholds:
            thresholds[technique] = load_threshold(technique, results_dir)

    first = techniques[0]
    reference = frames[first].set_index("row_id")["y_true"]
    columns = {}
    for technique in techniques:
        frame = frames[technique].set_index("row_id")
        if set(frame.index) != set(reference.index):
            only_here = len(set(frame.index) - set(reference.index))
            only_ref = len(set(reference.index) - set(frame.index))
            raise McNemarError(
                f"{technique} and {first} were not scored on the same rows "
                f"({only_here} row ids only in {technique}, {only_ref} only in {first}); "
                "a paired comparison is invalid"
            )
        frame = frame.loc[reference.index]
        if (frame["y_true"].to_numpy() != reference.to_numpy()).any():
            raise McNemarError(f"{technique} and {first} disagree on y_true for some rows")
        predicted = (frame["y_prob"] >= thresholds[technique]).astype(int)
        columns[technique] = predicted.to_numpy() == frame["y_true"].to_numpy()

    matrix = pd.DataFrame(columns, index=reference.index)
    return matrix, reference, thresholds


def mcnemar_exact_p(n_only_a_wrong, n_only_b_wrong):
    """Exact two-sided McNemar p-value: a binomial test over the discordant pairs."""
    discordant = int(n_only_a_wrong + n_only_b_wrong)
    if discordant == 0:
        return 1.0
    return float(binomtest(int(n_only_a_wrong), discordant, 0.5, alternative="two-sided").pvalue)


def holm_adjust(p_values):
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(np.argsort(p, kind="stable")):
        value = min(1.0, (m - rank) * p[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def pairwise_table(matrix, split, thresholds):
    techniques = list(matrix.columns)
    n_rows = len(matrix)
    rows = []
    for a, b in itertools.combinations(techniques, 2):
        correct_a, correct_b = matrix[a].to_numpy(), matrix[b].to_numpy()
        both_correct = int((correct_a & correct_b).sum())
        both_wrong = int((~correct_a & ~correct_b).sum())
        only_a_wrong = int((~correct_a & correct_b).sum())
        only_b_wrong = int((correct_a & ~correct_b).sum())
        union_wrong = both_wrong + only_a_wrong + only_b_wrong
        rows.append(
            {
                "technique_a": a,
                "technique_b": b,
                "split": split,
                "n_rows": n_rows,
                "threshold_a": thresholds[a],
                "threshold_b": thresholds[b],
                "accuracy_a": float(correct_a.mean()),
                "accuracy_b": float(correct_b.mean()),
                "accuracy_gap_rows": only_b_wrong - only_a_wrong,
                "n_both_correct": both_correct,
                "n_both_wrong": both_wrong,
                "n_only_a_wrong": only_a_wrong,
                "n_only_b_wrong": only_b_wrong,
                "n_discordant": only_a_wrong + only_b_wrong,
                "error_jaccard": float(both_wrong / union_wrong) if union_wrong else float("nan"),
                "mcnemar_p": mcnemar_exact_p(only_a_wrong, only_b_wrong),
            }
        )
    table = pd.DataFrame(rows, columns=[c for c in PAIR_COLUMNS if c not in ("holm_p", "significant_holm")])
    table["holm_p"] = holm_adjust(table["mcnemar_p"]) if len(table) else []
    table["significant_holm"] = table["holm_p"] < ALPHA
    return table[list(PAIR_COLUMNS)]


def per_technique_table(matrix, y_true, split, thresholds):
    wrong = ~matrix
    n_wrong_per_row = wrong.sum(axis=1)
    rows = []
    for technique in matrix.columns:
        errors = wrong[technique]
        n_errors = int(errors.sum())
        unique = int((errors & (n_wrong_per_row == 1)).sum())
        rows.append(
            {
                "technique": technique,
                "split": split,
                "threshold": thresholds[technique],
                "n_rows": len(matrix),
                "n_errors": n_errors,
                "n_false_positives": int((errors & (y_true != POSITIVE_LABEL)).sum()),
                "n_false_negatives": int((errors & (y_true == POSITIVE_LABEL)).sum()),
                "n_errors_unique": unique,
                "n_errors_shared": n_errors - unique,
                "share_errors_unique": float(unique / n_errors) if n_errors else float("nan"),
            }
        )
    return pd.DataFrame(rows, columns=list(OVERLAP_COLUMNS))


def shared_errors_table(matrix, y_true, split):
    n_techniques = matrix.shape[1]
    n_wrong_per_row = (~matrix).sum(axis=1)
    rows = []
    for k in range(n_techniques + 1):
        mask = n_wrong_per_row == k
        rows.append(
            {
                "split": split,
                "n_techniques": n_techniques,
                "n_techniques_wrong": k,
                "n_posts": int(mask.sum()),
                "n_negative_posts": int((mask & (y_true != POSITIVE_LABEL)).sum()),
                "n_positive_posts": int((mask & (y_true == POSITIVE_LABEL)).sum()),
                "share_of_posts": float(mask.mean()),
            }
        )
    return pd.DataFrame(rows, columns=list(SHARED_COLUMNS))


def build(techniques=None, split=DEFAULT_SPLIT, results_dir=None, thresholds=None):
    """Return (pairs, overlap, shared, matrix, y_true) for the techniques."""
    techniques = list(techniques) if techniques else available_techniques(split)
    matrix, y_true, thresholds = correctness_matrix(techniques, split, results_dir, thresholds)
    pairs = pairwise_table(matrix, split, thresholds)
    overlap = per_technique_table(matrix, y_true, split, thresholds)
    shared = shared_errors_table(matrix, y_true, split)
    return pairs, overlap, shared, matrix, y_true


def write_tables(pairs, overlap, shared, pairs_path=RQ1_MCNEMAR_PAIRS_CSV,
                 overlap_path=RQ1_ERROR_OVERLAP_CSV, shared_path=RQ1_SHARED_ERRORS_CSV):
    for table, path in ((pairs, pairs_path), (overlap, overlap_path), (shared, shared_path)):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False, float_format="%.6g")
    return Path(pairs_path), Path(overlap_path), Path(shared_path)


def write_per_post(matrix, y_true, path):
    """Write the row-keyed correctness matrix outside the committed tree."""
    path = Path(path)
    if RESULTS_DIR in path.resolve().parents:
        raise McNemarError("the per-post matrix carries row ids and may not be written under results_summary/")
    path.parent.mkdir(parents=True, exist_ok=True)
    out = matrix.astype(int).copy()
    out.insert(0, "y_true", y_true.to_numpy())
    out.reset_index().to_csv(path, index=False)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--techniques", nargs="*", default=None,
                        help="technique names; default: every technique with an artifact for the split")
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--per-post-out", default=None,
                        help="also write the row-keyed correctness matrix here (keep it under external/)")
    args = parser.parse_args()

    try:
        pairs, overlap, shared, matrix, y_true = build(args.techniques, args.split)
        if args.per_post_out:
            write_per_post(matrix, y_true, args.per_post_out)
    except (ProbsArtifactError, McNemarError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    paths = write_tables(pairs, overlap, shared)
    for _, row in overlap.iterrows():
        print(f"{row['technique']}: {row['n_errors']} errors "
              f"({row['n_false_positives']} FP, {row['n_false_negatives']} FN), "
              f"{row['n_errors_unique']} made by no other technique")
    for _, row in shared.iterrows():
        print(f"wrong for {row['n_techniques_wrong']} of {row['n_techniques']} techniques: "
              f"{row['n_posts']} posts ({row['n_negative_posts']} negative, {row['n_positive_posts']} positive)")
    n_sig = int(pairs["significant_holm"].sum())
    print(f"{len(pairs)} pairs; {n_sig} significant after Holm at alpha={ALPHA}")
    for path in paths:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
