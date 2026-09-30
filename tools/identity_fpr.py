"""RQ4: false-positive rate on non-extremist posts that mention identity terms.

The shortcut hypothesis predicts that a model leaning on identity words will
flag benign posts that merely mention a community. This tool measures that,
per technique, from two committed inputs and one local one:

* the technique's probability artifact for the split (research_loop/probs/),
* its locked threshold (results_summary/<TECHNIQUE>/best_config.json), and
* the dataset text, rebuilt locally through split_protocol so row ids line up.

For each row the artifact's y_true is checked against the dataset label; any
disagreement aborts, because it means the artifact and the dataset are not the
same rows. Text is read into memory to test for identity terms and never
written out; the outputs carry only counts, rates, and the matching terms.

Outputs, in results_summary/<TECHNIQUE>/:
    identity_fpr_<split>.json         summary rates and a Fisher exact test
    identity_fpr_terms_<split>.csv    per-term counts (terms with enough support)
and a cross-technique table results_summary/rq/rq4_identity_fpr.csv.

USAGE
-----
    python3 tools/identity_fpr.py --all
    python3 tools/identity_fpr.py --technique 11_MULTI-CHECKPOINT_LOGIT-POOL --split test
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lexicons import load_lexicons, phrase_pattern
from probs_artifact import ProbsArtifactError, load_technique_probs
from repo_paths import (
    IDENTITY_FPR_FILENAME,
    IDENTITY_FPR_TERMS_FILENAME,
    POSITIVE_LABEL,
    RESULTS_DIR,
    RQ4_IDENTITY_FPR_CSV,
    technique_dirs,
)
from split_protocol import build_processed_dataset

DEFAULT_SPLIT = "test"
# A per-term row is only reported when at least this many non-extremist posts
# contain the term; below that a rate is noise.
DEFAULT_MIN_TERM_SUPPORT = 5

SUMMARY_COLUMNS = (
    "technique",
    "split",
    "threshold",
    "n_negative",
    "n_negative_with_identity_terms",
    "fpr_with_identity_terms",
    "fpr_without_identity_terms",
    "fpr_overall",
    "fpr_ratio",
    "fisher_odds_ratio",
    "fisher_p_value",
    "n_positive_with_identity_terms",
    "recall_with_identity_terms",
    "recall_without_identity_terms",
)


class IdentityFPRError(RuntimeError):
    pass


def load_threshold(technique, results_dir=None):
    base = RESULTS_DIR if results_dir is None else Path(results_dir)
    path = base / technique / "best_config.json"
    if not path.exists():
        raise IdentityFPRError(f"no best_config.json for {technique}; pass --threshold")
    with open(path) as handle:
        return float(json.load(handle)["selected_threshold"])


def identity_matches(texts, terms):
    """For each text, the sorted list of identity terms it contains."""
    pattern = phrase_pattern(terms)
    if pattern is None:
        return [[] for _ in texts]
    return [sorted({m.lower() for m in pattern.findall(t)}) for t in texts]


def _rate(numerator, denominator):
    return float(numerator / denominator) if denominator else float("nan")


def analyze(technique, split=DEFAULT_SPLIT, threshold=None, processed=None, probs=None,
            terms=None, min_term_support=DEFAULT_MIN_TERM_SUPPORT, results_dir=None):
    """Compute the RQ4 summary and per-term table for one technique.

    `processed`, `probs`, and `terms` can be injected (tests); by default they
    come from the dataset, the committed artifact, and the identity lexicon.
    """
    processed = build_processed_dataset() if processed is None else processed
    probs = load_technique_probs(technique, split) if probs is None else probs
    terms = load_lexicons()["identity_term"] if terms is None else terms
    threshold = load_threshold(technique, results_dir) if threshold is None else float(threshold)

    merged = probs.merge(processed[["row_id", "text", "label"]], on="row_id", how="left")
    if merged["text"].isna().any():
        raise IdentityFPRError(f"{technique}: some row_ids in the artifact are not in the dataset")
    if (merged["y_true"] != merged["label"]).any():
        raise IdentityFPRError(
            f"{technique}: y_true disagrees with the dataset label for "
            f"{int((merged['y_true'] != merged['label']).sum())} rows; wrong artifact or dataset version"
        )

    merged["pred"] = (merged["y_prob"] >= threshold).astype(int)
    merged["terms"] = identity_matches(merged["text"].tolist(), terms)
    merged["has_identity"] = merged["terms"].map(bool)

    negative = merged[merged["y_true"] != POSITIVE_LABEL]
    positive = merged[merged["y_true"] == POSITIVE_LABEL]
    neg_with = negative[negative["has_identity"]]
    neg_without = negative[~negative["has_identity"]]
    pos_with = positive[positive["has_identity"]]
    pos_without = positive[~positive["has_identity"]]

    fp_with, fp_without = int(neg_with["pred"].sum()), int(neg_without["pred"].sum())
    table = [[fp_with, len(neg_with) - fp_with], [fp_without, len(neg_without) - fp_without]]
    if min(len(neg_with), len(neg_without)) > 0:
        odds, p_value = fisher_exact(table)
    else:
        odds, p_value = float("nan"), float("nan")

    fpr_with = _rate(fp_with, len(neg_with))
    fpr_without = _rate(fp_without, len(neg_without))
    summary = {
        "technique": technique,
        "split": split,
        "threshold": threshold,
        "n_negative": int(len(negative)),
        "n_negative_with_identity_terms": int(len(neg_with)),
        "fpr_with_identity_terms": fpr_with,
        "fpr_without_identity_terms": fpr_without,
        "fpr_overall": _rate(int(negative["pred"].sum()), len(negative)),
        "fpr_ratio": (fpr_with / fpr_without) if fpr_without and not np.isnan(fpr_without) else float("nan"),
        "fisher_odds_ratio": float(odds),
        "fisher_p_value": float(p_value),
        "n_positive_with_identity_terms": int(len(pos_with)),
        "recall_with_identity_terms": _rate(int(pos_with["pred"].sum()), len(pos_with)),
        "recall_without_identity_terms": _rate(int(pos_without["pred"].sum()), len(pos_without)),
        "min_term_support": int(min_term_support),
        "n_identity_terms_in_lexicon": int(len(terms)),
    }

    per_term = (
        neg_with[["terms", "pred"]]
        .explode("terms")
        .groupby("terms")["pred"]
        .agg(n_negative_posts="size", n_false_positives="sum")
        .reset_index()
        .rename(columns={"terms": "term"})
    )
    per_term = per_term[per_term["n_negative_posts"] >= min_term_support].copy()
    per_term["fpr"] = per_term["n_false_positives"] / per_term["n_negative_posts"]
    per_term = per_term.sort_values(["fpr", "n_negative_posts"], ascending=[False, False]).reset_index(drop=True)
    return summary, per_term


def write_outputs(summary, per_term, results_dir=None):
    base = RESULTS_DIR if results_dir is None else Path(results_dir)
    folder = base / summary["technique"]
    folder.mkdir(parents=True, exist_ok=True)
    json_path = folder / IDENTITY_FPR_FILENAME.format(split=summary["split"])
    cleaned = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in summary.items()}
    with open(json_path, "w") as handle:
        json.dump(cleaned, handle, indent=2, sort_keys=True)
    per_term.to_csv(folder / IDENTITY_FPR_TERMS_FILENAME.format(split=summary["split"]),
                    index=False, float_format="%.6g")
    return json_path


def write_rq4_table(summaries, path=RQ4_IDENTITY_FPR_CSV):
    """Merge summaries into the cross-technique table, replacing same technique/split rows."""
    new = pd.DataFrame(summaries)[list(SUMMARY_COLUMNS)]
    path = Path(path)
    if path.exists():
        existing = pd.read_csv(path)
        keep = ~existing.set_index(["technique", "split"]).index.isin(
            new.set_index(["technique", "split"]).index
        )
        new = pd.concat([existing[keep], new], ignore_index=True)
    new = new.sort_values(["split", "technique"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False, float_format="%.6g")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--technique")
    group.add_argument("--all", action="store_true", help="every technique with an artifact for the split")
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min-term-support", type=int, default=DEFAULT_MIN_TERM_SUPPORT)
    args = parser.parse_args()

    techniques = [args.technique] if args.technique else [d.name for d in technique_dirs()]
    processed = build_processed_dataset()
    terms = load_lexicons()["identity_term"]
    if not terms:
        print("ERROR: identity lexicon is empty", file=sys.stderr)
        return 1

    summaries = []
    for technique in techniques:
        try:
            summary, per_term = analyze(
                technique, args.split, args.threshold, processed=processed, terms=terms,
                min_term_support=args.min_term_support,
            )
        except ProbsArtifactError as error:
            if args.all:
                print(f"skip {technique}: {error}")
                continue
            print(f"ERROR: {error}", file=sys.stderr)
            return 1
        except IdentityFPRError as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 1
        write_outputs(summary, per_term)
        summaries.append(summary)
        print(
            f"{technique}: FPR with identity terms {summary['fpr_with_identity_terms']:.4f} "
            f"(n={summary['n_negative_with_identity_terms']}) vs without "
            f"{summary['fpr_without_identity_terms']:.4f}; Fisher p={summary['fisher_p_value']:.4f}"
        )

    if not summaries:
        print("no technique had a probability artifact for this split", file=sys.stderr)
        return 1
    print(f"wrote {write_rq4_table(summaries)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
