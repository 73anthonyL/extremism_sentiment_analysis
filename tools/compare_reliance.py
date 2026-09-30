"""RQ3: does pretraining or ensembling change what models rely on?

Reads the per-run category shares written by categorize_attributions.py and
produces two cross-technique tables in results_summary/rq/:

rq3_reliance_comparison.csv
    One row per run in wide form (one column per category for the share of
    positive attribution mass and of absolute mass), with a `role` column:
    `whole` for a model or whole-ensemble run, `member` for an ensemble
    member. For every technique that has member runs, an extra `member_mean`
    row averages the members so per-member vs whole-ensemble reliance is one
    subtraction away.

rq3_stability.csv
    For every technique with two or more `whole` runs (different seeds or
    background sets), the standard deviation of each category share across
    runs, the mean pairwise Spearman correlation of the shared words' mean
    absolute attributions, and the mean pairwise Jaccard overlap of the top-K
    word sets. Reliance claims should only be made where these are stable.

With --techniques, also prints the pairwise differences in positive-mass share
between the named techniques (e.g. RoBERTa vs HateBERT vs Twitter-RoBERTa).

USAGE
-----
    python3 tools/compare_reliance.py
    python3 tools/compare_reliance.py --techniques 07_TWITTER-ROBERTA_FINE-TUNE 12_HATEBERT_FINE-TUNE
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attributions import list_runs, load_run, top_words
from repo_paths import (
    CATEGORIES,
    RQ2_CATEGORY_SHARES_CSV,
    RQ3_RELIANCE_CSV,
    RQ3_STABILITY_CSV,
    technique_dirs,
)

DEFAULT_TOP_K = 50
ROLE_WHOLE = "whole"
ROLE_MEMBER = "member"
ROLE_MEMBER_MEAN = "member_mean"


def _is_member(value):
    return not (value is None or (isinstance(value, float) and np.isnan(value)) or value == "")


def widen_shares(long_table):
    """Pivot the long RQ2 table to one row per run with per-category share columns."""
    if long_table.empty:
        return pd.DataFrame()
    keys = ["technique", "run_id", "member", "split", "seed", "top_k"]
    long_table = long_table.copy()
    long_table["member"] = long_table["member"].where(long_table["member"].notna(), "")
    wide = long_table.pivot_table(
        index=keys,
        columns="category",
        values=["share_of_positive_mass", "share_of_abs_mass"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}__{category}" for metric, category in wide.columns]
    wide = wide.reset_index()
    for category in CATEGORIES:
        for metric in ("share_of_positive_mass", "share_of_abs_mass"):
            column = f"{metric}__{category}"
            if column not in wide.columns:
                wide[column] = 0.0
    wide["role"] = [ROLE_MEMBER if _is_member(m) else ROLE_WHOLE for m in wide["member"]]
    return wide


def add_member_means(wide):
    """Append a member_mean row per technique that has member runs."""
    if wide.empty:
        return wide
    share_columns = [c for c in wide.columns if c.startswith("share_of_")]
    extra = []
    for technique, group in wide.groupby("technique"):
        members = group[group["role"] == ROLE_MEMBER]
        if members.empty:
            continue
        row = {c: members[c].mean() for c in share_columns}
        row.update(
            technique=technique,
            run_id=f"{technique}__member_mean",
            member="",
            split=members["split"].iloc[0],
            seed=None,
            top_k=members["top_k"].iloc[0],
            role=ROLE_MEMBER_MEAN,
        )
        extra.append(row)
    if extra:
        wide = pd.concat([wide, pd.DataFrame(extra)], ignore_index=True)
    return wide.sort_values(["technique", "role", "run_id"]).reset_index(drop=True)


def _pairwise_rank_stability(technique, run_ids, top_k, results_dir=None):
    """Mean pairwise Spearman on shared words and Jaccard on top-K sets."""
    frames = {r: load_run(technique, r, results_dir)[0] for r in run_ids}
    rhos, jaccards = [], []
    for a, b in itertools.combinations(run_ids, 2):
        merged = frames[a].merge(frames[b], on="word", suffixes=("_a", "_b"))
        if len(merged) >= 3:
            rho, _ = spearmanr(merged["mean_abs_attribution_a"], merged["mean_abs_attribution_b"])
            rhos.append(float(rho))
        top_a = set(top_words(frames[a], top_k)["word"])
        top_b = set(top_words(frames[b], top_k)["word"])
        union = top_a | top_b
        jaccards.append(len(top_a & top_b) / len(union) if union else 0.0)
    return (
        float(np.mean(rhos)) if rhos else np.nan,
        float(np.mean(jaccards)) if jaccards else np.nan,
    )


def stability_table(wide, top_k=DEFAULT_TOP_K, results_dir=None):
    """Per-technique dispersion across whole-model runs."""
    rows = []
    if wide.empty:
        return pd.DataFrame(rows)
    whole = wide[wide["role"] == ROLE_WHOLE]
    for technique, group in whole.groupby("technique"):
        if len(group) < 2:
            continue
        rho, jaccard = _pairwise_rank_stability(technique, list(group["run_id"]), top_k, results_dir)
        row = {"technique": technique, "n_runs": int(len(group))}
        for category in CATEGORIES:
            row[f"std_share_of_positive_mass__{category}"] = float(
                group[f"share_of_positive_mass__{category}"].std(ddof=0)
            )
        row["mean_pairwise_spearman_abs_attribution"] = rho
        row[f"mean_pairwise_jaccard_top{top_k}"] = jaccard
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_differences(wide, techniques):
    """Positive-mass share differences between whole-model runs of named techniques."""
    whole = wide[(wide["role"] == ROLE_WHOLE) & wide["technique"].isin(techniques)]
    means = whole.groupby("technique")[[f"share_of_positive_mass__{c}" for c in CATEGORIES]].mean()
    lines = []
    for a, b in itertools.combinations([t for t in techniques if t in means.index], 2):
        delta = means.loc[b] - means.loc[a]
        parts = ", ".join(
            f"{c} {delta[f'share_of_positive_mass__{c}']:+.3f}" for c in CATEGORIES
        )
        lines.append(f"{b} minus {a}: {parts}")
    return lines


def build(top_k=DEFAULT_TOP_K, results_dir=None, shares_path=RQ2_CATEGORY_SHARES_CSV):
    shares_path = Path(shares_path)
    if not shares_path.exists():
        raise FileNotFoundError(
            f"{shares_path} not found; run categorize_attributions.py --all first"
        )
    long_table = pd.read_csv(shares_path)
    wide = add_member_means(widen_shares(long_table))
    stability = stability_table(wide, top_k=top_k, results_dir=results_dir)
    return wide, stability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--techniques", nargs="*", default=None)
    args = parser.parse_args()

    try:
        wide, stability = build(top_k=args.top_k)
    except FileNotFoundError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    if wide.empty:
        print("ERROR: no category shares to compare", file=sys.stderr)
        return 1

    RQ3_RELIANCE_CSV.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(RQ3_RELIANCE_CSV, index=False, float_format="%.8g")
    stability.to_csv(RQ3_STABILITY_CSV, index=False, float_format="%.8g")
    print(f"wrote {RQ3_RELIANCE_CSV} ({len(wide)} rows)")
    print(f"wrote {RQ3_STABILITY_CSV} ({len(stability)} techniques with >= 2 whole runs)")

    if args.techniques:
        for line in pairwise_differences(wide, args.techniques):
            print(line)
    known = set(d.name for d in technique_dirs())
    unknown = set(wide["technique"]) - known
    if unknown:
        print(f"warning: shares reference techniques with no results folder: {sorted(unknown)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
