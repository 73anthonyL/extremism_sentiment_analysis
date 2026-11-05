"""RQ2 x RQ4: do the techniques that rely more on identity terms also have the larger identity-term FPR gap?

Joins three committed cross-technique tables on technique, for the test split
and the techniques in the comparison only, and writes one text-free table,
results_summary/rq/rq2_rq4_relation.csv, with one row per (reliance measure,
outcome) pair:

Reliance measures (x)
    identity_share_positive_mass
        RQ2: the identity-term category's share of positive attribution mass
        in the top-K words of the whole-model test run
        (`rq2_category_shares.csv`, member empty; the mean if a technique has
        several whole-model test runs). Depends on the identity lexicon only
        through which words are counted as identity terms.
    identity_share_positive_mass_validation
        The same share from the whole-model validation run: a sensitivity
        row whose attributions come from posts other than the test posts the
        FPR ratio is computed on.
    counterfactual_mean_delta
        WP7: the mean drop in p(extremist) on the non-extremist posts with an
        identity-lexicon match when those matches are removed
        (`rq4_identity_counterfactual.csv`, population non_extremist,
        operation remove). Lexicon-light: the lexicon picks the posts and the
        words to remove, the model's own response does the rest. It is in
        probability units, so it also reflects how far each model's
        probabilities sit from 0 on these posts.
    counterfactual_share_delta_positive
        The same rows' share of posts whose p(extremist) drops: a scale-free
        sensitivity companion to the mean delta.

Outcome (y)
    fpr_ratio: RQ4, FPR on non-extremist test posts with identity terms over
    FPR on those without (`rq4_identity_fpr.csv`).

A last row correlates the two main reliance measures with each other
(identity_share_positive_mass against counterfactual_mean_delta).

For each row: n (techniques with a finite value on both sides; any excluded
technique is named), Spearman's rho, a two-sided permutation p (exact over
all n! orderings up to n = 9, otherwise 100,000 seeded Monte Carlo draws with
the +1 correction), the smallest |rho| that would reach p <= 0.05 at this n,
and Holm-adjusted p over the rows of the table. With eight techniques the
threshold is about |rho| = 0.74: a correlation below it is reported as a
number, not as a relation.

Refusals: a missing input table or column; a technique in the comparison
missing from any input; duplicate rows for a technique; a share outside
[0, 1]; a split other than test; fewer than four techniques with finite
values on both sides.

USAGE
-----
    python3 tools/rq2_rq4_relation.py
"""

import argparse
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcnemar_pairs import holm_adjust
from repo_paths import (
    RANDOM_SEED,
    RQ2_CATEGORY_SHARES_CSV,
    RQ2_RQ4_RELATION_CSV,
    RQ4_IDENTITY_COUNTERFACTUAL_CSV,
    RQ4_IDENTITY_FPR_CSV,
    comparison_technique_dirs,
)

SPLIT = "test"
IDENTITY_CATEGORY = "identity_term"
MIN_TECHNIQUES = 4
MAX_EXACT_N = 9
MONTE_CARLO_DRAWS = 100_000
ALPHA = 0.05

RELATION_COLUMNS = (
    "x_measure",
    "y_measure",
    "split",
    "n_techniques",
    "techniques",
    "excluded_techniques",
    "spearman_rho",
    "permutation_p",
    "permutation_method",
    "n_permutations",
    "abs_rho_needed_p05",
    "holm_p",
)

PAIRS = (
    ("identity_share_positive_mass", "fpr_ratio"),
    ("identity_share_positive_mass_validation", "fpr_ratio"),
    ("counterfactual_mean_delta", "fpr_ratio"),
    ("counterfactual_share_delta_positive", "fpr_ratio"),
    ("identity_share_positive_mass", "counterfactual_mean_delta"),
)


class RelationError(RuntimeError):
    pass


def _require(frame, columns, label):
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise RelationError(f"{label}: missing columns {missing}")


def _read(path, label):
    path = Path(path)
    if not path.is_file():
        raise RelationError(f"{label}: {path} not found; run its tool first")
    return pd.read_csv(path, keep_default_na=True)


def _one_per_technique(frame, label):
    duplicated = frame["technique"][frame["technique"].duplicated()].unique().tolist()
    if duplicated:
        raise RelationError(f"{label}: more than one row for {sorted(duplicated)}")
    return frame.set_index("technique")


def identity_shares(shares, split=SPLIT):
    _require(shares, ("technique", "member", "split", "category", "share_of_positive_mass"), "rq2_category_shares")
    member = shares["member"]
    whole = member.isna() | member.astype(str).str.strip().eq("")
    rows = shares[whole & (shares["split"] == split) & (shares["category"] == IDENTITY_CATEGORY)]
    values = pd.to_numeric(rows["share_of_positive_mass"], errors="coerce")
    if values.isna().any() or (values < 0).any() or (values > 1).any():
        raise RelationError("rq2_category_shares: identity-term share_of_positive_mass must lie in [0, 1]")
    return rows.assign(share_of_positive_mass=values).groupby("technique")["share_of_positive_mass"].mean()


def fpr_ratios(fpr):
    _require(fpr, ("technique", "split", "fpr_ratio"), "rq4_identity_fpr")
    if (fpr["split"] != SPLIT).any():
        raise RelationError(f"rq4_identity_fpr: rows for a split other than {SPLIT}")
    table = _one_per_technique(fpr, "rq4_identity_fpr")
    return pd.to_numeric(table["fpr_ratio"], errors="coerce")


def counterfactual(cf):
    _require(cf, ("technique", "split", "population", "operation", "mean_delta", "share_delta_positive"),
             "rq4_identity_counterfactual")
    rows = cf[(cf["split"] == SPLIT) & (cf["population"] == "non_extremist") & (cf["operation"] == "remove")]
    table = _one_per_technique(rows, "rq4_identity_counterfactual (non_extremist, remove)")
    share = pd.to_numeric(table["share_delta_positive"], errors="coerce")
    if ((share < 0) | (share > 1)).any():
        raise RelationError("rq4_identity_counterfactual: share_delta_positive must lie in [0, 1]")
    return pd.to_numeric(table["mean_delta"], errors="coerce"), share


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def permutation_distribution(x, y, seed=RANDOM_SEED):
    """|rho| for every reordering of y (exact up to MAX_EXACT_N) or for Monte Carlo draws."""
    rx = rankdata(x)
    ry = rankdata(y)
    n = len(rx)
    if n <= MAX_EXACT_N:
        orders = np.array(list(itertools.permutations(range(n))))
        method = "exact"
    else:
        rng = np.random.default_rng(seed)
        orders = np.array([rng.permutation(n) for _ in range(MONTE_CARLO_DRAWS)])
        method = "monte_carlo"
    permuted = ry[orders]
    cx = rx - rx.mean()
    cy = permuted - permuted.mean(axis=1, keepdims=True)
    rho = (cy @ cx) / (np.sqrt((cx ** 2).sum()) * np.sqrt((cy ** 2).sum(axis=1)))
    return np.abs(rho), method


def relation_row(x_name, y_name, x, y):
    joined = pd.concat({"x": x, "y": y}, axis=1)
    finite = np.isfinite(joined["x"].astype(float)) & np.isfinite(joined["y"].astype(float))
    excluded = sorted(joined.index[~finite])
    used = joined[finite].sort_index()
    if len(used) < MIN_TECHNIQUES:
        raise RelationError(
            f"{x_name} vs {y_name}: only {len(used)} techniques with finite values on both sides; "
            f"need at least {MIN_TECHNIQUES}"
        )
    xs, ys = used["x"].to_numpy(float), used["y"].to_numpy(float)
    rho = spearman(xs, ys)
    null, method = permutation_distribution(xs, ys)
    tol = 1e-12
    if np.isnan(rho):
        p = float("nan")
    elif method == "exact":
        p = float(np.mean(null >= abs(rho) - tol))
    else:
        p = float((np.sum(null >= abs(rho) - tol) + 1) / (len(null) + 1))
    # Smallest |rho| in the null whose two-sided p is at most ALPHA.
    values = np.unique(np.round(null, 12))
    needed = float("nan")
    for value in values:
        if np.mean(null >= value - tol) <= ALPHA:
            needed = float(value)
            break
    return {
        "x_measure": x_name,
        "y_measure": y_name,
        "split": SPLIT,
        "n_techniques": int(len(used)),
        "techniques": ";".join(used.index),
        "excluded_techniques": ";".join(excluded),
        "spearman_rho": rho,
        "permutation_p": p,
        "permutation_method": method,
        "n_permutations": int(len(null)) if method == "monte_carlo" else math.factorial(len(used)),
        "abs_rho_needed_p05": needed,
    }


def build(shares, fpr, cf, techniques):
    """The relation table from the three input frames, over `techniques`."""
    techniques = sorted(techniques)
    measures = {
        "identity_share_positive_mass": identity_shares(shares),
        "identity_share_positive_mass_validation": identity_shares(shares, "validation"),
        "fpr_ratio": fpr_ratios(fpr),
    }
    measures["counterfactual_mean_delta"], measures["counterfactual_share_delta_positive"] = counterfactual(cf)
    for name, series in measures.items():
        missing = [t for t in techniques if t not in series.index]
        if missing:
            raise RelationError(f"{name}: no value for {missing}; run the chain for every technique first")
        measures[name] = series.reindex(techniques)
    rows = [relation_row(x, y, measures[x], measures[y]) for x, y in PAIRS]
    table = pd.DataFrame(rows)
    p = table["permutation_p"]
    table["holm_p"] = np.nan
    present = p.notna()
    if present.any():
        table.loc[present, "holm_p"] = holm_adjust(p[present].to_numpy())
    return table[list(RELATION_COLUMNS)], pd.DataFrame(measures)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    try:
        techniques = [d.name for d in comparison_technique_dirs()]
        table, measures = build(
            _read(RQ2_CATEGORY_SHARES_CSV, "rq2_category_shares"),
            _read(RQ4_IDENTITY_FPR_CSV, "rq4_identity_fpr"),
            _read(RQ4_IDENTITY_COUNTERFACTUAL_CSV, "rq4_identity_counterfactual"),
            techniques,
        )
    except RelationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    path = Path(RQ2_RQ4_RELATION_CSV)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, float_format="%.6g")
    with pd.option_context("display.width", 200, "display.precision", 4):
        print(measures)
    for _, row in table.iterrows():
        print(f"{row['x_measure']} vs {row['y_measure']}: n={row['n_techniques']}, rho={row['spearman_rho']:.3f}, "
              f"permutation p={row['permutation_p']:.4f} ({row['permutation_method']}), "
              f"Holm p={row['holm_p']:.4f}; |rho| >= {row['abs_rho_needed_p05']:.3f} needed for p <= 0.05")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
