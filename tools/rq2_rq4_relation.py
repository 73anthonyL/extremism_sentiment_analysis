"""RQ2 x RQ4: does the ranking of techniques by identity-term reliance match their ranking by FPR ratio?

Joins the committed cross-technique tables on technique, for the test split
and the techniques in the comparison only, and writes two text-free tables:

results_summary/rq/rq2_rq4_relation_inputs.csv
    One row per technique: every measure the correlations use, and the
    false-positive counts behind each FPR ratio (n_fp = FPR x n_negative,
    refused unless it is a whole number), so a reader can see what drives a
    ratio: with 67 identity-term negatives and 213 others, a ratio moves as
    much through a low baseline FPR as through identity-term false positives.

results_summary/rq/rq2_rq4_relation.csv
    One row per (reliance measure, outcome, subset) with a `role`:

    primary       identity_share_positive_mass vs fpr_ratio, all techniques.
                  The relation WP5 asks for.
    secondary     counterfactual_mean_delta vs fpr_ratio: the lexicon-light
                  WP7 measure the WP5 brief also named before any result.
    sensitivity   the primary pair with the share computed on the common
                  200 posts every technique explained
                  (identity_share_positive_mass_common_posts, from
                  same_post_stability.py), with the validation-run share, and
                  over the techniques whose attributions cover the whole test
                  split only (subset full_split_posts; the current runs give
                  the six classical techniques); and the scale-free
                  counterfactual share against the ratio.
    descriptive   identity share against each part of the ratio
                  (fpr_with_identity_terms, fpr_without_identity_terms), and
                  identity share against the counterfactual mean delta.

    The Holm family is fixed as the primary and secondary rows, the two the
    WP5 brief named before results were seen; `holm_p` is empty for every
    other row. Sensitivity and descriptive rows are read for direction and
    size, never cited as tests.

Reliance measures
    identity_share_positive_mass
        RQ2: the identity-term category's share of positive attribution mass
        in the top-K words of the whole-model test run
        (`rq2_category_shares.csv`, member empty). Classical runs explain all
        450 test posts, transformer runs a 200-post sample, so this measure
        mixes post sets; `n_posts_attribution_test` in the inputs table says
        which, and the common-posts measure removes the difference.
    identity_share_positive_mass_common_posts
        The same share on the common post set
        (`rq2_same_post_category_shares.csv`, post_set common).
    identity_share_positive_mass_validation
        The same share from the whole-model validation run (other posts).
    counterfactual_mean_delta, counterfactual_share_delta_positive
        WP7 (`rq4_identity_counterfactual.csv`, population non_extremist,
        operation remove): the mean drop in p(extremist) when identity-lexicon
        matches are removed, and the share of posts whose p(extremist) drops.

Outcomes
    fpr_ratio, fpr_with_identity_terms, fpr_without_identity_terms
        RQ4 (`rq4_identity_fpr.csv`).

For each row: n (techniques with a finite value on both sides; any excluded
technique is named), Spearman's rho, a two-sided permutation p (exact over
all n! orderings up to n = 9, otherwise 100,000 seeded Monte Carlo draws with
the +1 correction), the smallest |rho| that reaches p <= 0.05 at this n (0.7306
at n = 8 without ties; the value is computed per row because ties change it),
and the leave-one-out range of rho with the technique whose removal gives the
lowest value. With eight techniques a rank correlation is a statement about
an ordering of eight models, not about mechanism.

Refusals: a missing input table or column; a technique in the comparison
missing from any input; duplicate rows for a technique, including more than
one whole-model run for a technique and split; a share outside [0, 1]; a
split other than test in the RQ4 tables; a false-positive count that is not a
whole number; own-post shares in the same-post table that differ from
rq2_category_shares.csv (a stale table); fewer than four techniques with
finite values on both sides.

USAGE
-----
    python3 tools/same_post_stability.py     # writes the same-post category table
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
    RQ2_RQ4_RELATION_INPUTS_CSV,
    RQ2_SAME_POST_CATEGORY_SHARES_CSV,
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
COUNT_TOLERANCE = 1e-3
SHARE_MATCH_ATOL = 1e-6
HOLM_ROLES = ("primary", "secondary")

RELATION_COLUMNS = (
    "role",
    "x_measure",
    "y_measure",
    "subset",
    "split",
    "n_techniques",
    "techniques",
    "excluded_techniques",
    "spearman_rho",
    "permutation_p",
    "permutation_method",
    "n_permutations",
    "abs_rho_needed_p05",
    "loo_rho_min",
    "loo_rho_max",
    "loo_rho_min_dropped",
    "holm_p",
)

INPUT_COLUMNS = (
    "technique",
    "n_posts_attribution_test",
    "identity_share_positive_mass",
    "identity_share_positive_mass_common_posts",
    "n_posts_common",
    "identity_share_positive_mass_validation",
    "n_negative_with_identity_terms",
    "n_fp_with_identity_terms",
    "fpr_with_identity_terms",
    "n_negative_without_identity_terms",
    "n_fp_without_identity_terms",
    "fpr_without_identity_terms",
    "fpr_ratio",
    "counterfactual_mean_delta",
    "counterfactual_share_delta_positive",
)

ALL = "all"
FULL_SPLIT = "full_split_posts"

# (role, x, y, subset). The order is the order of the written table.
PAIRS = (
    ("primary", "identity_share_positive_mass", "fpr_ratio", ALL),
    ("secondary", "counterfactual_mean_delta", "fpr_ratio", ALL),
    ("sensitivity", "identity_share_positive_mass_common_posts", "fpr_ratio", ALL),
    ("sensitivity", "identity_share_positive_mass", "fpr_ratio", FULL_SPLIT),
    ("sensitivity", "identity_share_positive_mass_validation", "fpr_ratio", ALL),
    ("sensitivity", "counterfactual_share_delta_positive", "fpr_ratio", ALL),
    ("descriptive", "identity_share_positive_mass", "fpr_with_identity_terms", ALL),
    ("descriptive", "identity_share_positive_mass", "fpr_without_identity_terms", ALL),
    ("descriptive", "identity_share_positive_mass_common_posts", "fpr_with_identity_terms", ALL),
    ("descriptive", "identity_share_positive_mass_common_posts", "fpr_without_identity_terms", ALL),
    ("descriptive", "identity_share_positive_mass", "counterfactual_mean_delta", ALL),
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


def _unit_interval(values, label):
    values = pd.to_numeric(values, errors="coerce")
    if values.isna().any() or (values < 0).any() or (values > 1).any():
        raise RelationError(f"{label}: identity-term share_of_positive_mass must lie in [0, 1]")
    return values


def identity_shares(shares, split=SPLIT):
    """Identity-term share per technique from the whole-model run of `split`; exactly one run each."""
    _require(shares, ("technique", "member", "split", "category", "share_of_positive_mass"), "rq2_category_shares")
    member = shares["member"]
    whole = member.isna() | member.astype(str).str.strip().eq("")
    rows = shares[whole & (shares["split"] == split) & (shares["category"] == IDENTITY_CATEGORY)]
    table = _one_per_technique(rows, f"rq2_category_shares (whole-model {split} runs, identity_term)")
    return _unit_interval(table["share_of_positive_mass"], "rq2_category_shares")


def same_post_shares(same_post, own_test_share):
    """(common-post identity share, own-post n_posts, common n_posts) from the same-post category table.

    The own-post test rows must match rq2_category_shares, or the table is stale.
    """
    _require(same_post, ("technique", "split", "post_set", "n_posts", "category", "share_of_positive_mass"),
             "rq2_same_post_category_shares")
    rows = same_post[(same_post["split"] == SPLIT) & (same_post["category"] == IDENTITY_CATEGORY)]
    out = {}
    for post_set in ("own", "common"):
        part = _one_per_technique(rows[rows["post_set"] == post_set],
                                  f"rq2_same_post_category_shares ({post_set}, identity_term)")
        out[post_set] = part
    own = out["own"]
    shared = own.index.intersection(own_test_share.index)
    own_share = _unit_interval(own.loc[shared, "share_of_positive_mass"], "rq2_same_post_category_shares")
    if not np.allclose(own_share, own_test_share.loc[shared], atol=SHARE_MATCH_ATOL, rtol=0):
        raise RelationError("rq2_same_post_category_shares: own-post shares differ from rq2_category_shares.csv; "
                            "re-run same_post_stability.py")
    common = out["common"]
    return (_unit_interval(common["share_of_positive_mass"], "rq2_same_post_category_shares"),
            pd.to_numeric(own["n_posts"]), pd.to_numeric(common["n_posts"]))


def _count(rate, n, label):
    counts = rate * n
    rounded = counts.round()
    if (~np.isfinite(counts) | ((counts - rounded).abs() > COUNT_TOLERANCE)).any():
        raise RelationError(f"rq4_identity_fpr: {label} x its negative count is not a whole number")
    return rounded.astype(int)


def fpr_table(fpr):
    """FPR ratio, its two parts and the false-positive counts behind them."""
    columns = ("technique", "split", "n_negative", "n_negative_with_identity_terms",
               "fpr_with_identity_terms", "fpr_without_identity_terms", "fpr_ratio")
    _require(fpr, columns, "rq4_identity_fpr")
    if (fpr["split"] != SPLIT).any():
        raise RelationError(f"rq4_identity_fpr: rows for a split other than {SPLIT}")
    table = _one_per_technique(fpr, "rq4_identity_fpr")
    n_with = pd.to_numeric(table["n_negative_with_identity_terms"]).astype(int)
    n_without = pd.to_numeric(table["n_negative"]).astype(int) - n_with
    with_rate = pd.to_numeric(table["fpr_with_identity_terms"], errors="coerce")
    without_rate = pd.to_numeric(table["fpr_without_identity_terms"], errors="coerce")
    return pd.DataFrame({
        "n_negative_with_identity_terms": n_with,
        "n_fp_with_identity_terms": _count(with_rate, n_with, "fpr_with_identity_terms"),
        "fpr_with_identity_terms": with_rate,
        "n_negative_without_identity_terms": n_without,
        "n_fp_without_identity_terms": _count(without_rate, n_without, "fpr_without_identity_terms"),
        "fpr_without_identity_terms": without_rate,
        "fpr_ratio": pd.to_numeric(table["fpr_ratio"], errors="coerce"),
    })


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
    with np.errstate(invalid="ignore", divide="ignore"):  # a constant measure gives nan, reported as such
        rho = (cy @ cx) / (np.sqrt((cx ** 2).sum()) * np.sqrt((cy ** 2).sum(axis=1)))
    return np.abs(rho), method


def leave_one_out(xs, ys, names):
    """(min rho, max rho, technique whose removal gives the min) over every drop-one subset."""
    if len(xs) - 1 < 3:
        return float("nan"), float("nan"), ""
    values = []
    for i in range(len(xs)):
        keep = np.arange(len(xs)) != i
        values.append(spearman(xs[keep], ys[keep]))
    values = np.array(values)
    if np.isnan(values).all():
        return float("nan"), float("nan"), ""
    low = int(np.nanargmin(values))
    return float(np.nanmin(values)), float(np.nanmax(values)), names[low]


def relation_row(role, x_name, y_name, subset, x, y, members):
    joined = pd.concat({"x": x, "y": y}, axis=1).reindex(members)
    finite = np.isfinite(joined["x"].astype(float)) & np.isfinite(joined["y"].astype(float))
    excluded = sorted(joined.index[~finite])
    used = joined[finite].sort_index()
    if len(used) < MIN_TECHNIQUES:
        raise RelationError(
            f"{x_name} vs {y_name} ({subset}): only {len(used)} techniques with finite values on both sides; "
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
    loo_min, loo_max, loo_dropped = leave_one_out(xs, ys, list(used.index))
    return {
        "role": role,
        "x_measure": x_name,
        "y_measure": y_name,
        "subset": subset,
        "split": SPLIT,
        "n_techniques": int(len(used)),
        "techniques": ";".join(used.index),
        "excluded_techniques": ";".join(excluded),
        "spearman_rho": rho,
        "permutation_p": p,
        "permutation_method": method,
        "n_permutations": int(len(null)) if method == "monte_carlo" else math.factorial(len(used)),
        "abs_rho_needed_p05": needed,
        "loo_rho_min": loo_min,
        "loo_rho_max": loo_max,
        "loo_rho_min_dropped": loo_dropped,
    }


def build(shares, fpr, cf, same_post, techniques):
    """(relation table, per-technique inputs table) from the input frames, over `techniques`."""
    techniques = sorted(techniques)
    test_share = identity_shares(shares)
    common_share, n_own, n_common = same_post_shares(same_post, test_share)
    measures = {
        "n_posts_attribution_test": n_own,
        "identity_share_positive_mass": test_share,
        "identity_share_positive_mass_common_posts": common_share,
        "n_posts_common": n_common,
        "identity_share_positive_mass_validation": identity_shares(shares, "validation"),
    }
    for column, series in fpr_table(fpr).items():
        measures[column] = series
    measures["counterfactual_mean_delta"], measures["counterfactual_share_delta_positive"] = counterfactual(cf)
    for name, series in measures.items():
        missing = [t for t in techniques if t not in series.index]
        if missing:
            raise RelationError(f"{name}: no value for {missing}; run the chain for every technique first")
        measures[name] = series.reindex(techniques)
    inputs = pd.DataFrame(measures)
    inputs.index.name = "technique"
    inputs = inputs.reset_index()[list(INPUT_COLUMNS)]

    n_posts = measures["n_posts_attribution_test"]
    subsets = {ALL: techniques, FULL_SPLIT: [t for t in techniques if n_posts[t] == n_posts.max()]}
    rows = [relation_row(role, x, y, subset, measures[x], measures[y], subsets[subset])
            for role, x, y, subset in PAIRS]
    table = pd.DataFrame(rows)
    table["holm_p"] = np.nan
    family = table["role"].isin(HOLM_ROLES) & table["permutation_p"].notna()
    if family.any():
        table.loc[family, "holm_p"] = holm_adjust(table.loc[family, "permutation_p"].to_numpy())
    return table[list(RELATION_COLUMNS)], inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    try:
        techniques = [d.name for d in comparison_technique_dirs()]
        table, inputs = build(
            _read(RQ2_CATEGORY_SHARES_CSV, "rq2_category_shares"),
            _read(RQ4_IDENTITY_FPR_CSV, "rq4_identity_fpr"),
            _read(RQ4_IDENTITY_COUNTERFACTUAL_CSV, "rq4_identity_counterfactual"),
            _read(RQ2_SAME_POST_CATEGORY_SHARES_CSV, "rq2_same_post_category_shares"),
            techniques,
        )
    except RelationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    paths = []
    for frame, path in ((table, RQ2_RQ4_RELATION_CSV), (inputs, RQ2_RQ4_RELATION_INPUTS_CSV)):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, float_format="%.6g")
        paths.append(path)
    with pd.option_context("display.width", 250, "display.precision", 4, "display.max_columns", 30):
        print(inputs.set_index("technique"))
    for _, row in table.iterrows():
        holm = "" if np.isnan(row["holm_p"]) else f", Holm p={row['holm_p']:.4f}"
        print(f"[{row['role']}] {row['x_measure']} vs {row['y_measure']} ({row['subset']}): "
              f"n={row['n_techniques']}, rho={row['spearman_rho']:.3f}, "
              f"permutation p={row['permutation_p']:.4f} ({row['permutation_method']}){holm}; "
              f"leave-one-out {row['loo_rho_min']:.3f} to {row['loo_rho_max']:.3f}; "
              f"|rho| >= {row['abs_rho_needed_p05']:.3f} needed for p <= 0.05")
    for path in paths:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
