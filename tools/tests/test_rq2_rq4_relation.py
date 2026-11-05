"""Behavioural and adversarial tests for tools/rq2_rq4_relation.py on synthetic tables."""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIR))

from rq2_rq4_relation import RelationError, build, permutation_distribution, spearman

TECHNIQUES = [f"0{i}_T{i}" for i in range(1, 9)]


N_WITH, N_WITHOUT = 67, 213


def _tables(share=None, share_val=None, ratio=None, delta=None, share_delta=None, techniques=TECHNIQUES,
            share_common=None, n_posts=None):
    n = len(techniques)
    share = np.linspace(0.05, 0.2, n) if share is None else share
    share_val = share if share_val is None else share_val
    share_common = share if share_common is None else share_common
    n_posts = [450] * (n - 2) + [200] * 2 if n_posts is None else n_posts
    ratio = np.linspace(1.0, 3.0, n) if ratio is None else ratio
    delta = np.linspace(0.01, 0.05, n) if delta is None else delta
    share_delta = np.linspace(0.5, 0.9, n) if share_delta is None else share_delta
    shares, same_post = [], []
    for split, values in (("test", share), ("validation", share_val)):
        for t, v in zip(techniques, values):
            for category in ("extremist_framing", "identity_term", "slur", "topical"):
                shares.append({"technique": t, "member": np.nan, "split": split, "category": category,
                               "share_of_positive_mass": v if category == "identity_term" else (1 - v) / 3})
    for post_set, values in (("own", share), ("common", share_common)):
        for t, v, k in zip(techniques, values, n_posts):
            same_post.append({"technique": t, "split": "test", "post_set": post_set,
                              "n_posts": k if post_set == "own" else 200, "category": "identity_term",
                              "share_of_positive_mass": v})
    # FPR without identity terms fixed at 10/213, so the ratio sets the identity-term count.
    fp_without = np.full(n, 10)
    fpr_without = fp_without / N_WITHOUT
    fp_with = np.round(np.asarray(ratio) * fpr_without * N_WITH)
    fp_with = np.where(np.isfinite(fp_with), fp_with, 5)
    fpr = pd.DataFrame({"technique": techniques, "split": "test", "n_negative": N_WITH + N_WITHOUT,
                        "n_negative_with_identity_terms": N_WITH, "fpr_with_identity_terms": fp_with / N_WITH,
                        "fpr_without_identity_terms": fpr_without, "fpr_ratio": ratio})
    cf = []
    for t, d, s in zip(techniques, delta, share_delta):
        for population in ("non_extremist", "extremist"):
            for operation in ("remove", "substitute"):
                cf.append({"technique": t, "split": "test", "population": population, "operation": operation,
                           "mean_delta": d if (population, operation) == ("non_extremist", "remove") else -9.0,
                           "share_delta_positive": s})
    return pd.DataFrame(shares), fpr, pd.DataFrame(cf), pd.DataFrame(same_post)


def _row(table, x, y="fpr_ratio", subset="all"):
    return table[(table["x_measure"] == x) & (table["y_measure"] == y) & (table["subset"] == subset)].iloc[0]


class TestRelation:
    def test_monotone_relation_has_rho_one_and_exact_p(self):
        table, _ = build(*_tables(), TECHNIQUES)
        row = _row(table, "identity_share_positive_mass")
        assert row["spearman_rho"] == pytest.approx(1.0)
        assert row["permutation_method"] == "exact" and row["n_permutations"] == math.factorial(8)
        assert row["permutation_p"] == pytest.approx(2 / math.factorial(8))
        assert row["n_techniques"] == 8 and row["excluded_techniques"] == ""
        assert 0.7 < row["abs_rho_needed_p05"] < 0.76
        assert row["role"] == "primary"
        assert row["loo_rho_min"] == pytest.approx(1.0) and row["loo_rho_max"] == pytest.approx(1.0)

    def test_only_the_non_extremist_remove_rows_are_used(self):
        # Every other counterfactual row carries -9; if any were read, the delta could not be monotone.
        table, measures = build(*_tables(), TECHNIQUES)
        assert (measures["counterfactual_mean_delta"] > 0).all()
        assert _row(table, "counterfactual_mean_delta")["spearman_rho"] == pytest.approx(1.0)

    def test_rho_matches_scipy_with_ties(self):
        x = np.array([0.1, 0.1, 0.3, 0.2, 0.5, 0.4, 0.4, 0.0])
        y = np.array([1.0, 2.0, 2.0, 1.5, 3.0, 2.5, 1.0, 0.5])
        assert spearman(x, y) == pytest.approx(spearmanr(x, y)[0])
        null, method = permutation_distribution(x, y)
        assert method == "exact" and len(null) == math.factorial(8)

    def test_non_finite_ratio_is_excluded_and_named(self):
        ratio = np.linspace(1.0, 3.0, 8)
        ratio[2] = np.inf
        table, _ = build(*_tables(ratio=ratio), TECHNIQUES)
        row = _row(table, "identity_share_positive_mass")
        assert row["n_techniques"] == 7 and row["excluded_techniques"] == TECHNIQUES[2]
        other = _row(table, "identity_share_positive_mass", "counterfactual_mean_delta")
        assert other["n_techniques"] == 8

    def test_member_rows_are_ignored(self):
        shares, fpr, cf, same = _tables()
        member = shares[shares["split"] == "test"].assign(member="A", share_of_positive_mass=0.99)
        table, measures = build(pd.concat([shares, member]), fpr, cf, same, TECHNIQUES)
        assert measures["identity_share_positive_mass"].max() < 0.99

    def test_holm_is_over_the_primary_and_secondary_rows_only(self):
        table, _ = build(*_tables(), TECHNIQUES)
        family = table["role"].isin(["primary", "secondary"])
        assert family.sum() == 2 and table.loc[family, "holm_p"].notna().all()
        assert table.loc[~family, "holm_p"].isna().all()
        assert (table.loc[family, "holm_p"] >= table.loc[family, "permutation_p"]).all()
        assert set(table["role"]) == {"primary", "secondary", "sensitivity", "descriptive"}

    def test_false_positive_counts_are_reported(self):
        _, inputs = build(*_tables(), TECHNIQUES)
        assert (inputs["n_fp_without_identity_terms"] == 10).all()
        assert (inputs["n_negative_without_identity_terms"] == N_WITHOUT).all()
        expected = np.round(np.linspace(1.0, 3.0, 8) * 10 / N_WITHOUT * N_WITH).astype(int)
        assert inputs["n_fp_with_identity_terms"].tolist() == expected.tolist()

    def test_full_split_subset_drops_the_sampled_techniques(self):
        table, _ = build(*_tables(), TECHNIQUES)
        row = _row(table, "identity_share_positive_mass", subset="full_split_posts")
        assert row["n_techniques"] == 6 and row["techniques"] == ";".join(TECHNIQUES[:6])

    def test_common_post_share_is_its_own_measure(self):
        reversed_common = np.linspace(0.2, 0.05, 8)
        table, inputs = build(*_tables(share_common=reversed_common), TECHNIQUES)
        assert _row(table, "identity_share_positive_mass_common_posts")["spearman_rho"] == pytest.approx(-1.0)
        assert _row(table, "identity_share_positive_mass")["spearman_rho"] == pytest.approx(1.0)
        assert inputs["n_posts_common"].eq(200).all()

    def test_leave_one_out_range_reaches_one_without_the_outlier(self):
        share = np.linspace(0.05, 0.2, 8)
        ratio = np.linspace(1.0, 3.0, 8)
        ratio[0] = 5.0  # the lowest share has the highest ratio
        table, _ = build(*_tables(share=share, ratio=ratio), TECHNIQUES)
        row = _row(table, "identity_share_positive_mass")
        assert row["spearman_rho"] < 1.0 and row["loo_rho_max"] == pytest.approx(1.0)
        assert row["loo_rho_min"] <= row["spearman_rho"] <= row["loo_rho_max"]


class TestRefusals:
    def test_technique_missing_from_rq4_is_refused(self):
        shares, fpr, cf, same = _tables()
        with pytest.raises(RelationError, match="no value for"):
            build(shares, fpr.iloc[1:], cf, same, TECHNIQUES)

    def test_duplicate_fpr_rows_are_refused(self):
        shares, fpr, cf, same = _tables()
        with pytest.raises(RelationError, match="more than one row"):
            build(shares, pd.concat([fpr, fpr.iloc[:1]]), cf, same, TECHNIQUES)

    def test_duplicate_counterfactual_rows_are_refused(self):
        shares, fpr, cf, same = _tables()
        with pytest.raises(RelationError, match="more than one row"):
            build(shares, fpr, pd.concat([cf, cf.iloc[:1]]), same, TECHNIQUES)

    def test_share_outside_unit_interval_is_refused(self):
        share = np.linspace(0.05, 0.2, 8)
        share[0] = 1.4
        with pytest.raises(RelationError, match=r"\[0, 1\]"):
            build(*_tables(share=share), TECHNIQUES)

    def test_fpr_rows_for_another_split_are_refused(self):
        shares, fpr, cf, same = _tables()
        with pytest.raises(RelationError, match="split other than test"):
            build(shares, fpr.assign(split="validation"), cf, same, TECHNIQUES)

    def test_too_few_techniques_are_refused(self):
        few = TECHNIQUES[:3]
        with pytest.raises(RelationError, match="need at least 4"):
            build(*_tables(techniques=few), few)

    def test_missing_column_is_refused(self):
        shares, fpr, cf, same = _tables()
        with pytest.raises(RelationError, match="missing columns"):
            build(shares, fpr.drop(columns="fpr_ratio"), cf, same, TECHNIQUES)

    def test_two_whole_model_test_runs_for_a_technique_are_refused(self):
        shares, fpr, cf, same = _tables()
        extra = shares[(shares["technique"] == TECHNIQUES[0]) & (shares["split"] == "test")]
        with pytest.raises(RelationError, match="more than one row"):
            build(pd.concat([shares, extra]), fpr, cf, same, TECHNIQUES)

    def test_stale_same_post_table_is_refused(self):
        shares, fpr, cf, same = _tables()
        own = (same["post_set"] == "own") & (same["technique"] == TECHNIQUES[3])
        same.loc[own, "share_of_positive_mass"] += 0.01
        with pytest.raises(RelationError, match="differ from rq2_category_shares"):
            build(shares, fpr, cf, same, TECHNIQUES)

    def test_fpr_that_is_not_a_whole_count_is_refused(self):
        shares, fpr, cf, same = _tables()
        fpr.loc[2, "fpr_without_identity_terms"] = 10.5 / N_WITHOUT
        with pytest.raises(RelationError, match="not a whole number"):
            build(shares, fpr, cf, same, TECHNIQUES)

    def test_technique_missing_from_same_post_table_is_refused(self):
        shares, fpr, cf, same = _tables()
        with pytest.raises(RelationError, match="no value for"):
            build(shares, fpr, cf, same[same["technique"] != TECHNIQUES[5]], TECHNIQUES)


def test_more_than_nine_techniques_use_seeded_monte_carlo():
    techniques = [f"{i:02d}_T{i}" for i in range(1, 11)]
    table, _ = build(*_tables(techniques=techniques), techniques)
    row = _row(table, "identity_share_positive_mass")
    assert row["permutation_method"] == "monte_carlo" and row["n_permutations"] == 100_000
    assert 0 < row["permutation_p"] < 1e-3
    again, _ = build(*_tables(techniques=techniques), techniques)
    assert again["permutation_p"].equals(table["permutation_p"])
