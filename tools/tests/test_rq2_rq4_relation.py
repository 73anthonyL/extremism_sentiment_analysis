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


def _tables(share=None, share_val=None, ratio=None, delta=None, share_delta=None, techniques=TECHNIQUES):
    n = len(techniques)
    share = np.linspace(0.05, 0.2, n) if share is None else share
    share_val = share if share_val is None else share_val
    ratio = np.linspace(1.0, 3.0, n) if ratio is None else ratio
    delta = np.linspace(0.01, 0.05, n) if delta is None else delta
    share_delta = np.linspace(0.5, 0.9, n) if share_delta is None else share_delta
    shares = []
    for split, values in (("test", share), ("validation", share_val)):
        for t, v in zip(techniques, values):
            for category in ("extremist_framing", "identity_term", "slur", "topical"):
                shares.append({"technique": t, "member": np.nan, "split": split, "category": category,
                               "share_of_positive_mass": v if category == "identity_term" else (1 - v) / 3})
    fpr = pd.DataFrame({"technique": techniques, "split": "test", "fpr_ratio": ratio})
    cf = []
    for t, d, s in zip(techniques, delta, share_delta):
        for population in ("non_extremist", "extremist"):
            for operation in ("remove", "substitute"):
                cf.append({"technique": t, "split": "test", "population": population, "operation": operation,
                           "mean_delta": d if (population, operation) == ("non_extremist", "remove") else -9.0,
                           "share_delta_positive": s})
    return pd.DataFrame(shares), fpr, pd.DataFrame(cf)


def _row(table, x, y="fpr_ratio"):
    return table[(table["x_measure"] == x) & (table["y_measure"] == y)].iloc[0]


class TestRelation:
    def test_monotone_relation_has_rho_one_and_exact_p(self):
        table, _ = build(*_tables(), TECHNIQUES)
        row = _row(table, "identity_share_positive_mass")
        assert row["spearman_rho"] == pytest.approx(1.0)
        assert row["permutation_method"] == "exact" and row["n_permutations"] == math.factorial(8)
        assert row["permutation_p"] == pytest.approx(2 / math.factorial(8))
        assert row["n_techniques"] == 8 and row["excluded_techniques"] == ""
        assert 0.7 < row["abs_rho_needed_p05"] < 0.76
        assert (table["holm_p"] >= table["permutation_p"]).all()

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
        shares, fpr, cf = _tables()
        member = shares[shares["split"] == "test"].assign(member="A", share_of_positive_mass=0.99)
        table, measures = build(pd.concat([shares, member]), fpr, cf, TECHNIQUES)
        assert measures["identity_share_positive_mass"].max() < 0.99


class TestRefusals:
    def test_technique_missing_from_rq4_is_refused(self):
        shares, fpr, cf = _tables()
        with pytest.raises(RelationError, match="no value for"):
            build(shares, fpr.iloc[1:], cf, TECHNIQUES)

    def test_duplicate_fpr_rows_are_refused(self):
        shares, fpr, cf = _tables()
        with pytest.raises(RelationError, match="more than one row"):
            build(shares, pd.concat([fpr, fpr.iloc[:1]]), cf, TECHNIQUES)

    def test_duplicate_counterfactual_rows_are_refused(self):
        shares, fpr, cf = _tables()
        with pytest.raises(RelationError, match="more than one row"):
            build(shares, fpr, pd.concat([cf, cf.iloc[:1]]), TECHNIQUES)

    def test_share_outside_unit_interval_is_refused(self):
        share = np.linspace(0.05, 0.2, 8)
        share[0] = 1.4
        with pytest.raises(RelationError, match=r"\[0, 1\]"):
            build(*_tables(share=share), TECHNIQUES)

    def test_fpr_rows_for_another_split_are_refused(self):
        shares, fpr, cf = _tables()
        with pytest.raises(RelationError, match="split other than test"):
            build(shares, fpr.assign(split="validation"), cf, TECHNIQUES)

    def test_too_few_techniques_are_refused(self):
        few = TECHNIQUES[:3]
        with pytest.raises(RelationError, match="need at least 4"):
            build(*_tables(techniques=few), few)

    def test_missing_column_is_refused(self):
        shares, fpr, cf = _tables()
        with pytest.raises(RelationError, match="missing columns"):
            build(shares, fpr.drop(columns="fpr_ratio"), cf, TECHNIQUES)


def test_more_than_nine_techniques_use_seeded_monte_carlo():
    techniques = [f"{i:02d}_T{i}" for i in range(1, 11)]
    table, _ = build(*_tables(techniques=techniques), techniques)
    row = _row(table, "identity_share_positive_mass")
    assert row["permutation_method"] == "monte_carlo" and row["n_permutations"] == 100_000
    assert 0 < row["permutation_p"] < 1e-3
    again, _ = build(*_tables(techniques=techniques), techniques)
    assert again["permutation_p"].equals(table["permutation_p"])
