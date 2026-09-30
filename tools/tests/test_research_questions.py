"""Behavioural tests for the RQ2-RQ4 tools on synthetic, text-free inputs."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIR))

from conftest import FRAMING_WORDS, IDENTITY_WORDS, TOPICAL_WORDS, synthetic_attributions
from attributions import write_run
from conftest import base_meta

ALL_WORDS = FRAMING_WORDS + IDENTITY_WORDS + TOPICAL_WORDS


class TestCategorize:
    def test_words_take_lexicon_categories(self):
        from categorize_attributions import build_word_index, categorize_word
        from lexicons import load_lexicons

        index, _ = build_word_index(load_lexicons())
        assert categorize_word("KKK", index)[0] == "extremist_framing"
        assert categorize_word("muslim", index)[0] == "identity_term"
        assert categorize_word("coffee", index)[0] == "topical"

    def test_precedence_and_overlap_are_reported(self):
        from categorize_attributions import categorize_word

        index = {"slur": {"x"}, "extremist_framing": {"x", "y"}, "identity_term": {"y"}}
        category, matched = categorize_word("x", index)
        assert category == "slur" and matched == ["slur", "extremist_framing"]
        category, matched = categorize_word("y", index)
        assert category == "extremist_framing" and matched == ["extremist_framing", "identity_term"]

    def test_shares_sum_to_one_and_cover_all_categories(self, results_dir, make_run):
        from categorize_attributions import categorize_all

        make_run("01_T", "run_a", ALL_WORDS)
        table, _ = categorize_all(["01_T"], top_k=50, results_dir=results_dir)
        assert sorted(table["category"]) == sorted(["slur", "extremist_framing", "identity_term", "topical"])
        assert table["share_of_words"].sum() == pytest.approx(1.0)
        assert table["share_of_abs_mass"].sum() == pytest.approx(1.0)
        assert table["share_of_positive_mass"].sum() == pytest.approx(1.0)
        assert table.loc[table["category"] == "slur", "n_words"].item() == 0
        derived = results_dir / "01_T" / "attributions"
        assert (derived / "run_a__categorized.csv").exists()
        assert (derived / "run_a__category_shares.csv").exists()

    def test_top_k_limits_the_words_considered(self, results_dir, make_run):
        from categorize_attributions import categorize_all

        make_run("01_T", "run_a", ALL_WORDS)
        table, _ = categorize_all(["01_T"], top_k=3, results_dir=results_dir)
        assert table["n_words"].sum() == 3


class TestValidateShap:
    def _coefficients(self, words, signs, results_dir, technique):
        folder = results_dir / technique / "attributions"
        folder.mkdir(parents=True, exist_ok=True)
        coef = pd.DataFrame({"word": words, "coefficient": signs})
        coef.to_csv(folder / "coefficients.csv", index=False)

    def test_attributions_matching_coefficients_pass(self, results_dir):
        from validate_shap import validate

        rng = np.random.default_rng(1)
        words = [f"w{i}" for i in range(60)]
        coef = rng.normal(0, 1, 60)
        frame = pd.DataFrame(
            {"word": words, "mean_attribution": coef * 0.1 + rng.normal(0, 0.005, 60),
             "support": rng.integers(1, 50, 60)}
        )
        frame["mean_abs_attribution"] = frame["mean_attribution"].abs() + 0.01
        write_run("01_T", "run_a", frame, base_meta("01_T", "run_a"), results_dir=results_dir)
        self._coefficients(words, coef, results_dir, "01_T")
        stats, _ = validate("01_T", results_dir=results_dir)
        assert stats["passed"]
        assert stats["top_k_sign_agreement"] == 1.0

    def test_sign_flipped_attributions_fail(self, results_dir):
        from validate_shap import validate

        rng = np.random.default_rng(2)
        words = [f"w{i}" for i in range(60)]
        coef = rng.normal(0, 1, 60)
        frame = pd.DataFrame(
            {"word": words, "mean_attribution": -coef * 0.1, "support": rng.integers(1, 50, 60)}
        )
        frame["mean_abs_attribution"] = frame["mean_attribution"].abs() + 0.01
        write_run("01_T", "run_a", frame, base_meta("01_T", "run_a"), results_dir=results_dir)
        self._coefficients(words, coef, results_dir, "01_T")
        stats, out = validate("01_T", results_dir=results_dir)
        assert not stats["passed"]
        assert out.exists()


class TestCompareReliance:
    def test_member_mean_and_stability_rows(self, results_dir, make_run, tmp_path):
        from categorize_attributions import categorize_all
        from compare_reliance import ROLE_MEMBER_MEAN, ROLE_WHOLE, add_member_means, stability_table, widen_shares

        make_run("11_E", "whole_s1", ALL_WORDS, seed=1)
        make_run("11_E", "whole_s2", ALL_WORDS, seed=2)
        make_run("11_E", "member_a", ALL_WORDS, seed=3, member="A_anchor")
        make_run("11_E", "member_b", ALL_WORDS, seed=4, member="B_dynabench")
        table, _ = categorize_all(["11_E"], top_k=50, results_dir=results_dir)

        wide = add_member_means(widen_shares(table))
        assert (wide["role"] == ROLE_MEMBER_MEAN).sum() == 1
        assert (wide["role"] == ROLE_WHOLE).sum() == 2
        share_columns = [c for c in wide.columns if c.startswith("share_of_positive_mass__")]
        assert wide[share_columns].sum(axis=1).round(6).eq(1.0).all()

        stability = stability_table(wide, top_k=50, results_dir=results_dir)
        assert len(stability) == 1
        assert stability.iloc[0]["n_runs"] == 2
        assert 0.0 <= stability.iloc[0]["mean_pairwise_jaccard_top50"] <= 1.0


class TestIdentityFPR:
    def _dataset(self, n_neg=280, n_pos=170):
        # Synthetic posts: no real dataset text. Identity term in every other negative.
        rows = []
        for i in range(n_neg):
            rows.append(("ex_%06d" % i, "benign post about muslim neighbours" if i % 2 == 0 else "benign post about gardening", 0))
        for j in range(n_pos):
            rows.append(("ex_%06d" % (n_neg + j), "hostile post", 1))
        return pd.DataFrame(rows, columns=["row_id", "text", "label"])

    def test_rates_are_split_by_identity_terms(self, make_probs):
        from identity_fpr import analyze
        from probs_artifact import load_probs

        paths = make_probs("11_T")
        probs = load_probs(paths["test"])
        processed = self._dataset()
        # Force false positives only on the identity posts.
        probs.loc[(probs["y_true"] == 0) & (probs.index % 2 == 0), "y_prob"] = 0.99
        probs.loc[(probs["y_true"] == 0) & (probs.index % 2 == 1), "y_prob"] = 0.01
        summary, per_term = analyze("11_T", "test", threshold=0.5, processed=processed,
                                    probs=probs, terms=["muslim"])
        assert summary["n_negative"] == 280
        assert summary["n_negative_with_identity_terms"] == 140
        assert summary["fpr_with_identity_terms"] == pytest.approx(1.0)
        assert summary["fpr_without_identity_terms"] == pytest.approx(0.0)
        assert summary["fisher_p_value"] < 1e-6
        assert per_term.iloc[0]["term"] == "muslim"
        assert "text" not in per_term.columns

    def test_label_mismatch_aborts(self, make_probs):
        from identity_fpr import IdentityFPRError, analyze
        from probs_artifact import load_probs

        paths = make_probs("11_T")
        probs = load_probs(paths["test"])
        processed = self._dataset()
        processed.loc[0, "label"] = 1  # dataset disagrees with the artifact
        with pytest.raises(IdentityFPRError, match="disagrees"):
            analyze("11_T", "test", threshold=0.5, processed=processed, probs=probs, terms=["muslim"])

    def test_phrase_terms_match_at_word_boundaries(self):
        from identity_fpr import identity_matches

        texts = ["the asylum seekers arrived", "asylumseekers", "Trans rights"]
        matches = identity_matches(texts, ["asylum seekers", "trans"])
        assert matches == [["asylum seekers"], [], ["trans"]]
