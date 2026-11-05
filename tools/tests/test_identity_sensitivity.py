"""Behavioural and adversarial tests for tools/identity_sensitivity.py (synthetic data, fake predictor)."""

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIR))

import identity_sensitivity as isens
from identity_fpr import analyze as fpr_analyze
from identity_sensitivity import (
    AMBIGUOUS_IDENTITY_TERMS,
    CF_TABLE_COLUMNS,
    IdentitySensitivityError,
    build_restricted_population,
    check_ambiguous_terms,
    counterfactual_sensitivity,
    fpr_sensitivity,
    gap_rule,
    lexicon_gaps,
    match_classes,
    write_cf_table,
)
from lexicons import load_lexicons

ROW_ID_RE = re.compile(r"\bex_\d{6}\b")
TERMS = ["muslim", "white", "man", "asylum seekers"]
AMBIGUOUS = frozenset({"white", "man"})
GROUP_TEXT = "a benign note about muslim neighbours"
COLOUR_TEXT = "a benign note about a white fence"
MIXED_TEXT = "a white muslim man wrote a benign note"
PLAIN_TEXT = "a benign note about gardening"
HOSTILE_TEXT = "a hostile rant naming asylum seekers"
HOSTILE_COLOUR = "a hostile rant, man"
TEXT_PROB = {GROUP_TEXT: 0.7, COLOUR_TEXT: 0.6, MIXED_TEXT: 0.65, PLAIN_TEXT: 0.2,
             HOSTILE_TEXT: 0.9, HOSTILE_COLOUR: 0.85, "a hostile rant": 0.8}


def _dataset(n_neg=280, n_pos=170):
    """Synthetic posts with the frozen test shape; no dataset text."""
    neg_cycle = [GROUP_TEXT, COLOUR_TEXT, MIXED_TEXT, PLAIN_TEXT]
    pos_cycle = [HOSTILE_TEXT, HOSTILE_COLOUR, "a hostile rant", "a hostile rant"]
    rows = [(f"ex_{i:06d}", neg_cycle[i % 4], 0) for i in range(n_neg)]
    rows += [(f"ex_{n_neg + j:06d}", pos_cycle[j % 4], 1) for j in range(n_pos)]
    return pd.DataFrame(rows, columns=["row_id", "text", "label"])


def _probs(make_probs, processed):
    from probs_artifact import load_probs

    probs = load_probs(make_probs("11_T")["test"])
    probs["y_prob"] = processed["text"].map(TEXT_PROB).to_numpy()
    return probs


def _fake_predictor(base, threshold=0.5):
    """Scores the originals as the artifact; any edited text scores 0.1."""
    return SimpleNamespace(technique="11_T", threshold=threshold,
                           predict=lambda texts: np.array([base.get(t, 0.1) for t in texts], dtype=float))


class TestAmbiguousList:
    def test_declared_list_is_a_strict_subset_of_the_committed_lexicon(self):
        terms = load_lexicons()["identity_term"]
        ambiguous = check_ambiguous_terms(AMBIGUOUS_IDENTITY_TERMS, terms)
        assert ambiguous == frozenset(AMBIGUOUS_IDENTITY_TERMS)
        assert all(reason.strip() for reason in AMBIGUOUS_IDENTITY_TERMS.values())

    def test_term_outside_the_lexicon_is_refused(self):
        # 'whites' is a gap candidate, not a lexicon entry: declaring it ambiguous would be silent drift.
        with pytest.raises(IdentitySensitivityError, match="not in the identity lexicon.*whites"):
            check_ambiguous_terms(["white", "whites"], TERMS)

    def test_empty_list_is_refused(self):
        with pytest.raises(IdentitySensitivityError, match="empty"):
            check_ambiguous_terms([], TERMS)

    def test_list_covering_the_whole_lexicon_is_refused(self):
        with pytest.raises(IdentitySensitivityError, match="whole identity lexicon"):
            check_ambiguous_terms(TERMS, TERMS)

    def test_match_classes(self):
        texts = [GROUP_TEXT, COLOUR_TEXT, MIXED_TEXT, PLAIN_TEXT, "White MAN"]
        assert match_classes(texts, TERMS, AMBIGUOUS) == [
            "unambiguous", "ambiguous_only", "unambiguous", "none", "ambiguous_only"]


class TestFPR:
    def test_variants_on_synthetic_data(self, make_probs):
        processed = _dataset()
        probs = _probs(make_probs, processed)
        rows = {r["variant"]: r for r in fpr_sensitivity("11_T", processed, probs, TERMS, AMBIGUOUS, 0.5)}
        full, _ = fpr_analyze("11_T", "test", 0.5, processed=processed, probs=probs, terms=TERMS)
        assert rows["full_lexicon"]["fpr_with_identity_terms"] == full["fpr_with_identity_terms"]
        assert rows["full_lexicon"]["fisher_p_value"] == full["fisher_p_value"]
        assert rows["full_lexicon"]["n_negative_with_identity_terms"] == 210
        excl = rows["exclude_ambiguous_only"]
        assert excl["n_negative_excluded"] == 70 and excl["n_negative"] == 210
        assert excl["n_negative_with_identity_terms"] == 140
        assert excl["n_positive_excluded"] == 43 and excl["n_positive_with_identity_terms"] == 43
        reduced = rows["reduced_lexicon"]
        assert reduced["n_negative"] == 280 and reduced["n_negative_with_identity_terms"] == 140
        # The colour posts (all flagged) move into the "without" group.
        assert reduced["fpr_without_identity_terms"] == pytest.approx(0.5)


class TestCounterfactual:
    def _setup(self, make_probs):
        processed = _dataset()
        probs = _probs(make_probs, processed)
        return build_restricted_population("11_T", processed, probs, TERMS, AMBIGUOUS)

    def test_restricted_population_and_operations(self, make_probs):
        population, pattern, unamb, full_counts = self._setup(make_probs)
        assert full_counts == {"non_extremist": 210, "extremist": 86}
        assert len(population) == 140 + 43
        assert COLOUR_TEXT not in set(population["text"]) and HOSTILE_COLOUR not in set(population["text"])
        # remove_unambiguous keeps the ambiguous words; remove deletes every match.
        assert unamb.sub("", MIXED_TEXT).split() == ["a", "white", "man", "wrote", "a", "benign", "note"]
        rows = counterfactual_sensitivity("11_T", _fake_predictor(TEXT_PROB), population, pattern,
                                          unamb, 0.5, full_counts)
        by = {(r["population"], r["operation"]): r for r in rows}
        neg = by[("non_extremist", "remove")]
        assert neg["n_posts"] == 140 and neg["n_posts_full_population"] == 210
        assert neg["n_flagged_original"] == 140 and neg["n_flip_pos_to_neg"] == 140

    def test_predictor_disagreeing_with_artifact_is_refused(self, make_probs):
        population, pattern, unamb, full_counts = self._setup(make_probs)
        drifted = {t: min(1.0, p + 0.05) for t, p in TEXT_PROB.items()}
        with pytest.raises(IdentitySensitivityError, match="disagrees with the committed"):
            counterfactual_sensitivity("11_T", _fake_predictor(drifted), population, pattern,
                                       unamb, 0.5, full_counts)

    def test_threshold_mismatch_is_refused(self, make_probs):
        population, pattern, unamb, full_counts = self._setup(make_probs)
        with pytest.raises(IdentitySensitivityError, match="threshold"):
            counterfactual_sensitivity("11_T", _fake_predictor(TEXT_PROB, threshold=0.4), population,
                                       pattern, unamb, 0.5, full_counts)

    def test_empty_restricted_population_is_refused(self, make_probs):
        population, pattern, unamb, full_counts = self._setup(make_probs)
        with pytest.raises(IdentitySensitivityError, match="restricted population is empty"):
            counterfactual_sensitivity("11_T", _fake_predictor(TEXT_PROB), population.iloc[0:0],
                                       pattern, unamb, 0.5, full_counts)

    def test_table_is_text_free_and_refuses_a_foreign_schema(self, make_probs, tmp_path):
        population, pattern, unamb, full_counts = self._setup(make_probs)
        rows = counterfactual_sensitivity("11_T", _fake_predictor(TEXT_PROB), population, pattern,
                                          unamb, 0.5, full_counts)
        path = write_cf_table(rows, path=tmp_path / "cf.csv", results_dir=tmp_path)
        assert list(pd.read_csv(path).columns) == list(CF_TABLE_COLUMNS)
        content = path.read_text()
        assert not ROW_ID_RE.search(content)
        for text in TEXT_PROB:
            assert text not in content
        foreign = tmp_path / "foreign.csv"
        pd.DataFrame({"technique": ["x"], "row_id": ["ex_000001"]}).to_csv(foreign, index=False)
        with pytest.raises(IdentitySensitivityError, match="columns differ"):
            write_cf_table(rows, path=foreign, results_dir=tmp_path)


    def test_holm_families_and_flip_rate(self, tmp_path):
        """The three Holm families are computed over the right rows and the flip rate from the counts."""
        rows = []
        for t, technique in enumerate(("01_A", "02_B")):
            for label, name in ((0, "non_extremist"), (1, "extremist")):
                for k, op in enumerate(("remove", "remove_unambiguous")):
                    row = {c: 0.0 for c in isens.CF_ROW_COLUMNS}
                    row.update(technique=technique, split="test", population=name, y_true=label,
                               operation=op, n_flagged_original=20, n_flip_pos_to_neg=5 - k,
                               n_flip_neg_to_pos=0, wilcoxon_p=0.5,
                               mcnemar_exact_p=0.001 * (1 + t + 2 * label + 4 * k))
                    rows.append(row)
        table = pd.read_csv(write_cf_table(rows, path=tmp_path / "cf.csv", results_dir=tmp_path))
        # Both techniques are in one family: 8 rows, 4 remove rows, 2 extremist remove rows.
        by = table.set_index(["technique", "population", "operation"])
        r = by.loc[("01_A", "extremist", "remove")]
        assert r["flagged_flip_rate"] == pytest.approx(5 / 20)
        assert by.loc[("01_A", "extremist", "remove_unambiguous"), "flagged_flip_rate"] == pytest.approx(4 / 20)
        assert r["mcnemar_p_holm"] == pytest.approx(0.003 * 6)                   # rank 3 of 8
        assert r["mcnemar_p_holm_remove"] == pytest.approx(0.003 * 2)            # rank 3 of 4
        assert r["mcnemar_p_holm_remove_extremist"] == pytest.approx(0.003 * 2)  # rank 1 of 2
        assert table.loc[table["operation"] == "remove_unambiguous", "mcnemar_p_holm_remove"].isna().all()
        assert table.loc[table["y_true"] == 0, "mcnemar_p_holm_remove_extremist"].isna().all()


class TestLexiconGaps:
    def test_rules(self):
        terms = ["white", "gay", "islam", "refugees", "woman"]
        assert gap_rule("whites", terms) == ("inflection", "white")
        assert gap_rule("gays", terms) == ("inflection", "gay")
        assert gap_rule("women", terms) == ("inflection", "woman")
        assert gap_rule("refugeesnotwelcome", terms) == ("compound", "refugees")
        assert gap_rule("mulatto", terms) == ("group_noun", "")
        assert gap_rule("white", terms) is None       # already in the lexicon
        assert gap_rule("island", terms) is None      # 'islam' is not a substring
        assert gap_rule("gayness", terms) is None     # 'gay' is too short to count as a compound

    def test_lexicon_gaps_on_a_synthetic_run(self):
        lexicons = {"slur": [], "extremist_framing": ["stopwhitegenocide"],
                    "identity_term": ["white", "muslim"]}
        frame = pd.DataFrame({
            "word": ["whites", "white", "kill", "stopwhitegenocide", "muslim"],
            "mean_abs_attribution": [0.5, 0.4, 0.3, 0.2, 0.1],
            "mean_attribution": [0.5, 0.4, 0.3, 0.2, 0.1],
            "support": [2, 3, 4, 1, 2],
        })
        found, summary = lexicon_gaps("11_T", frame, "run", lexicons, frozenset({"white"}), top_k=5)
        assert [c["word"] for c in found] == ["whites", "stopwhitegenocide"]
        assert found[1]["other_lexicon_category"] == "extremist_framing"
        assert summary["n_identity_term"] == 2 and summary["n_identity_term_ambiguous"] == 1
        assert summary["ambiguous_words"] == "white"


def test_cli_refuses_a_bad_ambiguous_list(monkeypatch, capsys, tmp_path):
    # Point the outputs at tmp_path: if the subset guard regresses, the CLI must not
    # overwrite the committed results_summary/rq tables.
    gaps_csv, top_k_csv = tmp_path / "gaps.csv", tmp_path / "top_k.csv"
    monkeypatch.setattr(isens, "RQ2_IDENTITY_LEXICON_GAPS_SENSITIVITY_CSV", gaps_csv)
    monkeypatch.setattr(isens, "RQ2_IDENTITY_TOP_K_SENSITIVITY_CSV", top_k_csv)
    monkeypatch.setattr(isens, "AMBIGUOUS_IDENTITY_TERMS", {"whites": "not a lexicon entry"})
    assert isens.main(["lexicon-gaps"]) == 1
    assert "not in the identity lexicon" in capsys.readouterr().err
    assert not gaps_csv.exists() and not top_k_csv.exists()
