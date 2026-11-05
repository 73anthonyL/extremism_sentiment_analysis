"""rq2_lexicon_coverage.csv: counts per category, and refusal when stale."""

import pandas as pd
import pytest

from categorize_attributions import (
    COVERAGE_COLUMNS,
    check_coverage,
    lexicon_coverage,
    write_coverage,
)
from lexicons import load_lexicons
from repo_paths import RQ2_LEXICON_COVERAGE_CSV

LEXICONS = {
    "slur": [],
    "extremist_framing": ["jihad", "white genocide", "great replacement"],
    "identity_term": ["muslims", "refugees"],
}


def test_counts_single_and_multi_word_entries():
    table = lexicon_coverage(LEXICONS).set_index("category")
    assert list(table.reset_index().columns) == list(COVERAGE_COLUMNS)
    assert table.loc["extremist_framing", "n_entries"] == 3
    assert table.loc["extremist_framing", "n_single_word"] == 1
    assert table.loc["extremist_framing", "n_multi_word_unmatchable"] == 2
    assert table.loc["identity_term", "n_multi_word_unmatchable"] == 0
    assert table.loc["slur", "n_entries"] == 0


def test_round_trip_passes(tmp_path):
    path = write_coverage(LEXICONS, tmp_path / "coverage.csv")
    assert check_coverage(LEXICONS, path) == path


def test_stale_table_after_lexicon_change_is_refused(tmp_path):
    path = write_coverage(LEXICONS, tmp_path / "coverage.csv")
    changed = dict(LEXICONS, identity_term=LEXICONS["identity_term"] + ["asylum seekers"])
    with pytest.raises(ValueError, match="does not match"):
        check_coverage(changed, path)


def test_hand_edited_count_is_refused(tmp_path):
    path = write_coverage(LEXICONS, tmp_path / "coverage.csv")
    table = pd.read_csv(path)
    table.loc[table.category == "extremist_framing", "n_multi_word_unmatchable"] = 0
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="does not match"):
        check_coverage(LEXICONS, path)


def test_missing_table_is_refused(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        check_coverage(LEXICONS, tmp_path / "absent.csv")


def test_committed_table_matches_committed_lexicons():
    assert check_coverage(load_lexicons(), RQ2_LEXICON_COVERAGE_CSV)


# --- entries the word-level masker splits (WP3 review) ---------------------

def test_hyphenated_and_punctuated_entries_are_counted_unmatchable():
    """A hyphen or a slash splits an entry under the masker exactly as a space does."""
    lexicons = {
        "slur": ["w1", "w2-w3"],
        "extremist_framing": ["jihad", "anti-x", "9/11", "/pol/", "((()))", "two words"],
        "identity_term": ["muslims"],
    }
    table = lexicon_coverage(lexicons).set_index("category")
    assert table.loc["slur", "n_single_word"] == 1
    assert table.loc["slur", "n_multi_word_unmatchable"] == 1
    assert table.loc["extremist_framing", "n_single_word"] == 1
    assert table.loc["extremist_framing", "n_multi_word_unmatchable"] == 5
    assert table.loc["identity_term", "n_multi_word_unmatchable"] == 0


def test_a_table_that_counts_only_spaced_entries_is_refused(tmp_path):
    """The old space-only count under-reports; a table written that way must fail the check."""
    lexicons = {"slur": ["w2-w3"], "extremist_framing": ["two words"], "identity_term": []}
    path = write_coverage(lexicons, tmp_path / "coverage.csv")
    table = pd.read_csv(path)
    table.loc[table.category == "slur", "n_multi_word_unmatchable"] = 0
    table.loc[table.category == "slur", "n_single_word"] = 1
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="does not match"):
        check_coverage(lexicons, path)


def test_word_index_never_holds_an_entry_the_masker_would_split():
    from categorize_attributions import build_word_index
    from lexicons import masker_tokens

    index, skipped = build_word_index(load_lexicons())
    for category, words in index.items():
        for word in words:
            assert masker_tokens(word) == [word], (category, "an unmatchable entry is in the word index")
    assert skipped == int(pd.read_csv(RQ2_LEXICON_COVERAGE_CSV)["n_multi_word_unmatchable"].sum())


def test_lexicon_masker_pattern_equals_the_notebook_kit_pattern():
    import lexicons
    import notebook_kit

    assert lexicons.MASKER_PATTERN == notebook_kit.MASKER_PATTERN
