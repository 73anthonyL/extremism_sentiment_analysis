"""Behavioural and adversarial tests for tools/identity_swap.py with a fake predictor."""

import copy
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIR))

import identity_swap as isw
from identity_swap import (
    TABLE_COLUMNS,
    IdentitySwapError,
    analyze,
    build_population,
    check_swap_map,
    check_text_free,
    lexicon_sections,
    match_case,
    read_swap_map,
    remove_text,
    substitute_text,
    swap_one_source,
    swap_text,
    write_directions,
    write_outputs,
    write_table,
)
from lexicons import load_lexicons, phrase_pattern

ROW_ID_RE = re.compile(r"\bex_\d{6}\b")

# The preregistered panels (docs/preregistrations/WP7b_identity_swap.md).
PANELS = {
    ("religion", "person"): ["muslim", "christian", "hindu", "buddhist"],
    ("religion", "person_plural"): ["muslims", "christians", "hindus", "buddhists"],
    ("religion", "faith"): ["islam", "christianity", "judaism", "hinduism"],
    ("ethnicity_nationality", "person"): ["african", "asian", "mexican", "european"],
    ("ethnicity_nationality", "person_plural"): ["africans", "asians", "mexicans", "europeans"],
    ("migration", "person"): ["immigrant", "refugee", "foreigner", "migrant"],
    ("migration", "person_plural"): ["immigrants", "refugees", "foreigners", "migrants"],
    ("gender_sexuality", "identity"): ["gay", "lesbian", "bisexual", "transgender"],
}

# A small synthetic lexicon and map for the refusal tests.
LEXICONS = {"slur": [], "extremist_framing": ["jihad"],
            "identity_term": ["muslim", "christian", "hindu", "buddhist", "jewish", "refugee", "white"]}
SECTIONS = {"muslim": "religion", "christian": "religion", "hindu": "religion", "buddhist": "religion",
            "jewish": "religion", "refugee": "migration", "white": "ethnicity_nationality"}


def _row(category, form, group, source, targets=("", "", ""), note=""):
    return {"category": category, "form": form, "group": group, "source": source,
            "target_1": targets[0], "target_2": targets[1], "target_3": targets[2], "note": note}


def _good_rows():
    return [
        _row("religion", "person", "islam", "muslim", ("christian", "hindu", "buddhist")),
        _row("religion", "person", "christianity", "christian", ("muslim", "hindu", "buddhist")),
        _row("religion", "person", "hinduism", "hindu", ("muslim", "christian", "buddhist")),
        _row("religion", "person", "buddhism", "buddhist", ("muslim", "christian", "hindu")),
        _row("religion", "person", "judaism", "jewish", ("muslim", "christian", "hindu")),
        _row("migration", "excluded", "", "refugee", note="no same-form target in this test"),
        _row("ethnicity_nationality", "excluded", "", "white", note="ambiguous"),
    ]


def _with(rows, source, **changes):
    rows = copy.deepcopy(rows)
    for row in rows:
        if row["source"] == source:
            row.update(changes)
    return rows


class TestCommittedMap:
    def test_committed_map_passes_and_covers_the_lexicon(self):
        lexicons = load_lexicons()
        swapped, excluded = check_swap_map(read_swap_map(), lexicons, lexicon_sections())
        assert set(swapped) | set(excluded) == set(lexicons["identity_term"])
        assert not set(swapped) & set(excluded)

    def test_committed_map_follows_the_panel_rule(self):
        swapped, _ = check_swap_map(read_swap_map(), load_lexicons(), lexicon_sections())
        for source, spec in swapped.items():
            panel = PANELS[(spec["category"], spec["form"])]
            expected = [p for p in panel if swapped[p]["group"] != spec["group"]][:3]
            assert spec["targets"] == expected, source

    def test_lexicon_sections_cover_every_identity_entry(self):
        sections = lexicon_sections()
        assert set(sections) == set(load_lexicons()["identity_term"])
        assert sections["muslims"] == "religion" and sections["refugees"] == "migration"


class TestMapRefusals:
    def test_good_synthetic_map_is_accepted(self):
        swapped, excluded = check_swap_map(_good_rows(), LEXICONS, SECTIONS, ambiguous={"white"})
        assert swapped["jewish"]["targets"] == ["muslim", "christian", "hindu"]
        assert set(excluded) == {"refugee", "white"}

    def test_target_outside_the_lexicon_is_refused(self):
        rows = _with(_good_rows(), "muslim", target_3="people")
        with pytest.raises(IdentitySwapError, match="'people'.*not an identity-lexicon entry"):
            check_swap_map(rows, LEXICONS, SECTIONS, ambiguous={"white"})

    def test_source_outside_the_lexicon_is_refused(self):
        rows = _good_rows() + [_row("religion", "person", "x", "pagan", ("muslim", "christian", "hindu"))]
        with pytest.raises(IdentitySwapError, match="source 'pagan' is not an identity-lexicon entry"):
            check_swap_map(rows, LEXICONS, SECTIONS, ambiguous={"white"})

    def test_cross_category_swap_is_refused(self):
        lexicons = copy.deepcopy(LEXICONS)
        rows = _with(_good_rows(), "refugee", form="person", group="refugee",
                     target_1="muslim", target_2="christian", target_3="hindu")
        with pytest.raises(IdentitySwapError, match="target 'muslim' of 'refugee'.*must not cross categories"):
            check_swap_map(rows, lexicons, SECTIONS, ambiguous={"white"})

    def test_row_category_disagreeing_with_the_lexicon_section_is_refused(self):
        rows = _with(_good_rows(), "jewish", category="ethnicity_nationality")
        with pytest.raises(IdentitySwapError, match="must not cross categories"):
            check_swap_map(rows, LEXICONS, SECTIONS, ambiguous={"white"})

    def test_form_change_is_refused(self):
        rows = _with(_good_rows(), "buddhist", form="person_plural")
        with pytest.raises(IdentitySwapError, match="changes the form"):
            check_swap_map(rows, LEXICONS, SECTIONS, ambiguous={"white"})

    def test_same_group_target_is_refused(self):
        rows = _with(_good_rows(), "jewish", group="islam")
        with pytest.raises(IdentitySwapError, match="same group"):
            check_swap_map(rows, LEXICONS, SECTIONS, ambiguous={"white"})

    def test_ambiguous_source_or_target_is_refused(self):
        with pytest.raises(IdentitySwapError, match="ambiguous"):
            check_swap_map(_good_rows(), LEXICONS, SECTIONS, ambiguous={"white", "hindu"})

    def test_target_in_another_lexicon_is_refused(self):
        lexicons = copy.deepcopy(LEXICONS)
        lexicons["extremist_framing"].append("hindu")
        with pytest.raises(IdentitySwapError, match="matched by the extremist_framing lexicon"):
            check_swap_map(_good_rows(), lexicons, SECTIONS, ambiguous={"white"})

    def test_missing_row_and_duplicate_targets_are_refused(self):
        rows = [r for r in _good_rows() if r["source"] != "refugee"]
        with pytest.raises(IdentitySwapError, match="without a row"):
            check_swap_map(rows, LEXICONS, SECTIONS, ambiguous={"white"})
        rows = _with(_good_rows(), "muslim", target_3="christian")
        with pytest.raises(IdentitySwapError, match="3 distinct targets"):
            check_swap_map(rows, LEXICONS, SECTIONS, ambiguous={"white"})


SWAP_MAP = {
    "muslims": {"category": "religion", "form": "person_plural", "group": "islam",
                "targets": ["christians", "hindus", "buddhists"]},
    "refugees": {"category": "migration", "form": "person_plural", "group": "refugee",
                 "targets": ["immigrants", "foreigners", "migrants"]},
}


class TestEditedText:
    def test_case_follows_the_match(self):
        assert match_case("MUSLIMS", "christians") == "CHRISTIANS"
        assert match_case("Muslims", "christians") == "Christians"
        assert match_case("muslims", "christians") == "christians"

    def test_operations_edit_only_swapped_sources(self):
        pattern = phrase_pattern(list(SWAP_MAP))
        text = "Muslims and white refugees,  muslims"
        assert swap_text(text, pattern, SWAP_MAP, 1) == "Hindus and white foreigners, hindus"
        assert remove_text(text, pattern) == "and white ,"
        assert substitute_text(text, pattern, "group") == "Group and white group, group"
        assert swap_one_source(text, pattern, "refugees", "migrants") == "Muslims and white migrants, muslims"


IDENTITY_TEXT = "a benign note about muslims nearby"
PLAIN_TEXT = "a benign note about gardening"
HOSTILE_TEXT = "a hostile rant naming refugees"


def _dataset(n_neg=280, n_pos=170):
    rows = [(f"ex_{i:06d}", IDENTITY_TEXT if i % 2 == 0 else PLAIN_TEXT, 0) for i in range(n_neg)]
    rows += [(f"ex_{n_neg + j:06d}", HOSTILE_TEXT if j % 2 == 0 else "a hostile rant", 1) for j in range(n_pos)]
    return pd.DataFrame(rows, columns=["row_id", "text", "label"])


def _fake_predictor(base, threshold=0.5, technique="11_T", edited=None):
    """Originals score as the artifact; an edited text scores `edited(text)` (default 0.1)."""
    def predict(texts):
        return np.array([base[t] if t in base else (edited(t) if edited else 0.1) for t in texts], dtype=float)

    return SimpleNamespace(technique=technique, threshold=threshold, predict=predict)


def _setup(make_probs):
    from probs_artifact import load_probs

    probs = load_probs(make_probs("11_T")["test"])
    processed = _dataset()
    text_prob = {IDENTITY_TEXT: 0.7, PLAIN_TEXT: 0.2, HOSTILE_TEXT: 0.9, "a hostile rant": 0.8}
    probs["y_prob"] = processed["text"].map(text_prob).to_numpy()
    population, pattern = build_population("11_T", processed, probs, SWAP_MAP)
    return population, pattern, text_prob


def _group_specific(text):
    """Swapping to hindus keeps the score; every other edit lowers it."""
    return 0.7 if "hindus" in text else 0.1


class TestAnalyze:
    def test_population_and_rows(self, make_probs):
        population, pattern, text_prob = _setup(make_probs)
        assert len(population) == 140 + 85
        payload = analyze("11_T", _fake_predictor(text_prob, edited=_group_specific),
                          population, pattern, SWAP_MAP, 0.5)
        rows = {(r["y_true"], r["operation"]): r for r in payload["rows"]}
        assert rows[(0, "remove")]["mean_delta"] == pytest.approx(0.6)
        # Two of three swap variants lower the score by 0.6, one keeps it.
        assert rows[(0, "swap")]["mean_delta"] == pytest.approx(0.4)
        assert rows[(0, "swap")]["mean_abs_delta"] == pytest.approx(0.4)
        assert rows[(0, "swap")]["mean_variant_spread"] == pytest.approx(0.6)
        assert rows[(0, "substitute")]["mean_variant_spread"] == pytest.approx(0.0)
        assert rows[(0, "swap")]["n_variants"] == 3 and rows[(0, "remove")]["n_variants"] == 1
        assert rows[(0, "swap")]["n_flip_pos_to_neg"] == 140  # mean 0.3 < 0.5
        directions = {(d["population"], d["source"], d["target"]): d for d in payload["directions"]}
        assert directions[("non_extremist", "muslims", "hindus")]["mean_delta"] == pytest.approx(0.0)
        assert directions[("non_extremist", "muslims", "christians")]["mean_delta"] == pytest.approx(0.6)
        assert directions[("extremist", "refugees", "migrants")]["n_posts"] == 85

    def test_table_holm_families(self, make_probs, tmp_path):
        population, pattern, text_prob = _setup(make_probs)
        payloads = [analyze(name, _fake_predictor(text_prob, technique=name), population, pattern, SWAP_MAP, 0.5)
                    for name in ("11_A", "11_B")]
        table = pd.read_csv(write_table(payloads, path=tmp_path / "t.csv"))
        assert list(table.columns) == list(TABLE_COLUMNS) and len(table) == 12
        swap = table["operation"] == "swap"
        assert table.loc[swap, "mcnemar_p_holm_swap"].notna().all()
        assert table.loc[~swap, "mcnemar_p_holm_swap"].isna().all()
        assert (table["mcnemar_p_holm"] >= table["mcnemar_exact_p"]).all()
        write_table(payloads[:1], path=tmp_path / "t.csv")
        assert len(pd.read_csv(tmp_path / "t.csv")) == 12


class TestAdversarial:
    def test_drifted_predictor_is_refused(self, make_probs):
        population, pattern, text_prob = _setup(make_probs)
        drifted = {text: min(1.0, p + 0.05) for text, p in text_prob.items()}
        with pytest.raises(IdentitySwapError, match="disagrees with the committed"):
            analyze("11_T", _fake_predictor(drifted), population, pattern, SWAP_MAP, 0.5)

    def test_threshold_mismatch_is_refused(self, make_probs):
        population, pattern, text_prob = _setup(make_probs)
        with pytest.raises(IdentitySwapError, match="threshold"):
            analyze("11_T", _fake_predictor(text_prob, threshold=0.4), population, pattern, SWAP_MAP, 0.5)

    def test_payload_with_a_row_id_is_refused(self, make_probs, results_dir):
        population, pattern, text_prob = _setup(make_probs)
        payload = analyze("11_T", _fake_predictor(text_prob), population, pattern, SWAP_MAP, 0.5)
        tampered = copy.deepcopy(payload)
        tampered["directions"][0]["note"] = "worst post ex_000012"
        with pytest.raises(IdentitySwapError, match="row id"):
            write_outputs(tampered, results_dir=results_dir)
        assert not list(results_dir.rglob("*.json"))

    def test_payload_with_text_keys_hashes_or_text_is_refused(self, make_probs):
        population, pattern, text_prob = _setup(make_probs)
        payload = analyze("11_T", _fake_predictor(text_prob), population, pattern, SWAP_MAP, 0.5)
        for key, value in (("row_id", "x"), ("example", "0123456789abcdef"), ("example", HOSTILE_TEXT)):
            tampered = copy.deepcopy(payload)
            tampered["rows"][0][key] = value
            with pytest.raises(IdentitySwapError, match="nothing was written"):
                check_text_free(tampered, population["text"].tolist())

    def test_outputs_carry_no_row_id_and_no_text(self, make_probs, results_dir, tmp_path):
        population, pattern, text_prob = _setup(make_probs)
        payload = analyze("11_T", _fake_predictor(text_prob), population, pattern, SWAP_MAP, 0.5)
        paths = [write_outputs(payload, results_dir=results_dir),
                 write_table([payload], path=tmp_path / "t.csv"),
                 write_directions([payload], path=tmp_path / "d.csv")]
        for path in paths:
            content = path.read_text()
            assert not ROW_ID_RE.search(content)
            for text in (IDENTITY_TEXT, PLAIN_TEXT, HOSTILE_TEXT, "benign note", "hostile rant"):
                assert text not in content
        assert json.loads(paths[0].read_text())["primary_operation"] == "swap"


class TestAbortWritesNothing:
    def test_second_technique_failing_verification_writes_nothing(self, monkeypatch, make_probs,
                                                                   results_dir, tmp_path):
        from probs_artifact import load_probs

        _, _, text_prob = _setup(make_probs)
        probs = load_probs(make_probs("11_T")["test"])
        probs["y_prob"] = _dataset()["text"].map(text_prob).to_numpy()
        drifted = {text: min(1.0, p + 0.05) for text, p in text_prob.items()}
        predictors = {"11_A": _fake_predictor(text_prob, technique="11_A"),
                      "11_B": _fake_predictor(drifted, technique="11_B")}

        import split_protocol

        monkeypatch.setattr(split_protocol, "build_processed_dataset", lambda: _dataset())
        monkeypatch.setattr(isw, "check_swap_map", lambda rows, lexicons, sections: (SWAP_MAP, {}))
        monkeypatch.setattr(isw, "check_substitute_word", lambda lexicons: None)
        monkeypatch.setattr(isw, "term_profile", lambda processed, swap_map: None)
        monkeypatch.setattr(isw, "load_technique_probs", lambda technique, split: probs)
        monkeypatch.setattr(isw, "load_threshold", lambda technique: 0.5)
        monkeypatch.setattr(isw, "load_technique", lambda technique, folder, device, resident: predictors[technique])
        monkeypatch.setattr(isw, "RESULTS_DIR", results_dir)
        csv_path = tmp_path / "rq" / "rq4_identity_swap.csv"
        monkeypatch.setattr(isw, "RQ4_IDENTITY_SWAP_CSV", csv_path)
        monkeypatch.setattr(isw, "RQ4_IDENTITY_SWAP_DIRECTIONS_CSV", tmp_path / "rq" / "d.csv")

        assert isw.main(["--technique", "11_A", "--technique", "11_B"]) == 1
        assert not list(results_dir.rglob("*.json")) and not csv_path.exists()
        assert isw.main(["--technique", "11_A"]) == 0
        assert csv_path.exists() and len(list(results_dir.rglob("identity_swap_test.json"))) == 1
