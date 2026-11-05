"""Behavioural and adversarial tests for tools/identity_counterfactual.py with a fake predictor."""

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

import identity_counterfactual as ic
from identity_counterfactual import (
    SUBSTITUTE_WORDS,
    TABLE_COLUMNS,
    IdentityCounterfactualError,
    analyze,
    build_population,
    check_substitute_profile,
    check_substitute_word,
    counterfactual_text,
    paired_metrics,
    recorded_device,
    substitute_profile,
    write_outputs,
    write_table,
)
from lexicons import load_lexicons, phrase_pattern

ROW_ID_RE = re.compile(r"\bex_\d{6}\b")
TERMS = ["muslim", "asylum seekers", "/pol/"]
IDENTITY_TEXT = "a benign note about muslim neighbours"
PLAIN_TEXT = "a benign note about gardening"
HOSTILE_TEXT = "a hostile rant naming asylum seekers"


def _dataset(n_neg=280, n_pos=170):
    """Synthetic posts, no dataset text: identity term in every other row."""
    rows = []
    for i in range(n_neg):
        rows.append((f"ex_{i:06d}", IDENTITY_TEXT if i % 2 == 0 else PLAIN_TEXT, 0))
    for j in range(n_pos):
        rows.append((f"ex_{n_neg + j:06d}", HOSTILE_TEXT if j % 2 == 0 else "a hostile rant", 1))
    return pd.DataFrame(rows, columns=["row_id", "text", "label"])


def _fake_predictor(base, threshold=0.5, technique="11_T"):
    """Scores the original texts exactly as the artifact; any edited text scores 0.1 lower."""
    def predict(texts):
        return np.array([base.get(t, 0.1) for t in texts], dtype=float)

    return SimpleNamespace(technique=technique, threshold=threshold, predict=predict)


def _setup(make_probs):
    from probs_artifact import load_probs

    probs = load_probs(make_probs("11_T")["test"])
    processed = _dataset()
    # One probability per distinct text, so the fake predictor can reproduce the artifact.
    text_prob = {IDENTITY_TEXT: 0.7, PLAIN_TEXT: 0.2, HOSTILE_TEXT: 0.9, "a hostile rant": 0.8}
    probs["y_prob"] = processed["text"].map(text_prob).to_numpy()
    population, pattern = build_population("11_T", processed, probs, TERMS)
    return population, pattern, text_prob


class TestCounterfactualText:
    def test_multi_word_case_insensitive_and_multiple_matches(self):
        pattern = phrase_pattern(TERMS)
        text = "Muslim and ASYLUM Seekers  met; muslim again"
        assert counterfactual_text(text, pattern, "remove") == "and met; again"
        assert counterfactual_text(text, pattern, "substitute", "group") == "group and group met; group again"

    def test_punctuation_boundaries(self):
        pattern = phrase_pattern(TERMS)
        assert counterfactual_text("(muslim), muslims!", pattern, "remove") == "(), muslims!"
        assert counterfactual_text("on /pol/.", pattern, "substitute", "group") == "on group."
        assert counterfactual_text("muslimness", pattern, "remove") == "muslimness"

    def test_unknown_operation_is_refused(self):
        with pytest.raises(IdentityCounterfactualError, match="unknown operation"):
            counterfactual_text("muslim", phrase_pattern(TERMS), "mask")


class TestSubstituteWord:
    def test_committed_lexicons_accept_the_panel(self):
        assert check_substitute_word(load_lexicons()) == SUBSTITUTE_WORDS

    def test_substitute_in_a_lexicon_is_refused(self):
        lexicons = {"slur": [], "extremist_framing": [], "identity_term": ["muslim", "group"]}
        with pytest.raises(IdentityCounterfactualError, match="'group'.*identity_term"):
            check_substitute_word(lexicons)

    @staticmethod
    def _profile(word, n_extremist, n_benign, coefficient):
        rows, labels = [], []
        for i in range(n_extremist):
            rows.append(f"they {word} x"); labels.append(1)
        for i in range(n_benign):
            rows.append(f"some {word} y"); labels.append(0)
        rows += ["filler"] * 60; labels += [1] * 20 + [0] * 40
        processed = pd.DataFrame({"row_id": [f"r{i}" for i in range(len(rows))], "text": rows, "label": labels})
        assignments = pd.DataFrame({"row_id": processed["row_id"], "split": "train"})
        coefficients = pd.DataFrame({"word": [word, "filler"], "coefficient": [coefficient, 0.0]})
        return substitute_profile([word], processed, assignments, coefficients)

    def test_label_imbalanced_substitute_is_refused(self):
        # Like "people" in the real data: most of its training posts are extremist.
        profile = self._profile("people", n_extremist=16, n_benign=4, coefficient=0.1)
        assert profile["words"]["people"]["train_extremist_share"] == pytest.approx(0.8)
        with pytest.raises(IdentityCounterfactualError, match="training extremist share"):
            check_substitute_profile(profile)

    def test_strongly_weighted_substitute_is_refused(self):
        profile = self._profile("crowd", n_extremist=7, n_benign=13, coefficient=3.2)
        with pytest.raises(IdentityCounterfactualError, match="LR coefficient"):
            check_substitute_profile(profile)

    def test_balanced_near_zero_substitute_is_accepted(self):
        profile = self._profile("group", n_extremist=7, n_benign=13, coefficient=0.1)
        assert check_substitute_profile(profile) is profile
        assert profile["words"]["group"]["lr_coefficient_rank_desc"] == 1


class TestMetrics:
    def test_known_flips_and_deltas(self):
        p_orig = [0.9, 0.8, 0.6, 0.3, 0.2]
        p_cf = [0.4, 0.85, 0.3, 0.6, 0.2]
        metrics = paired_metrics(p_orig, p_cf, threshold=0.5, y_true=0)
        assert metrics["n_posts"] == 5
        assert metrics["mean_delta"] == pytest.approx((0.5 - 0.05 + 0.3 - 0.3 + 0.0) / 5)
        assert metrics["median_delta"] == pytest.approx(0.0)
        assert metrics["share_delta_positive"] == pytest.approx(2 / 5)
        assert metrics["n_flagged_original"] == 3 and metrics["n_flagged_counterfactual"] == 2
        assert metrics["n_flip_pos_to_neg"] == 2 and metrics["n_flip_neg_to_pos"] == 1
        assert metrics["fp_share_cleared"] == pytest.approx(2 / 3)
        assert metrics["mcnemar_exact_p"] == pytest.approx(1.0)
        assert 0.0 < metrics["wilcoxon_p"] <= 1.0

    def test_no_change_gives_nan_wilcoxon_and_unit_mcnemar(self):
        metrics = paired_metrics([0.7, 0.2], [0.7, 0.2], threshold=0.5, y_true=1)
        assert np.isnan(metrics["wilcoxon_p"]) and metrics["mcnemar_exact_p"] == 1.0
        assert np.isnan(metrics["fp_share_cleared"])  # recall population has no FP share

    def test_analyze_on_synthetic_population(self, make_probs):
        population, pattern, text_prob = _setup(make_probs)
        assert len(population) == 140 + 85
        payload = analyze("11_T", _fake_predictor(text_prob), population, pattern, 0.5, len(TERMS))
        rows = {(r["y_true"], r["operation"]): r for r in payload["rows"]}
        neg = rows[(0, "remove")]
        assert neg["n_posts"] == 140 and neg["n_flagged_original"] == 140
        assert neg["n_flip_pos_to_neg"] == 140 and neg["fp_share_cleared"] == 1.0
        assert neg["mean_delta"] == pytest.approx(0.6)
        assert rows[(1, "substitute")]["mean_delta"] == pytest.approx(0.8)
        assert payload["verification"]["max_abs_diff"] == 0.0
        assert payload["n_population_by_label"] == {"non_extremist": 140, "extremist": 85}
        assert payload["substitute_words"] == list(SUBSTITUTE_WORDS)
        assert {(r["population"], r["word"]) for r in payload["substitute_per_word"]} == {
            (pop, w) for pop in ("non_extremist", "extremist") for w in SUBSTITUTE_WORDS
        }

    def test_device_is_recorded_only_for_torch_families(self):
        assert recorded_device(SimpleNamespace(family="sklearn_pipeline"), "mps") is None
        assert recorded_device(SimpleNamespace(family="fasttext_logreg"), "auto") is None
        assert recorded_device(SimpleNamespace(family="transformer"), "mps") == "mps"


class TestAdversarial:
    def test_predictor_disagreeing_with_artifact_is_refused(self, make_probs):
        population, pattern, text_prob = _setup(make_probs)
        drifted = {text: min(1.0, p + 0.05) for text, p in text_prob.items()}
        with pytest.raises(IdentityCounterfactualError, match="disagrees with the committed"):
            analyze("11_T", _fake_predictor(drifted), population, pattern, 0.5, len(TERMS))

    def test_threshold_mismatch_is_refused(self, make_probs):
        population, pattern, text_prob = _setup(make_probs)
        with pytest.raises(IdentityCounterfactualError, match="threshold"):
            analyze("11_T", _fake_predictor(text_prob, threshold=0.4), population, pattern, 0.5, len(TERMS))

    def test_outputs_carry_no_row_id_and_no_text(self, make_probs, results_dir, tmp_path):
        population, pattern, text_prob = _setup(make_probs)
        payload = analyze("11_T", _fake_predictor(text_prob), population, pattern, 0.5, len(TERMS))
        json_path = write_outputs(payload, results_dir=results_dir)
        csv_path = write_table([payload], path=tmp_path / "rq4_identity_counterfactual.csv")

        table = pd.read_csv(csv_path)
        assert list(table.columns) == list(TABLE_COLUMNS)
        assert "row_id" not in table.columns and "text" not in table.columns
        for path in (json_path, csv_path):
            content = path.read_text()
            assert not ROW_ID_RE.search(content)
            for text in (IDENTITY_TEXT, PLAIN_TEXT, HOSTILE_TEXT, "benign note", "hostile rant"):
                assert text not in content
        payload_back = json.loads(json_path.read_text())
        assert all("row_id" not in row and "text" not in row for row in payload_back["rows"])

    def test_table_replaces_rows_of_the_same_technique(self, make_probs, tmp_path):
        population, pattern, text_prob = _setup(make_probs)
        payload = analyze("11_T", _fake_predictor(text_prob), population, pattern, 0.5, len(TERMS))
        path = tmp_path / "table.csv"
        write_table([payload], path=path)
        write_table([payload], path=path)
        assert len(pd.read_csv(path)) == 4

    def test_holm_columns_cover_every_row(self, make_probs, tmp_path):
        population, pattern, text_prob = _setup(make_probs)
        payloads = []
        for name in ("11_A", "11_B"):
            payload = analyze(name, _fake_predictor(text_prob), population, pattern, 0.5, len(TERMS))
            payloads.append(payload)
        table = pd.read_csv(write_table(payloads, path=tmp_path / "t.csv"))
        assert len(table) == 8
        assert (table["mcnemar_p_holm"] >= table["mcnemar_exact_p"]).all()
        assert (table["wilcoxon_p_holm"] >= table["wilcoxon_p"]).all()
        assert (table["mcnemar_p_holm"] <= 1.0).all()


class TestAbortWritesNothing:
    """A failure partway through a run must not leave new JSONs next to a stale table."""

    def _wire(self, monkeypatch, make_probs, results_dir, tmp_path, predictors):
        population_probs = {}
        population, pattern, text_prob = _setup(make_probs)
        from probs_artifact import load_probs

        probs = load_probs(make_probs("11_T")["test"])
        probs["y_prob"] = _dataset()["text"].map(text_prob).to_numpy()
        population_probs.update({name: probs for name in predictors})
        lexicons = {"identity_term": TERMS, "slur": [], "extremist_framing": [], "topical": []}
        csv_path = tmp_path / "rq" / "rq4_identity_counterfactual.csv"

        import split_protocol

        monkeypatch.setattr(split_protocol, "build_processed_dataset", lambda: _dataset())
        monkeypatch.setattr(ic, "load_lexicons", lambda: lexicons)
        monkeypatch.setattr(ic, "load_substitute_profile", lambda processed: {
            "train_base_rate": 0.4, "max_share_gap": 0.1, "max_abs_coef": 0.5, "words": {}})
        monkeypatch.setattr(ic, "load_technique_probs", lambda technique, split: population_probs[technique])
        monkeypatch.setattr(ic, "load_threshold", lambda technique: 0.5)
        monkeypatch.setattr(ic, "load_technique", lambda technique, folder, device, resident: predictors[technique])
        monkeypatch.setattr(ic, "RESULTS_DIR", results_dir)
        monkeypatch.setattr(ic, "RQ4_IDENTITY_COUNTERFACTUAL_CSV", csv_path)
        return text_prob, csv_path

    def test_second_technique_failing_verification_writes_nothing(self, monkeypatch, make_probs,
                                                                   results_dir, tmp_path):
        population, pattern, text_prob = _setup(make_probs)
        drifted = {text: min(1.0, p + 0.05) for text, p in text_prob.items()}
        predictors = {"11_A": _fake_predictor(text_prob), "11_B": _fake_predictor(drifted)}
        _, csv_path = self._wire(monkeypatch, make_probs, results_dir, tmp_path, predictors)

        code = ic.main(["--technique", "11_A", "--technique", "11_B"])
        assert code == 1
        assert not list(results_dir.rglob("*.json"))
        assert not csv_path.exists()

    def test_all_techniques_passing_write_json_and_table(self, monkeypatch, make_probs, results_dir, tmp_path):
        population, pattern, text_prob = _setup(make_probs)
        predictors = {"11_A": _fake_predictor(text_prob), "11_B": _fake_predictor(text_prob)}
        _, csv_path = self._wire(monkeypatch, make_probs, results_dir, tmp_path, predictors)

        assert ic.main(["--technique", "11_A", "--technique", "11_B"]) == 0
        assert len(list(results_dir.rglob("identity_counterfactual_test.json"))) == 2
        assert set(pd.read_csv(csv_path)["technique"]) == {"11_A", "11_B"}
