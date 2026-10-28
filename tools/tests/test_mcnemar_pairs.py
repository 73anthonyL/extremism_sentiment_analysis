"""Behavioural and adversarial tests for tools/mcnemar_pairs.py on synthetic artifacts."""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIR))

from mcnemar_pairs import (
    McNemarError,
    build,
    holm_adjust,
    mcnemar_exact_p,
    write_per_post,
    write_tables,
)

ROW_ID_RE = re.compile(r"\bex_\d{6}\b")


def _lock_threshold(results_dir, technique, threshold=0.5):
    folder = results_dir / technique
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "best_config.json", "w") as handle:
        json.dump({"selected_threshold": threshold}, handle)


def _copy_artifact(probs_dir, source, target, edit=None):
    frame = pd.read_csv(probs_dir / f"{source}__test.csv")
    if edit is not None:
        frame = edit(frame)
    frame.to_csv(probs_dir / f"{target}__test.csv", index=False)


class TestStatistics:
    def test_exact_p_matches_binomial(self):
        assert mcnemar_exact_p(0, 0) == 1.0
        assert mcnemar_exact_p(0, 10) == pytest.approx(2 * 0.5 ** 10)
        assert mcnemar_exact_p(10, 0) == mcnemar_exact_p(0, 10)
        assert mcnemar_exact_p(5, 5) == pytest.approx(1.0)

    def test_holm_is_monotone_and_bounded(self):
        adjusted = holm_adjust([0.01, 0.04, 0.03, 0.5])
        assert adjusted.tolist() == pytest.approx([0.04, 0.09, 0.09, 0.5])
        assert (adjusted <= 1.0).all()
        assert holm_adjust([0.2, 0.9]).tolist() == pytest.approx([0.4, 0.9])


class TestBuild:
    def test_identical_artifacts_have_no_discordant_pairs(self, probs_dir, make_probs, results_dir):
        make_probs("01_A", seed=1)
        _copy_artifact(probs_dir, "01_A", "02_B")
        for technique in ("01_A", "02_B"):
            _lock_threshold(results_dir, technique)
        pairs, overlap, shared, matrix, y_true = build(["01_A", "02_B"], results_dir=results_dir)
        row = pairs.iloc[0]
        assert row["n_discordant"] == 0 and row["mcnemar_p"] == 1.0 and row["accuracy_gap_rows"] == 0
        assert row["error_jaccard"] == 1.0
        assert not row["significant_holm"]
        assert overlap["n_errors_unique"].eq(0).all()
        assert shared["n_posts"].sum() == 450 == len(matrix)
        assert shared.loc[shared["n_techniques_wrong"] == 1, "n_posts"].item() == 0

    def test_flipped_rows_give_the_exact_binomial_p(self, probs_dir, make_probs, results_dir):
        make_probs("01_A", seed=2)
        base = pd.read_csv(probs_dir / "01_A__test.csv")
        right_negatives = base.index[(base["y_true"] == 0) & (base["y_prob"] < 0.5)][:10]

        def break_ten(frame):
            frame = frame.copy()
            frame.loc[right_negatives, "y_prob"] = 0.9
            return frame

        _copy_artifact(probs_dir, "01_A", "02_B", edit=break_ten)
        for technique in ("01_A", "02_B"):
            _lock_threshold(results_dir, technique)
        pairs, overlap, shared, _, _ = build(["01_A", "02_B"], results_dir=results_dir)
        row = pairs.iloc[0]
        assert row["n_only_a_wrong"] == 0 and row["n_only_b_wrong"] == 10
        assert row["accuracy_gap_rows"] == 10
        assert row["mcnemar_p"] == pytest.approx(2 * 0.5 ** 10)
        assert row["holm_p"] == pytest.approx(row["mcnemar_p"])
        assert row["significant_holm"]
        b_row = overlap.set_index("technique").loc["02_B"]
        assert b_row["n_errors_unique"] == 10
        assert b_row["n_false_positives"] - overlap.set_index("technique").loc["01_A", "n_false_positives"] == 10
        assert shared.loc[shared["n_techniques_wrong"] == 1, "n_negative_posts"].item() == 10

    def test_holm_runs_over_every_pair(self, probs_dir, make_probs, results_dir):
        make_probs("01_A", seed=3)
        _copy_artifact(probs_dir, "01_A", "02_B")
        _copy_artifact(probs_dir, "01_A", "03_C")
        for technique in ("01_A", "02_B", "03_C"):
            _lock_threshold(results_dir, technique)
        pairs, _, shared, _, _ = build(["01_A", "02_B", "03_C"], results_dir=results_dir)
        assert len(pairs) == 3
        assert (pairs["holm_p"] >= pairs["mcnemar_p"]).all()
        assert shared["n_techniques"].eq(3).all() and len(shared) == 4


class TestRefusals:
    def test_mismatched_row_ids_are_refused(self, probs_dir, make_probs, results_dir):
        make_probs("01_A", seed=4)

        def rename_one_row(frame):
            frame = frame.copy()
            frame.loc[0, "row_id"] = "ex_999999"
            return frame

        _copy_artifact(probs_dir, "01_A", "02_B", edit=rename_one_row)
        for technique in ("01_A", "02_B"):
            _lock_threshold(results_dir, technique)
        with pytest.raises(McNemarError, match="not scored on the same rows"):
            build(["01_A", "02_B"], results_dir=results_dir)

    def test_label_disagreement_is_refused(self, probs_dir, make_probs, results_dir):
        make_probs("01_A", seed=5)

        def swap_two_labels(frame):
            frame = frame.copy()
            first_negative = frame.index[frame["y_true"] == 0][0]
            first_positive = frame.index[frame["y_true"] == 1][0]
            frame.loc[first_negative, "y_true"], frame.loc[first_positive, "y_true"] = 1, 0
            return frame

        _copy_artifact(probs_dir, "01_A", "02_B", edit=swap_two_labels)
        for technique in ("01_A", "02_B"):
            _lock_threshold(results_dir, technique)
        with pytest.raises(McNemarError, match="disagree on y_true"):
            build(["01_A", "02_B"], results_dir=results_dir)

    def test_missing_locked_threshold_is_refused(self, probs_dir, make_probs, results_dir):
        make_probs("01_A", seed=6)
        _copy_artifact(probs_dir, "01_A", "02_B")
        _lock_threshold(results_dir, "01_A")
        with pytest.raises(McNemarError, match="no best_config.json"):
            build(["01_A", "02_B"], results_dir=results_dir)

    def test_single_technique_is_refused(self, probs_dir, make_probs, results_dir):
        make_probs("01_A", seed=7)
        _lock_threshold(results_dir, "01_A")
        with pytest.raises(McNemarError, match="at least two"):
            build(["01_A"], results_dir=results_dir)


class TestOutputs:
    def test_committed_tables_carry_no_row_ids(self, probs_dir, make_probs, results_dir, tmp_path):
        make_probs("01_A", seed=8)
        _copy_artifact(probs_dir, "01_A", "02_B")
        for technique in ("01_A", "02_B"):
            _lock_threshold(results_dir, technique)
        pairs, overlap, shared, matrix, y_true = build(["01_A", "02_B"], results_dir=results_dir)
        rq_dir = tmp_path / "rq"
        paths = write_tables(
            pairs, overlap, shared,
            pairs_path=rq_dir / "pairs.csv", overlap_path=rq_dir / "overlap.csv", shared_path=rq_dir / "shared.csv",
        )
        for path in paths:
            assert not ROW_ID_RE.search(path.read_text())

    def test_per_post_matrix_is_refused_under_results_summary(self, probs_dir, make_probs, results_dir, monkeypatch):
        import mcnemar_pairs

        make_probs("01_A", seed=9)
        _copy_artifact(probs_dir, "01_A", "02_B")
        for technique in ("01_A", "02_B"):
            _lock_threshold(results_dir, technique)
        _, _, _, matrix, y_true = build(["01_A", "02_B"], results_dir=results_dir)
        monkeypatch.setattr(mcnemar_pairs, "RESULTS_DIR", results_dir)
        with pytest.raises(McNemarError, match="row ids"):
            write_per_post(matrix, y_true, results_dir / "rq" / "matrix.csv")
        out = write_per_post(matrix, y_true, results_dir.parent / "external" / "matrix.csv")
        written = pd.read_csv(out)
        assert list(written.columns) == ["row_id", "y_true", "01_A", "02_B"]
        assert len(written) == 450
