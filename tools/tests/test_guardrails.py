"""Adversarial tests: each asserts that a specific failure is CAUGHT.

A verification toolkit that only ever passes is worthless. Every test here
introduces a defect that has either already occurred in this repository or is the
direct analogue of one, and asserts the tooling refuses it.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIR))

from conftest import corrupt_json_field
from metrics_core import compute_binary_metrics, recompute_from_confusion
from probs_artifact import ProbsArtifactError, load_probs
from split_protocol import (
    reconstruct_split_assignments,
    split_counts,
    verify_assignments,
)
from validate_results_folder import validate_folder


class TestSplitProtocol:
    """The frozen split must be reproducible, and deviations must be rejected."""

    def test_reconstruction_reproduces_published_counts(self):
        assignments = reconstruct_split_assignments()
        ok, problems = verify_assignments(assignments)
        assert ok, f"reconstruction no longer matches the foundation: {problems}"
        assert split_counts(assignments) == {
            "train": (2099, 1309, 790),
            "validation": (450, 281, 169),
            "test": (450, 280, 170),
        }

    def test_reconstruction_is_deterministic(self):
        first = reconstruct_split_assignments()
        second = reconstruct_split_assignments()
        pd.testing.assert_frame_equal(first, second)

    def test_wrong_split_counts_are_rejected(self):
        assignments = reconstruct_split_assignments()
        # Move one frozen test row into train and expect a complaint about test.
        index = assignments.index[assignments["split"] == "test"][0]
        assignments.loc[index, "split"] = "train"
        ok, problems = verify_assignments(assignments)
        assert not ok
        assert any("test" in problem for problem in problems)


class TestMetricConsistency:
    """Stored metrics must agree with the stored confusion matrix."""

    def test_recompute_agrees_with_sklearn(self):
        import numpy as np

        rng = np.random.default_rng(30)
        y_true = rng.integers(0, 2, 450)
        y_prob = rng.random(450)
        metrics = compute_binary_metrics(y_true, y_prob, 0.5)
        derived = recompute_from_confusion(
            metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]
        )
        for field, value in derived.items():
            assert metrics[field] == pytest.approx(value, abs=1e-9), field

    def test_corrupted_confusion_count_is_detected(self, golden_results_folder):
        """The archetype: a metric file edited so it no longer matches its matrix."""
        corrupt_json_field(golden_results_folder / "metrics_test.json", "tn", 7)
        problems = validate_folder(golden_results_folder)
        assert problems, "a corrupted tn must not validate"
        assert any("accuracy" in problem for problem in problems)
        assert any("450 rows" in problem for problem in problems)

    def test_corrupted_accuracy_is_detected(self, golden_results_folder):
        corrupt_json_field(golden_results_folder / "metrics_test.json", "accuracy", 0.05)
        problems = validate_folder(golden_results_folder)
        assert any("accuracy" in problem for problem in problems)

    def test_unmodified_golden_folder_passes(self, golden_results_folder):
        assert validate_folder(golden_results_folder) == []


class TestProbsArtifact:
    """The probability artifact is the only channel for per-row data."""

    def test_text_column_is_refused(self, probs_dir):
        """Text must never re-enter the repository through this artifact."""
        frame = pd.DataFrame(
            {
                "row_id": ["ex_000000"],
                "split": ["test"],
                "y_true": [1],
                "y_prob": [0.9],
                "text": ["some dataset text"],
            }
        )
        path = probs_dir / "leaky__test.csv"
        frame.to_csv(path, index=False)
        with pytest.raises(ProbsArtifactError, match="text-bearing"):
            load_probs(path)

    def test_wrong_row_count_is_refused(self, probs_dir):
        """A truncated download must fail loudly, not produce a plausible metric."""
        frame = pd.DataFrame(
            {
                "row_id": [f"ex_{i:06d}" for i in range(100)],
                "split": "test",
                "y_true": [0, 1] * 50,
                "y_prob": [0.5] * 100,
            }
        )
        path = probs_dir / "truncated__test.csv"
        frame.to_csv(path, index=False)
        with pytest.raises(ProbsArtifactError, match="frozen protocol"):
            load_probs(path)

    def test_valid_artifact_loads(self, make_probs):
        paths = make_probs("synthetic")
        frame = load_probs(paths["test"], expected_split="test")
        assert len(frame) == 450
        assert int((frame["y_true"] == 1).sum()) == 170

    def test_out_of_range_probability_is_refused(self, probs_dir, make_probs):
        paths = make_probs("synthetic")
        frame = pd.read_csv(paths["test"])
        frame.loc[0, "y_prob"] = 1.5
        frame.to_csv(paths["test"], index=False)
        with pytest.raises(ProbsArtifactError, match=r"\[0, 1\]"):
            load_probs(paths["test"])


class TestThresholdPolicy:
    """Fitting a threshold on test is blocked in code, not in review."""

    def test_threshold_selection_on_test_is_refused(self, make_probs):
        from eval_from_probs import TestSplitSelectionError, derive_threshold
        from probs_artifact import load_probs

        paths = make_probs("synthetic")
        test_frame = load_probs(paths["test"])
        with pytest.raises(TestSplitSelectionError, match="only 'validation'"):
            derive_threshold(test_frame)

    def test_threshold_selection_on_validation_succeeds(self, make_probs):
        from eval_from_probs import derive_threshold
        from probs_artifact import load_probs

        paths = make_probs("synthetic")
        threshold, sweep = derive_threshold(load_probs(paths["validation"]))
        assert 0.0 < threshold < 1.0
        assert not sweep.empty


class TestLedger:
    """The test split is unlocked once per technique, and history is tamper-evident."""

    @pytest.fixture
    def ledger_env(self, tmp_path, monkeypatch):
        import ledger
        import repo_paths

        path = tmp_path / "test_ledger.jsonl"
        monkeypatch.setattr(ledger, "TEST_LEDGER_JSONL", path, raising=False)
        monkeypatch.setattr(ledger, "LOOP_DIR", tmp_path, raising=False)
        monkeypatch.setattr(repo_paths, "TEST_LEDGER_JSONL", path, raising=False)

        results = tmp_path / "results"
        monkeypatch.setattr(ledger, "RESULTS_DIR", results, raising=False)
        return ledger, results

    def _write_metrics(self, results_dir, technique, accuracy=0.88):
        folder = results_dir / technique
        folder.mkdir(parents=True, exist_ok=True)
        payload = {
            "accuracy": accuracy,
            "balanced_accuracy": accuracy - 0.01,
            "positive_f1": accuracy - 0.03,
            "pr_auc": accuracy + 0.04,
            "false_positive_rate": 0.10,
            "tn": 252, "fp": 28, "fn": 22, "tp": 148,
        }
        with open(folder / "metrics_test.json", "w") as handle:
            json.dump(payload, handle)

    def test_second_unlock_for_same_technique_is_refused(self, ledger_env):
        ledger, results = ledger_env
        self._write_metrics(results, "09_CANDIDATE")

        ledger.append_entry("09_CANDIDATE", "01", prereg_sha="abc")
        with pytest.raises(ledger.LedgerError, match="already has a test evaluation"):
            ledger.append_entry("09_CANDIDATE", "02", prereg_sha="def")

    def test_chain_detects_tampering(self, ledger_env):
        ledger, results = ledger_env
        self._write_metrics(results, "09_A", accuracy=0.88)
        self._write_metrics(results, "10_B", accuracy=0.89)
        ledger.append_entry("09_A", "01", prereg_sha="abc")
        ledger.append_entry("10_B", "02", prereg_sha="def")

        assert ledger.verify_chain()[0]

        # Tamper with a recorded metric after the fact and rewrite the file.
        entries = ledger.read_entries()
        entries[0]["accuracy"] = 0.95
        with open(ledger.TEST_LEDGER_JSONL, "w") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, sort_keys=True) + "\n")

        ok, problems = ledger.verify_chain()
        assert not ok
        assert any("modified after being written" in problem for problem in problems)

    def test_family_size_grows_with_each_unlock(self, ledger_env):
        ledger, results = ledger_env
        self._write_metrics(results, "09_A")
        self._write_metrics(results, "10_B")
        assert ledger.family_size() == 0
        ledger.append_entry("09_A", "01", prereg_sha="abc")
        assert ledger.family_size() == 1
        ledger.append_entry("10_B", "02", prereg_sha="def")
        assert ledger.family_size() == 2


class TestStatistics:
    """The judge's arithmetic, including the bound that decides cycle 00."""

    def test_mcnemar_matches_known_bound(self):
        """07 vs 08 differ by 5 test rows; even the best case cannot reach p<0.05."""
        import numpy as np

        from compare_techniques import mcnemar_exact

        # All 5 disagreements favor one side: the best case for significance.
        correct_champion = np.array([True] * 5 + [False] * 445)
        correct_candidate = np.array([False] * 5 + [False] * 445)
        b, c, p_value = mcnemar_exact(correct_champion, correct_candidate)
        assert (b, c) == (5, 0)
        assert p_value == pytest.approx(0.0625, abs=1e-4)
        assert p_value > 0.05, "a 5-row gap must not reach significance at n=450"

    def test_holm_adjustment_penalizes_repeated_attempts(self):
        from compare_techniques import holm_adjusted_p

        assert holm_adjusted_p(0.01, 1) == pytest.approx(0.01)
        assert holm_adjusted_p(0.01, 5) == pytest.approx(0.05)
        assert holm_adjusted_p(0.5, 20) == 1.0

    def test_identical_predictions_give_no_evidence(self):
        import numpy as np

        from compare_techniques import mcnemar_exact

        correct = np.array([True] * 400 + [False] * 50)
        b, c, p_value = mcnemar_exact(correct, correct)
        assert (b, c) == (0, 0)
        assert p_value == 1.0
