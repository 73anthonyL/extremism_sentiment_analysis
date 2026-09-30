"""Adversarial tests: each asserts that a specific failure is CAUGHT.

A verification toolkit that only ever passes is worthless. Every test here
introduces a defect that has either already occurred in this repository or is
the direct analogue of one, and asserts the tooling refuses it.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIR))

from conftest import (
    FRAMING_WORDS,
    IDENTITY_WORDS,
    TOPICAL_WORDS,
    base_meta,
    corrupt_json_field,
    synthetic_attributions,
)
from attributions import (
    AttributionArtifactError,
    list_runs,
    load_attributions,
    load_run,
    validate_meta,
    write_run,
)
from metrics_core import compute_binary_metrics, recompute_from_confusion
from probs_artifact import ProbsArtifactError, load_probs
from split_protocol import reconstruct_split_assignments, split_counts, verify_assignments
from validate_results_folder import validate_folder

ALL_WORDS = FRAMING_WORDS + IDENTITY_WORDS + TOPICAL_WORDS


class TestSplitProtocol:
    def test_reconstruction_reproduces_published_counts(self):
        assignments = reconstruct_split_assignments()
        ok, problems = verify_assignments(assignments)
        assert ok, f"reconstruction no longer matches the foundation: {problems}"
        assert split_counts(assignments) == {
            "train": (2099, 1309, 790),
            "validation": (450, 281, 169),
            "test": (450, 280, 170),
        }

    def test_committed_mirror_matches_reconstruction(self):
        """The mirror has been stale before; the tests are where that is caught now."""
        from repo_paths import SPLIT_ASSIGNMENTS_CSV

        committed = pd.read_csv(SPLIT_ASSIGNMENTS_CSV).sort_values("row_id").reset_index(drop=True)
        reconstructed = reconstruct_split_assignments()
        pd.testing.assert_frame_equal(
            committed[["row_id", "label", "text_hash", "split"]],
            reconstructed[["row_id", "label", "text_hash", "split"]],
        )

    def test_wrong_split_counts_are_rejected(self):
        assignments = reconstruct_split_assignments()
        index = assignments.index[assignments["split"] == "test"][0]
        assignments.loc[index, "split"] = "train"
        ok, problems = verify_assignments(assignments)
        assert not ok
        assert any("test" in problem for problem in problems)


class TestMetricConsistency:
    def test_recompute_agrees_with_sklearn(self):
        rng = np.random.default_rng(30)
        y_true = rng.integers(0, 2, 450)
        y_prob = rng.random(450)
        metrics = compute_binary_metrics(y_true, y_prob, 0.5)
        derived = recompute_from_confusion(metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"])
        for field, value in derived.items():
            assert metrics[field] == pytest.approx(value, abs=1e-9), field

    def test_corrupted_confusion_count_is_detected(self, golden_results_folder):
        corrupt_json_field(golden_results_folder / "metrics_test.json", "tn", 7)
        problems = validate_folder(golden_results_folder)
        assert any("accuracy" in problem for problem in problems)
        assert any("450 rows" in problem for problem in problems)

    def test_unmodified_golden_folder_passes(self, golden_results_folder):
        assert validate_folder(golden_results_folder) == []


class TestProbsArtifact:
    def test_text_column_is_refused(self, probs_dir):
        frame = pd.DataFrame(
            {"row_id": ["ex_000000"], "split": ["test"], "y_true": [1], "y_prob": [0.9],
             "text": ["some dataset text"]}
        )
        path = probs_dir / "leaky__test.csv"
        frame.to_csv(path, index=False)
        with pytest.raises(ProbsArtifactError, match="text-bearing"):
            load_probs(path)

    def test_wrong_row_count_is_refused(self, probs_dir):
        frame = pd.DataFrame(
            {"row_id": [f"ex_{i:06d}" for i in range(100)], "split": "test",
             "y_true": [0, 1] * 50, "y_prob": [0.5] * 100}
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


class TestAttributionArtifact:
    """Only word-level aggregates may be committed; anything per-post is refused."""

    def test_valid_run_round_trips(self, results_dir, make_run):
        make_run("01_LOG-REG_TF-IDF", "run_a", ALL_WORDS)
        assert list_runs("01_LOG-REG_TF-IDF", results_dir) == ["run_a"]
        frame, meta = load_run("01_LOG-REG_TF-IDF", "run_a", results_dir)
        assert set(frame["word"]) == set(ALL_WORDS)
        assert meta["aggregation"] == "word"
        # sorted by mean_abs_attribution descending
        assert frame["mean_abs_attribution"].is_monotonic_decreasing

    def test_row_id_column_is_refused(self, tmp_path):
        frame = synthetic_attributions(ALL_WORDS)
        frame["row_id"] = "ex_000001"
        path = tmp_path / "leaky.csv"
        frame.to_csv(path, index=False)
        with pytest.raises(AttributionArtifactError, match="per-post"):
            load_attributions(path)

    def test_text_column_is_refused(self, tmp_path):
        frame = synthetic_attributions(ALL_WORDS)
        frame["text"] = "a post"
        path = tmp_path / "leaky.csv"
        frame.to_csv(path, index=False)
        with pytest.raises(AttributionArtifactError, match="per-post"):
            load_attributions(path)

    def test_phrase_in_word_column_is_refused(self, tmp_path):
        """A pasted fragment of a post cannot hide in the word column."""
        frame = synthetic_attributions(ALL_WORDS)
        frame.loc[0, "word"] = "should burn to the ground"
        path = tmp_path / "phrase.csv"
        frame.to_csv(path, index=False)
        with pytest.raises(AttributionArtifactError, match="whitespace"):
            load_attributions(path)

    def test_extra_column_is_refused(self, tmp_path):
        frame = synthetic_attributions(ALL_WORDS)
        frame["std_attribution"] = 0.1
        path = tmp_path / "extra.csv"
        frame.to_csv(path, index=False)
        with pytest.raises(AttributionArtifactError, match="unexpected columns"):
            load_attributions(path)

    def test_signed_mean_cannot_exceed_abs_mean(self, tmp_path):
        frame = synthetic_attributions(ALL_WORDS)
        frame.loc[0, "mean_attribution"] = frame.loc[0, "mean_abs_attribution"] + 1.0
        path = tmp_path / "inconsistent.csv"
        frame.to_csv(path, index=False)
        with pytest.raises(AttributionArtifactError, match="exceeds mean_abs_attribution"):
            load_attributions(path)

    def test_support_beyond_posts_explained_is_refused(self, results_dir):
        frame = synthetic_attributions(ALL_WORDS)
        frame["support"] = 500
        meta = base_meta("T", "r", n_posts=200)
        with pytest.raises(AttributionArtifactError, match="n_posts_explained"):
            write_run("T", "r", frame, meta, results_dir=results_dir)

    def test_subword_aggregation_is_refused(self):
        meta = base_meta("T", "r")
        meta["aggregation"] = "subword"
        with pytest.raises(AttributionArtifactError, match="aggregation"):
            validate_meta(meta)

    def test_missing_meta_field_is_refused(self):
        meta = base_meta("T", "r")
        del meta["background_size"]
        with pytest.raises(AttributionArtifactError, match="background_size"):
            validate_meta(meta)

    def test_run_without_sidecar_is_not_a_run(self, results_dir):
        folder = results_dir / "T" / "attributions"
        folder.mkdir(parents=True)
        synthetic_attributions(ALL_WORDS).to_csv(folder / "orphan.csv", index=False)
        assert list_runs("T", results_dir) == []

    def test_validate_folder_flags_broken_run(self, results_dir, make_run):
        make_run("T", "run_a", ALL_WORDS)
        csv_path = results_dir / "T" / "attributions" / "run_a.csv"
        frame = pd.read_csv(csv_path)
        frame["row_id"] = "ex_000000"
        frame.to_csv(csv_path, index=False)
        problems = validate_folder(results_dir / "T")
        assert any("per-post" in p for p in problems)


class TestRunManifest:
    """Committed artifacts are hashed; edits after the fact are detected."""

    @pytest.fixture
    def technique_folder(self, results_dir):
        folder = results_dir / "07_T"
        folder.mkdir()
        (folder / "metrics_test.json").write_text('{"accuracy": 0.88}')
        return folder

    def test_init_then_check_passes(self, results_dir, technique_folder):
        from run_manifest import check, init_manifest

        init_manifest("07_T", "07_T_run", "https://www.kaggle.com/code/u/nb", 3, "abc123",
                      results_dir=results_dir)
        assert check("07_T", results_dir) == []

    def test_edit_after_init_is_detected(self, results_dir, technique_folder):
        from run_manifest import check, init_manifest

        init_manifest("07_T", "07_T_run", "https://www.kaggle.com/code/u/nb", 3, "abc123",
                      results_dir=results_dir)
        (technique_folder / "metrics_test.json").write_text('{"accuracy": 0.95}')
        problems = check("07_T", results_dir)
        assert any("content changed" in p for p in problems)

    def test_text_bearing_asset_in_repo_is_refused(self, results_dir, technique_folder):
        from run_manifest import ManifestError, add_asset, init_manifest

        init_manifest("07_T", "07_T_run", "https://www.kaggle.com/code/u/nb", 3, "abc123",
                      results_dir=results_dir)
        with pytest.raises(ManifestError, match="git host"):
            add_asset("07_T", "local_attributions",
                      "https://github.com/u/repo/blob/main/local.csv", "0" * 64, 10,
                      contains_text=True, results_dir=results_dir)

    def test_relative_path_location_is_refused(self, results_dir, technique_folder):
        from run_manifest import ManifestError, add_asset, init_manifest

        init_manifest("07_T", "07_T_run", "https://www.kaggle.com/code/u/nb", 3, "abc123",
                      results_dir=results_dir)
        with pytest.raises(ManifestError, match="must use one of"):
            add_asset("07_T", "weights", "outputs/model.bin", "0" * 64, 10, results_dir=results_dir)
