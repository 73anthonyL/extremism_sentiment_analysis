"""Behavioural and adversarial tests for tools/same_post_stability.py on synthetic per-post files."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIR))

from attributions import write_run
from conftest import base_meta
from notebook_kit import aggregate_word_attributions
from same_post_stability import SamePostError, build, concentration

VOCAB = [f"w{i}" for i in range(40)]


def _posts(row_ids, seed):
    """Synthetic per-post tokens and attributions: six tokens per post from VOCAB."""
    rng = np.random.default_rng(seed)
    posts = {}
    for row_id in row_ids:
        tokens = [f"{w} " for w in rng.choice(VOCAB, size=6, replace=True)]
        posts[row_id] = (tokens, rng.normal(0.0, 0.1, len(tokens)))
    return posts


def _write(results_dir, external_dir, technique, split, posts, run_id=None, local=None, n_posts=None):
    """Write the committed word table and sidecar, and the external per-post file."""
    run_id = run_id or f"shap-partition_{split}_seed30"
    frame = aggregate_word_attributions([p[0] for p in posts.values()], [p[1] for p in posts.values()])
    meta = base_meta(technique, run_id, split=split, n_posts=n_posts or len(posts))
    write_run(technique, run_id, frame, meta, results_dir=results_dir)
    rows = [
        {"row_id": r, "position": i, "token": t, "attribution": v}
        for r, (tokens, values) in posts.items()
        for i, (t, v) in enumerate(zip(tokens, values))
    ]
    table = pd.DataFrame(rows) if local is None else local(pd.DataFrame(rows))
    folder = external_dir / technique / "attributions_local"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{run_id}.csv"
    table.to_csv(path, index=False)
    return path


TEST_ALL = [f"ex_{i:06d}" for i in range(30)]
TEST_SAMPLE = TEST_ALL[::3]          # 10 of the 30
VAL_ALL = [f"ex_{i:06d}" for i in range(100, 130)]
VAL_SAMPLE = VAL_ALL[::3]


@pytest.fixture
def external_dir(tmp_path):
    directory = tmp_path / "external"
    directory.mkdir()
    return directory


def _standard(results_dir, external_dir):
    classical_test, classical_val = _posts(TEST_ALL, 1), _posts(VAL_ALL, 2)
    _write(results_dir, external_dir, "01_A", "test", classical_test)
    _write(results_dir, external_dir, "01_A", "validation", classical_val)
    _write(results_dir, external_dir, "07_T", "test", {r: classical_test[r] for r in TEST_SAMPLE})
    _write(results_dir, external_dir, "07_T", "validation", {r: classical_val[r] for r in VAL_SAMPLE})
    return classical_test


class TestSamePost:
    def test_common_set_is_the_smallest_sample(self, results_dir, external_dir):
        _standard(results_dir, external_dir)
        support, agreement = build(top_k=5, results_dir=results_dir, external_dir=external_dir)
        rows = support.set_index(["technique", "split", "post_set"])
        assert rows.loc[("01_A", "test", "own"), "n_posts"] == 30
        assert rows.loc[("01_A", "test", "common"), "n_posts"] == 10
        assert rows.loc[("07_T", "test", "common"), "n_posts"] == 10
        assert set(agreement["post_set"]) == {"own", "common"} and len(agreement) == 4

    def test_identical_attributions_on_the_same_posts_give_identical_rows(self, results_dir, external_dir):
        # 07_T's per-post values are 01_A's on the sample, so on the common set the two must agree exactly.
        _standard(results_dir, external_dir)
        support, _ = build(top_k=5, results_dir=results_dir, external_dir=external_dir)
        common = support[support["post_set"] == "common"].set_index(["technique", "split"])
        numeric = [c for c in common.columns if c not in ("run_id", "post_set")]
        pd.testing.assert_series_equal(
            common.loc[("01_A", "test"), numeric], common.loc[("07_T", "test"), numeric], check_names=False
        )

    def test_own_rows_match_the_committed_table(self, results_dir, external_dir):
        posts = _standard(results_dir, external_dir)
        support, _ = build(top_k=5, results_dir=results_dir, external_dir=external_dir)
        frame = aggregate_word_attributions([p[0] for p in posts.values()], [p[1] for p in posts.values()])
        top = frame.nlargest(5, "mean_attribution")
        row = support.set_index(["technique", "split", "post_set"]).loc[("01_A", "test", "own")]
        assert row["n_words"] == len(frame)
        assert row["share_single_post_top_k_positive"] == pytest.approx((top["support"] == 1).mean())

    def test_concentration_of_a_single_word(self):
        frame = pd.DataFrame({"word": ["a", "b", "c"], "mean_abs_attribution": [1.0, 0.1, 0.1],
                              "mean_attribution": [1.0, -0.1, 0.0], "support": [3, 2, 1]})
        result = concentration(frame, top_k=2)
        assert result["n_words_positive_mass"] == 1 and result["n_words_half_positive_mass"] == 1
        assert result["share_positive_mass_top_k"] == pytest.approx(1.0)
        even = pd.DataFrame({"word": list("abcd"), "mean_abs_attribution": [1.0] * 4,
                             "mean_attribution": [1.0] * 4, "support": [1] * 4})
        result = concentration(even, top_k=2)
        assert result["n_words_half_positive_mass"] == 2
        assert result["positive_mass_entropy_normalized"] == pytest.approx(1.0)
        assert result["effective_n_words_positive_mass"] == pytest.approx(4.0)


class TestRefusals:
    def test_missing_per_post_file_is_refused(self, results_dir, external_dir):
        path = _write(results_dir, external_dir, "01_A", "test", _posts(TEST_ALL, 1))
        _write(results_dir, external_dir, "02_B", "test", _posts(TEST_ALL, 3))
        path.unlink()
        with pytest.raises(SamePostError, match="per-post file not found"):
            build(top_k=5, results_dir=results_dir, external_dir=external_dir)

    def test_per_post_file_that_does_not_reproduce_the_committed_table_is_refused(self, results_dir, external_dir):
        def tamper(table):
            table.loc[0, "attribution"] += 0.5
            return table

        _write(results_dir, external_dir, "01_A", "test", _posts(TEST_ALL, 1), local=tamper)
        _write(results_dir, external_dir, "02_B", "test", _posts(TEST_ALL, 3))
        with pytest.raises(SamePostError, match="does not reproduce the committed"):
            build(top_k=5, results_dir=results_dir, external_dir=external_dir)

    def test_per_post_file_carrying_text_is_refused(self, results_dir, external_dir):
        _write(results_dir, external_dir, "01_A", "test", _posts(TEST_ALL, 1),
               local=lambda t: t.assign(text="a whole post"))
        _write(results_dir, external_dir, "02_B", "test", _posts(TEST_ALL, 3))
        with pytest.raises(SamePostError, match="columns are"):
            build(top_k=5, results_dir=results_dir, external_dir=external_dir)

    def test_duplicate_positions_are_refused(self, results_dir, external_dir):
        def duplicate(table):
            table.loc[1, "position"] = 0
            return table

        _write(results_dir, external_dir, "01_A", "test", _posts(TEST_ALL, 1), local=duplicate)
        _write(results_dir, external_dir, "02_B", "test", _posts(TEST_ALL, 3))
        with pytest.raises(SamePostError, match="duplicate or missing token positions"):
            build(top_k=5, results_dir=results_dir, external_dir=external_dir)

    def test_post_count_different_from_the_sidecar_is_refused(self, results_dir, external_dir):
        _write(results_dir, external_dir, "01_A", "test", _posts(TEST_ALL, 1), n_posts=31)
        _write(results_dir, external_dir, "02_B", "test", _posts(TEST_ALL, 3))
        with pytest.raises(SamePostError, match="n_posts_explained=31"):
            build(top_k=5, results_dir=results_dir, external_dir=external_dir)

    def test_samples_that_are_not_nested_are_refused(self, results_dir, external_dir):
        _write(results_dir, external_dir, "01_A", "test", _posts(TEST_ALL[:20], 1))
        _write(results_dir, external_dir, "02_B", "test", _posts(TEST_ALL[10:], 3))
        with pytest.raises(SamePostError, match="not nested"):
            build(top_k=5, results_dir=results_dir, external_dir=external_dir)

    def test_row_ids_from_another_split_are_refused(self, results_dir, external_dir):
        _write(results_dir, external_dir, "01_A", "test", _posts(TEST_ALL, 1))
        _write(results_dir, external_dir, "02_B", "test", _posts(TEST_ALL, 3))
        assignments = pd.DataFrame({"row_id": TEST_ALL, "split": ["test"] * 29 + ["validation"]})
        with pytest.raises(SamePostError, match="not in the test split"):
            build(top_k=5, results_dir=results_dir, external_dir=external_dir, assignments=assignments)

    def test_member_runs_are_not_read(self, results_dir, external_dir):
        _standard(results_dir, external_dir)
        frame = aggregate_word_attributions([["w1 "]], [np.array([0.2])])
        meta = base_meta("07_T", "shap-partition_test_seed30_member-A", member="A", n_posts=1)
        write_run("07_T", "shap-partition_test_seed30_member-A", frame, meta, results_dir=results_dir)
        # No per-post file exists for the member run; it must not be asked for.
        support, _ = build(top_k=5, results_dir=results_dir, external_dir=external_dir)
        assert len(support) == 8
