"""Behavioural and adversarial tests for tools/bootstrap_uncertainty.py on synthetic per-post files."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIR))

from attributions import load_run, write_run
from bootstrap_uncertainty import (
    BootstrapError,
    PostMatrix,
    build,
    check_text_free,
    fpr_inputs,
    fpr_ratios,
    write_tables,
)
from conftest import base_meta
from identity_fpr import analyze
from notebook_kit import aggregate_word_attributions
from rq2_rq4_relation import spearman
from same_post_stability import SamePostError, build_all, load_whole_runs

VOCAB = [f"w{i}" for i in range(40)]
LEXICONS = {"slur": ["w1"], "extremist_framing": ["w2", "w3"], "identity_term": ["w4", "w5", "w6"]}
TOP_K = 5
TEST_ALL = [f"ex_{i:06d}" for i in range(30)]
TEST_SAMPLE = TEST_ALL[::3]
VAL_ALL = [f"ex_{i:06d}" for i in range(100, 130)]
VAL_SAMPLE = VAL_ALL[::3]
CLASSICAL = ("01_A", "02_B", "03_C")
TRANSFORMER = "07_T"
FPR_ROWS = [f"ex_{i:06d}" for i in range(200, 260)]


def _posts(row_ids, seed):
    rng = np.random.default_rng(seed)
    posts = {}
    for row_id in row_ids:
        tokens = [f"{w} " for w in rng.choice(VOCAB, size=6, replace=True)]
        posts[row_id] = (tokens, rng.normal(0.05 * (seed % 3), 0.1, len(tokens)))
    return posts


def _write(results_dir, external_dir, technique, split, posts):
    run_id = f"shap-partition_{split}_seed30"
    frame = aggregate_word_attributions([p[0] for p in posts.values()], [p[1] for p in posts.values()])
    write_run(technique, run_id, frame, base_meta(technique, run_id, split=split, n_posts=len(posts)),
              results_dir=results_dir)
    rows = [{"row_id": r, "position": i, "token": t, "attribution": v}
            for r, (tokens, values) in posts.items() for i, (t, v) in enumerate(zip(tokens, values))]
    folder = external_dir / technique / "attributions_local"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{run_id}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _fpr_fixture(techniques):
    """A synthetic processed dataset and probability artifacts with false positives on both sides."""
    labels = np.array([0, 1] * (len(FPR_ROWS) // 2))
    texts = [("w4 here" if i % 4 < 2 else "nothing here") for i in range(len(FPR_ROWS))]
    processed = pd.DataFrame({"row_id": FPR_ROWS, "text": texts, "label": labels})
    probs, thresholds = {}, {}
    for k, technique in enumerate(techniques):
        rng = np.random.default_rng(100 + k)
        y_prob = np.where(labels == 1, 0.8, rng.uniform(0.0, 0.62 + 0.03 * k, len(labels)))
        probs[technique] = pd.DataFrame({"row_id": FPR_ROWS, "split": "test", "y_true": labels, "y_prob": y_prob})
        thresholds[technique] = 0.5
    return processed, probs, thresholds


@pytest.fixture
def external_dir(tmp_path):
    directory = tmp_path / "external"
    directory.mkdir()
    return directory


@pytest.fixture
def setup(results_dir, external_dir):
    """Four techniques (three on every post, one on a sample) with committed tables built by the real tools."""
    paths = {}
    for k, technique in enumerate(CLASSICAL):
        test, val = _posts(TEST_ALL, 1 + 3 * k), _posts(VAL_ALL, 2 + 3 * k)
        paths[(technique, "test")] = _write(results_dir, external_dir, technique, "test", test)
        paths[(technique, "validation")] = _write(results_dir, external_dir, technique, "validation", val)
    test, val = _posts(TEST_SAMPLE, 11), _posts(VAL_SAMPLE, 12)
    paths[(TRANSFORMER, "test")] = _write(results_dir, external_dir, TRANSFORMER, "test", test)
    paths[(TRANSFORMER, "validation")] = _write(results_dir, external_dir, TRANSFORMER, "validation", val)

    support, agreement, categories = build_all(top_k=TOP_K, results_dir=results_dir, external_dir=external_dir,
                                               lexicons=LEXICONS)
    runs = load_whole_runs(results_dir=results_dir, external_dir=external_dir)
    techniques = sorted(runs)
    processed, probs, thresholds = _fpr_fixture(techniques)
    summaries = [analyze(t, "test", thresholds[t], processed=processed, probs=probs[t],
                         terms=LEXICONS["identity_term"])[0] for t in techniques]
    fpr_table = pd.DataFrame(summaries)
    identity = categories[(categories["split"] == "test") & (categories["category"] == "identity_term")]
    ratio = fpr_table.set_index("technique")["fpr_ratio"].reindex(techniques).to_numpy()
    relation = []
    for post_set, name in (("own", "identity_share_positive_mass"),
                           ("common", "identity_share_positive_mass_common_posts")):
        share = identity[identity["post_set"] == post_set].set_index("technique")["share_of_positive_mass"]
        relation.append({"x_measure": name, "y_measure": "fpr_ratio", "subset": "all",
                         "spearman_rho": spearman(share.reindex(techniques).to_numpy(), ratio)})
    committed = {
        "word_tables": {(t, s): load_run(t, runs[t][s][0], results_dir)[0] for t in runs for s in runs[t]},
        "rq2_category_shares": categories[categories["post_set"] == "own"].assign(member=np.nan),
        "rq2_same_post_category_shares": categories,
        "rq2_same_post_support": support,
        "rq3_same_post_split_agreement": agreement,
        "rq4_identity_fpr": fpr_table,
        "rq2_rq4_relation": pd.DataFrame(relation),
    }
    fpr = fpr_inputs(techniques, processed, LEXICONS["identity_term"], probs, thresholds)
    return {"runs": runs, "committed": committed, "fpr": fpr, "paths": paths,
            "results_dir": results_dir, "external_dir": external_dir}


def _build(setup, reps=30, seed=30, committed=None, runs=None):
    return build(runs or setup["runs"], LEXICONS, committed or setup["committed"], top_k=TOP_K, reps=reps,
                 seed=seed, fpr=setup["fpr"], contrasts=(TRANSFORMER,))


class TestIdentity:
    def test_point_column_is_the_committed_number(self, setup):
        tables = _build(setup)
        cats = tables["categories"].set_index(["technique", "split", "post_set", "category"])["point"]
        committed = setup["committed"]["rq2_same_post_category_shares"].set_index(
            ["technique", "split", "post_set", "category"])["share_of_positive_mass"]
        assert np.allclose(cats.sort_index(), committed.reindex(cats.sort_index().index))
        same = tables["same_post"].set_index(["technique", "split", "post_set", "measure"])["point"]
        support = setup["committed"]["rq2_same_post_support"].set_index(["technique", "split", "post_set"])
        for (t, s, p, m), value in same.items():
            assert value == pytest.approx(support.loc[(t, s, p), m])
        agree = tables["agreement"].set_index(["technique", "post_set", "measure"])["point"]
        committed_agree = setup["committed"]["rq3_same_post_split_agreement"].set_index(["technique", "post_set"])
        for (t, p, m), value in agree.items():
            assert value == pytest.approx(committed_agree.loc[(t, p), m])

    def test_weights_reproduce_the_kit_aggregation_with_repeats(self, setup):
        # A replicate is the committed aggregation on a list of posts with repeats.
        posts = setup["runs"]["01_A"]["test"][1]
        matrix = PostMatrix(posts)
        rng = np.random.default_rng(5)
        draw = rng.choice(matrix.row_ids, size=len(matrix.row_ids), replace=True)
        counts = pd.Series(draw).value_counts()
        table = matrix.table(matrix.weights(list(counts.index), counts.to_numpy(float)))
        expected = aggregate_word_attributions([posts[r][0] for r in draw], [posts[r][1] for r in draw])
        expected = expected.set_index("word").loc[matrix.vocab[table.idx]]
        assert np.allclose(table.mean, expected["mean_attribution"])
        assert np.allclose(table.mean_abs, expected["mean_abs_attribution"])
        assert np.array_equal(table.support, expected["support"].to_numpy(float))

    def test_intervals_bracket_a_finite_sample(self, setup):
        tables = _build(setup)
        for name in ("categories", "same_post", "agreement", "differences"):
            frame = tables[name]
            assert len(frame) and (frame["ci_low"] <= frame["ci_high"] + 1e-12).all()
            assert (frame["n_reps"] == 30).all()
        assert set(tables["differences"]["reference_technique"]) == set(CLASSICAL)
        assert (tables["differences"]["role"] == "descriptive").all()
        rho = tables["relation"][tables["relation"]["quantity"] == "spearman_rho"]
        assert set(rho["resampled"]) == {"posts_and_rows", "posts_only", "rows_only"}


class TestDeterminism:
    def test_same_seed_same_tables(self, setup):
        a, b = _build(setup, seed=7), _build(setup, seed=7)
        for name in a:
            pd.testing.assert_frame_equal(a[name], b[name])

    def test_different_seed_different_intervals(self, setup):
        a, b = _build(setup, seed=7), _build(setup, seed=8)
        assert not a["categories"]["ci_low"].equals(b["categories"]["ci_low"])
        assert np.allclose(a["categories"]["point"], b["categories"]["point"])


class TestRefusals:
    def test_per_post_file_that_does_not_reproduce_the_committed_table_is_refused(self, setup):
        path = setup["paths"][("02_B", "test")]
        frame = pd.read_csv(path, dtype={"row_id": str, "token": str}, keep_default_na=False)
        frame.loc[0, "attribution"] += 0.5
        frame.to_csv(path, index=False)
        with pytest.raises(SamePostError, match="does not reproduce the committed"):
            load_whole_runs(results_dir=setup["results_dir"], external_dir=setup["external_dir"])

    def test_tampered_committed_word_table_is_refused(self, setup):
        committed = dict(setup["committed"])
        committed["word_tables"] = dict(committed["word_tables"])
        frame = committed["word_tables"][("01_A", "test")].copy()
        frame.loc[0, "mean_attribution"] += 0.01
        committed["word_tables"][("01_A", "test")] = frame
        with pytest.raises(BootstrapError, match="per-post matrix does not reproduce"):
            _build(setup, reps=2, committed=committed)

    def test_stale_category_table_is_refused(self, setup):
        committed = dict(setup["committed"])
        shares = committed["rq2_category_shares"].copy()
        row = shares.index[(shares["technique"] == "03_C") & (shares["category"] == "topical")][0]
        shares.loc[row, "share_of_positive_mass"] += 0.05
        committed["rq2_category_shares"] = shares
        with pytest.raises(BootstrapError, match="rq2_category_shares"):
            _build(setup, reps=2, committed=committed)

    def test_stale_split_agreement_is_refused(self, setup):
        committed = dict(setup["committed"])
        agree = committed["rq3_same_post_split_agreement"].copy()
        agree.loc[0, "spearman_mean_abs_attribution"] += 0.01
        committed["rq3_same_post_split_agreement"] = agree
        with pytest.raises(BootstrapError, match="rq3_same_post_split_agreement"):
            _build(setup, reps=2, committed=committed)

    def test_relation_rho_that_does_not_reproduce_is_refused(self, setup):
        committed = dict(setup["committed"])
        relation = committed["rq2_rq4_relation"].copy()
        relation.loc[0, "spearman_rho"] += 0.2
        committed["rq2_rq4_relation"] = relation
        with pytest.raises(BootstrapError, match="rq2_rq4_relation"):
            _build(setup, reps=2, committed=committed)

    def test_missing_committed_table_is_refused(self, setup):
        committed = {k: v for k, v in setup["committed"].items() if k != "rq2_same_post_support"}
        with pytest.raises(BootstrapError, match="rq2_same_post_support is missing"):
            _build(setup, reps=2, committed=committed)

    def test_one_replicate_is_refused(self, setup):
        with pytest.raises(BootstrapError, match="at least two replicates"):
            _build(setup, reps=1)

    def test_payload_with_a_row_id_is_refused(self, setup, tmp_path):
        tables = _build(setup, reps=2)
        tables["categories"].loc[0, "technique"] = "ex_000123"
        paths = {name: tmp_path / f"{name}.csv" for name in tables}
        with pytest.raises(BootstrapError, match="row id"):
            write_tables(tables, paths)
        assert not any(p.exists() for p in paths.values())

    def test_word_or_text_column_is_refused(self):
        for column in ("word", "row_id", "text"):
            with pytest.raises(BootstrapError, match="refuses column"):
                check_text_free(pd.DataFrame({column: ["a"], "point": [0.1]}), "fixture")

    def test_artifacts_on_different_rows_are_refused(self):
        processed, probs, thresholds = _fpr_fixture(["01_A", "02_B"])
        probs["02_B"] = probs["02_B"].iloc[:-2]
        with pytest.raises(BootstrapError, match="other test rows"):
            fpr_inputs(["01_A", "02_B"], processed, LEXICONS["identity_term"], probs, thresholds)


class TestFprRatio:
    def test_identity_counts_reproduce_identity_fpr(self, setup):
        inputs, n_rows = setup["fpr"]
        ratios = fpr_ratios(inputs, np.ones((1, n_rows)))
        committed = setup["committed"]["rq4_identity_fpr"].set_index("technique")["fpr_ratio"]
        for technique, value in ratios.items():
            assert value[0] == pytest.approx(committed[technique])

    def test_zero_baseline_is_infinite_and_zero_over_zero_is_missing(self):
        with_id = np.array([1.0, 1.0, 0.0, 0.0])
        without_id = np.array([0.0, 0.0, 1.0, 1.0])
        inputs = {"a": (with_id, without_id, np.array([1.0, 0.0, 0.0, 0.0])),
                  "b": (with_id, without_id, np.zeros(4))}
        ratios = fpr_ratios(inputs, np.ones((1, 4)))
        assert np.isinf(ratios["a"][0]) and np.isnan(ratios["b"][0])
