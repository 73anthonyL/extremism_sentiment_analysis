"""Behavioural and adversarial tests for tools/word_agreement.py on synthetic runs."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIR))

from attributions import write_run
from conftest import base_meta, synthetic_attributions
from word_agreement import WordAgreementError, build

WORDS = [f"w{i}" for i in range(60)]
OTHER_WORDS = [f"v{i}" for i in range(60)]


def _write(results_dir, technique, run_id, frame, split="test", member=None, n_posts=200):
    meta = base_meta(technique, run_id, split=split, member=member, n_posts=n_posts)
    return write_run(technique, run_id, frame, meta, results_dir=results_dir)


class TestAgreement:
    def test_identical_runs_agree_perfectly(self, results_dir):
        frame = synthetic_attributions(WORDS, seed=1)
        _write(results_dir, "01_A", "run", frame)
        _write(results_dir, "02_B", "run", frame)
        pairs, shared, support = build(top_k=10, results_dir=results_dir)
        row = pairs.iloc[0]
        assert len(pairs) == 1 and row["n_shared_words"] == 60
        assert row["spearman_mean_attribution"] == pytest.approx(1.0)
        assert row["spearman_mean_abs_attribution"] == pytest.approx(1.0)
        assert row["jaccard_top_k_positive"] == 1.0 and row["jaccard_top_k_abs"] == 1.0
        assert len(shared) == 10 and shared["n_in_top_k"].eq(2).all()
        assert shared["n_columns"].eq(2).all()
        assert len(support) == 2 and support["member"].eq("whole").all()

    def test_sign_flip_reverses_signed_agreement_only(self, results_dir):
        frame = synthetic_attributions(WORDS, seed=2)
        flipped = frame.assign(mean_attribution=-frame["mean_attribution"])
        _write(results_dir, "01_A", "run", frame)
        _write(results_dir, "02_B", "run", flipped)
        pairs, _, _ = build(top_k=10, results_dir=results_dir)
        row = pairs.iloc[0]
        assert row["spearman_mean_attribution"] == pytest.approx(-1.0)
        assert row["spearman_mean_abs_attribution"] == pytest.approx(1.0)
        assert row["jaccard_top_k_abs"] == 1.0
        assert row["jaccard_top_k_positive"] < 0.2

    def test_single_post_share_is_counted(self, results_dir):
        frame = synthetic_attributions(WORDS, seed=3)
        frame = frame.sort_values("mean_attribution", ascending=False).reset_index(drop=True)
        frame.loc[:4, "support"] = 1       # five of the top ten positive words are single-post words
        frame.loc[5:, "support"] = 7
        _write(results_dir, "01_A", "run", frame)
        _write(results_dir, "02_B", "run", synthetic_attributions(WORDS, seed=4))
        _, _, support = build(top_k=10, results_dir=results_dir)
        a = support.set_index("technique").loc["01_A"]
        assert a["share_single_post_top_k_positive"] == pytest.approx(0.5)
        assert a["share_single_post_vocabulary"] == pytest.approx(5 / 60)


class TestSelection:
    def test_member_runs_are_left_out_unless_asked(self, results_dir):
        _write(results_dir, "01_A", "run", synthetic_attributions(WORDS, seed=5))
        _write(results_dir, "11_E", "whole", synthetic_attributions(WORDS, seed=6))
        _write(results_dir, "11_E", "member_a", synthetic_attributions(WORDS, seed=7), member="A_anchor")
        pairs, _, support = build(top_k=10, results_dir=results_dir)
        assert len(pairs) == 1 and set(support["member"]) == {"whole"}
        pairs, _, support = build(top_k=10, results_dir=results_dir, include_members=True)
        assert len(pairs) == 3 and "A_anchor" in set(support["member"])

    def test_runs_from_another_split_are_not_used(self, results_dir):
        _write(results_dir, "01_A", "run", synthetic_attributions(WORDS, seed=8), split="test")
        _write(results_dir, "02_B", "run", synthetic_attributions(WORDS, seed=9), split="validation")
        with pytest.raises(WordAgreementError, match="fewer than two"):
            build(split="test", top_k=10, results_dir=results_dir)
        _write(results_dir, "02_B", "run_test", synthetic_attributions(WORDS, seed=10), split="test")
        pairs, _, _ = build(split="test", top_k=10, results_dir=results_dir)
        assert len(pairs) == 1


class TestRefusals:
    def test_disjoint_vocabularies_are_refused(self, results_dir):
        _write(results_dir, "01_A", "run", synthetic_attributions(WORDS, seed=11))
        _write(results_dir, "02_B", "run", synthetic_attributions(OTHER_WORDS, seed=12))
        with pytest.raises(WordAgreementError, match="share only 0 words"):
            build(top_k=10, results_dir=results_dir)

    def test_too_small_shared_vocabulary_is_refused(self, results_dir):
        _write(results_dir, "01_A", "run", synthetic_attributions(WORDS, seed=13))
        _write(results_dir, "02_B", "run", synthetic_attributions(WORDS[:20] + OTHER_WORDS[:40], seed=14))
        with pytest.raises(WordAgreementError, match="minimum 30"):
            build(top_k=10, results_dir=results_dir)
        pairs, _, _ = build(top_k=10, results_dir=results_dir, min_shared_words=20)
        assert pairs.iloc[0]["n_shared_words"] == 20

    def test_a_broken_run_is_refused_by_the_loader(self, results_dir):
        from attributions import AttributionArtifactError

        _write(results_dir, "01_A", "run", synthetic_attributions(WORDS, seed=15))
        path = _write(results_dir, "02_B", "run", synthetic_attributions(WORDS, seed=16))
        broken = pd.read_csv(path)
        broken["row_id"] = [f"ex_{i:06d}" for i in range(len(broken))]
        broken.to_csv(path, index=False)
        with pytest.raises(AttributionArtifactError):
            build(top_k=10, results_dir=results_dir)
