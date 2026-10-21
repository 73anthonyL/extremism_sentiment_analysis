"""Tests for tools/split_beeswarms.py.

Synthetic text only. Two techniques and two splits of made-up posts: the
per-post files and the committed word tables are written the way the kit
writes them (the tables by notebook_kit.aggregate_word_attributions), and the
probability artifacts are supplied by a stand-in loader. Adversarial tests
prove the tool refuses, before drawing anything: a per-post file that no
longer reproduces its committed word table, a post that is not a row of the
split, a gap in a post's token positions, and a missing per-post file.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIR))

pytest.importorskip("matplotlib")

import notebook_kit as nk  # noqa: E402
import split_beeswarms as sb  # noqa: E402

TECHNIQUES = ("01_T", "07_T")
WORDS = ("alarm", "calm", "filler", "march", "quiet", "storm")


def make_posts(seed, n_posts=12):
    rng = np.random.default_rng(seed)
    posts = []
    for i in range(n_posts):
        tokens = [f"{w} " for w in rng.choice(WORDS, size=rng.integers(3, 7))]
        posts.append((f"syn_{i:04d}", tokens, rng.normal(0, 0.05, size=len(tokens))))
    return posts


def write_world(root):
    external, results = root / "external", root / "results"
    probs = {}
    for t_index, technique in enumerate(TECHNIQUES):
        for s_index, split in enumerate(sb.SPLITS):
            posts = make_posts(10 * t_index + s_index)
            name = f"{sb.run_id(split)}.csv"
            local = pd.DataFrame(
                [{"row_id": row_id, "position": p, "token": token, "attribution": value}
                 for row_id, tokens, values in posts for p, (token, value) in enumerate(zip(tokens, values))]
            )
            (external / technique / sb.LOCAL_SUBDIR).mkdir(parents=True, exist_ok=True)
            local.to_csv(external / technique / sb.LOCAL_SUBDIR / name, index=False, float_format="%.8g")
            table = nk.aggregate_word_attributions([p[1] for p in posts], [p[2] for p in posts])
            (results / technique / sb.ATTRIBUTIONS_SUBDIR).mkdir(parents=True, exist_ok=True)
            table.to_csv(results / technique / sb.ATTRIBUTIONS_SUBDIR / name, index=False, float_format="%.8g")
            probs[(technique, split)] = pd.DataFrame(
                {"row_id": [p[0] for p in posts] + ["syn_9999"], "split": split,
                 "y_true": [i % 2 for i in range(len(posts) + 1)], "y_prob": 0.5}
            )
    return {"external": external, "results": results, "probs": probs, "out": root / "out"}


@pytest.fixture
def world(tmp_path):
    return write_world(tmp_path)


def run(world, **kwargs):
    return sb.run_split_beeswarms(
        world["out"], world["external"], world["results"],
        probs_loader=lambda technique, split: world["probs"][(technique, split)], log=lambda *_: None, **kwargs,
    )


def local_path(world, technique=TECHNIQUES[0], split="test"):
    return world["external"] / technique / sb.LOCAL_SUBDIR / f"{sb.run_id(split)}.csv"


class TestDrawing:
    def test_every_technique_and_split_is_drawn_and_checked(self, world):
        checks = run(world)
        for technique in TECHNIQUES:
            for split in sb.SPLITS:
                assert (world["out"] / technique / sb.beeswarm_filename(split)).stat().st_size > 0
        saved = pd.read_csv(world["out"] / sb.CHECKS_FILENAME)
        assert list(saved.columns) == list(sb.CHECK_COLUMNS) and len(saved) == len(TECHNIQUES) * len(sb.SPLITS)
        assert (saved["n_posts"] == 12).all() and (saved["max_table_diff"] < 1e-7).all()
        assert checks["technique"].tolist() == [t for t in TECHNIQUES for _ in sb.SPLITS]

    def test_repeated_words_in_a_post_make_one_dot(self):
        posts = [("syn_0001", ["alarm ", "calm ", "Alarm."], np.array([0.1, -0.2, 0.05]))]
        frame = sb.post_contributions("T", posts).set_index("word")["attribution"]
        assert frame.to_dict() == pytest.approx({"alarm": 0.15, "calm": -0.2})

    def test_only_downloaded_techniques_with_committed_runs_are_found(self, world):
        (world["external"] / "99_NOT_COMMITTED" / sb.LOCAL_SUBDIR).mkdir(parents=True)
        assert sb.available_techniques(world["external"], world["results"]) == list(TECHNIQUES)


class TestRefusals:
    def test_a_file_that_no_longer_reproduces_its_table_is_refused(self, world):
        path = local_path(world, TECHNIQUES[1], "validation")
        frame = pd.read_csv(path, keep_default_na=False)
        frame.loc[0, "attribution"] += 0.01
        frame.to_csv(path, index=False)
        with pytest.raises(sb.SplitBeeswarmsError, match="does not reproduce"):
            run(world)
        assert not world["out"].exists()

    def test_a_post_outside_the_split_is_refused(self, world):
        world["probs"][(TECHNIQUES[0], "test")] = world["probs"][(TECHNIQUES[0], "test")].iloc[1:]
        with pytest.raises(sb.SplitBeeswarmsError, match="not rows of the committed test"):
            run(world)
        assert not world["out"].exists()

    def test_a_gap_in_token_positions_is_refused(self, world):
        path = local_path(world)
        frame = pd.read_csv(path, keep_default_na=False)
        frame = frame.drop(index=frame.index[(frame["row_id"] == "syn_0003") & (frame["position"] == 1)])
        frame.to_csv(path, index=False)
        with pytest.raises(sb.SplitBeeswarmsError, match="token positions"):
            run(world)

    def test_a_missing_per_post_file_is_refused(self, world):
        local_path(world, TECHNIQUES[1], "test").unlink()
        with pytest.raises(sb.SplitBeeswarmsError, match="no per-post attributions"):
            run(world)
        assert not world["out"].exists()
