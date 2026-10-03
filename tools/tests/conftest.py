"""Shared fixtures for the toolkit tests.

Fixtures build synthetic artifacts with the frozen split's exact shape (450
rows, 170 positive for test) so shape assertions are exercised, and synthetic
attribution runs whose words come from the real lexicons so categorization is
exercised end to end without any dataset text.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIR))

from notebook_kit import EXPLAINER_ID
from repo_paths import EXPECTED_SPLIT_COUNTS

SEED = 30


def _synthetic_probs(split, separation=1.0, seed=SEED):
    """A probability artifact with the frozen class balance for `split`."""
    total, negatives, positives = EXPECTED_SPLIT_COUNTS[split]
    rng = np.random.default_rng(seed)
    y_true = np.array([0] * negatives + [1] * positives)
    noise = rng.normal(0.0, 0.25, total)
    logits = y_true * separation + noise
    y_prob = 1.0 / (1.0 + np.exp(-logits * 3.0))
    return pd.DataFrame(
        {
            "row_id": [f"ex_{i:06d}" for i in range(total)],
            "split": split,
            "y_true": y_true,
            "y_prob": np.clip(y_prob, 0.0, 1.0),
        }
    )


@pytest.fixture
def probs_dir(tmp_path, monkeypatch):
    """A temporary probs directory wired into the tools' paths."""
    directory = tmp_path / "probs"
    directory.mkdir(parents=True)
    import probs_artifact
    import repo_paths

    monkeypatch.setattr(repo_paths, "PROBS_DIR", directory, raising=False)
    monkeypatch.setattr(probs_artifact, "PROBS_DIR", directory, raising=False)
    return directory


@pytest.fixture
def make_probs(probs_dir):
    def _make(technique, separation=1.0, seed=SEED):
        paths = {}
        for split in ("validation", "test"):
            frame = _synthetic_probs(split, separation=separation, seed=seed)
            path = probs_dir / f"{technique}__{split}.csv"
            frame.to_csv(path, index=False)
            paths[split] = path
        return paths

    return _make


@pytest.fixture
def results_dir(tmp_path):
    """An empty results_summary/ stand-in."""
    directory = tmp_path / "results_summary"
    directory.mkdir()
    return directory


def synthetic_attributions(words, seed=SEED, n_posts=200):
    """A valid attribution frame over the given words with random values."""
    rng = np.random.default_rng(seed)
    mean_attr = rng.normal(0.0, 0.05, len(words))
    mean_abs = np.abs(mean_attr) + rng.uniform(0.0, 0.02, len(words))
    support = rng.integers(1, n_posts + 1, len(words))
    return pd.DataFrame(
        {
            "word": words,
            "mean_abs_attribution": mean_abs,
            "mean_attribution": mean_attr,
            "support": support,
        }
    )


def base_meta(technique, run_id, split="test", seed=SEED, member=None, n_posts=200):
    return {
        "technique": technique,
        "run_id": run_id,
        "split": split,
        "explainer": EXPLAINER_ID,
        "background_size": 0,
        "seed": seed,
        "aggregation": "word",
        "n_posts_explained": n_posts,
        "member": member,
    }


@pytest.fixture
def make_run(results_dir):
    """Factory writing a synthetic attribution run into the temp results dir."""
    from attributions import write_run

    def _make(technique, run_id, words, seed=SEED, member=None, split="test"):
        frame = synthetic_attributions(words, seed=seed)
        meta = base_meta(technique, run_id, split=split, seed=seed, member=member)
        return write_run(technique, run_id, frame, meta, results_dir=results_dir)

    return _make


@pytest.fixture
def golden_results_folder(tmp_path):
    """A copy of a real committed result folder, for regression checks."""
    source = TOOLS_DIR.parent / "results_summary" / "01_LOG-REG_TF-IDF"
    if not source.exists():
        pytest.skip("committed golden result folder not present")
    destination = tmp_path / "01_LOG-REG_TF-IDF"
    destination.mkdir()
    for item in source.iterdir():
        if item.is_file():
            destination.joinpath(item.name).write_bytes(item.read_bytes())
    return destination


def corrupt_json_field(path, field, delta):
    with open(path) as handle:
        payload = json.load(handle)
    payload[field] = payload[field] + delta
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    return payload[field]


# Words for synthetic runs: two from each lexicon category plus topical filler.
FRAMING_WORDS = ["kkk", "qanon"]
IDENTITY_WORDS = ["muslim", "immigrants"]
TOPICAL_WORDS = ["weather", "football", "tuesday", "coffee"]
