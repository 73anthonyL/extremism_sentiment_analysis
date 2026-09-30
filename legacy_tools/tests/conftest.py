"""Shared fixtures for the toolkit tests.

Fixtures build synthetic artifacts with the frozen split's exact shape (450 rows,
170 positive for test) so that shape assertions in the tools are exercised rather
than bypassed.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIR))

from repo_paths import EXPECTED_SPLIT_COUNTS

SEED = 30


def _synthetic_probs(split, separation=1.0, seed=SEED):
    """Build a probability artifact with the frozen class balance for `split`.

    `separation` controls how far the positive class's probabilities are shifted
    upward, so tests can produce a deliberately strong or weak classifier.
    """
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
    """A temporary research_loop/probs directory wired into the tools' paths."""
    directory = tmp_path / "probs"
    directory.mkdir(parents=True)

    import probs_artifact
    import repo_paths

    monkeypatch.setattr(repo_paths, "PROBS_DIR", directory, raising=False)
    monkeypatch.setattr(probs_artifact, "PROBS_DIR", directory, raising=False)
    return directory


@pytest.fixture
def make_probs(probs_dir):
    """Factory writing a technique's validation and test artifacts to disk."""

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
    """Nudge a numeric field in a JSON artifact, simulating a transcription error."""
    with open(path) as handle:
        payload = json.load(handle)
    payload[field] = payload[field] + delta
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    return payload[field]
