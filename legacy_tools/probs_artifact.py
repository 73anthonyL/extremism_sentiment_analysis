"""Read/write the sanitized probability artifact.

This artifact is the keystone of the hybrid compute design. A transformer run on
Kaggle exports, per split, one row per example:

    row_id, split, y_true, y_prob

No text, no text_hash. `y_true` is a single bit already public via the committed
confusion matrices, and it is required locally -- without it no paired statistical
test, threshold re-derivation, or calibration study can run off-GPU. With it,
essentially all downstream research (calibration, thresholding, ensembling,
stacking, McNemar) becomes CPU work over committed files.

The loader refuses any file carrying a text-bearing column. That refusal is the
mechanism preventing dataset text from re-entering the repository through the
artifact that everything else is built on.
"""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from repo_paths import EXPECTED_SPLIT_COUNTS, POSITIVE_LABEL, PROBS_DIR, SPLIT_NAMES

# Exactly the columns a probability artifact must carry. `row_id` ties a row
# back to the frozen split assignment; `y_true` is the single bit that lets
# paired tests and threshold work run off-GPU; `y_prob` is the positive-class
# probability. Keeping the contract this small is what makes the artifact safe
# to commit.
REQUIRED_COLUMNS = ("row_id", "split", "y_true", "y_prob")

# Column names that mean dataset text (or a proxy for it) has leaked into the
# artifact. Presence of any of these is an immediate refusal, never a warning.
FORBIDDEN_COLUMNS = ("text", "text_hash", "Original_Message", "text_preview", "message")


class ProbsArtifactError(ValueError):
    """Raised when a probability artifact violates its schema or split contract."""


def probs_path(technique, split):
    """Return the canonical artifact path for a technique/split pair."""
    return PROBS_DIR / f"{technique}__{split}.csv"


def meta_path(technique):
    """Return the canonical sidecar metadata path for a technique."""
    return PROBS_DIR / f"{technique}__meta.json"


def sha256_file(path, n_chars=16):
    """Short SHA-256 of a file's bytes, for provenance records."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:n_chars]


def load_probs(path, expected_split=None):
    """Load and validate a probability artifact.

    Enforces: required columns present, no text-bearing column, probabilities in
    [0, 1], labels in {0, 1}, unique row_ids, a single split value, and -- when
    the split is one of the frozen splits -- the exact frozen row and class
    counts. A stale or truncated Kaggle download fails here rather than silently
    producing a plausible wrong metric.
    """
    path = Path(path)
    if not path.exists():
        raise ProbsArtifactError(f"probability artifact not found: {path}")

    frame = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ProbsArtifactError(f"{path.name}: missing required columns {missing}")

    present_forbidden = [c for c in FORBIDDEN_COLUMNS if c in frame.columns]
    if present_forbidden:
        raise ProbsArtifactError(
            f"{path.name}: carries text-bearing column(s) {present_forbidden}."
            " Probability artifacts must never contain dataset text."
        )

    if frame["row_id"].duplicated().any():
        raise ProbsArtifactError(f"{path.name}: duplicate row_id values")

    splits_present = sorted(frame["split"].unique())
    if len(splits_present) != 1:
        raise ProbsArtifactError(f"{path.name}: expected one split, found {splits_present}")
    split = splits_present[0]

    if expected_split is not None and split != expected_split:
        raise ProbsArtifactError(f"{path.name}: split is '{split}', expected '{expected_split}'")

    bad_labels = set(frame["y_true"].unique()) - {0, 1}
    if bad_labels:
        raise ProbsArtifactError(f"{path.name}: y_true contains non-binary values {bad_labels}")

    if not frame["y_prob"].between(0.0, 1.0).all():
        raise ProbsArtifactError(f"{path.name}: y_prob values outside [0, 1]")

    if split in EXPECTED_SPLIT_COUNTS:
        expected_total, expected_neg, expected_pos = EXPECTED_SPLIT_COUNTS[split]
        actual_pos = int((frame["y_true"] == POSITIVE_LABEL).sum())
        actual_neg = int((frame["y_true"] != POSITIVE_LABEL).sum())
        actual = (len(frame), actual_neg, actual_pos)
        if actual != (expected_total, expected_neg, expected_pos):
            raise ProbsArtifactError(
                f"{path.name}: split '{split}' has counts {actual} "
                "(total, negative, positive) but the frozen protocol requires "
                f"{(expected_total, expected_neg, expected_pos)}"
            )

    return frame.sort_values("row_id").reset_index(drop=True)


def load_technique_probs(technique, split):
    """Load one technique's artifact for one split by canonical path."""
    return load_probs(probs_path(technique, split), expected_split=split)


def available_techniques(split="test"):
    """List techniques that have a committed artifact for the given split."""
    if not PROBS_DIR.exists():
        return []
    suffix = f"__{split}.csv"
    return sorted(p.name[: -len(suffix)] for p in PROBS_DIR.glob(f"*{suffix}"))


def align_pair(frame_a, frame_b):
    """Inner-join two artifacts on row_id, asserting identical row_id sets.

    Paired statistics (McNemar, paired bootstrap) are only valid when both
    techniques were scored on exactly the same examples; this refuses to align
    artifacts that were not.
    """
    ids_a, ids_b = set(frame_a["row_id"]), set(frame_b["row_id"])
    if ids_a != ids_b:
        raise ProbsArtifactError(
            f"row_id sets differ: {len(ids_a - ids_b)} only in first, "
            f"{len(ids_b - ids_a)} only in second. Paired comparison is invalid."
        )
    merged = frame_a.merge(frame_b, on=["row_id", "split", "y_true"], suffixes=("_a", "_b"))
    if len(merged) != len(frame_a):
        raise ProbsArtifactError(
            "join changed row count; the two artifacts disagree on y_true for some rows"
        )
    return merged.sort_values("row_id").reset_index(drop=True)


def write_meta(technique, payload):
    """Write the sidecar metadata JSON, stamping artifact hashes for provenance."""
    PROBS_DIR.mkdir(parents=True, exist_ok=True)
    record = dict(payload)
    record["technique"] = technique
    record["contains_text"] = False
    record["artifact_sha256"] = {
        split: sha256_file(probs_path(technique, split))
        for split in SPLIT_NAMES
        if probs_path(technique, split).exists()
    }
    path = meta_path(technique)
    with open(path, "w") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
    return path


def read_meta(technique):
    """Read a technique's sidecar metadata, or {} when absent."""
    path = meta_path(technique)
    if not path.exists():
        return {}
    with open(path) as handle:
        return json.load(handle)
