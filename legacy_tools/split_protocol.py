"""Deterministic reconstruction and verification of the frozen dataset split.

The split assignment is frozen, but it is also *derivable*: notebook 00 builds it
from `data/dataset.csv` with a fixed seed and a fixed stratified 70/15/15 recipe.
That makes the committed `splits/split_assignments.csv` a mirror of a computation
rather than an irreplaceable primary artifact -- which matters, because the
committed mirror was found to be a stale pre-correction file whose class counts
(test 240/210) disagree with the published foundation counts (test 280/170).

Reconstruction here is verified against `results_summary/foundation/` before it is
trusted. If the regenerated counts do not match the published ones exactly, the
reconstruction is rejected rather than the foundation being overwritten.
"""

import hashlib
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from repo_paths import (
    DATASET_CSV,
    EXPECTED_SPLIT_COUNTS,
    PROCESSED_ROW_COUNT,
    RANDOM_SEED,
    SPLIT_LABEL_DISTRIBUTION_CSV,
    TEST_SIZE,
    TRAIN_SIZE,
    VALIDATION_SIZE,
)

# Raw column names in data/dataset.csv.
RAW_TEXT_COL = "Original_Message"
RAW_LABEL_COL = "Extremism_Label"

LABEL_MAP = {
    "NON_EXTREMIST": 0,
    "NON-EXTREMIST": 0,
    "NON EXTREMIST": 0,
    "NOT_EXTREMIST": 0,
    "NOT EXTREMIST": 0,
    "0": 0,
    0: 0,
    "EXTREMIST": 1,
    "1": 1,
    1: 1,
}

TEXT_HASH_CHARS = 16


def sha256_text(value, n_chars=TEXT_HASH_CHARS):
    """Stable short SHA-256 of a text value, matching notebook 00's recipe."""
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:n_chars]


def clean_text(text):
    """Collapse whitespace and strip -- notebook 00's only text normalization."""
    text = "" if pd.isna(text) else str(text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_label(value):
    """Map a raw label cell to 0/1, or NaN when unmappable."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, np.integer)):
        return LABEL_MAP.get(int(value), np.nan)
    if isinstance(value, (float, np.floating)) and value in (0.0, 1.0):
        return LABEL_MAP.get(int(value), np.nan)
    return LABEL_MAP.get(str(value).strip().upper(), np.nan)


def build_processed_dataset(dataset_csv=DATASET_CSV):
    """Rebuild the processed dataset exactly as notebook 00 section 6 does.

    Drops rows with empty text or an unmappable label, then assigns the stable
    `ex_NNNNNN` row IDs that every downstream artifact keys on. Row ID assignment
    depends on post-cleaning row order, so the drop must happen first.
    """
    raw = pd.read_csv(dataset_csv)
    processed = raw[[RAW_TEXT_COL, RAW_LABEL_COL]].copy()
    processed["text"] = processed[RAW_TEXT_COL].apply(clean_text)
    processed["label"] = processed[RAW_LABEL_COL].apply(normalize_label)

    keep = ~(processed["text"].eq("") | processed["label"].isna())
    processed = processed.loc[keep].reset_index(drop=True)
    processed["label"] = processed["label"].astype(int)
    processed["row_id"] = [f"ex_{i:06d}" for i in range(len(processed))]
    processed["text_hash"] = processed["text"].apply(sha256_text)

    return processed[["row_id", "text", "label", "text_hash"]]


def reconstruct_split_assignments(processed=None):
    """Reproduce the frozen stratified 70/15/15 split assignment.

    Uses two nested `train_test_split` calls with RANDOM_SEED, mirroring notebook
    00: first peel off the training rows, then halve the remainder into
    validation and test. Returns a frame with columns row_id,label,text_hash,split.
    """
    if processed is None:
        processed = build_processed_dataset()

    train_df, holdout_df = train_test_split(
        processed,
        train_size=TRAIN_SIZE,
        random_state=RANDOM_SEED,
        stratify=processed["label"],
    )

    # The remaining 30% is halved: validation's share of the holdout is
    # 0.15 / (0.15 + 0.15) = 0.5, derived from the frozen constants rather
    # than hardcoded so the recipe reads as 70/15/15.
    validation_share = VALIDATION_SIZE / (VALIDATION_SIZE + TEST_SIZE)
    validation_df, test_df = train_test_split(
        holdout_df,
        train_size=validation_share,
        random_state=RANDOM_SEED,
        stratify=holdout_df["label"],
    )

    labelled = []
    for split_name, frame in (
        ("train", train_df),
        ("validation", validation_df),
        ("test", test_df),
    ):
        part = frame[["row_id", "label", "text_hash"]].copy()
        part["split"] = split_name
        labelled.append(part)

    assignments = pd.concat(labelled, ignore_index=True)
    return assignments.sort_values("row_id").reset_index(drop=True)


def split_counts(assignments):
    """Return {split: (total, negatives, positives)} for a split assignment frame."""
    counts = {}
    for split_name, group in assignments.groupby("split"):
        value_counts = group["label"].value_counts()
        counts[split_name] = (
            int(len(group)),
            int(value_counts.get(0, 0)),
            int(value_counts.get(1, 0)),
        )
    return counts


def published_split_counts(distribution_csv=SPLIT_LABEL_DISTRIBUTION_CSV):
    """Read the authoritative per-split class counts from the foundation artifact."""
    distribution = pd.read_csv(distribution_csv)
    counts = {}
    for split_name, group in distribution.groupby("split"):
        by_label = dict(zip(group["label"], group["count"]))
        counts[split_name] = (
            int(group["split_count"].iloc[0]),
            int(by_label.get(0, 0)),
            int(by_label.get(1, 0)),
        )
    return counts


def verify_assignments(assignments):
    """Check a split assignment frame against the frozen protocol constants.

    Cross-checks against BOTH the hardcoded EXPECTED_SPLIT_COUNTS and the
    published foundation CSV when available, so a reconstruction is only trusted
    when two independent records agree. Returns (ok, list_of_problem_strings).
    """
    problems = []

    if len(assignments) != PROCESSED_ROW_COUNT:
        problems.append(
            f"row count {len(assignments)} != expected {PROCESSED_ROW_COUNT}"
        )
    if assignments["row_id"].duplicated().any():
        problems.append("duplicate row_id values present")

    actual = split_counts(assignments)
    for split_name, expected in EXPECTED_SPLIT_COUNTS.items():
        if split_name not in actual:
            problems.append(f"split '{split_name}' missing")
        elif actual[split_name] != expected:
            problems.append(
                f"split '{split_name}' counts {actual[split_name]} != expected {expected} "
                "(total, negative, positive)"
            )

    if SPLIT_LABEL_DISTRIBUTION_CSV.exists():
        published = published_split_counts()
        for split_name, expected in published.items():
            if actual.get(split_name) != expected:
                problems.append(
                    f"split '{split_name}' counts {actual.get(split_name)} disagree with "
                    f"published foundation counts {expected}"
                )

    return (not problems), problems
