"""Canonical repository paths and frozen protocol constants.

Every tool imports its paths from here so that a single edit relocates the whole
toolkit, and so that the frozen protocol values (split counts, seed, version
strings) exist in exactly one place rather than being retyped per script.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Data and split artifacts.
DATA_DIR = REPO_ROOT / "data"
DATASET_CSV = DATA_DIR / "dataset.csv"
SPLITS_DIR = REPO_ROOT / "splits"
SPLIT_ASSIGNMENTS_CSV = SPLITS_DIR / "split_assignments.csv"

# Word lexicons used to categorize attributions (RQ2) and to find identity
# terms (RQ4). One entry per line, '#' comments, case-insensitive.
LEXICON_DIR = DATA_DIR / "lexicons"
LEXICON_FILES = {
    "slur": LEXICON_DIR / "slurs.txt",
    "extremist_framing": LEXICON_DIR / "extremist_framing.txt",
    "identity_term": LEXICON_DIR / "identity_terms.txt",
}
# A word matched by several lexicons takes the first category in this order.
CATEGORY_PRECEDENCE = ("slur", "extremist_framing", "identity_term")
DEFAULT_CATEGORY = "topical"
CATEGORIES = CATEGORY_PRECEDENCE + (DEFAULT_CATEGORY,)

NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
DOCS_DIR = REPO_ROOT / "docs"

RESULTS_DIR = REPO_ROOT / "results_summary"
FOUNDATION_DIR = RESULTS_DIR / "foundation"
SPLIT_LABEL_DISTRIBUTION_CSV = FOUNDATION_DIR / "split_label_distribution.csv"
DATASET_MANIFEST_JSON = FOUNDATION_DIR / "dataset_manifest.json"

# Cross-technique tables, one file per research question.
RQ_DIR = RESULTS_DIR / "rq"
RQ2_CATEGORY_SHARES_CSV = RQ_DIR / "rq2_category_shares.csv"
RQ3_RELIANCE_CSV = RQ_DIR / "rq3_reliance_comparison.csv"
RQ3_STABILITY_CSV = RQ_DIR / "rq3_stability.csv"
RQ4_IDENTITY_FPR_CSV = RQ_DIR / "rq4_identity_fpr.csv"

# Per-technique files the tools read or write inside results_summary/<TECHNIQUE>/.
ATTRIBUTIONS_SUBDIR = "attributions"
COEFFICIENTS_FILENAME = "coefficients.csv"
RUN_MANIFEST_FILENAME = "run_manifest.json"
IDENTITY_FPR_FILENAME = "identity_fpr_{split}.json"
IDENTITY_FPR_TERMS_FILENAME = "identity_fpr_terms_{split}.csv"

# Probability artifacts (row_id, split, y_true, y_prob) exported by notebooks.
LOOP_DIR = REPO_ROOT / "research_loop"
PROBS_DIR = LOOP_DIR / "probs"

TOOLS_DIR = REPO_ROOT / "tools"

# ---------------------------------------------------------------------------
# Frozen protocol constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 30
DATASET_VERSION = "extremism_dataset_clean_v1"
SPLIT_VERSION = "split_v1_stratified_70_15_15_seed30"

POSITIVE_LABEL = 1
POSITIVE_CLASS_NAME = "EXTREMIST"
NEGATIVE_CLASS_NAME = "NON_EXTREMIST"

PROCESSED_ROW_COUNT = 2999

# Per split: (total, negative, positive) row counts.
EXPECTED_SPLIT_COUNTS = {
    "train": (2099, 1309, 790),
    "validation": (450, 281, 169),
    "test": (450, 280, 170),
}

SPLIT_NAMES = ("train", "validation", "test")

# Stratified 70/15/15 recipe.
TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

# A technique folder is named <two digits>_<NAME>; anything else under
# results_summary/ (foundation/, rq/) is a shared artifact folder.
TECHNIQUE_DIR_RE = re.compile(r"^\d{2}_")


def technique_dirs(results_dir=None):
    """Return sorted results_summary/ technique folders (NN_NAME), nothing else."""
    results_dir = RESULTS_DIR if results_dir is None else Path(results_dir)
    if not results_dir.exists():
        return []
    return sorted(
        d for d in results_dir.iterdir() if d.is_dir() and TECHNIQUE_DIR_RE.match(d.name)
    )
