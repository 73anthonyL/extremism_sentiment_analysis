"""Canonical repository paths and frozen protocol constants.

Every tool imports its paths from here so that a single edit relocates the whole
toolkit, and so that the frozen protocol values (split counts, seed, version
strings) exist in exactly one place rather than being retyped per script.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Data and split artifacts.
DATA_DIR = REPO_ROOT / "data"
DATASET_CSV = DATA_DIR / "dataset.csv"
SPLITS_DIR = REPO_ROOT / "splits"
SPLIT_ASSIGNMENTS_CSV = SPLITS_DIR / "split_assignments.csv"

NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
DOCS_DIR = REPO_ROOT / "docs"

RESULTS_DIR = REPO_ROOT / "results_summary"
FOUNDATION_DIR = RESULTS_DIR / "foundation"
SPLIT_LABEL_DISTRIBUTION_CSV = FOUNDATION_DIR / "split_label_distribution.csv"
DATASET_MANIFEST_JSON = FOUNDATION_DIR / "dataset_manifest.json"

# Research-loop state.
LOOP_DIR = REPO_ROOT / "research_loop"
STATE_MD = LOOP_DIR / "STATE.md"
REGISTRY_JSON = LOOP_DIR / "registry.json"
PROBS_DIR = LOOP_DIR / "probs"
PREREG_DIR = LOOP_DIR / "prereg"
DECISIONS_DIR = LOOP_DIR / "decisions"
CYCLES_DIR = LOOP_DIR / "cycles"
TEST_LEDGER_JSONL = LOOP_DIR / "test_ledger.jsonl"
VAL_LOG_JSONL = LOOP_DIR / "val_log.jsonl"

TOOLS_DIR = REPO_ROOT / "tools"
FROZEN_MANIFEST_JSON = TOOLS_DIR / "frozen_manifest.json"

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


def technique_dirs():
    """Return sorted results_summary/ technique folders, excluding foundation/."""
    if not RESULTS_DIR.exists():
        return []
    return sorted(
        d
        for d in RESULTS_DIR.iterdir()
        if d.is_dir() and d.name != "foundation" and not d.name.startswith(".")
    )
