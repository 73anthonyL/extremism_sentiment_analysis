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
RQ1_MCNEMAR_PAIRS_CSV = RQ_DIR / "rq1_mcnemar_pairs.csv"
RQ1_ERROR_OVERLAP_CSV = RQ_DIR / "rq1_error_overlap.csv"
RQ1_SHARED_ERRORS_CSV = RQ_DIR / "rq1_shared_errors.csv"
RQ2_CATEGORY_SHARES_CSV = RQ_DIR / "rq2_category_shares.csv"
RQ2_WORD_AGREEMENT_CSV = RQ_DIR / "rq2_word_agreement_pairs.csv"
RQ2_SHARED_TOP_WORDS_CSV = RQ_DIR / "rq2_shared_top_words.csv"
RQ2_TOP_WORD_SUPPORT_CSV = RQ_DIR / "rq2_top_word_support.csv"
RQ3_RELIANCE_CSV = RQ_DIR / "rq3_reliance_comparison.csv"
RQ3_STABILITY_CSV = RQ_DIR / "rq3_stability.csv"
RQ4_IDENTITY_FPR_CSV = RQ_DIR / "rq4_identity_fpr.csv"
RQ4_IDENTITY_COUNTERFACTUAL_CSV = RQ_DIR / "rq4_identity_counterfactual.csv"
# Deletion faithfulness of the per-post SHAP attributions (tools/faithfulness.py).
RQ2_FAITHFULNESS_CSV = RQ_DIR / "rq2_faithfulness.csv"
# Same-post re-aggregation of the per-post attribution files (same_post_stability.py).
RQ2_SAME_POST_SUPPORT_CSV = RQ_DIR / "rq2_same_post_support.csv"
RQ3_SAME_POST_SPLIT_AGREEMENT_CSV = RQ_DIR / "rq3_same_post_split_agreement.csv"
RQ2_SAME_POST_CATEGORY_SHARES_CSV = RQ_DIR / "rq2_same_post_category_shares.csv"
# Rank correlation of RQ2 reliance measures with the RQ4 FPR ratio (rq2_rq4_relation.py).
RQ2_RQ4_RELATION_CSV = RQ_DIR / "rq2_rq4_relation.csv"
RQ2_RQ4_RELATION_INPUTS_CSV = RQ_DIR / "rq2_rq4_relation_inputs.csv"

# Pairs between a technique outside the comparison (best_config.json
# "in_comparison": false) and every technique in it. Kept apart so an audited
# classifier never enters the paper's Holm family or its cross-technique counts.
RQ1_MCNEMAR_PAIRS_OUTSIDE_CSV = RQ_DIR / "rq1_mcnemar_pairs_outside_comparison.csv"
RQ2_WORD_AGREEMENT_OUTSIDE_CSV = RQ_DIR / "rq2_word_agreement_pairs_outside_comparison.csv"

# One text-free report card per technique, written by tools/audit.py report.
REPORT_CARDS_DIR = RQ_DIR / "report_cards"
REPORT_CARD_MD_FILENAME = "{technique}.md"
REPORT_CARD_JSON_FILENAME = "{technique}.json"

# Per-technique files the tools read or write inside results_summary/<TECHNIQUE>/.
ATTRIBUTIONS_SUBDIR = "attributions"
COEFFICIENTS_FILENAME = "coefficients.csv"
RUN_MANIFEST_FILENAME = "run_manifest.json"
IDENTITY_FPR_FILENAME = "identity_fpr_{split}.json"
IDENTITY_FPR_TERMS_FILENAME = "identity_fpr_terms_{split}.csv"
IDENTITY_COUNTERFACTUAL_FILENAME = "identity_counterfactual_{split}.json"
FAITHFULNESS_FILENAME = "faithfulness_{split}.json"

# Probability artifacts (row_id, split, y_true, y_prob) exported by notebooks.
LOOP_DIR = REPO_ROOT / "research_loop"
PROBS_DIR = LOOP_DIR / "probs"

TOOLS_DIR = REPO_ROOT / "tools"

# Untracked downloads: weights, tokenizers and per-post files, one folder per
# technique. Never committed.
EXTERNAL_DIR = REPO_ROOT / "external"

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


def in_comparison(technique, results_dir=None):
    """Whether a technique takes part in the paper's controlled comparison.

    Read from results_summary/<T>/best_config.json. Only an explicit
    `"in_comparison": false` takes a technique out; a missing file or key
    leaves it in, so a fixture or a half-exported folder is never silently
    dropped from a family.
    """
    import json

    results_dir = RESULTS_DIR if results_dir is None else Path(results_dir)
    path = results_dir / technique / "best_config.json"
    if not path.is_file():
        return True
    try:
        with open(path) as handle:
            config = json.load(handle)
    except (OSError, ValueError):
        return True
    return not (isinstance(config, dict) and config.get("in_comparison") is False)


def comparison_technique_dirs(results_dir=None):
    """Technique folders in the comparison: the family every Holm correction,
    cross-technique count and rendered doc table is computed over."""
    return [d for d in technique_dirs(results_dir) if in_comparison(d.name, results_dir)]


def outside_comparison_technique_dirs(results_dir=None):
    """Technique folders with `"in_comparison": false` (an audited classifier
    that must not move the paper's numbers)."""
    return [d for d in technique_dirs(results_dir) if not in_comparison(d.name, results_dir)]

# Sensitivity of RQ4 / WP7 to ambiguous identity terms, and identity-lexicon
# gap candidates for WP3; written by tools/identity_sensitivity.py.
RQ4_IDENTITY_FPR_SENSITIVITY_CSV = RQ_DIR / "rq4_identity_fpr_sensitivity.csv"
RQ4_IDENTITY_COUNTERFACTUAL_SENSITIVITY_CSV = RQ_DIR / "rq4_identity_counterfactual_sensitivity.csv"
RQ2_IDENTITY_LEXICON_GAPS_SENSITIVITY_CSV = RQ_DIR / "rq2_identity_lexicon_gaps_sensitivity.csv"
RQ2_IDENTITY_TOP_K_SENSITIVITY_CSV = RQ_DIR / "rq2_identity_top_k_sensitivity.csv"
