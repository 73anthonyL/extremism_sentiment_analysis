"""Shared helpers for the experiment notebooks. One file, no repo imports.

Every notebook under notebooks/ loads this module in its bootstrap cell and
calls it as `nk`. The kit owns everything that must be identical across
techniques so their outputs are comparable:

* loading the frozen dataset and split, and asserting the frozen counts;
* the metric function (a pinned copy of tools/metrics_core.py);
* threshold selection on validation only, and the one test evaluation;
* the probability artifact, the word-level SHAP attribution artifact, the
  coefficient file, and the result folder, each written under its contract;
* the output tree, which mirrors the repository so a run's folders copy
  straight in.

It deliberately imports nothing from this repository. On Kaggle the repo is a
read-only input dataset, the tools' module-level paths would point at it, and
a notebook must not depend on which other tool modules happen to import. The
contracts are instead pinned by tools/tests/test_notebook_kit.py: the metric
functions are compared character for character with tools/metrics_core.py,
and every artifact the kit writes is loaded back through the real loaders
(tools/probs_artifact.py, tools/attributions.py,
tools/validate_results_folder.py).

No function here prints, returns for display, or writes into the committed
tree any dataset text, row id, or text hash. Per-post files go under
external/, which is uploaded to a Kaggle Dataset and never committed.

Attach this file to a Kaggle kernel as an input (a dataset built from the
repository, or from tools/ alone). The bootstrap cell finds it under
/kaggle/input.
"""

import hashlib
import importlib.util
import json
import os
import platform
import random
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

KIT_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Frozen protocol constants (identical to tools/repo_paths.py; pinned by test)
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

ROLES = ("foundation", "classical", "transformer", "ensemble")
FOUNDATION_TECHNIQUE = "00_create_dataset_and_splits"
TECHNIQUE_RE = re.compile(r"^\d{2}_[A-Z0-9-]+(_[A-Z0-9-]+)*$")
CLAIM_TOKENS = ("BEST", "FINAL", "WINNER", "CHAMPION")

# ---------------------------------------------------------------------------
# Artifact contracts
# ---------------------------------------------------------------------------
PROBS_REQUIRED_COLUMNS = ("row_id", "split", "y_true", "y_prob")
# Superset of tools/probs_artifact.py::FORBIDDEN_COLUMNS.
PROBS_FORBIDDEN_COLUMNS = (
    "text",
    "text_hash",
    "Original_Message",
    "text_preview",
    "message",
    "content",
    "raw_text",
    "model_input_text",
)

# Columns that hold dataset text. Nothing under results_summary/ or
# research_loop/ may carry one, in any notebook.
TEXT_BEARING_COLUMNS = (
    "text",
    "Original_Message",
    "text_preview",
    "message",
    "content",
    "raw_text",
    "model_input_text",
)
# Row-level identity. A technique's result folder carries neither; the
# probability artifact is keyed by row_id by contract, and the foundation's
# duplicate report is keyed by both.
ROW_IDENTITY_COLUMNS = ("row_id", "text_hash")

ATTR_REQUIRED_COLUMNS = ("word", "mean_abs_attribution", "mean_attribution", "support")
# Same tuple as tools/attributions.py::FORBIDDEN_COLUMNS.
ATTR_FORBIDDEN_COLUMNS = (
    "row_id",
    "text",
    "text_hash",
    "Original_Message",
    "message",
    "post",
    "text_preview",
    "position",
    "token_index",
    "y_true",
    "y_prob",
)
ATTR_REQUIRED_META_FIELDS = (
    "technique",
    "run_id",
    "split",
    "explainer",
    "background_size",
    "seed",
    "aggregation",
    "n_posts_explained",
    "member",
)
WORD_AGGREGATION = "word"
MAX_WORD_LENGTH = 40
ATTRIBUTIONS_SUBDIR = "attributions"
COEFFICIENTS_FILENAME = "coefficients.csv"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9-]+(_[A-Za-z0-9-]+)*$")

REQUIRED_METRIC_FIELDS = (
    "threshold",
    "support",
    "positive_support",
    "negative_support",
    "accuracy",
    "balanced_accuracy",
    "positive_precision",
    "positive_recall",
    "positive_f1",
    "f1_macro",
    "f1_weighted",
    "roc_auc",
    "pr_auc",
    "tn",
    "fp",
    "fn",
    "tp",
    "false_positive_rate",
    "false_negative_rate",
)

# 0.05 to 0.95 in steps of 0.005: the grid every technique's threshold is
# selected from, so thresholds are comparable across notebooks.
THRESHOLD_GRID = tuple(round(float(x), 3) for x in np.linspace(0.05, 0.95, 181))
THRESHOLD_STRATEGY = "grid_search_on_validation"

# One attribution method for every model family: SHAP's partition explainer
# over a word-level text masker, applied to a black-box function from a list
# of strings to P(EXTREMIST). The masker tokenizes at word level, so no
# subword aggregation is needed and every family lands in one vocabulary.
MASKER_PATTERN = r"\W+"
MASK_TOKEN = "..."
EXPLAINER_ID = "shap.PartitionExplainer(maskers.Text(r'\\W+'))"

REQUIRED_CONFIG_KEYS = (
    "project_name",
    "technique_name",
    "role",
    "dataset_version",
    "split_version",
    "random_seed",
)
REQUIRED_MODEL_CONFIG_KEYS = (
    "in_comparison",
    "model_family",
    "feature_family",
    "threshold",
    "explainer",
    "error_analysis",
)
REQUIRED_EXPLAINER_KEYS = ("max_evals", "batch_size", "seed", "n_posts_per_split", "per_member_runs")

# external/<technique>/<subfolder> -> (run_manifest asset kind, contains_text)
EXTERNAL_KINDS = {
    "weights": ("weights", False),
    "tokenizer": ("tokenizer", False),
    "training_logs": ("training_log", False),
    "predictions": ("predictions_with_text", True),
    "attributions_local": ("local_attributions", True),
    "review": ("error_review_queue", True),
    "probs_members": ("other", False),
    "auxiliary": ("other", False),
    "plots": ("other", False),
}
EXTERNAL_ASSETS_FILENAME = "external_assets.json"


class KitError(RuntimeError):
    """Base class for every refusal raised by the kit."""


class ConfigError(KitError):
    """CONFIG violates a notebook convention."""


class InputNotFoundError(KitError):
    """A required input file is not attached to the kernel."""


class SplitIntegrityError(KitError):
    """The loaded data does not match the frozen split."""


class TestSplitError(KitError):
    """The test split was about to be used for selection or evaluated twice."""

    __test__ = False  # not a pytest test class


class ArtifactContractError(KitError):
    """An artifact about to be written violates its committed contract."""


# ---------------------------------------------------------------------------
# Run context
# ---------------------------------------------------------------------------
@dataclass
class RunContext:
    """Where one notebook run reads from and writes to."""

    technique: str
    run_id: str
    role: str
    config: dict
    working_root: Path
    results_dir: Path
    probs_dir: Path
    external_dir: Path
    foundation_dir: Path
    input_roots: tuple
    test_evaluated: bool = False
    written: list = field(default_factory=list)

    def external(self, name):
        """A subfolder of external/<technique>/, created on first use."""
        path = self.external_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path


def _now_utc():
    return datetime.now(timezone.utc).replace(microsecond=0)


def sha256_file(path, n_chars=None):
    """SHA-256 of a file's bytes; `n_chars` truncates for display."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    return value if n_chars is None else value[:n_chars]


def _json_safe(value):
    """Convert numpy scalars and NaN so json.dump writes valid, portable JSON."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, payload):
    """Write JSON with sorted keys; NaN becomes null."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
    return path


def seed_everything(seed):
    """Seed Python, NumPy and (when installed) PyTorch, with deterministic cuDNN."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    # Required by cuBLAS for deterministic matrix products on GPU.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def _any_true_text_preview(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "include_text_preview" and value:
                return True
            if _any_true_text_preview(value):
                return True
    elif isinstance(node, (list, tuple)):
        return any(_any_true_text_preview(item) for item in node)
    return False


def validate_config(config):
    """Refuse a CONFIG that breaks a notebook convention. Returns the role."""
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in config]
    if missing:
        raise ConfigError(f"CONFIG is missing required keys {missing}")

    role = config["role"]
    if role not in ROLES:
        raise ConfigError(f"CONFIG['role'] is '{role}', must be one of {ROLES}")

    technique = config["technique_name"]
    if role == "foundation":
        if technique != FOUNDATION_TECHNIQUE:
            raise ConfigError(
                f"the foundation notebook's technique_name must be '{FOUNDATION_TECHNIQUE}'"
            )
        if config.get("overwrite_existing_split") is not False:
            raise ConfigError(
                "CONFIG['overwrite_existing_split'] must be False: the split assignment is "
                "frozen and is never regenerated over an existing one"
            )
    else:
        if not TECHNIQUE_RE.match(str(technique)):
            raise ConfigError(
                f"technique_name '{technique}' must be the notebook filename stem, "
                "numeric prefix included (e.g. 01_LOG-REG_TF-IDF)"
            )
        claims = [t for t in CLAIM_TOKENS if t in re.split(r"[-_]", technique)]
        if claims:
            raise ConfigError(f"technique_name encodes a claim {claims}; name it for what it is")

    if config["split_version"] != SPLIT_VERSION:
        raise ConfigError(
            f"split_version is '{config['split_version']}', must be the full frozen string "
            f"'{SPLIT_VERSION}'"
        )
    if config["dataset_version"] != DATASET_VERSION:
        raise ConfigError(
            f"dataset_version is '{config['dataset_version']}', must be '{DATASET_VERSION}'"
        )
    if _any_true_text_preview(config):
        raise ConfigError(
            "include_text_preview must stay False: cell outputs are committed, so no "
            "cell may show dataset text"
        )

    if role != "foundation":
        missing = [k for k in REQUIRED_MODEL_CONFIG_KEYS if k not in config]
        if missing:
            raise ConfigError(f"CONFIG is missing required keys {missing}")
        if "metric" not in config["threshold"]:
            raise ConfigError("CONFIG['threshold'] must declare 'metric'")
        missing = [k for k in REQUIRED_EXPLAINER_KEYS if k not in config["explainer"]]
        if missing:
            raise ConfigError(f"CONFIG['explainer'] is missing keys {missing}")
        if config["error_analysis"].get("include_text_preview") is not False:
            raise ConfigError("CONFIG['error_analysis']['include_text_preview'] must be False")
    return role


def describe_environment():
    """Library versions and accelerator, for the run record. No data."""
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    try:
        import sklearn

        info["sklearn"] = sklearn.__version__
    except ImportError:
        pass
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        import shap

        info["shap"] = shap.__version__
    except ImportError:
        pass
    return info


def bootstrap(config, working_root=None, input_roots=None):
    """Validate CONFIG, seed, create the output tree, and return the RunContext.

    The output tree mirrors the repository:

        results_summary/<TECHNIQUE>/     copy into the repo
        research_loop/probs/             copy into the repo
        external/<TECHNIQUE>/            upload to a Kaggle Dataset, never commit
    """
    role = validate_config(config)
    technique = config["technique_name"]

    if working_root is None:
        kaggle_working = Path("/kaggle/working")
        working_root = kaggle_working if kaggle_working.exists() else Path.cwd() / "kit_output"
    working_root = Path(working_root)

    if input_roots is None:
        kaggle_input = Path("/kaggle/input")
        input_roots = (kaggle_input,) if kaggle_input.exists() else (Path.cwd(),)
    input_roots = tuple(Path(p) for p in input_roots)

    results_name = "foundation" if role == "foundation" else technique
    ctx = RunContext(
        technique=technique,
        run_id=f"{technique}_{_now_utc().strftime('%Y%m%dT%H%M%SZ')}",
        role=role,
        config=config,
        working_root=working_root,
        results_dir=working_root / "results_summary" / results_name,
        probs_dir=working_root / "research_loop" / "probs",
        external_dir=working_root / "external" / technique,
        foundation_dir=working_root / "research_foundation",
        input_roots=input_roots,
    )
    ctx.results_dir.mkdir(parents=True, exist_ok=True)
    if role == "foundation":
        ctx.foundation_dir.mkdir(parents=True, exist_ok=True)
    else:
        ctx.probs_dir.mkdir(parents=True, exist_ok=True)
        ctx.external_dir.mkdir(parents=True, exist_ok=True)

    if role != "foundation" and importlib.util.find_spec("shap") is None:
        raise KitError(
            "the shap package is not installed in this kernel. Every model notebook ends "
            "with SHAP attribution runs, so this is checked before any training starts. "
            "Enable internet and add a `%pip install shap` line to the bootstrap cell."
        )

    seed_everything(config["random_seed"])

    kit_hash = sha256_file(Path(__file__).resolve(), n_chars=16)
    print(f"notebook_kit {KIT_VERSION} (sha256 {kit_hash})")
    print(f"technique: {technique}   role: {role}   run_id: {ctx.run_id}")
    print(f"working root: {ctx.working_root}")
    for key, value in describe_environment().items():
        print(f"  {key}: {value}")
    return ctx


# ---------------------------------------------------------------------------
# Inputs and the frozen split
# ---------------------------------------------------------------------------
def _find_all(ctx, name):
    matches = []
    for root in ctx.input_roots:
        if root.is_file():
            if root.name == name:
                matches.append(root)
        elif root.exists():
            matches.extend(p for p in root.rglob(name) if p.is_file())
    return sorted(set(matches), key=lambda p: (len(p.parts), str(p)))


def find_input(ctx, name, prefer=None):
    """Locate one attached input file by name.

    `prefer` narrows to paths with that folder name in them (the notebook-00
    output lives in a folder called research_foundation). Several matches are
    fine when they are byte-identical; different files under one name are
    refused rather than guessed between.
    """
    matches = _find_all(ctx, name)
    if not matches:
        roots = ", ".join(str(r) for r in ctx.input_roots)
        raise InputNotFoundError(
            f"no file named '{name}' under {roots}. Attach the input that provides it."
        )
    if prefer:
        preferred = [p for p in matches if prefer in p.parts]
        if preferred:
            matches = preferred
    if len({sha256_file(p) for p in matches}) > 1:
        listing = "\n  ".join(str(p) for p in matches)
        raise KitError(
            f"{len(matches)} different files named '{name}' are attached:\n  {listing}\n"
            "Detach the stale one."
        )
    return matches[0]


def split_counts(frame, split_col="split", label_col="label"):
    """{split: (total, negative, positive)} for a frame."""
    counts = {}
    for split_name, group in frame.groupby(split_col):
        positives = int((group[label_col] == POSITIVE_LABEL).sum())
        counts[str(split_name)] = (int(len(group)), int(len(group) - positives), positives)
    return counts


def assert_split_counts(frame, split_col="split", label_col="label"):
    """Refuse any frame whose per-split class counts differ from the frozen split."""
    actual = split_counts(frame, split_col, label_col)
    problems = []
    for split_name, expected in EXPECTED_SPLIT_COUNTS.items():
        if split_name not in actual:
            problems.append(f"split '{split_name}' is missing")
        elif actual[split_name] != expected:
            problems.append(
                f"split '{split_name}' has counts {actual[split_name]}, frozen protocol "
                f"requires {expected} (total, negative, positive)"
            )
    extra = sorted(set(actual) - set(EXPECTED_SPLIT_COUNTS))
    if extra:
        problems.append(f"unknown split value(s) {extra}")
    if problems:
        raise SplitIntegrityError("; ".join(problems))
    return actual


def compare_with_committed_split(ctx, assignments, exclude=()):
    """Assert `assignments` equals every other attached split_assignments.csv.

    When the repository is attached to the kernel, its committed mirror
    (splits/split_assignments.csv) is found here, and a foundation output that
    disagrees with it is refused. Returns how many mirrors were compared.
    """
    columns = ["row_id", "label", "text_hash", "split"]
    reference = assignments[columns].sort_values("row_id").reset_index(drop=True)
    excluded = {Path(p).resolve() for p in exclude}
    compared = 0
    for path in _find_all(ctx, "split_assignments.csv"):
        if path.resolve() in excluded:
            continue
        other = pd.read_csv(path)
        if not set(columns).issubset(other.columns):
            raise SplitIntegrityError(f"{path}: not a split assignment file (columns differ)")
        other = other[columns].sort_values("row_id").reset_index(drop=True)
        if len(other) != len(reference) or not other.equals(reference):
            differing = (
                int((other.values != reference.values).any(axis=1).sum())
                if len(other) == len(reference)
                else abs(len(other) - len(reference))
            )
            raise SplitIntegrityError(
                f"the split assignment differs from the attached mirror at {path} "
                f"({differing} row(s)). The split is frozen; do not regenerate it."
            )
        compared += 1
    return compared


def load_foundation(ctx, prefer="research_foundation"):
    """Load the processed dataset joined to the frozen split, fully cross-checked.

    Returns (frame, info). `frame` has row_id, text, label, split, text_hash,
    sorted by row_id. `info` records input paths, file hashes and counts.
    """
    processed_path = find_input(ctx, "processed_dataset.csv", prefer=prefer)
    split_path = find_input(ctx, "split_assignments.csv", prefer=prefer)
    manifest_path = find_input(ctx, "dataset_manifest.json", prefer=prefer)

    processed = pd.read_csv(processed_path)
    assignments = pd.read_csv(split_path)
    with open(manifest_path) as handle:
        manifest = json.load(handle)

    for label, frame, required in (
        ("processed_dataset.csv", processed, ("row_id", "text", "label", "text_hash")),
        ("split_assignments.csv", assignments, ("row_id", "label", "text_hash", "split")),
    ):
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise SplitIntegrityError(f"{label}: missing columns {missing}")
        if frame["row_id"].duplicated().any():
            raise SplitIntegrityError(f"{label}: duplicate row ids")

    if manifest.get("dataset_version") != DATASET_VERSION:
        raise SplitIntegrityError(
            f"dataset_manifest.json declares dataset_version "
            f"'{manifest.get('dataset_version')}', expected '{DATASET_VERSION}'. "
            "Attach the output of the current notebook 00."
        )
    if manifest.get("split_version") != SPLIT_VERSION:
        raise SplitIntegrityError(
            f"dataset_manifest.json declares split_version '{manifest.get('split_version')}', "
            f"expected '{SPLIT_VERSION}'"
        )
    if len(processed) != PROCESSED_ROW_COUNT or manifest.get("processed_rows") != len(processed):
        raise SplitIntegrityError(
            f"processed dataset has {len(processed)} rows, manifest says "
            f"{manifest.get('processed_rows')}, frozen protocol requires {PROCESSED_ROW_COUNT}"
        )

    merged = processed[["row_id", "text", "label", "text_hash"]].merge(
        assignments[["row_id", "label", "text_hash", "split"]],
        on="row_id",
        how="outer",
        suffixes=("", "_split"),
        validate="one_to_one",
        indicator=True,
    )
    unmatched = int((merged["_merge"] != "both").sum())
    if unmatched:
        raise SplitIntegrityError(
            f"{unmatched} row(s) appear in only one of the dataset and the split assignment"
        )
    label_mismatch = int((merged["label"] != merged["label_split"]).sum())
    hash_mismatch = int((merged["text_hash"] != merged["text_hash_split"]).sum())
    if label_mismatch or hash_mismatch:
        raise SplitIntegrityError(
            f"dataset and split assignment disagree on {label_mismatch} label(s) and "
            f"{hash_mismatch} text hash(es)"
        )
    if not set(merged["label"].unique()).issubset({0, 1}):
        raise SplitIntegrityError("labels are not binary")

    frame = (
        merged[["row_id", "text", "label", "split", "text_hash"]]
        .sort_values("row_id")
        .reset_index(drop=True)
    )
    frame["label"] = frame["label"].astype(int)
    frame["text"] = frame["text"].astype(str)
    counts = assert_split_counts(frame)
    mirrors = compare_with_committed_split(ctx, assignments, exclude=(split_path,))

    info = {
        "paths": {
            "processed_dataset": str(processed_path),
            "split_assignments": str(split_path),
            "dataset_manifest": str(manifest_path),
        },
        "file_sha256_16": {
            "processed_dataset": sha256_file(processed_path, 16),
            "split_assignments": sha256_file(split_path, 16),
            "dataset_manifest": sha256_file(manifest_path, 16),
        },
        "split_counts_total_negative_positive": counts,
        "committed_mirrors_compared": mirrors,
        "dataset_version": manifest["dataset_version"],
        "split_version": manifest["split_version"],
    }
    print("Frozen split verified (total, negative, positive):")
    for split_name in SPLIT_NAMES:
        print(f"  {split_name:<10} {counts[split_name]}")
    print(f"Committed split mirrors compared: {mirrors}")
    return frame, info


def split_frames(frame):
    """(train_df, val_df, test_df), each sorted by row id with a clean index."""
    parts = []
    for split_name in SPLIT_NAMES:
        part = frame[frame["split"] == split_name].sort_values("row_id").reset_index(drop=True)
        parts.append(part)
    if sum(len(p) for p in parts) != len(frame):
        raise SplitIntegrityError("the three splits do not partition the frame")
    return tuple(parts)


# ---------------------------------------------------------------------------
# Metrics. The three functions below are character-for-character copies of
# tools/metrics_core.py and are pinned to it by tools/tests/test_notebook_kit.py.
# Do not edit them here.
# ---------------------------------------------------------------------------
def safe_roc_auc(y_true, y_prob):
    """ROC-AUC that returns NaN rather than raising on a single-class input."""
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, y_prob)
    except Exception:
        return np.nan


def safe_pr_auc(y_true, y_prob):
    """Average precision that returns NaN rather than raising on a single-class input."""
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return average_precision_score(y_true, y_prob)
    except Exception:
        return np.nan


def compute_binary_metrics(y_true, y_prob, threshold, positive_label=POSITIVE_LABEL):
    """Compute the full standard metric set for one split at one threshold.

    Verbatim from notebook 07 cell 6. Returns the superset of fields required by
    docs/RESULTS_SCHEMA.md; predictions are `y_prob >= threshold`.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()

    metrics = {
        "threshold": float(threshold),
        "support": int(len(y_true)),
        "positive_support": int((y_true == positive_label).sum()),
        "negative_support": int((y_true != positive_label).sum()),
        "positive_rate": float((y_true == positive_label).mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "positive_precision": float(
            precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
        ),
        "positive_recall": float(
            recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
        ),
        "positive_f1": float(f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
        "roc_auc": float(safe_roc_auc(y_true, y_prob)),
        "pr_auc": float(safe_pr_auc(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else np.nan,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else np.nan,
    }
    return metrics


def _checked_probabilities(frame, y_prob, label):
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)
    if len(y_prob) != len(frame):
        raise ArtifactContractError(
            f"{label}: {len(y_prob)} probabilities for {len(frame)} rows"
        )
    if not np.isfinite(y_prob).all():
        raise ArtifactContractError(f"{label}: non-finite probabilities")
    if (y_prob < 0).any() or (y_prob > 1).any():
        raise ArtifactContractError(f"{label}: probabilities outside [0, 1]")
    return y_prob


def _single_split(frame, label):
    values = sorted(str(v) for v in frame["split"].unique())
    if len(values) != 1:
        raise KitError(f"{label}: expected rows from one split, found {values}")
    return values[0]


def select_threshold(validation_frame, y_prob, metric="accuracy", grid=None):
    """Select the decision threshold on the validation split, and only there.

    The frame, not just an array, is required so the split can be checked: a
    threshold fitted on test rows is the one mistake this function exists to
    make impossible. Ties break on positive recall, positive precision, then
    balanced accuracy, as in tools/metrics_core.py::select_threshold.

    Returns (threshold, sweep) where `sweep` is ordered by threshold and marks
    the selected row.
    """
    split = _single_split(validation_frame, "select_threshold")
    if split != "validation":
        raise TestSplitError(
            f"threshold selection attempted on the '{split}' split; only 'validation' "
            "may be used to select a threshold"
        )
    y_prob = _checked_probabilities(validation_frame, y_prob, "select_threshold")
    grid = THRESHOLD_GRID if grid is None else tuple(float(t) for t in grid)
    y_true = validation_frame["label"].to_numpy()

    sweep = pd.DataFrame([compute_binary_metrics(y_true, y_prob, t) for t in grid])
    if metric not in sweep.columns:
        raise ConfigError(f"threshold metric '{metric}' is not a computed metric")
    ranked = sweep.sort_values(
        by=[metric, "positive_recall", "positive_precision", "balanced_accuracy"],
        ascending=[False, False, False, False],
    )
    threshold = float(ranked.iloc[0]["threshold"])
    sweep = sweep.sort_values("threshold").reset_index(drop=True)
    sweep["selected"] = sweep["threshold"].eq(threshold)
    return threshold, sweep


def evaluate(ctx, frame, y_prob, threshold, split):
    """Compute the standard metrics for one whole split at one threshold.

    The test split can be evaluated once per run; a second call raises. The
    frame must be the complete split, so the frozen class counts are checked
    on every evaluation.
    """
    if split not in SPLIT_NAMES:
        raise KitError(f"unknown split '{split}'")
    actual_split = _single_split(frame, "evaluate")
    if actual_split != split:
        raise KitError(f"evaluate(split='{split}') was given rows from '{actual_split}'")
    y_prob = _checked_probabilities(frame, y_prob, f"evaluate[{split}]")
    counts = split_counts(frame)[split]
    if counts != EXPECTED_SPLIT_COUNTS[split]:
        raise SplitIntegrityError(
            f"evaluate[{split}]: counts {counts} are not the frozen "
            f"{EXPECTED_SPLIT_COUNTS[split]}; evaluate the whole split"
        )
    if split == "test":
        if ctx.test_evaluated:
            raise TestSplitError(
                f"the test split was already evaluated for {ctx.technique}. "
                "Test is evaluated once per technique."
            )
        ctx.test_evaluated = True

    metrics = compute_binary_metrics(frame["label"].to_numpy(), y_prob, threshold)
    metrics["technique"] = ctx.technique
    metrics["split"] = split
    return metrics


def ablation_row(ctx, config_id, hyperparameters, validation_metrics, notes=None):
    """One standard row of ablation_results.csv, from validation metrics only."""
    if validation_metrics.get("split") != "validation":
        raise TestSplitError("ablation rows are built from validation metrics only")
    return {
        "technique": ctx.technique,
        "config_id": config_id,
        "model_family": ctx.config["model_family"],
        "feature_family": ctx.config["feature_family"],
        "hyperparameter_summary": json.dumps(_json_safe(hyperparameters), sort_keys=True),
        "threshold": validation_metrics["threshold"],
        "validation_accuracy": validation_metrics["accuracy"],
        "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
        "validation_positive_f1": validation_metrics["positive_f1"],
        "validation_f1_macro": validation_metrics["f1_macro"],
        "validation_roc_auc": validation_metrics["roc_auc"],
        "validation_pr_auc": validation_metrics["pr_auc"],
        "notes": notes,
    }


def select_best_config(ablation_results, metric="accuracy"):
    """The winning ablation row: best validation metric, ties on ROC-AUC then config id."""
    column = f"validation_{metric}"
    if column not in ablation_results.columns:
        raise ConfigError(f"ablation results have no column '{column}'")
    ranked = ablation_results.sort_values(
        by=[column, "validation_roc_auc", "config_id"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    return ranked.iloc[0]


# ---------------------------------------------------------------------------
# Probability artifacts
# ---------------------------------------------------------------------------
def _probability_frame(frame, y_prob, split):
    missing = [c for c in ("row_id", "label", "split") if c not in frame.columns]
    if missing:
        raise ArtifactContractError(f"probability export needs columns {missing}")
    actual_split = _single_split(frame, f"probabilities[{split}]")
    if actual_split != split:
        raise ArtifactContractError(f"expected '{split}' rows, got '{actual_split}'")
    y_prob = _checked_probabilities(frame, y_prob, f"probabilities[{split}]")
    out = pd.DataFrame(
        {
            "row_id": frame["row_id"].astype(str).to_numpy(),
            "split": split,
            "y_true": frame["label"].astype(int).to_numpy(),
            "y_prob": y_prob,
        }
    )
    if out["row_id"].duplicated().any():
        raise ArtifactContractError(f"probabilities[{split}]: duplicate row ids")
    if set(out["y_true"].unique()) - {0, 1}:
        raise ArtifactContractError(f"probabilities[{split}]: non-binary labels")
    positives = int((out["y_true"] == POSITIVE_LABEL).sum())
    actual = (len(out), len(out) - positives, positives)
    if actual != EXPECTED_SPLIT_COUNTS[split]:
        raise ArtifactContractError(
            f"probabilities[{split}]: counts {actual} are not the frozen "
            f"{EXPECTED_SPLIT_COUNTS[split]}"
        )
    if list(out.columns) != list(PROBS_REQUIRED_COLUMNS):
        raise ArtifactContractError("probability artifact columns are not the contract's four")
    return out.sort_values("row_id").reset_index(drop=True)


REQUIRED_PROBS_META_KEYS = (
    "model_family",
    "feature_family",
    "hyperparameters",
    "selected_threshold",
    "threshold_strategy",
    "threshold_metric",
)


def export_probabilities(ctx, val_df, val_prob, test_df, test_prob, meta):
    """Write the committed probability artifact for both splits, plus its sidecar.

    Exactly row_id, split, y_true, y_prob. Any other column is a leak, not a
    formatting problem. Returns {"validation": path, "test": path, "meta": path}.
    """
    missing = [k for k in REQUIRED_PROBS_META_KEYS if k not in meta]
    if missing:
        raise ArtifactContractError(f"probability meta is missing keys {missing}")

    paths = {}
    for split, frame, prob in (("validation", val_df, val_prob), ("test", test_df, test_prob)):
        artifact = _probability_frame(frame, prob, split)
        path = ctx.probs_dir / f"{ctx.technique}__{split}.csv"
        artifact.to_csv(path, index=False)
        paths[split] = path

    record = dict(meta)
    record.update(
        {
            "technique": ctx.technique,
            "run_id": ctx.run_id,
            "dataset_version": DATASET_VERSION,
            "split_version": SPLIT_VERSION,
            "random_seed": ctx.config["random_seed"],
            "threshold_selected_on_split": "validation",
            "test_set_used_for_selection": False,
            "contains_text": False,
            "schema": list(PROBS_REQUIRED_COLUMNS),
            "row_counts": {s: EXPECTED_SPLIT_COUNTS[s][0] for s in ("validation", "test")},
            "artifact_sha256": {s: sha256_file(paths[s], 16) for s in ("validation", "test")},
            "kit_version": KIT_VERSION,
        }
    )
    paths["meta"] = write_json(ctx.probs_dir / f"{ctx.technique}__meta.json", record)
    ctx.written.extend(paths.values())
    print(f"Probability artifacts written: {[p.name for p in paths.values()]}")
    return paths


def export_member_probabilities(ctx, member, val_df, val_prob, test_df, test_prob):
    """Write one ensemble member's probabilities to external/ (not committed).

    Same four-column contract as the committed artifact, so the files can be
    used for later CPU-side ensembling studies without a rerun.
    """
    if not RUN_ID_RE.match(str(member)):
        raise ArtifactContractError(f"member id '{member}' is not a simple identifier")
    folder = ctx.external("probs_members")
    paths = {}
    for split, frame, prob in (("validation", val_df, val_prob), ("test", test_df, test_prob)):
        artifact = _probability_frame(frame, prob, split)
        path = folder / f"{ctx.technique}__member-{member}__{split}.csv"
        artifact.to_csv(path, index=False)
        paths[split] = path
    return paths


# ---------------------------------------------------------------------------
# Word-level SHAP attributions
# ---------------------------------------------------------------------------
@dataclass
class AttributionBundle:
    """In-memory result of one explanation run. Nothing here is written yet."""

    row_ids: list
    split: str
    tokens_per_post: list
    values_per_post: list
    seed: int
    max_evals: int
    n_posts: int


def sample_for_explanation(frame, n_posts, seed):
    """A label-stratified sample of one split for explanation, or the whole split.

    `n_posts=None` (or at least the split size) returns every row. The sample
    depends only on the row ids, the labels and the seed, so every member of an
    ensemble is explained on the same posts.
    """
    _single_split(frame, "sample_for_explanation")
    ordered = frame.sort_values("row_id").reset_index(drop=True)
    if n_posts is None or n_posts >= len(ordered):
        return ordered
    if n_posts < 2:
        raise ConfigError("n_posts_per_split must be at least 2")
    rng = np.random.default_rng(seed)
    chosen = []
    labels = sorted(ordered["label"].unique())
    remaining = n_posts
    for index, label in enumerate(labels):
        group = ordered.index[ordered["label"] == label].to_numpy()
        if index == len(labels) - 1:
            take = remaining
        else:
            take = int(round(n_posts * len(group) / len(ordered)))
        take = max(1, min(take, len(group)))
        remaining -= take
        chosen.extend(rng.choice(group, size=take, replace=False).tolist())
    return ordered.loc[sorted(chosen)].reset_index(drop=True)


def _word_tokens(text):
    return [piece for piece in re.split(MASKER_PATTERN, str(text)) if piece]


def explain(ctx, predict_fn, frame, seed=None, max_evals=None, batch_size=None):
    """Explain a black-box text classifier on the rows of one split.

    `predict_fn(list_of_str) -> array of P(EXTREMIST)`, one value per string.
    That signature is the whole interface: a scikit-learn pipeline, an
    embedding model, a fine-tuned transformer and a pooled ensemble are all
    explained by the same call, which is what makes their attributions
    comparable.

    Returns an AttributionBundle held in memory. Pass it to
    export_attribution_run to write the committed word-level aggregate.
    """
    settings = ctx.config["explainer"]
    seed = settings["seed"] if seed is None else seed
    max_evals = settings["max_evals"] if max_evals is None else max_evals
    batch_size = settings["batch_size"] if batch_size is None else batch_size
    split = _single_split(frame, "explain")
    texts = [str(t) for t in frame["text"].tolist()]
    if len(texts) < 2:
        raise ArtifactContractError("explain needs at least two posts")

    def predict(batch):
        values = np.asarray(predict_fn([str(item) for item in batch]), dtype=float)
        return values.reshape(-1)

    probe = predict(texts[:2])
    if probe.shape != (2,) or not np.isfinite(probe).all():
        raise ArtifactContractError(
            "predict_fn must return one finite probability per input string "
            f"(got shape {probe.shape} for 2 inputs)"
        )
    if (probe < 0).any() or (probe > 1).any():
        raise ArtifactContractError("predict_fn must return probabilities in [0, 1]")

    import shap

    masker = shap.maskers.Text(MASKER_PATTERN, mask_token=MASK_TOKEN)
    explainer = shap.Explainer(predict, masker, algorithm="partition", seed=seed)

    # The partition explainer needs at least two tokens to build a hierarchy.
    # A one-word post is attributed directly: its single word carries the whole
    # difference between the post and the fully masked input.
    multi_index = [i for i, text in enumerate(texts) if len(_word_tokens(text)) >= 2]
    single_index = [i for i, text in enumerate(texts) if len(_word_tokens(text)) == 1]

    tokens_per_post = [[] for _ in texts]
    values_per_post = [np.zeros(0) for _ in texts]

    if multi_index:
        explanation = explainer(
            [texts[i] for i in multi_index],
            max_evals=max_evals,
            batch_size=batch_size,
            silent=True,
        )
        for position, post_index in enumerate(multi_index):
            tokens_per_post[post_index] = [str(t) for t in explanation.data[position]]
            values_per_post[post_index] = np.asarray(
                explanation.values[position], dtype=float
            ).reshape(-1)

    if single_index:
        baseline = float(predict([MASK_TOKEN])[0])
        outputs = predict([texts[i] for i in single_index])
        for position, post_index in enumerate(single_index):
            tokens_per_post[post_index] = _word_tokens(texts[post_index])
            values_per_post[post_index] = np.asarray([outputs[position] - baseline])

    explained = len(multi_index) + len(single_index)
    print(
        f"Explained {explained} of {len(texts)} {split} posts "
        f"(max_evals={max_evals}, seed={seed}); "
        f"{len(texts) - explained} had no word tokens."
    )
    return AttributionBundle(
        row_ids=frame["row_id"].astype(str).tolist(),
        split=split,
        tokens_per_post=tokens_per_post,
        values_per_post=values_per_post,
        seed=int(seed),
        max_evals=int(max_evals),
        n_posts=len(texts),
    )


def normalize_word(token):
    """Lowercase a masker segment and strip its surrounding non-word characters."""
    return re.sub(r"^\W+|\W+$", "", str(token)).lower()


def aggregate_word_attributions_with_stats(tokens_per_post, values_per_post):
    """Aggregate per-token attributions to the committed word-level table.

    Within a post, repeated occurrences of a word are summed, so a word's
    per-post attribution is its total contribution to that post. Across posts,
    both means are taken over the posts that contain the word, never over all
    posts: for a linear model the attribution of an absent word has the
    opposite sign to its coefficient, so an all-post mean would cancel.

    Returns (frame, stats): the frame has word, mean_abs_attribution,
    mean_attribution, support; stats counts the tokens that were dropped.
    """
    if len(tokens_per_post) != len(values_per_post):
        raise ArtifactContractError("tokens and values cover different numbers of posts")
    signed_sum = {}
    absolute_sum = {}
    support = {}
    dropped_empty = 0
    dropped_long = 0
    for tokens, values in zip(tokens_per_post, values_per_post):
        values = np.asarray(values, dtype=float).reshape(-1)
        if len(tokens) != len(values):
            raise ArtifactContractError("a post has a different number of tokens and values")
        per_post = {}
        for token, value in zip(tokens, values):
            word = normalize_word(token)
            if not word:
                dropped_empty += 1
                continue
            if len(word) > MAX_WORD_LENGTH or re.search(r"\s", word):
                dropped_long += 1
                continue
            per_post[word] = per_post.get(word, 0.0) + float(value)
        for word, total in per_post.items():
            signed_sum[word] = signed_sum.get(word, 0.0) + total
            absolute_sum[word] = absolute_sum.get(word, 0.0) + abs(total)
            support[word] = support.get(word, 0) + 1

    words = sorted(signed_sum)
    frame = pd.DataFrame(
        {
            "word": words,
            "mean_abs_attribution": [absolute_sum[w] / support[w] for w in words],
            "mean_attribution": [signed_sum[w] / support[w] for w in words],
            "support": [support[w] for w in words],
        }
    )
    frame = frame.sort_values(
        ["mean_abs_attribution", "word"], ascending=[False, True]
    ).reset_index(drop=True)
    return frame, {"tokens_dropped_empty": dropped_empty, "tokens_dropped_long": dropped_long}


def aggregate_word_attributions(tokens_per_post, values_per_post):
    """The word-level table alone: word, mean_abs_attribution, mean_attribution, support."""
    return aggregate_word_attributions_with_stats(tokens_per_post, values_per_post)[0]


def validate_attribution_frame(frame, label="attributions"):
    """Mirror of tools/attributions.py::validate_frame. Returns a clean, sorted copy."""
    missing = [c for c in ATTR_REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ArtifactContractError(f"{label}: missing required columns {missing}")
    forbidden = [c for c in ATTR_FORBIDDEN_COLUMNS if c in frame.columns]
    if forbidden:
        raise ArtifactContractError(
            f"{label}: carries per-post column(s) {forbidden}. Only word-level "
            "aggregates may be committed."
        )
    extra = [c for c in frame.columns if c not in ATTR_REQUIRED_COLUMNS]
    if extra:
        raise ArtifactContractError(
            f"{label}: unexpected columns {extra}; the contract is exactly "
            f"{list(ATTR_REQUIRED_COLUMNS)}"
        )

    clean = frame.copy()
    clean["word"] = clean["word"].astype(str)
    bad_words = clean["word"][
        clean["word"].str.strip().eq("")
        | clean["word"].str.contains(r"\s", regex=True)
        | (clean["word"].str.len() > MAX_WORD_LENGTH)
    ]
    if len(bad_words):
        raise ArtifactContractError(
            f"{label}: {len(bad_words)} 'word' value(s) are empty, contain whitespace, or "
            f"exceed {MAX_WORD_LENGTH} characters; the artifact is word-level"
        )
    if clean["word"].duplicated().any():
        raise ArtifactContractError(f"{label}: duplicate word values")

    for column in ("mean_abs_attribution", "mean_attribution"):
        values = pd.to_numeric(clean[column], errors="coerce")
        if not np.isfinite(values).all():
            raise ArtifactContractError(f"{label}: non-finite values in {column}")
        clean[column] = values.astype(float)
    if (clean["mean_abs_attribution"] < 0).any():
        raise ArtifactContractError(f"{label}: mean_abs_attribution must be non-negative")
    if (clean["mean_attribution"].abs() > clean["mean_abs_attribution"] + 1e-9).any():
        raise ArtifactContractError(
            f"{label}: |mean_attribution| exceeds mean_abs_attribution for some words"
        )

    support = pd.to_numeric(clean["support"], errors="coerce")
    if support.isna().any() or (support < 1).any() or (support != support.round()).any():
        raise ArtifactContractError(f"{label}: support must be a positive integer")
    clean["support"] = support.astype(int)

    return clean.sort_values(
        ["mean_abs_attribution", "word"], ascending=[False, True]
    ).reset_index(drop=True)


def export_attribution_run(ctx, bundle, run_id, member=None, write_local=True):
    """Write one committed attribution run: <run_id>.csv and its sidecar <run_id>.json.

    Mirrors tools/attributions.py::write_run. `member` is None for a whole
    model or whole ensemble, and the member id for one ensemble member.

    With `write_local`, the per-post token table is also written under
    external/attributions_local/. It carries row ids and tokens, so it is
    uploaded to external storage and never committed.

    Returns the validated word-level frame.
    """
    if not RUN_ID_RE.match(str(run_id)) or "__" in str(run_id):
        raise ArtifactContractError(
            f"run_id '{run_id}' must be letters, digits and '-' in '_'-separated parts, "
            "with no '__' (derived files use '__' and such stems are not runs)"
        )
    if member is not None and not RUN_ID_RE.match(str(member)):
        raise ArtifactContractError(f"member id '{member}' is not a simple identifier")

    frame, stats = aggregate_word_attributions_with_stats(
        bundle.tokens_per_post, bundle.values_per_post
    )
    clean = validate_attribution_frame(frame, label=f"{run_id}.csv")
    if clean.empty:
        raise ArtifactContractError(f"{run_id}: no words were attributed")
    if (clean["support"] > bundle.n_posts).any():
        raise ArtifactContractError("a word's support exceeds n_posts_explained")

    meta = {
        "technique": ctx.technique,
        "run_id": run_id,
        "split": bundle.split,
        "explainer": EXPLAINER_ID,
        "background_size": 0,
        "seed": bundle.seed,
        "aggregation": WORD_AGGREGATION,
        "n_posts_explained": int(bundle.n_posts),
        "member": member,
        "explainer_settings": {
            "algorithm": "partition",
            "masker": MASKER_PATTERN,
            "mask_token": MASK_TOKEN,
            "max_evals": bundle.max_evals,
            "background": "mask token; the text masker uses no background sample",
        },
        "n_words": int(len(clean)),
        "tokens_dropped": stats,
        "kit_version": KIT_VERSION,
    }
    missing = [f for f in ATTR_REQUIRED_META_FIELDS if f not in meta]
    if missing or meta["split"] not in SPLIT_NAMES:
        raise ArtifactContractError(f"{run_id}.json: invalid meta (missing {missing})")

    folder = ctx.results_dir / ATTRIBUTIONS_SUBDIR
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / f"{run_id}.csv"
    clean.to_csv(csv_path, index=False, float_format="%.8g")
    json_path = write_json(folder / f"{run_id}.json", meta)
    ctx.written.extend([csv_path, json_path])

    if write_local:
        rows = []
        for row_id, tokens, values in zip(
            bundle.row_ids, bundle.tokens_per_post, bundle.values_per_post
        ):
            for position, (token, value) in enumerate(zip(tokens, values)):
                rows.append(
                    {"row_id": row_id, "position": position, "token": token, "attribution": value}
                )
        local = pd.DataFrame(rows, columns=["row_id", "position", "token", "attribution"])
        local.to_csv(ctx.external("attributions_local") / f"{run_id}.csv", index=False)

    print(
        f"Attribution run '{run_id}': {len(clean)} words over {bundle.n_posts} "
        f"{bundle.split} posts (member={member})."
    )
    return clean


def top_words(frame, k=50):
    """The k words with the largest mean absolute attribution. Word-level; safe to display."""
    return frame.nlargest(k, "mean_abs_attribution").reset_index(drop=True)


def export_coefficients(ctx, words, coefficients):
    """Write attributions/coefficients.csv (word, coefficient) for a linear model.

    tools/validate_shap.py compares these with the attribution run to confirm
    the pipeline recovers what a linear model demonstrably relies on. Features
    that are not single words (n-grams with a space) must be dropped by the
    caller, which should report how many were dropped.
    """
    frame = pd.DataFrame(
        {"word": [str(w) for w in words], "coefficient": np.asarray(coefficients, dtype=float)}
    )
    if frame.empty:
        raise ArtifactContractError("no coefficients to export")
    if frame["word"].str.contains(r"\s", regex=True).any() or frame["word"].eq("").any():
        raise ArtifactContractError("coefficient words must be single non-empty words")
    if (frame["word"].str.len() > MAX_WORD_LENGTH).any():
        raise ArtifactContractError(f"a coefficient word exceeds {MAX_WORD_LENGTH} characters")
    if frame["word"].duplicated().any():
        raise ArtifactContractError("duplicate words in coefficients")
    if not np.isfinite(frame["coefficient"]).all():
        raise ArtifactContractError("non-finite coefficients")
    folder = ctx.results_dir / ATTRIBUTIONS_SUBDIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / COEFFICIENTS_FILENAME
    frame.sort_values("word").to_csv(path, index=False, float_format="%.8g")
    ctx.written.append(path)
    print(f"Coefficients written: {len(frame)} words.")
    return path


# ---------------------------------------------------------------------------
# Result folder
# ---------------------------------------------------------------------------
REQUIRED_BEST_CONFIG_KEYS = (
    "model_family",
    "feature_family",
    "hyperparameters",
    "threshold_strategy",
    "threshold_metric",
    "selected_threshold",
)


def confusion_matrix_frame(metrics):
    """The long-form confusion matrix documented in docs/RESULTS_SCHEMA.md."""
    return pd.DataFrame(
        [
            {"actual": NEGATIVE_CLASS_NAME, "predicted": NEGATIVE_CLASS_NAME, "count": metrics["tn"]},
            {"actual": NEGATIVE_CLASS_NAME, "predicted": POSITIVE_CLASS_NAME, "count": metrics["fp"]},
            {"actual": POSITIVE_CLASS_NAME, "predicted": NEGATIVE_CLASS_NAME, "count": metrics["fn"]},
            {"actual": POSITIVE_CLASS_NAME, "predicted": POSITIVE_CLASS_NAME, "count": metrics["tp"]},
        ]
    )


def classification_report_payload(metrics):
    """Per-class precision, recall, F1 and support, derived from the confusion counts."""
    tn, fp, fn = metrics["tn"], metrics["fp"], metrics["fn"]
    negative_precision = tn / (tn + fn) if (tn + fn) else 0.0
    negative_recall = tn / (tn + fp) if (tn + fp) else 0.0
    negative_f1 = (
        2 * negative_precision * negative_recall / (negative_precision + negative_recall)
        if (negative_precision + negative_recall)
        else 0.0
    )
    return {
        NEGATIVE_CLASS_NAME: {
            "precision": negative_precision,
            "recall": negative_recall,
            "f1-score": negative_f1,
            "support": metrics["negative_support"],
        },
        POSITIVE_CLASS_NAME: {
            "precision": metrics["positive_precision"],
            "recall": metrics["positive_recall"],
            "f1-score": metrics["positive_f1"],
            "support": metrics["positive_support"],
        },
        "accuracy": metrics["accuracy"],
        "macro avg": {
            "precision": metrics["precision_macro"],
            "recall": metrics["recall_macro"],
            "f1-score": metrics["f1_macro"],
            "support": metrics["support"],
        },
        "weighted avg": {
            "precision": metrics["precision_weighted"],
            "recall": metrics["recall_weighted"],
            "f1-score": metrics["f1_weighted"],
            "support": metrics["support"],
        },
    }


def _check_metrics(metrics, split, label):
    missing = [f for f in REQUIRED_METRIC_FIELDS if f not in metrics]
    if missing:
        raise ArtifactContractError(f"{label}: missing metric fields {missing}")
    if metrics.get("split") != split:
        raise ArtifactContractError(f"{label}: metrics are for split '{metrics.get('split')}'")
    total, negatives, positives = EXPECTED_SPLIT_COUNTS[split]
    actual = (metrics["support"], metrics["negative_support"], metrics["positive_support"])
    if actual != (total, negatives, positives):
        raise ArtifactContractError(f"{label}: support {actual} is not the frozen split")


def export_results_folder(
    ctx, best_config, validation_metrics, test_metrics, ablation_results, threshold_sweep
):
    """Write results_summary/<TECHNIQUE>/ from metrics the kit itself computed.

    Files: best_config.json, metrics_validation.json, metrics_test.json,
    confusion_matrix_test.csv, classification_report_test.json,
    ablation_results.csv, threshold_sweep_validation.csv.
    """
    missing = [k for k in REQUIRED_BEST_CONFIG_KEYS if k not in best_config]
    if missing:
        raise ArtifactContractError(f"best_config is missing keys {missing}")
    _check_metrics(validation_metrics, "validation", "metrics_validation")
    _check_metrics(test_metrics, "test", "metrics_test")

    threshold = float(best_config["selected_threshold"])
    for label, metrics in (("validation", validation_metrics), ("test", test_metrics)):
        if abs(float(metrics["threshold"]) - threshold) > 1e-9:
            raise ArtifactContractError(
                f"selected_threshold {threshold} differs from the {label} metrics threshold "
                f"{metrics['threshold']}; evaluate both splits at the locked threshold"
            )

    if ablation_results is None or len(ablation_results) == 0:
        raise ArtifactContractError("ablation_results is empty")
    if "config_id" not in ablation_results.columns:
        raise ArtifactContractError("ablation_results needs a config_id column")
    if not any(str(c).startswith("validation_") for c in ablation_results.columns):
        raise ArtifactContractError("ablation_results needs validation_* metric columns")
    test_columns = [c for c in ablation_results.columns if str(c).startswith("test_")]
    if test_columns:
        raise ArtifactContractError(
            f"ablation_results carries test columns {test_columns}; configurations are "
            "compared on validation only"
        )

    stamp = {
        "technique": ctx.technique,
        "run_id": ctx.run_id,
        "provenance": "derived_from_probs",
        "recomputable": True,
    }
    config_record = dict(best_config)
    config_record.update(stamp)
    config_record.update(
        {
            "dataset_version": DATASET_VERSION,
            "split_version": SPLIT_VERSION,
            "random_seed": ctx.config["random_seed"],
            "in_comparison": ctx.config["in_comparison"],
            "threshold_selected_on_split": "validation",
            "test_set_used_for_selection": False,
            "derived_by": "tools/notebook_kit.py",
            "kit_version": KIT_VERSION,
        }
    )

    folder = ctx.results_dir
    folder.mkdir(parents=True, exist_ok=True)
    paths = [
        write_json(folder / "best_config.json", config_record),
        write_json(folder / "metrics_validation.json", {**validation_metrics, **stamp}),
        write_json(folder / "metrics_test.json", {**test_metrics, **stamp}),
        write_json(
            folder / "classification_report_test.json",
            classification_report_payload(test_metrics),
        ),
    ]
    confusion_path = folder / "confusion_matrix_test.csv"
    confusion_matrix_frame(test_metrics).to_csv(confusion_path, index=False)
    ablation_path = folder / "ablation_results.csv"
    pd.DataFrame(ablation_results).to_csv(ablation_path, index=False)
    sweep_path = folder / "threshold_sweep_validation.csv"
    pd.DataFrame(threshold_sweep).to_csv(sweep_path, index=False)
    paths.extend([confusion_path, ablation_path, sweep_path])
    ctx.written.extend(paths)
    print(f"Result folder written: {folder} ({len(paths)} files)")
    return folder


def error_counts(frame, y_prob, threshold):
    """Counts of TP/TN/FP/FN by confidence bucket. No ids, no text.

    This is the table a notebook shows when it interprets its errors.
    """
    y_prob = _checked_probabilities(frame, y_prob, "error_counts")
    y_true = frame["label"].to_numpy().astype(int)
    y_pred = (y_prob >= threshold).astype(int)
    outcome = np.where(
        y_true == 1,
        np.where(y_pred == 1, "true_positive", "false_negative"),
        np.where(y_pred == 1, "false_positive", "true_negative"),
    )
    confidence = np.maximum(y_prob, 1.0 - y_prob)
    bucket = pd.cut(
        confidence,
        bins=[0.5, 0.6, 0.8, 1.0],
        labels=["0.5-0.6", "0.6-0.8", "0.8-1.0"],
        include_lowest=True,
    )
    table = (
        pd.DataFrame({"outcome": outcome, "confidence": bucket})
        .groupby(["outcome", "confidence"], observed=False)
        .size()
        .rename("count")
        .reset_index()
    )
    return table


# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------
def _forbidden_columns_in(path, forbidden):
    try:
        header = pd.read_csv(path, nrows=0)
    except Exception:
        return []
    return sorted(c for c in header.columns if c in set(forbidden))


def finalize(ctx, zip_external=False):
    """Inventory external assets, guard the committed tree, and zip the repo files.

    Writes external/<TECHNIQUE>/external_assets.json, which
    `tools/run_manifest.py add-assets-from` turns into manifest entries, and
    <TECHNIQUE>_repo_files.zip, whose contents unpack at the repository root.
    Returns the zip path.
    """
    if ctx.role == "foundation":
        committed_roots = [(ctx.results_dir, TEXT_BEARING_COLUMNS)]
    else:
        committed_roots = [
            (ctx.results_dir, TEXT_BEARING_COLUMNS + ROW_IDENTITY_COLUMNS),
            (ctx.probs_dir, PROBS_FORBIDDEN_COLUMNS),
        ]
    for root, forbidden in committed_roots:
        for path in root.rglob("*.csv"):
            leaked = _forbidden_columns_in(path, forbidden)
            if leaked:
                raise ArtifactContractError(
                    f"{path.relative_to(ctx.working_root)} carries text-bearing or row-level "
                    f"column(s) {leaked}; nothing under results_summary/ or research_loop/ may "
                    "hold dataset text, and a result folder holds no per-post rows"
                )

    assets = []
    if ctx.external_dir.exists():
        for path in sorted(p for p in ctx.external_dir.rglob("*") if p.is_file()):
            relative = path.relative_to(ctx.external_dir).as_posix()
            if relative == EXTERNAL_ASSETS_FILENAME:
                continue
            kind, contains_text = EXTERNAL_KINDS.get(relative.split("/", 1)[0], ("other", True))
            assets.append(
                {
                    "kind": kind,
                    "relative_path": relative,
                    "sha256": sha256_file(path),
                    "size_bytes": int(path.stat().st_size),
                    "contains_text": contains_text,
                }
            )
        write_json(
            ctx.external_dir / EXTERNAL_ASSETS_FILENAME,
            {"technique": ctx.technique, "run_id": ctx.run_id, "assets": assets},
        )

    zip_path = ctx.working_root / f"{ctx.technique}_repo_files.zip"
    members = []
    for path in sorted(p for p in ctx.results_dir.rglob("*") if p.is_file()):
        members.append(path)
    if ctx.role == "foundation":
        mirror = ctx.working_root / "splits" / "split_assignments.csv"
        if mirror.exists():
            members.append(mirror)
    else:
        members.extend(sorted(ctx.probs_dir.glob(f"{ctx.technique}__*")))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in members:
            archive.write(path, path.relative_to(ctx.working_root).as_posix())
        if zip_external and ctx.external_dir.exists():
            for path in sorted(p for p in ctx.external_dir.rglob("*") if p.is_file()):
                archive.write(path, path.relative_to(ctx.working_root).as_posix())

    print(f"Repository files ({len(members)}), zipped to {zip_path.name}:")
    for path in members:
        print(f"  {path.relative_to(ctx.working_root).as_posix()}")
    if assets:
        total_mb = sum(a["size_bytes"] for a in assets) / 1e6
        with_text = sum(1 for a in assets if a["contains_text"])
        print(
            f"External assets: {len(assets)} file(s), {total_mb:.1f} MB, {with_text} with "
            f"dataset text. Upload external/{ctx.technique}/ to a Kaggle Dataset, then run "
            "tools/run_manifest.py add-assets-from."
        )
    return zip_path
