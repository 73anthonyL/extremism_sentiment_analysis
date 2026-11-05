"""Helpers for the probe role (WP9, notebook 13): layer-wise linear probes on
frozen transformers.

Notebook 13 loads this file next to `notebook_kit` (as `pk`, after `nk`), from
the same folder. It builds on the kit and changes nothing in it: the kit owns
the run context, the frozen split, the probability-artifact contract and the
output tree; this module owns what only the probe notebook needs.

The probe notebook fits no post classifier and selects no threshold. What it
shares with the model notebooks is the discipline: the test split is opened
once, in the evaluation section, by `begin_test_evaluation`; every function
here that looks at a split refuses test rows until then. Its outputs are three
text-free tables and a run record under results_summary/rq/, written by
`export_probe_tables` under the contracts in docs/RESULTS_SCHEMA.md and read
locally by tools/layer_probes_report.py.

Like the kit it imports nothing else from the repository (on Kaggle the repo
is a read-only input). Its copies are pinned by tools/tests/test_probe_kit.py:
the lexicon matching to tools/lexicons.py and tools/identity_fpr.py, and the
McNemar and Holm functions character for character to tools/mcnemar_pairs.py.
"""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from notebook_kit import (
    ArtifactContractError,
    ConfigError,
    DATASET_VERSION,
    EXPECTED_SPLIT_COUNTS,
    InputNotFoundError,
    KIT_VERSION,
    KitError,
    POSITIVE_LABEL,
    PROBE_ROLE,
    PROBS_REQUIRED_COLUMNS,
    ROW_IDENTITY_COLUMNS,
    SPLIT_VERSION,
    TECHNIQUE_RE,
    TEXT_BEARING_COLUMNS,
    TestSplitError,
    _checked_probabilities,
    _find_all,
    _json_safe,
    _single_split,
    _word_tokens,
    safe_roc_auc,
    sha256_file,
    write_json,
)

PROBE_KIT_VERSION = "1.1.0"
PROBE_TABLE_FILES = {
    "probes": "rq3_layer_probes.csv",
    "contrasts": "rq3_layer_probe_contrasts.csv",
    "ablation": "rq4_identity_direction_ablation.csv",
    "run": "rq3_layer_probes_run.json",
}
PROBE_TASKS = ("extremist_label", "identity_term_present", "control_word_type")
CONTROL_TASK = "control_word_type"
PROBE_SPLITS = ("validation", "test")
PROBE_MODEL_KINDS = ("fine_tuned", "pretrained_base")
PROBE_POOLING = "mean"
PROBE_CONTRASTS = ("fine_tuned_minus_base", "selectivity")
ABLATION_POPULATIONS = ("non_extremist", "extremist")
ABLATION_DIRECTIONS = ("identity_probe", "random")
ABLATION_THRESHOLD_KINDS = ("locked", "screening")
REPRODUCTION_TOLERANCE = 1e-2

REQUIRED_PROBE_CONFIG_KEYS = (
    "models",
    "probe",
    "control_task",
    "bootstrap",
    "identity_direction",
    "error_analysis",
)
REQUIRED_PROBE_SETTINGS = ("pooling", "c_grid", "cv_folds", "cv_seed", "max_iter")
REQUIRED_BOOTSTRAP_SETTINGS = ("n_resamples", "seed", "ci_level")

PROBE_COLUMNS = (
    "model_id",
    "model_kind",
    "source_technique",
    "component",
    "checkpoint",
    "seed",
    "text_normalization",
    "n_layers",
    "hidden_size",
    "layer",
    "task",
    "split",
    "n_posts",
    "n_positive",
    "probe_c",
    "cv_auc_train",
    "roc_auc",
    "roc_auc_low",
    "roc_auc_high",
    "accuracy",
    "accuracy_low",
    "accuracy_high",
    "majority_accuracy",
    "n_saturated",
    "n_boot",
    "n_boot_valid",
    "ci_level",
)
CONTRAST_COLUMNS = (
    "contrast",
    "model_id",
    "reference_id",
    "layer",
    "task",
    "split",
    "n_posts",
    "difference",
    "difference_low",
    "difference_high",
    "n_boot",
    "n_boot_valid",
    "ci_level",
)
ABLATION_COLUMNS = (
    "model_id",
    "source_technique",
    "seed",
    "population",
    "direction",
    "n_posts",
    "threshold",
    "threshold_kind",
    "mean_p_before",
    "mean_p_after",
    "mean_delta",
    "median_delta",
    "random_delta_p95",
    "n_random_directions",
    "n_flagged_before",
    "n_flagged_after",
    "n_flip_pos_to_neg",
    "n_flip_neg_to_pos",
    "wilcoxon_p",
    "mcnemar_p",
    "wilcoxon_p_holm",
    "mcnemar_p_holm",
)
_ROW_ID_IN_TEXT_RE = re.compile(r"\bex_\d{6}\b")
_MODEL_ID_RE = re.compile(r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")


def validate_probe_config(config):
    missing = [k for k in REQUIRED_PROBE_CONFIG_KEYS if k not in config]
    if missing:
        raise ConfigError(f"CONFIG is missing required keys {missing}")
    if config["error_analysis"].get("include_text_preview") is not False:
        raise ConfigError("CONFIG['error_analysis']['include_text_preview'] must be False")
    missing = [k for k in REQUIRED_PROBE_SETTINGS if k not in config["probe"]]
    if missing:
        raise ConfigError(f"CONFIG['probe'] is missing keys {missing}")
    if config["probe"]["pooling"] != PROBE_POOLING:
        raise ConfigError(
            f"CONFIG['probe']['pooling'] is {config['probe']['pooling']!r}; the preregistered "
            f"pooling is {PROBE_POOLING!r}"
        )
    grid = config["probe"]["c_grid"]
    if not grid or any(float(c) <= 0 for c in grid):
        raise ConfigError("CONFIG['probe']['c_grid'] must be a non-empty list of positive values")
    missing = [k for k in REQUIRED_BOOTSTRAP_SETTINGS if k not in config["bootstrap"]]
    if missing:
        raise ConfigError(f"CONFIG['bootstrap'] is missing keys {missing}")
    if not 0 < float(config["bootstrap"]["ci_level"]) < 1:
        raise ConfigError("CONFIG['bootstrap']['ci_level'] must lie strictly between 0 and 1")
    if not config["models"]:
        raise ConfigError("CONFIG['models'] is empty")
    ids = [m.get("model_id") for m in config["models"]]
    if len(set(ids)) != len(ids) or not all(_MODEL_ID_RE.match(str(i)) for i in ids):
        raise ConfigError("every CONFIG['models'] entry needs a unique lower-case model_id")


def _require_probe_role(ctx, label):
    if ctx.role != PROBE_ROLE:
        raise KitError(f"{label} belongs to the probe role; model notebooks use nk.evaluate")


def _refuse_unopened_test(ctx, frame, label):
    """The split of `frame`; test rows are refused until begin_test_evaluation."""
    split = _single_split(frame, label)
    if split == "test" and not ctx.test_evaluated:
        raise TestSplitError(
            f"{label}: test rows before the test split was opened. Call "
            "nk.begin_test_evaluation(ctx) once, in the evaluation section."
        )
    return split


def begin_test_evaluation(ctx):
    """Open the test split, once per probe run. A second call raises."""
    _require_probe_role(ctx, "begin_test_evaluation")
    if ctx.test_evaluated:
        raise TestSplitError(
            f"the test split was already opened for {ctx.technique}. It is evaluated once per run."
        )
    ctx.test_evaluated = True
    print("Test split opened for its single evaluation.")


# --- inputs ------------------------------------------------------------------
def find_input_dir(ctx, relative, prefer=None):
    """Locate one attached input folder whose path ends with `relative`.

    `prefer` narrows to paths with that folder name in them. Several distinct
    folders under one name are refused rather than guessed between.
    """
    parts = Path(relative).parts
    matches = set()
    for root in ctx.input_roots:
        if not root.exists() or root.is_file():
            continue
        if root.parts[-len(parts):] == parts and root.is_dir():
            matches.add(root)
        for path in root.rglob(parts[-1]):
            if path.is_dir() and path.parts[-len(parts):] == parts:
                matches.add(path)
    matches = sorted(matches, key=lambda p: (len(p.parts), str(p)))
    if prefer:
        preferred = [p for p in matches if prefer in p.parts]
        if preferred:
            matches = preferred
    if not matches:
        roots = ", ".join(str(r) for r in ctx.input_roots)
        raise InputNotFoundError(
            f"no folder ending in '{relative}' under {roots}. Attach the input that provides it."
        )
    if len(matches) > 1:
        listing = "\n  ".join(str(p) for p in matches)
        raise KitError(f"{len(matches)} folders end in '{relative}':\n  {listing}\nDetach the extra one.")
    return matches[0]


def _find_in_folder(ctx, name, folder):
    """An attached file called `name` inside a folder called `folder`, required."""
    matches = [p for p in _find_all(ctx, name) if folder in p.parts]
    if not matches:
        raise InputNotFoundError(
            f"no '{name}' inside a '{folder}' folder among the inputs. Attach the repository."
        )
    if len({sha256_file(p) for p in matches}) > 1:
        raise KitError(f"different files named '{name}' under '{folder}' are attached")
    return matches[0]


def committed_best_config(ctx, technique):
    """A technique's committed best_config.json (locked threshold, hyperparameters)."""
    path = _find_in_folder(ctx, "best_config.json", technique)
    with open(path) as handle:
        config = json.load(handle)
    threshold = config.get("selected_threshold")
    if threshold is None or not 0.0 <= float(threshold) <= 1.0:
        raise ArtifactContractError(f"{path}: no locked threshold in [0, 1]")
    return config


def committed_probabilities(ctx, technique, frame):
    """The committed y_prob of `technique` for the rows of `frame`, in frame order."""
    split = _refuse_unopened_test(ctx, frame, "committed_probabilities")
    path = _find_in_folder(ctx, f"{technique}__{split}.csv", "probs")
    artifact = pd.read_csv(path)
    if list(artifact.columns) != list(PROBS_REQUIRED_COLUMNS):
        raise ArtifactContractError(f"{path.name}: not a probability artifact")
    merged = frame[["row_id", "label"]].merge(
        artifact, on="row_id", how="left", validate="one_to_one"
    )
    if merged["y_prob"].isna().any():
        raise ArtifactContractError(
            f"{path.name}: {int(merged['y_prob'].isna().sum())} row(s) of the split are missing"
        )
    if (merged["y_true"].astype(int) != merged["label"].astype(int)).any():
        raise ArtifactContractError(f"{path.name}: labels disagree with the foundation")
    return merged["y_prob"].to_numpy(dtype=float)


def verify_reproduction(ctx, technique, frame, y_prob, tolerance=REPRODUCTION_TOLERANCE):
    """Refuse restored weights whose probabilities differ from the committed artifact."""
    committed = committed_probabilities(ctx, technique, frame)
    y_prob = _checked_probabilities(frame, y_prob, f"verify_reproduction[{technique}]")
    difference = np.abs(y_prob - committed)
    result = {
        "technique": technique,
        "split": _single_split(frame, "verify_reproduction"),
        "n_rows": int(len(frame)),
        "max_abs_diff": float(difference.max()),
        "mean_abs_diff": float(difference.mean()),
        "tolerance": float(tolerance),
    }
    if result["max_abs_diff"] > tolerance:
        raise ArtifactContractError(
            f"{technique}: restored weights differ from the committed {result['split']} "
            f"probabilities by up to {result['max_abs_diff']:.4f} (tolerance {tolerance}). "
            "These are not the weights the notebook evaluated."
        )
    print(
        f"{technique}: reproduces its committed {result['split']} probabilities "
        f"(max |diff| {result['max_abs_diff']:.5f} over {result['n_rows']} rows)."
    )
    return result


# --- lexicon matching. read_lexicon_entries and lexicon_pattern are copies of
# tools/lexicons.py::read_lexicon and ::phrase_pattern (identity terms are
# found exactly as RQ4 finds them); pinned by tools/tests/test_notebook_kit.py.
_LEXICON_WHITESPACE_RE = re.compile(r"\s+")


def read_lexicon_entries(path):
    """The ordered, de-duplicated, lower-cased entries of one lexicon file."""
    path = Path(path)
    seen = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0]
        entry = _LEXICON_WHITESPACE_RE.sub(" ", line.strip().lower())
        if entry and entry not in seen:
            seen.append(entry)
    return seen


def lexicon_pattern(entries):
    """One compiled regex matching any entry at non-word boundaries, or None."""
    entries = [e for e in entries if e]
    if not entries:
        return None
    alternation = "|".join(re.escape(e) for e in sorted(entries, key=len, reverse=True))
    return re.compile(r"(?<![\w])(?:" + alternation + r")(?![\w])", re.IGNORECASE)


def load_identity_terms(ctx):
    """The identity lexicon attached with the repository: (entries, record)."""
    path = _find_in_folder(ctx, "identity_terms.txt", "lexicons")
    entries = read_lexicon_entries(path)
    if not entries:
        raise InputNotFoundError(f"{path}: the identity lexicon is empty")
    record = {"path": str(path), "sha256_16": sha256_file(path, 16), "n_entries": len(entries)}
    print(f"Identity lexicon: {len(entries)} entries (sha256 {record['sha256_16']}).")
    return entries, record


def identity_term_present(ctx, frame, entries):
    """1 where a post contains at least one identity-lexicon match (RQ4's rule)."""
    _refuse_unopened_test(ctx, frame, "identity_term_present")
    pattern = lexicon_pattern(entries)
    if pattern is None:
        raise KitError("identity_term_present needs a non-empty lexicon")
    return np.array([1 if pattern.search(str(t)) else 0 for t in frame["text"]], dtype=int)


# --- control task (Hewitt and Liang 2019, adapted to posts) -------------------
def word_type_bit(word, seed):
    """A fixed pseudo-random bit for one lower-cased word type."""
    return hashlib.sha256(f"{int(seed)}|{word}".encode("utf-8")).digest()[0] & 1


def control_word_type_scores(frame, seed):
    """Per post, the mean bit over its words (0 for a post with no word)."""
    scores = []
    for text in frame["text"]:
        words = [w.lower() for w in _word_tokens(text)]
        scores.append(float(np.mean([word_type_bit(w, seed) for w in words])) if words else 0.0)
    return np.asarray(scores, dtype=float)


def control_word_type_cutoff(train_frame, seed):
    """The cutoff c that gives the control task the label's train base rate.

    c is the smallest observed train score whose share of train posts above it
    does not exceed the train extremist share. Fitted on the train split only.
    Returns (cutoff, record).
    """
    if _single_split(train_frame, "control_word_type_cutoff") != "train":
        raise TestSplitError("the control cutoff is fitted on the train split only")
    scores = control_word_type_scores(train_frame, seed)
    target = float((train_frame["label"] == POSITIVE_LABEL).mean())
    cutoff = None
    for value in np.unique(scores):
        if float((scores > value).mean()) <= target:
            cutoff = float(value)
            break
    if cutoff is None:  # pragma: no cover - the largest score always qualifies
        cutoff = float(scores.max())
    record = {
        "seed": int(seed),
        "cutoff": cutoff,
        "train_label_positive_share": target,
        "train_control_positive_share": float((scores > cutoff).mean()),
    }
    return cutoff, record


def control_word_type_labels(ctx, frame, seed, cutoff):
    """The control task's labels for one split, at the train cutoff."""
    _refuse_unopened_test(ctx, frame, "control_word_type_labels")
    return (control_word_type_scores(frame, seed) > float(cutoff)).astype(int)


# --- probe metrics -----------------------------------------------------------
def bootstrap_indices(n_posts, n_resamples, seed):
    """Resample indices shared by every model, layer and task on one split."""
    rng = np.random.default_rng(int(seed))
    return rng.integers(0, int(n_posts), size=(int(n_resamples), int(n_posts)))


def roc_auc_rows(y_matrix, score_matrix):
    """ROC-AUC of every row (Mann-Whitney with average ranks); NaN for one class."""
    from scipy.stats import rankdata

    y_matrix = np.atleast_2d(np.asarray(y_matrix)).astype(int)
    score_matrix = np.atleast_2d(np.asarray(score_matrix, dtype=float))
    ranks = rankdata(score_matrix, axis=1)
    n_positive = y_matrix.sum(axis=1).astype(float)
    n_negative = y_matrix.shape[1] - n_positive
    rank_sum = (ranks * y_matrix).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        auc = (rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative)
    auc[(n_positive == 0) | (n_negative == 0)] = np.nan
    return auc


def _percentile_interval(values, ci_level):
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    tail = (1.0 - float(ci_level)) / 2.0 * 100.0
    return float(np.percentile(values, tail)), float(np.percentile(values, 100.0 - tail))


SATURATION_TOLERANCE = 1e-12


def _checked_logits(frame, logits, label):
    """Finite probe decision values (log-odds), one per row."""
    logits = np.asarray(logits, dtype=float).reshape(-1)
    if len(logits) != len(frame):
        raise ArtifactContractError(f"{label}: {len(logits)} values for {len(frame)} rows")
    if not np.isfinite(logits).all():
        raise ArtifactContractError(f"{label}: non-finite probe logits")
    return logits


def logit_to_probability(logits):
    """The probe's probability from its log-odds."""
    return 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=float)))


def score_probe(ctx, frame, y_true, y_logit):
    """ROC-AUC and accuracy at 0.5 of one probe on one split, with bootstrap intervals.

    `y_logit` is the probe's decision value (log-odds). AUC ranks the logits,
    not the probabilities, so probabilities that saturate to 0 or 1 in float
    arithmetic do not become ties; `n_saturated` counts them. Accuracy uses
    p >= 0.5, which is logit >= 0.
    """
    _require_probe_role(ctx, "score_probe")
    split = _refuse_unopened_test(ctx, frame, "score_probe")
    y_true = np.asarray(y_true).astype(int)
    y_logit = _checked_logits(frame, y_logit, f"score_probe[{split}]")
    if len(y_true) != len(frame):
        raise KitError("score_probe: labels and rows differ in length")
    probability = logit_to_probability(y_logit)
    settings = ctx.config["bootstrap"]
    index = bootstrap_indices(len(frame), settings["n_resamples"], settings["seed"])
    correct = ((probability >= 0.5).astype(int) == y_true).astype(float)
    boot_auc = roc_auc_rows(y_true[index], y_logit[index])
    boot_accuracy = correct[index].mean(axis=1)
    auc_low, auc_high = _percentile_interval(boot_auc, settings["ci_level"])
    acc_low, acc_high = _percentile_interval(boot_accuracy, settings["ci_level"])
    positive_share = float(y_true.mean())
    saturated = (probability <= SATURATION_TOLERANCE) | (probability >= 1.0 - SATURATION_TOLERANCE)
    return {
        "split": split,
        "n_posts": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "roc_auc": float(safe_roc_auc(y_true, y_logit)),
        "roc_auc_low": auc_low,
        "roc_auc_high": auc_high,
        "accuracy": float(correct.mean()),
        "accuracy_low": acc_low,
        "accuracy_high": acc_high,
        "majority_accuracy": max(positive_share, 1.0 - positive_share),
        "n_saturated": int(saturated.sum()),
        "n_boot": int(settings["n_resamples"]),
        "n_boot_valid": int(np.isfinite(boot_auc).sum()),
        "ci_level": float(settings["ci_level"]),
    }


def paired_auc_difference(ctx, frame, y_a, logit_a, y_b, logit_b):
    """AUC(a) - AUC(b) on the same posts, ranked on probe logits, with a paired bootstrap interval."""
    _require_probe_role(ctx, "paired_auc_difference")
    split = _refuse_unopened_test(ctx, frame, "paired_auc_difference")
    y_a, y_b = np.asarray(y_a).astype(int), np.asarray(y_b).astype(int)
    logit_a = _checked_logits(frame, logit_a, "paired_auc_difference[a]")
    logit_b = _checked_logits(frame, logit_b, "paired_auc_difference[b]")
    if len(y_a) != len(frame) or len(y_b) != len(frame):
        raise KitError("paired_auc_difference: labels and rows differ in length")
    settings = ctx.config["bootstrap"]
    index = bootstrap_indices(len(frame), settings["n_resamples"], settings["seed"])
    boot = roc_auc_rows(y_a[index], logit_a[index]) - roc_auc_rows(y_b[index], logit_b[index])
    low, high = _percentile_interval(boot, settings["ci_level"])
    return {
        "split": split,
        "n_posts": int(len(frame)),
        "difference": float(safe_roc_auc(y_a, logit_a) - safe_roc_auc(y_b, logit_b)),
        "difference_low": low,
        "difference_high": high,
        "n_boot": int(settings["n_resamples"]),
        "n_boot_valid": int(np.isfinite(boot).sum()),
        "ci_level": float(settings["ci_level"]),
    }


# mcnemar_exact_p and holm_adjust are character-for-character copies of
# tools/mcnemar_pairs.py, pinned by tools/tests/test_probe_kit.py.
def mcnemar_exact_p(n_only_a_wrong, n_only_b_wrong):
    """Exact two-sided McNemar p-value: a binomial test over the discordant pairs."""
    discordant = int(n_only_a_wrong + n_only_b_wrong)
    if discordant == 0:
        return 1.0
    return float(binomtest(int(n_only_a_wrong), discordant, 0.5, alternative="two-sided").pvalue)


def holm_adjust(p_values):
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(np.argsort(p, kind="stable")):
        value = min(1.0, (m - rank) * p[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def direction_ablation_row(
    ctx,
    frame,
    p_before,
    p_after,
    threshold,
    threshold_kind,
    identity,
    random_delta_p95=float("nan"),
    n_random_directions=0,
):
    """One row of the identity-direction ablation table, from per-post probabilities.

    `identity` names the row: model_id, source_technique, seed, population,
    direction. Holm columns are filled by export_probe_tables' caller.
    """
    from scipy.stats import wilcoxon

    _require_probe_role(ctx, "direction_ablation_row")
    _refuse_unopened_test(ctx, frame, "direction_ablation_row")
    p_before = _checked_probabilities(frame, p_before, "direction_ablation_row[before]")
    p_after = _checked_probabilities(frame, p_after, "direction_ablation_row[after]")
    if threshold_kind not in ABLATION_THRESHOLD_KINDS:
        raise KitError(f"threshold_kind must be one of {ABLATION_THRESHOLD_KINDS}")
    delta = p_before - p_after
    flagged_before = p_before >= threshold
    flagged_after = p_after >= threshold
    pos_to_neg = int((flagged_before & ~flagged_after).sum())
    neg_to_pos = int((~flagged_before & flagged_after).sum())
    if len(delta) == 0 or np.all(delta == 0):
        wilcoxon_p = float("nan")
    else:
        wilcoxon_p = float(wilcoxon(p_before, p_after).pvalue)
    row = {key: identity.get(key) for key in ("model_id", "source_technique", "seed", "population", "direction")}
    row.update(
        {
            "n_posts": int(len(delta)),
            "threshold": float(threshold),
            "threshold_kind": threshold_kind,
            "mean_p_before": float(p_before.mean()),
            "mean_p_after": float(p_after.mean()),
            "mean_delta": float(delta.mean()),
            "median_delta": float(np.median(delta)),
            "random_delta_p95": float(random_delta_p95),
            "n_random_directions": int(n_random_directions),
            "n_flagged_before": int(flagged_before.sum()),
            "n_flagged_after": int(flagged_after.sum()),
            "n_flip_pos_to_neg": pos_to_neg,
            "n_flip_neg_to_pos": neg_to_pos,
            "wilcoxon_p": wilcoxon_p,
            "mcnemar_p": mcnemar_exact_p(pos_to_neg, neg_to_pos),
            "wilcoxon_p_holm": float("nan"),
            "mcnemar_p_holm": float("nan"),
        }
    )
    return row


# --- the probe tables --------------------------------------------------------
def _exact_columns(frame, columns, label):
    forbidden = [c for c in frame.columns if c in set(TEXT_BEARING_COLUMNS + ROW_IDENTITY_COLUMNS)]
    if forbidden:
        raise ArtifactContractError(f"{label}: carries per-post or text column(s) {forbidden}")
    if list(frame.columns) != list(columns):
        raise ArtifactContractError(
            f"{label}: columns must be exactly {list(columns)} (got {list(frame.columns)})"
        )


def _require_values(frame, column, allowed, label):
    bad = sorted(set(frame[column].astype(str)) - set(allowed))
    if bad:
        raise ArtifactContractError(f"{label}: {column} has values {bad} outside {list(allowed)}")


def _require_unit_interval(frame, columns, label, allow_nan=False):
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        finite = values[np.isfinite(values)]
        if not allow_nan and len(finite) != len(values):
            raise ArtifactContractError(f"{label}: {column} has missing or non-finite values")
        if ((finite < 0) | (finite > 1)).any():
            raise ArtifactContractError(f"{label}: {column} has values outside [0, 1]")


def _require_ordered_interval(frame, point, low, high, label):
    lows, highs = pd.to_numeric(frame[low]), pd.to_numeric(frame[high])
    both = np.isfinite(lows) & np.isfinite(highs)
    if (lows[both] > highs[both] + 1e-12).any():
        raise ArtifactContractError(f"{label}: {low} exceeds {high}")
    if not np.isfinite(pd.to_numeric(frame[point])).all():
        raise ArtifactContractError(f"{label}: {point} has non-finite values")


def _require_unique(frame, key, label):
    duplicated = frame.duplicated(subset=list(key))
    if duplicated.any():
        raise ArtifactContractError(f"{label}: {int(duplicated.sum())} duplicate {list(key)} row(s)")


def validate_probe_table(frame, label=PROBE_TABLE_FILES["probes"]):
    """Refuse a probe table that breaks its contract. Returns it sorted."""
    _exact_columns(frame, PROBE_COLUMNS, label)
    if frame.empty:
        raise ArtifactContractError(f"{label}: no rows")
    _require_values(frame, "task", PROBE_TASKS, label)
    _require_values(frame, "split", PROBE_SPLITS, label)
    _require_values(frame, "model_kind", PROBE_MODEL_KINDS, label)
    if not frame["model_id"].astype(str).str.match(_MODEL_ID_RE.pattern).all():
        raise ArtifactContractError(f"{label}: a model_id is not a lower-case identifier")
    _require_unit_interval(
        frame,
        ("roc_auc", "accuracy", "majority_accuracy", "cv_auc_train", "ci_level"),
        label,
    )
    _require_unit_interval(
        frame, ("roc_auc_low", "roc_auc_high", "accuracy_low", "accuracy_high"), label, allow_nan=True
    )
    _require_ordered_interval(frame, "roc_auc", "roc_auc_low", "roc_auc_high", label)
    _require_ordered_interval(frame, "accuracy", "accuracy_low", "accuracy_high", label)
    _require_unique(frame, ("model_id", "layer", "task", "split"), label)
    for split, group in frame.groupby("split"):
        expected = EXPECTED_SPLIT_COUNTS[str(split)][0]
        if (group["n_posts"] != expected).any():
            raise ArtifactContractError(
                f"{label}: {split} rows must be scored on all {expected} posts of the split"
            )
    if (frame["n_positive"] > frame["n_posts"]).any() or (frame["n_positive"] < 0).any():
        raise ArtifactContractError(f"{label}: n_positive outside [0, n_posts]")
    if (frame["n_boot_valid"] > frame["n_boot"]).any():
        raise ArtifactContractError(f"{label}: n_boot_valid exceeds n_boot")
    if ((frame["n_saturated"] < 0) | (frame["n_saturated"] > frame["n_posts"])).any():
        raise ArtifactContractError(f"{label}: n_saturated outside [0, n_posts]")
    if (pd.to_numeric(frame["probe_c"]) <= 0).any():
        raise ArtifactContractError(f"{label}: probe_c must be positive")
    fine_tuned = frame[frame["model_kind"] == "fine_tuned"]
    if (
        not fine_tuned["source_technique"].astype(str).str.match(TECHNIQUE_RE.pattern).all()
        or not np.isfinite(pd.to_numeric(fine_tuned["seed"], errors="coerce")).all()
    ):
        raise ArtifactContractError(
            f"{label}: fine-tuned rows need a source_technique and an integer seed"
        )
    for model_id, group in frame.groupby("model_id"):
        n_layers = group["n_layers"].unique()
        if len(n_layers) != 1:
            raise ArtifactContractError(f"{label}: {model_id} declares several n_layers")
        layers = set(range(int(n_layers[0])))
        for (task, split), cell in group.groupby(["task", "split"]):
            if set(cell["layer"].astype(int)) != layers:
                raise ArtifactContractError(
                    f"{label}: {model_id} {task} {split} does not cover layers 0..{int(n_layers[0]) - 1}"
                )
        if set(group["task"]) != set(PROBE_TASKS) or set(group["split"]) != set(PROBE_SPLITS):
            raise ArtifactContractError(f"{label}: {model_id} lacks a task or split")
    return frame.sort_values(["model_id", "task", "split", "layer"]).reset_index(drop=True)


def validate_contrast_table(frame, probes, label=PROBE_TABLE_FILES["contrasts"]):
    """Refuse a contrast table that breaks its contract or names unknown probe rows."""
    _exact_columns(frame, CONTRAST_COLUMNS, label)
    _require_values(frame, "contrast", PROBE_CONTRASTS, label)
    _require_values(frame, "split", PROBE_SPLITS, label)
    _require_values(frame, "task", PROBE_TASKS, label)
    _require_unique(frame, ("contrast", "model_id", "reference_id", "layer", "task", "split"), label)
    _require_ordered_interval(frame, "difference", "difference_low", "difference_high", label)
    if (pd.to_numeric(frame["difference"]).abs() > 1).any():
        raise ArtifactContractError(f"{label}: an AUC difference exceeds 1 in magnitude")
    selectivity = frame[frame["contrast"] == "selectivity"]
    if (selectivity["task"] == CONTROL_TASK).any() or (
        selectivity["model_id"] != selectivity["reference_id"]
    ).any():
        raise ArtifactContractError(
            f"{label}: selectivity rows compare a task with the control on the same model"
        )
    known = set(zip(probes["model_id"], probes["layer"], probes["task"], probes["split"]))
    for row in frame.itertuples():
        for model in (row.model_id, row.reference_id):
            if (model, row.layer, row.task, row.split) not in known:
                raise ArtifactContractError(
                    f"{label}: {row.contrast} names {model} layer {row.layer} {row.task} "
                    f"{row.split}, which the probe table does not have"
                )
    return frame.sort_values(["contrast", "model_id", "reference_id", "task", "split", "layer"]).reset_index(drop=True)


def validate_ablation_table(frame, label=PROBE_TABLE_FILES["ablation"]):
    """Refuse an ablation table that breaks its contract or its own arithmetic."""
    _exact_columns(frame, ABLATION_COLUMNS, label)
    if frame.empty:
        raise ArtifactContractError(f"{label}: no rows")
    _require_values(frame, "population", ABLATION_POPULATIONS, label)
    _require_values(frame, "direction", ABLATION_DIRECTIONS, label)
    _require_values(frame, "threshold_kind", ABLATION_THRESHOLD_KINDS, label)
    _require_unique(frame, ("model_id", "population", "direction"), label)
    _require_unit_interval(frame, ("threshold", "mean_p_before", "mean_p_after"), label)
    _require_unit_interval(
        frame, ("wilcoxon_p", "mcnemar_p", "wilcoxon_p_holm", "mcnemar_p_holm"), label, allow_nan=True
    )
    if (
        (frame["mean_p_before"] - frame["mean_p_after"] - frame["mean_delta"]).abs() > 1e-6
    ).any():
        raise ArtifactContractError(f"{label}: mean_delta is not mean_p_before - mean_p_after")
    expected_after = frame["n_flagged_before"] - frame["n_flip_pos_to_neg"] + frame["n_flip_neg_to_pos"]
    if (expected_after != frame["n_flagged_after"]).any():
        raise ArtifactContractError(f"{label}: flagged counts and flips disagree")
    for column in ("n_flagged_before", "n_flagged_after"):
        if ((frame[column] < 0) | (frame[column] > frame["n_posts"])).any():
            raise ArtifactContractError(f"{label}: {column} outside [0, n_posts]")
    return frame.sort_values(["model_id", "population", "direction"]).reset_index(drop=True)


def export_probe_tables(ctx, probes, contrasts, ablation, run_record):
    """Write the WP9 tables and their run record to results_summary/rq/.

    Every table is validated against its contract first; nothing is written
    unless all of them pass. The run record is refused if it carries a row id.
    Returns {name: path}.
    """
    _require_probe_role(ctx, "export_probe_tables")
    validate_probe_config(ctx.config)
    if not ctx.test_evaluated:
        raise TestSplitError("export_probe_tables: the tables include test rows; open test first")
    probes = validate_probe_table(pd.DataFrame(probes))
    contrasts = validate_contrast_table(pd.DataFrame(contrasts), probes)
    wants_ablation = bool(ctx.config["identity_direction"].get("enabled"))
    if wants_ablation and ablation is None:
        raise ArtifactContractError("identity_direction is enabled but no ablation table was given")
    if ablation is not None:
        ablation = validate_ablation_table(pd.DataFrame(ablation))

    record = dict(run_record)
    record.update(
        {
            "technique": ctx.technique,
            "run_id": ctx.run_id,
            "dataset_version": DATASET_VERSION,
            "split_version": SPLIT_VERSION,
            "random_seed": ctx.config["random_seed"],
            "test_set_used_for_selection": False,
            "contains_text": False,
            "kit_version": KIT_VERSION,
            "probe_kit_version": PROBE_KIT_VERSION,
            "config": ctx.config,
            "tables": {
                "probes": {"file": PROBE_TABLE_FILES["probes"], "columns": list(PROBE_COLUMNS), "rows": len(probes)},
                "contrasts": {"file": PROBE_TABLE_FILES["contrasts"], "columns": list(CONTRAST_COLUMNS), "rows": len(contrasts)},
                "ablation": (
                    {"file": PROBE_TABLE_FILES["ablation"], "columns": list(ABLATION_COLUMNS), "rows": len(ablation)}
                    if ablation is not None
                    else None
                ),
            },
        }
    )
    if _ROW_ID_IN_TEXT_RE.search(json.dumps(_json_safe(record))):
        raise ArtifactContractError("the run record carries a row id; it holds no per-post data")

    folder = ctx.results_dir
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, table in (("probes", probes), ("contrasts", contrasts), ("ablation", ablation)):
        if table is None:
            continue
        path = folder / PROBE_TABLE_FILES[name]
        table.to_csv(path, index=False, float_format="%.8g")
        paths[name] = path
    record["artifact_sha256"] = {name: sha256_file(path, 16) for name, path in paths.items()}
    paths["run"] = write_json(folder / PROBE_TABLE_FILES["run"], record)
    ctx.written.extend(paths.values())
    print(f"Probe tables written: {[p.name for p in paths.values()]}")
    return paths
