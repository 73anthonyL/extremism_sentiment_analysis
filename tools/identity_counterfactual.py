"""RQ4 / WP7: does removing an identity term change the prediction?

identity_fpr.py shows that non-extremist posts mentioning an identity term are
flagged more often. That is a correlation. This tool intervenes: for every
test post containing at least one identity-lexicon match it builds
counterfactual texts and re-scores them with the technique's restored
predictor:

* remove      every identity match is deleted and whitespace collapsed. This
              is the primary operation.
* substitute  every identity match is replaced by a control word, once per word
              of SUBSTITUTE_WORDS, and the probabilities are averaged over the
              panel. This is a sensitivity check.

The substitute panel must be neutral, and the tool refuses it at startup
otherwise: no panel word may be matched by any lexicon, its share of
extremist posts in the TRAINING split must lie within SUBSTITUTE_MAX_SHARE_GAP
of the training base rate, and its 01_LOG-REG_TF-IDF coefficient must satisfy
|coef| <= SUBSTITUTE_MAX_ABS_COEF. A first version used "people", which fails
both checks: 63% of the training posts that contain it are extremist, and its
LR coefficient is the 10th largest of 4,692. The class balance and coefficient
of each panel word are recorded in every JSON.

Inputs, per technique:

* its committed test probability artifact (research_loop/probs/),
* its locked threshold (results_summary/<TECHNIQUE>/best_config.json),
* the dataset text, rebuilt locally through split_protocol, with the same
  y_true-vs-label abort as identity_fpr.py, and
* its predictor restored from external/<TECHNIQUE> by model_loaders.

Before any counterfactual is scored, the restored predictor re-scores the
ORIGINAL texts of the population and must agree with the committed artifact
within curated_examples.DEFAULT_TOLERANCE. Otherwise the technique is refused.
p_original in every metric is the committed artifact probability.

delta = p_original - p_counterfactual, so a positive delta means the identity
term pushed the post toward EXTREMIST. Non-extremist posts (y_true=0) answer
the false-positive question. There, n_flip_pos_to_neg counts original false
positives that clear when the intervention is applied, and fp_share_cleared
is that count over the false positives flagged originally. Read it with
n_flagged_original -> n_flagged_counterfactual, the net change. Extremist posts
(y_true=1) answer the recall question. Paired tests: Wilcoxon signed-rank on
the probabilities and an exact McNemar test on the flips. Holm-adjusted
p-values are added in the cross-technique table, over every technique x
population x operation row of the techniques in the comparison (one family
per test), matching RQ1. A technique whose best_config.json says
"in_comparison": false (an audited classifier) gets its rows Holm-adjusted
within its own rows, so it never moves the comparison's p values.

A flip shows that this perturbation moved the post across the locked
threshold. That is threshold-local necessity, not proof that the term is the
cause. Removal also leaves ungrammatical text, and for TF-IDF models it
changes L2 normalisation and the character n-grams across the join.

Re-scoring frozen models at their locked thresholds is a perturbation
analysis, not a test evaluation: no threshold, model or configuration is
selected from it.

Text is held in memory only. Outputs carry counts, rates, and p-values; no
text, no row id, no hash. Nothing is written unless every requested technique
passes.

Outputs:
    results_summary/<TECHNIQUE>/identity_counterfactual_test.json
    results_summary/rq/rq4_identity_counterfactual.csv

USAGE
-----
    python3 tools/identity_counterfactual.py --dry-run-population --all
    python3 tools/identity_counterfactual.py --technique 01_LOG-REG_TF-IDF
    python3 tools/identity_counterfactual.py --technique 11_MULTI-CHECKPOINT_LOGIT-POOL \\
        --device mps --sequential-checkpoints
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attributions import AttributionArtifactError  # noqa: E402
from curated_examples import DEFAULT_TOLERANCE, CuratedExamplesError, verify_model  # noqa: E402
from identity_fpr import IdentityFPRError, load_threshold, merge_with_dataset  # noqa: E402
from lexicons import load_lexicons, phrase_pattern  # noqa: E402
from mcnemar_pairs import holm_adjust, mcnemar_exact_p  # noqa: E402
from model_loaders import ModelLoadError, load_technique  # noqa: E402
from probs_artifact import ProbsArtifactError, load_technique_probs  # noqa: E402
from validate_shap import load_coefficients  # noqa: E402
from repo_paths import (  # noqa: E402
    ATTRIBUTIONS_SUBDIR,
    COEFFICIENTS_FILENAME,
    IDENTITY_COUNTERFACTUAL_FILENAME,
    POSITIVE_LABEL,
    REPO_ROOT,
    RESULTS_DIR,
    RQ4_IDENTITY_COUNTERFACTUAL_CSV,
    SPLIT_ASSIGNMENTS_CSV,
    in_comparison,
    technique_dirs,
)

SPLIT = "test"
OPERATIONS = ("remove", "substitute")
POPULATIONS = ((0, "non_extremist"), (POSITIVE_LABEL, "extremist"))
EXTERNAL_ROOT = REPO_ROOT / "external"
WHITESPACE_RE = re.compile(r"\s+")
SIG_DIGITS = 6

# Substitute panel: group nouns that are label-balanced in the training split
# and carry a near-zero logistic-regression coefficient (checked at startup).
SUBSTITUTE_WORDS = ("group", "population", "humans")
SUBSTITUTE_REFERENCE_TECHNIQUE = "01_LOG-REG_TF-IDF"
SUBSTITUTE_MAX_SHARE_GAP = 0.10
SUBSTITUTE_MAX_ABS_COEF = 0.5
TORCH_FAMILIES = ("transformer", "pooled_transformer_ensemble")

ROW_COLUMNS = (
    "technique",
    "split",
    "population",
    "y_true",
    "operation",
    "threshold",
    "n_posts",
    "mean_p_original",
    "mean_p_counterfactual",
    "mean_delta",
    "median_delta",
    "share_delta_positive",
    "n_flagged_original",
    "n_flagged_counterfactual",
    "n_flip_pos_to_neg",
    "n_flip_neg_to_pos",
    "fp_share_cleared",
    "wilcoxon_p",
    "mcnemar_exact_p",
    "verification_max_abs_diff",
)
HOLM_COLUMNS = ("wilcoxon_p_holm", "mcnemar_p_holm")
TABLE_COLUMNS = ROW_COLUMNS + HOLM_COLUMNS


class IdentityCounterfactualError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Substitute panel
# ---------------------------------------------------------------------------
def check_substitute_word(lexicons, words=SUBSTITUTE_WORDS):
    """Refuse a substitute that any lexicon would itself match."""
    words = (words,) if isinstance(words, str) else tuple(words)
    if not words:
        raise IdentityCounterfactualError("substitute panel is empty")
    for word in words:
        for category, entries in lexicons.items():
            pattern = phrase_pattern(entries)
            if pattern is not None and pattern.search(word):
                raise IdentityCounterfactualError(
                    f"substitute word '{word}' is matched by the {category} lexicon; "
                    "the substitution would not be neutral"
                )
    return words


def substitute_profile(words, processed, assignments, coefficients):
    """Training-split class balance and LR coefficient of each panel word (counts only)."""
    train_ids = set(assignments.loc[assignments["split"] == "train", "row_id"])
    train = processed[processed["row_id"].isin(train_ids)]
    if train.empty:
        raise IdentityCounterfactualError("no training rows found for the substitute profile")
    base_rate = float((train["label"] == POSITIVE_LABEL).mean())
    coef = coefficients.set_index("word")["coefficient"]
    rank = coef.rank(ascending=False, method="min")
    profile = {}
    for word in words:
        pattern = phrase_pattern([word])
        has_word = train["text"].map(lambda t: pattern.search(t) is not None)
        n_posts = int(has_word.sum())
        n_extremist = int((train.loc[has_word, "label"] == POSITIVE_LABEL).sum())
        profile[word] = {
            "train_n_posts": n_posts,
            "train_n_extremist": n_extremist,
            "train_extremist_share": (n_extremist / n_posts) if n_posts else None,
            "lr_coefficient": float(coef[word]) if word in coef.index else None,
            "lr_coefficient_rank_desc": int(rank[word]) if word in rank.index else None,
        }
    return {
        "reference_technique": SUBSTITUTE_REFERENCE_TECHNIQUE,
        "train_base_rate": base_rate,
        "n_lr_coefficients": int(len(coef)),
        "max_share_gap": SUBSTITUTE_MAX_SHARE_GAP,
        "max_abs_coef": SUBSTITUTE_MAX_ABS_COEF,
        "words": profile,
    }


def check_substitute_profile(profile):
    """Refuse a panel word that is label-imbalanced in training or strongly weighted by LR."""
    base = profile["train_base_rate"]
    for word, stats in profile["words"].items():
        share = stats["train_extremist_share"]
        if share is not None and abs(share - base) > profile["max_share_gap"]:
            raise IdentityCounterfactualError(
                f"substitute word '{word}': training extremist share {share:.2f} is more than "
                f"{profile['max_share_gap']} from the base rate {base:.2f}"
            )
        coef = stats["lr_coefficient"]
        if coef is not None and abs(coef) > profile["max_abs_coef"]:
            raise IdentityCounterfactualError(
                f"substitute word '{word}': |LR coefficient| {abs(coef):.2f} exceeds {profile['max_abs_coef']}"
            )
    return profile


def load_substitute_profile(processed, words=SUBSTITUTE_WORDS, results_dir=None):
    base = RESULTS_DIR if results_dir is None else Path(results_dir)
    coefficients = load_coefficients(
        base / SUBSTITUTE_REFERENCE_TECHNIQUE / ATTRIBUTIONS_SUBDIR / COEFFICIENTS_FILENAME
    )
    assignments = pd.read_csv(SPLIT_ASSIGNMENTS_CSV, usecols=["row_id", "split"])
    return substitute_profile(words, processed, assignments, coefficients)


# ---------------------------------------------------------------------------
# Counterfactual text
# ---------------------------------------------------------------------------
def counterfactual_text(text, pattern, operation, substitute=SUBSTITUTE_WORDS[0]):
    """Apply one operation to every identity match in `text`."""
    if operation == "remove":
        replaced = pattern.sub("", text)
    elif operation == "substitute":
        replaced = pattern.sub(substitute, text)
    else:
        raise IdentityCounterfactualError(f"unknown operation '{operation}'")
    return WHITESPACE_RE.sub(" ", replaced).strip()


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------
def build_population(technique, processed, probs, terms):
    """Test rows with at least one identity match: row_id, text, y_true, y_prob (in memory only)."""
    pattern = phrase_pattern(terms)
    if pattern is None:
        raise IdentityCounterfactualError("identity lexicon is empty")
    merged = merge_with_dataset(technique, probs, processed)
    has_identity = merged["text"].map(lambda t: pattern.search(t) is not None)
    population = merged.loc[has_identity, ["row_id", "text", "y_true", "y_prob"]]
    return population.reset_index(drop=True), pattern


def population_counts(population):
    return {name: int((population["y_true"] == label).sum()) for label, name in POPULATIONS}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _nan_if_empty(value, n):
    return float(value) if n else float("nan")


def paired_metrics(p_original, p_counterfactual, threshold, y_true):
    """Shift and flip statistics for one population under one operation."""
    p_original = np.asarray(p_original, dtype=float)
    p_counterfactual = np.asarray(p_counterfactual, dtype=float)
    n = len(p_original)
    delta = p_original - p_counterfactual
    flagged_original = p_original >= threshold
    flagged_counterfactual = p_counterfactual >= threshold
    pos_to_neg = int((flagged_original & ~flagged_counterfactual).sum())
    neg_to_pos = int((~flagged_original & flagged_counterfactual).sum())
    if n == 0 or np.all(delta == 0):
        wilcoxon_p = float("nan")
    else:
        wilcoxon_p = float(wilcoxon(p_original, p_counterfactual).pvalue)
    n_flagged = int(flagged_original.sum())
    fp_share = (
        pos_to_neg / n_flagged if (y_true != POSITIVE_LABEL and n_flagged) else float("nan")
    )
    return {
        "n_posts": int(n),
        "mean_p_original": _nan_if_empty(p_original.mean() if n else 0, n),
        "mean_p_counterfactual": _nan_if_empty(p_counterfactual.mean() if n else 0, n),
        "mean_delta": _nan_if_empty(delta.mean() if n else 0, n),
        "median_delta": _nan_if_empty(np.median(delta) if n else 0, n),
        "share_delta_positive": _nan_if_empty((delta > 0).mean() if n else 0, n),
        "n_flagged_original": n_flagged,
        "n_flagged_counterfactual": int(flagged_counterfactual.sum()),
        "n_flip_pos_to_neg": pos_to_neg,
        "n_flip_neg_to_pos": neg_to_pos,
        "fp_share_cleared": float(fp_share),
        "wilcoxon_p": wilcoxon_p,
        "mcnemar_exact_p": mcnemar_exact_p(pos_to_neg, neg_to_pos),
    }


def _round(value):
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return float(f"{value:.{SIG_DIGITS}g}")
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    return value


def recorded_device(model, requested):
    """The device the predictor actually used, or None for techniques that ignore it."""
    if getattr(model, "family", None) not in TORCH_FAMILIES:
        return None
    if requested == "auto":
        from model_loaders import _torch_device

        return _torch_device(requested).type
    return requested


# ---------------------------------------------------------------------------
# One technique
# ---------------------------------------------------------------------------
def analyze(technique, model, population, pattern, threshold, n_lexicon_entries,
            device=None, tolerance=DEFAULT_TOLERANCE, substitutes=SUBSTITUTE_WORDS,
            substitute_profile=None):
    """Verify the predictor on the original texts, then score both counterfactuals."""
    substitutes = (substitutes,) if isinstance(substitutes, str) else tuple(substitutes)
    restored_threshold = getattr(model, "threshold", None)
    if restored_threshold is not None and not math.isclose(float(restored_threshold), float(threshold)):
        raise IdentityCounterfactualError(
            f"{technique}: restored threshold {restored_threshold} != committed {threshold}"
        )
    if population.empty:
        raise IdentityCounterfactualError(f"{technique}: no test post contains an identity term")
    try:
        verification = verify_model(
            model, population[["row_id", "text"]], population[["row_id", "y_prob"]], tolerance
        )
    except CuratedExamplesError as error:
        raise IdentityCounterfactualError(str(error)) from error

    texts = population["text"].tolist()
    n = len(texts)
    p_original = population["y_prob"].to_numpy(dtype=float)
    y_true = population["y_true"].to_numpy()
    batches = [[counterfactual_text(t, pattern, "remove") for t in texts]]
    batches += [[counterfactual_text(t, pattern, "substitute", word) for t in texts] for word in substitutes]
    scored = np.asarray(model.predict([t for batch in batches for t in batch]), dtype=float)
    per_batch = [scored[i * n:(i + 1) * n] for i in range(len(batches))]
    per_word = dict(zip(substitutes, per_batch[1:]))
    p_counterfactual = {"remove": per_batch[0], "substitute": np.mean(per_batch[1:], axis=0)}

    rows = []
    per_word_rows = []
    for label, name in POPULATIONS:
        mask = y_true == label
        for op in OPERATIONS:
            row = {
                "technique": technique,
                "split": SPLIT,
                "population": name,
                "y_true": int(label),
                "operation": op,
                "threshold": float(threshold),
            }
            row.update(paired_metrics(p_original[mask], p_counterfactual[op][mask], threshold, label))
            row["verification_max_abs_diff"] = verification["max_abs_diff"]
            rows.append({k: _round(v) for k, v in row.items()})
        for word, values in per_word.items():
            metrics = paired_metrics(p_original[mask], values[mask], threshold, label)
            per_word_rows.append(_round({
                "population": name,
                "word": word,
                "mean_delta": metrics["mean_delta"],
                "median_delta": metrics["median_delta"],
                "n_flagged_counterfactual": metrics["n_flagged_counterfactual"],
            }))

    return {
        "technique": technique,
        "split": SPLIT,
        "threshold": float(threshold),
        "device": device,
        "operations": list(OPERATIONS),
        "primary_operation": "remove",
        "substitute_words": list(substitutes),
        "substitute_aggregation": "mean probability over the substitute words",
        "substitute_profile": _round(substitute_profile) if substitute_profile else None,
        "substitute_per_word": per_word_rows,
        "n_identity_terms_in_lexicon": int(n_lexicon_entries),
        "n_population": int(n),
        "n_population_by_label": population_counts(population),
        "p_original_source": "committed probability artifact",
        "holm_adjusted_p": f"see {RQ4_IDENTITY_COUNTERFACTUAL_CSV.name} (family: every comparison row of that table, or this technique's own rows when it is outside the comparison)",
        "verification": {k: _round(v) for k, v in verification.items()},
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def write_outputs(payload, results_dir=None):
    base = RESULTS_DIR if results_dir is None else Path(results_dir)
    folder = base / payload["technique"]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / IDENTITY_COUNTERFACTUAL_FILENAME.format(split=payload["split"])
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def _holm_column(values):
    values = pd.to_numeric(values, errors="coerce")
    adjusted = pd.Series(np.nan, index=values.index)
    present = values.notna()
    if present.any():
        adjusted[present] = holm_adjust(values[present].to_numpy())
    return adjusted


def write_table(payloads, path=None, results_dir=None):
    """Merge rows into the cross-technique table and recompute Holm per family."""
    path = Path(RQ4_IDENTITY_COUNTERFACTUAL_CSV if path is None else path)
    new = pd.DataFrame([row for p in payloads for row in p["rows"]])[list(ROW_COLUMNS)]
    if path.exists():
        existing = pd.read_csv(path)
        if list(existing.columns) != list(TABLE_COLUMNS):
            raise IdentityCounterfactualError(
                f"{path.name}: columns differ from this version of the tool; rerun with --all"
            )
        existing = existing[~existing["technique"].isin(new["technique"])][list(ROW_COLUMNS)]
        new = pd.concat([existing, new], ignore_index=True)
    new = new.sort_values(["technique", "y_true", "operation"]).reset_index(drop=True)
    # One Holm family for the comparison; each outside-comparison technique is its own family.
    family = new["technique"].map(lambda t: "" if in_comparison(t, results_dir) else t)
    new["wilcoxon_p_holm"] = np.nan
    new["mcnemar_p_holm"] = np.nan
    for _, index in new.groupby(family, sort=False).groups.items():
        new.loc[index, "wilcoxon_p_holm"] = _holm_column(new.loc[index, "wilcoxon_p"])
        new.loc[index, "mcnemar_p_holm"] = _holm_column(new.loc[index, "mcnemar_exact_p"])
    path.parent.mkdir(parents=True, exist_ok=True)
    new[list(TABLE_COLUMNS)].to_csv(path, index=False, float_format=f"%.{SIG_DIGITS}g")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run(techniques, processed, terms, args, substitute_profile=None):
    """Analyse every technique in memory. Raises on the first fatal error; writes nothing."""
    payloads = []
    for technique in techniques:
        try:
            probs = load_technique_probs(technique, SPLIT)
            population, pattern = build_population(technique, processed, probs, terms)
            counts = population_counts(population)
            if args.dry_run_population:
                print(f"{technique}: population {len(population)} "
                      f"(non_extremist {counts['non_extremist']}, extremist {counts['extremist']})")
                continue
            threshold = load_threshold(technique)
            model = load_technique(technique, EXTERNAL_ROOT / technique, device=args.device,
                                   resident=not args.sequential_checkpoints)
            payload = analyze(technique, model, population, pattern, threshold, len(terms),
                              device=recorded_device(model, args.device),
                              substitute_profile=substitute_profile)
        except (ProbsArtifactError, ModelLoadError) as error:
            if args.all:
                print(f"skip {technique}: {error}")
                continue
            raise IdentityCounterfactualError(str(error)) from error
        except IdentityFPRError as error:
            raise IdentityCounterfactualError(str(error)) from error
        payloads.append(payload)
        neg = next(r for r in payload["rows"] if r["y_true"] == 0 and r["operation"] == "remove")
        print(f"{technique}: verified (max |diff| {payload['verification']['max_abs_diff']}); "
              f"non-extremist remove: mean delta {neg['mean_delta']}, flagged "
              f"{neg['n_flagged_original']} -> {neg['n_flagged_counterfactual']}")
    return payloads


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="every technique with a test artifact")
    group.add_argument("--technique", action="append", help="repeatable")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sequential-checkpoints", action="store_true",
                        help="load pooled-ensemble checkpoints one at a time (resident=False)")
    parser.add_argument("--dry-run-population", action="store_true",
                        help="print population counts only; no model is loaded, nothing is written")
    args = parser.parse_args(argv)

    from split_protocol import build_processed_dataset

    lexicons = load_lexicons()
    terms = lexicons["identity_term"]
    if not terms:
        print("ERROR: identity lexicon is empty", file=sys.stderr)
        return 1
    processed = build_processed_dataset()
    try:
        check_substitute_word(lexicons)
        profile = check_substitute_profile(load_substitute_profile(processed))
        techniques = args.technique if args.technique else [d.name for d in technique_dirs()]
        payloads = run(techniques, processed, terms, args, substitute_profile=profile)
    except (IdentityCounterfactualError, AttributionArtifactError) as error:
        print(f"ERROR: {error}; nothing was written", file=sys.stderr)
        return 1

    if args.dry_run_population:
        return 0
    if not payloads:
        print("no technique could be evaluated", file=sys.stderr)
        return 1
    for payload in payloads:
        write_outputs(payload)
    print(f"wrote {len(payloads)} technique JSON(s) and {write_table(payloads)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
