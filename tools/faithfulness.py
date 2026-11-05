"""RQ2 support: are the SHAP explanations faithful to the model they explain?

RQ2 compares which words each technique relies on, read from SHAP. That
comparison assumes the attributions are equally trustworthy for a linear
TF-IDF model, a character-level model and a transformer. This tool tests the
assumption with ERASER-style deletion (comprehensiveness; DeYoung et al.
2020): delete the words a post's own explanation ranks highest toward
EXTREMIST and measure how far the probability falls, against deleting the
same number of words at random from the same post.

Inputs, per technique:

* the per-post token attributions of the whole-model test run,
  external/<TECHNIQUE>/attributions_local/shap-partition_test_seed30.csv
  (row_id, position, token, attribution). The file must rebuild the committed
  word table results_summary/<TECHNIQUE>/attributions/<run_id>.csv with the
  kit's own aggregation (split_beeswarms.table_difference), its row ids must
  be rows of the committed test artifact, and each post's tokens, joined,
  must reproduce the dataset text exactly. Otherwise the technique is refused.
* its committed test probability artifact and locked threshold, and
* its predictor restored from external/<TECHNIQUE> by model_loaders.

The restored predictor scores the original texts of the explained posts in
the same call as the deletions and must agree with the committed artifact
within curated_examples.DEFAULT_TOLERANCE, or nothing is written.
p_original in every metric is the committed artifact probability.

Deletion. The text masker's tokens carry their trailing separator, so a post
is the concatenation of its tokens and deleting a token removes the word and
the separator after it. For k in K_VALUES the top-k set is the post's tokens
with a positive attribution, highest first (position breaks ties), cut to
m = min(k, number of positive tokens). Posts with no positive token are left
out of every row. The random baseline deletes m tokens drawn uniformly
without replacement from all of the post's tokens, once per seed of
RANDOM_SEEDS, with a generator seeded by (seed, k) and the posts in row-id
order, so it is reproducible. A post emptied by a deletion is scored as the
explainer's mask token, the input SHAP itself uses for a fully masked post.

Per technique x population x k:

* mean_drop_topk     mean of p_original - p_topk (comprehensiveness),
* mean_drop_random   the same for the random deletion, averaged over seeds,
* drop_minus_random  their difference, the faithfulness margin,
* mean_attributed_topk  mean summed SHAP value of the deleted tokens: the drop
                     additivity predicts if deleting a word acted like masking
                     it; drop_over_attributed_topk is mean_drop_topk over it
                     (mean_attributed_random: the same for the random tokens),
* flip_share_topk    share of posts flagged at the locked threshold that are
                     no longer flagged after the top-k deletion
                     (flip_share_random: expected share over the seeds),
* wilcoxon_p         paired two-sided Wilcoxon signed-rank test of the
                     per-post top-k drop against that post's mean random drop,
* wilcoxon_p_holm    Holm over every comparison row of the same population in
                     the cross-technique table; a technique outside the
                     comparison forms its own family per population.
                     `shared` is the primary family (32 rows). `explained`
                     is secondary: for the transformers it is the same posts,
                     so the two families overlap and are not corrected
                     jointly (64 tests in all).

Table-only columns, derived from a row's committed means (--rebuild-table
recomputes them without loading a model):

* drop_random_over_attributed_random  mean_drop_random over
                     mean_attributed_random: how much of the random drop the
                     SHAP values of the randomly deleted words already predict,
* drop_minus_random_flagged  the margin on posts flagged at the locked
                     threshold, the comparison least affected by calibration.

Populations: `explained` is every test post the technique explained (all 450
for the classical notebooks, a 200-post seeded sample for the transformers).
`shared` is the posts every technique in the comparison explained, which is
the population to compare techniques on: same posts, same labels. Within it
n_posts still varies by technique (196-200 in the committed run), because a
post with no positive token for that technique is left out.

Reading. Attributions are SHAP values of P(EXTREMIST) under word masking with
the mask token; deletion is a different perturbation (the word and its
separator vanish), so a gap between the two is part of what is measured.
A larger random drop does not by itself mean a model reacts to deletion
as such: read it next to mean_attributed_random. When attributions are
spread over many words, randomly chosen words carry more SHAP value and the
random drop is larger for that reason; only the part of the random drop
beyond what SHAP predicts is a reaction to deletion itself. The
drop-to-SHAP ratios (top-k and random) are descriptive; no cross-technique
test is run. A larger margin means the ranking finds words the model needs;
it does not show the model is right to need them.

The all-post means mix faithfulness with calibration. The drop and the margin
are both capped by p_original, so a model that puts most unflagged posts
near p = 0 has a small typical-post effect however faithful its ranking is.
Read median_paired_diff and share_topk_drop_gt_random with the mean, and
compare families on the flagged-post columns. random_drop_sd_over_seeds is
the spread of the random baseline over its seeds; at small k it is a large
fraction of the random drop itself.

Re-scoring frozen models at their locked thresholds is a perturbation
analysis, not a test evaluation: no threshold, model or configuration is
selected from it.

Text, tokens and row ids are held in memory only. Outputs carry counts,
means and p-values, and the writer refuses a payload carrying a row id.

Outputs:
    results_summary/<TECHNIQUE>/faithfulness_test.json
    results_summary/rq/rq2_faithfulness.csv

USAGE
-----
    python3 tools/faithfulness.py --technique 01_LOG-REG_TF-IDF
    python3 tools/faithfulness.py --technique 11_MULTI-CHECKPOINT_LOGIT-POOL \\
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

from curated_examples import DEFAULT_TOLERANCE  # noqa: E402
from identity_fpr import IdentityFPRError, load_threshold, merge_with_dataset  # noqa: E402
from mcnemar_pairs import holm_adjust  # noqa: E402
from model_loaders import ModelLoadError, load_technique, read_best_config  # noqa: E402
from notebook_kit import MASK_TOKEN  # noqa: E402
from probs_artifact import ProbsArtifactError, load_technique_probs  # noqa: E402
from split_beeswarms import (  # noqa: E402
    DEFAULT_TABLE_TOLERANCE,
    LOCAL_SUBDIR,
    SplitBeeswarmsError,
    read_posts,
    run_id,
    table_difference,
)
from repo_paths import (  # noqa: E402
    ATTRIBUTIONS_SUBDIR,
    FAITHFULNESS_FILENAME,
    RANDOM_SEED,
    REPO_ROOT,
    RESULTS_DIR,
    RQ2_FAITHFULNESS_CSV,
    comparison_technique_dirs,
    in_comparison,
)

SPLIT = "test"
K_VALUES = (1, 3, 5, 10)
N_RANDOM_SEEDS = 5
RANDOM_SEEDS = tuple(RANDOM_SEED + i for i in range(N_RANDOM_SEEDS))
POPULATIONS = ("explained", "shared")
EXTERNAL_ROOT = REPO_ROOT / "external"
TORCH_FAMILIES = ("transformer", "pooled_transformer_ensemble")
ROW_ID_RE = re.compile(r"\bex_\d{6}\b")
SIG_DIGITS = 6

ROW_COLUMNS = (
    "technique",
    "model_family",
    "split",
    "population",
    "k",
    "threshold",
    "n_posts_explained",
    "n_posts",
    "n_posts_fewer_than_k_positive",
    "mean_n_deleted",
    "mean_p_original",
    "mean_drop_topk",
    "mean_drop_random",
    "drop_minus_random",
    "median_paired_diff",
    "share_topk_drop_gt_random",
    "random_drop_sd_over_seeds",
    "mean_attributed_topk",
    "mean_attributed_random",
    "drop_over_attributed_topk",
    "n_flagged",
    "mean_drop_topk_flagged",
    "mean_drop_random_flagged",
    "flip_share_topk",
    "flip_share_random",
    "n_random_seeds",
    "wilcoxon_p",
    "verification_max_abs_diff",
)
# Table-only columns, pure functions of a row's committed means, so the table
# can be rebuilt from the committed JSONs without re-scoring any model.
DERIVED_COLUMNS = (
    "drop_random_over_attributed_random",
    "drop_minus_random_flagged",
)
HOLM_COLUMNS = ("wilcoxon_p_holm",)
TABLE_COLUMNS = ROW_COLUMNS + DERIVED_COLUMNS + HOLM_COLUMNS


class FaithfulnessError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Deletion sets and deleted text
# ---------------------------------------------------------------------------
def top_k_positive(values, k):
    """Positions of the at most k tokens with the largest positive attribution."""
    values = np.asarray(values, dtype=float)
    if k < 1:
        raise FaithfulnessError("k must be at least 1")
    positive = np.flatnonzero(values > 0)
    # Highest attribution first; the earlier position breaks a tie.
    order = positive[np.lexsort((positive, -values[positive]))]
    return order[:k]


def random_positions(n_tokens, m, rng):
    """m distinct token positions drawn uniformly from 0..n_tokens-1."""
    if m > n_tokens:
        raise FaithfulnessError(f"cannot draw {m} of {n_tokens} tokens")
    return np.sort(rng.choice(n_tokens, size=m, replace=False))


def delete_tokens(tokens, positions):
    """The post with the tokens at `positions` removed; an emptied post becomes the mask token."""
    drop = set(int(p) for p in positions)
    kept = "".join(token for i, token in enumerate(tokens) if i not in drop).strip()
    return kept if kept else MASK_TOKEN


# ---------------------------------------------------------------------------
# Posts
# ---------------------------------------------------------------------------
def local_path(technique, external_root=EXTERNAL_ROOT):
    return Path(external_root) / technique / LOCAL_SUBDIR / f"{run_id(SPLIT)}.csv"


def explained_row_ids(technique, external_root=EXTERNAL_ROOT):
    path = local_path(technique, external_root)
    if not path.is_file():
        raise FaithfulnessError(f"{technique}: no per-post attributions at {path}")
    return set(pd.read_csv(path, usecols=["row_id"])["row_id"].astype(str))


def shared_row_ids(external_root=EXTERNAL_ROOT, results_dir=None):
    """Posts every comparison technique explained; refuses when one technique's file is missing."""
    techniques = [d.name for d in comparison_technique_dirs(results_dir)]
    if not techniques:
        raise FaithfulnessError("no technique in the comparison")
    shared = None
    for technique in techniques:
        ids = explained_row_ids(technique, external_root)
        shared = ids if shared is None else shared & ids
    if not shared:
        raise FaithfulnessError("the comparison techniques share no explained post")
    return shared, techniques


def build_posts(technique, posts, processed, probs):
    """Join per-post tokens to text, label and committed probability (in memory only).

    Refuses row ids outside the committed artifact and token lists that do not
    concatenate to the dataset text, since deletion edits that concatenation.
    """
    if not posts:
        raise FaithfulnessError(f"{technique}: no explained post")
    merged = merge_with_dataset(technique, probs, processed).set_index("row_id")
    rows = []
    n_mismatch = 0
    for row_id, tokens, values in posts:
        if row_id not in merged.index:
            raise FaithfulnessError(
                f"{technique}: an explained post is not a row of the committed {SPLIT} artifact"
            )
        if len(tokens) != len(values) or not len(tokens):
            raise FaithfulnessError(f"{technique}: a post has no tokens or mismatched values")
        record = merged.loc[row_id]
        if "".join(tokens) != str(record["text"]):
            n_mismatch += 1
            continue
        rows.append({
            "row_id": row_id,
            "text": str(record["text"]),
            "tokens": list(tokens),
            "values": np.asarray(values, dtype=float),
            "y_true": int(record["y_true"]),
            "y_prob": float(record["y_prob"]),
        })
    if n_mismatch:
        raise FaithfulnessError(
            f"{technique}: the tokens of {n_mismatch} post(s) do not concatenate to the dataset "
            "text; the per-post file was explained on a different text"
        )
    return pd.DataFrame(rows).sort_values("row_id").reset_index(drop=True)


def load_posts(technique, processed, external_root=EXTERNAL_ROOT, results_dir=None,
               probs=None, tolerance=DEFAULT_TABLE_TOLERANCE):
    """Read and check the per-post file of the whole-model test run."""
    base = RESULTS_DIR if results_dir is None else Path(results_dir)
    label = f"{technique} {SPLIT}"
    try:
        posts = read_posts(local_path(technique, external_root), label)
        committed_path = base / technique / ATTRIBUTIONS_SUBDIR / f"{run_id(SPLIT)}.csv"
        if not committed_path.is_file():
            raise FaithfulnessError(f"{label}: no committed word table at {committed_path}")
        committed = pd.read_csv(committed_path, keep_default_na=False)
        difference = table_difference(posts, committed, label)
    except SplitBeeswarmsError as error:
        raise FaithfulnessError(str(error)) from error
    if difference > tolerance:
        raise FaithfulnessError(
            f"{label}: the per-post file does not reproduce the committed word table "
            f"(max |diff| {difference:.2g}, tolerance {tolerance:g})"
        )
    probs = load_technique_probs(technique, SPLIT) if probs is None else probs
    return build_posts(technique, posts, processed, probs), difference


# ---------------------------------------------------------------------------
# Scoring plan
# ---------------------------------------------------------------------------
def deletion_plan(posts, k_values=K_VALUES, seeds=RANDOM_SEEDS):
    """Texts to score: for each k, the top-k deletion and one random deletion per seed.

    Returns {k: {"m": array, "topk": [text], "random": [[text] per seed]}}. Posts
    with no positive token get m = 0 and their own text, and are dropped later.
    """
    plan = {}
    for k in k_values:
        m_values, topk_texts, topk_mass = [], [], []
        for tokens, values in zip(posts["tokens"], posts["values"]):
            chosen = top_k_positive(values, k)
            m_values.append(len(chosen))
            topk_texts.append(delete_tokens(tokens, chosen))
            topk_mass.append(float(np.asarray(values)[chosen].sum()))
        random_texts, random_mass = [], []
        for seed in seeds:
            rng = np.random.default_rng([int(seed), int(k)])
            texts, mass = [], []
            for tokens, values, m in zip(posts["tokens"], posts["values"], m_values):
                positions = random_positions(len(tokens), m, rng)
                texts.append(delete_tokens(tokens, positions))
                mass.append(float(np.asarray(values)[positions].sum()))
            random_texts.append(texts)
            random_mass.append(mass)
        plan[k] = {"m": np.asarray(m_values, dtype=int), "topk": topk_texts, "random": random_texts,
                   "topk_mass": np.asarray(topk_mass), "random_mass": np.asarray(random_mass)}
    return plan


def score_plan(model, posts, plan):
    """One predictor call over the originals and every deleted text (one call keeps
    the pooled ensemble from reloading its checkpoints per batch)."""
    texts = list(posts["text"])
    n = len(texts)
    for k in plan:
        texts += plan[k]["topk"]
        for batch in plan[k]["random"]:
            texts += batch
    scored = np.asarray(model.predict(texts), dtype=float)
    if scored.shape != (len(texts),):
        raise FaithfulnessError(f"predictor returned {scored.shape} for {len(texts)} texts")
    original, cursor = scored[:n], n
    out = {}
    for k in plan:
        topk = scored[cursor:cursor + n]
        cursor += n
        random = []
        for _ in plan[k]["random"]:
            random.append(scored[cursor:cursor + n])
            cursor += n
        out[k] = {"topk": topk, "random": np.vstack(random)}
    return original, out


def verify_original(technique, original, committed, tolerance=DEFAULT_TOLERANCE):
    differences = np.abs(np.asarray(original, dtype=float) - np.asarray(committed, dtype=float))
    result = {
        "n_rows": int(len(differences)),
        "max_abs_diff": float(differences.max()),
        "mean_abs_diff": float(differences.mean()),
        "tolerance": float(tolerance),
        "passed": bool(differences.max() <= tolerance),
    }
    if not result["passed"]:
        raise FaithfulnessError(
            f"{technique}: the restored model disagrees with the committed test probabilities "
            f"(max |diff| {result['max_abs_diff']:.4f} over {result['n_rows']} explained posts, "
            f"tolerance {tolerance}); its attributions do not describe this predictor"
        )
    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _mean(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()) if values.size else float("nan")


def deletion_metrics(p_original, p_topk, p_random, m, k, threshold, topk_mass=None, random_mass=None):
    """Comprehensiveness, random baseline, flips and the paired test for one k.

    `p_random` has one row per seed. Posts with m == 0 (no positive token) are
    excluded here. `topk_mass` / `random_mass` (one row per seed) are the summed
    SHAP values of the deleted tokens, the drop SHAP's additivity would predict
    if deleting a word acted like masking it.
    """
    keep = np.asarray(m) > 0
    p_original = np.asarray(p_original, dtype=float)[keep]
    p_topk = np.asarray(p_topk, dtype=float)[keep]
    p_random = np.asarray(p_random, dtype=float)[:, keep]
    m = np.asarray(m)[keep]
    n = int(keep.sum())
    topk_mass = np.full(len(keep), np.nan) if topk_mass is None else np.asarray(topk_mass, dtype=float)
    random_mass = np.full((1, len(keep)), np.nan) if random_mass is None else np.asarray(random_mass, dtype=float)
    attributed_topk = _mean(topk_mass[keep])
    attributed_random = _mean(random_mass[:, keep].mean(axis=0))

    drop_topk = p_original - p_topk
    drop_random_by_seed = p_original[None, :] - p_random
    drop_random = drop_random_by_seed.mean(axis=0)
    paired = drop_topk - drop_random
    flagged = p_original >= threshold
    n_flagged = int(flagged.sum())
    flip_topk = flagged & (p_topk < threshold)
    flip_random = flagged[None, :] & (p_random < threshold)

    if n == 0 or np.all(paired == 0):
        wilcoxon_p = float("nan")
    else:
        wilcoxon_p = float(wilcoxon(drop_topk, drop_random).pvalue)
    return {
        "k": int(k),
        "n_posts": n,
        "n_posts_fewer_than_k_positive": int((m < k).sum()),
        "mean_n_deleted": _mean(m),
        "mean_p_original": _mean(p_original),
        "mean_drop_topk": _mean(drop_topk),
        "mean_drop_random": _mean(drop_random),
        "drop_minus_random": _mean(paired),
        "median_paired_diff": float(np.median(paired)) if n else float("nan"),
        "share_topk_drop_gt_random": _mean(paired > 0),
        "random_drop_sd_over_seeds": (
            float(drop_random_by_seed.mean(axis=1).std(ddof=1)) if n and len(p_random) > 1 else float("nan")
        ),
        "mean_attributed_topk": attributed_topk,
        "mean_attributed_random": attributed_random,
        "drop_over_attributed_topk": (
            _mean(drop_topk) / attributed_topk if n and attributed_topk > 0 else float("nan")
        ),
        "n_flagged": n_flagged,
        "mean_drop_topk_flagged": _mean(drop_topk[flagged]),
        "mean_drop_random_flagged": _mean(drop_random[flagged]),
        "flip_share_topk": (float(flip_topk.sum()) / n_flagged) if n_flagged else float("nan"),
        "flip_share_random": (float(flip_random.sum()) / (n_flagged * len(p_random))) if n_flagged else float("nan"),
        "n_random_seeds": int(len(p_random)),
        "wilcoxon_p": wilcoxon_p,
    }


def _round(value):
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isnan(value):
            return None
        return float(f"{value:.{SIG_DIGITS}g}")
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round(v) for v in value]
    return value


def recorded_device(model, requested):
    if getattr(model, "family", None) not in TORCH_FAMILIES:
        return None
    if requested == "auto":
        from model_loaders import _torch_device

        return _torch_device(requested).type
    return requested


# ---------------------------------------------------------------------------
# One technique
# ---------------------------------------------------------------------------
def analyze(technique, model, posts, threshold, shared_ids, model_family=None, device=None,
            k_values=K_VALUES, seeds=RANDOM_SEEDS, tolerance=DEFAULT_TOLERANCE, table_diff=None):
    """Score every deletion, verify the originals, and build the rows."""
    restored_threshold = getattr(model, "threshold", None)
    if restored_threshold is not None and not math.isclose(float(restored_threshold), float(threshold)):
        raise FaithfulnessError(f"{technique}: restored threshold {restored_threshold} != committed {threshold}")
    if posts.empty:
        raise FaithfulnessError(f"{technique}: no explained post")
    plan = deletion_plan(posts, k_values, seeds)
    original, scored = score_plan(model, posts, plan)
    verification = verify_original(technique, original, posts["y_prob"], tolerance)

    p_original = posts["y_prob"].to_numpy(dtype=float)
    in_shared = posts["row_id"].isin(shared_ids).to_numpy()
    if not in_shared.any():
        raise FaithfulnessError(f"{technique}: none of its explained posts is in the shared population")
    masks = {"explained": np.ones(len(posts), dtype=bool), "shared": in_shared}

    rows = []
    for population in POPULATIONS:
        mask = masks[population]
        for k in k_values:
            row = {
                "technique": technique,
                "model_family": model_family,
                "split": SPLIT,
                "population": population,
                "threshold": float(threshold),
                "n_posts_explained": int(mask.sum()),
            }
            row.update(deletion_metrics(
                p_original[mask], scored[k]["topk"][mask], scored[k]["random"][:, mask],
                plan[k]["m"][mask], k, threshold,
                topk_mass=plan[k]["topk_mass"][mask], random_mass=plan[k]["random_mass"][:, mask],
            ))
            row["verification_max_abs_diff"] = verification["max_abs_diff"]
            rows.append({c: _round(row[c]) for c in ROW_COLUMNS})

    return {
        "technique": technique,
        "model_family": model_family,
        "split": SPLIT,
        "run_id": run_id(SPLIT),
        "threshold": float(threshold),
        "device": device,
        "k_values": list(k_values),
        "random_seeds": list(seeds),
        "deletion": "remove the token and its trailing separator; an emptied post is scored as the mask token",
        "top_k_rule": "tokens with positive attribution toward EXTREMIST, highest first, at most k",
        "random_rule": "same number of tokens drawn uniformly from all of the post's tokens, per seed",
        "mask_token": MASK_TOKEN,
        "n_posts_explained": int(len(posts)),
        "n_posts_shared": int(in_shared.sum()),
        "n_posts_no_positive_token": int((plan[k_values[0]]["m"] == 0).sum()),
        "n_texts_scored": int(len(posts) * (1 + len(k_values) * (1 + len(seeds)))),
        "p_original_source": "committed probability artifact",
        "word_table_max_diff": _round(table_diff) if table_diff is not None else None,
        "verification": _round(verification),
        "holm_adjusted_p": f"see {RQ2_FAITHFULNESS_CSV.name} (family: every comparison row of the same "
                           "population, or this technique's own rows of it when it is outside the comparison)",
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def check_text_free(payload):
    """Refuse a payload that carries a row id (the only per-post key in memory)."""
    serialized = json.dumps(payload, default=str)
    if ROW_ID_RE.search(serialized):
        raise FaithfulnessError(f"{payload.get('technique')}: the payload carries a row id; nothing written")
    for key in ("text", "tokens", "row_id", "values"):
        if key in payload:
            raise FaithfulnessError(f"{payload.get('technique')}: the payload carries per-post field '{key}'")
    return payload


def write_outputs(payload, results_dir=None):
    check_text_free(payload)
    base = RESULTS_DIR if results_dir is None else Path(results_dir)
    folder = base / payload["technique"]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / FAITHFULNESS_FILENAME.format(split=payload["split"])
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


def _ratio(numerator, denominator):
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    return (numerator / denominator).where(denominator > 0)


def derive_columns(table):
    """Add DERIVED_COLUMNS from a row's own means (ratio of means, like drop_over_attributed_topk)."""
    table = table.copy()
    table["drop_random_over_attributed_random"] = _ratio(table["mean_drop_random"], table["mean_attributed_random"])
    table["drop_minus_random_flagged"] = (
        pd.to_numeric(table["mean_drop_topk_flagged"], errors="coerce")
        - pd.to_numeric(table["mean_drop_random_flagged"], errors="coerce")
    )
    return table


def write_table(payloads, path=None, results_dir=None, replace=False):
    """Merge rows into the cross-technique table and recompute Holm per family.

    replace=True ignores any existing table (used by --rebuild-table, which
    passes every committed payload).
    """
    for payload in payloads:
        check_text_free(payload)
        missing = [c for row in payload["rows"] for c in ROW_COLUMNS if c not in row]
        if missing:
            raise FaithfulnessError(f"{payload.get('technique')}: rows lack {sorted(set(missing))}")
        if any(row["technique"] != payload.get("technique") for row in payload["rows"]):
            raise FaithfulnessError(f"{payload.get('technique')}: a row names another technique")
    path = Path(RQ2_FAITHFULNESS_CSV if path is None else path)
    new = pd.DataFrame([row for p in payloads for row in p["rows"]])[list(ROW_COLUMNS)]
    if path.exists() and not replace:
        # round_trip: pandas' default float parser can be one ulp off, and a Holm product that
        # lands on a rounding tie then prints differently from a rebuild from the JSONs.
        existing = pd.read_csv(path, float_precision="round_trip")
        if list(existing.columns) != list(TABLE_COLUMNS):
            raise FaithfulnessError(f"{path.name}: columns differ from this version of the tool; rerun every technique")
        existing = existing[~existing["technique"].isin(new["technique"])][list(ROW_COLUMNS)]
        new = pd.concat([existing, new], ignore_index=True)
    new = new.sort_values(["population", "technique", "k"]).reset_index(drop=True)
    owner = new["technique"].map(lambda t: "" if in_comparison(t, results_dir) else t)
    family = owner + "|" + new["population"].astype(str)
    new = derive_columns(new)
    new["wilcoxon_p_holm"] = np.nan
    for _, index in new.groupby(family, sort=False).groups.items():
        new.loc[index, "wilcoxon_p_holm"] = _holm_column(new.loc[index, "wilcoxon_p"])
    path.parent.mkdir(parents=True, exist_ok=True)
    new[list(TABLE_COLUMNS)].to_csv(path, index=False, float_format=f"%.{SIG_DIGITS}g")
    return path


def rebuild_table(results_dir=None, path=None):
    """Rewrite the table from every committed faithfulness JSON; no model is loaded."""
    base = RESULTS_DIR if results_dir is None else Path(results_dir)
    payloads = []
    for json_path in sorted(base.glob(f"*/{FAITHFULNESS_FILENAME.format(split=SPLIT)}")):
        with open(json_path) as handle:
            payload = json.load(handle)
        if payload.get("technique") != json_path.parent.name:
            raise FaithfulnessError(f"{json_path.parent.name}: its {json_path.name} names "
                                    f"technique {payload.get('technique')!r}")
        if payload.get("split") != SPLIT:
            raise FaithfulnessError(f"{json_path.parent.name}: {json_path.name} has split {payload.get('split')!r}")
        payloads.append(payload)
    if not payloads:
        raise FaithfulnessError(f"no {FAITHFULNESS_FILENAME.format(split=SPLIT)} under {base}")
    return write_table(payloads, path, results_dir, replace=True), len(payloads)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run(techniques, processed, args, external_root=EXTERNAL_ROOT):
    """Analyse every technique in memory. Raises on the first error; writes nothing."""
    shared_ids, shared_from = shared_row_ids(external_root)
    print(f"shared population: {len(shared_ids)} test posts explained by all {len(shared_from)} comparison techniques")
    payloads = []
    for technique in techniques:
        try:
            posts, table_diff = load_posts(technique, processed, external_root)
            threshold = load_threshold(technique)
            family = read_best_config(technique).get("model_family")
            model = load_technique(technique, Path(external_root) / technique, device=args.device,
                                   resident=not args.sequential_checkpoints)
            payload = analyze(technique, model, posts, threshold, shared_ids, model_family=family,
                              device=recorded_device(model, args.device), table_diff=table_diff)
        except (ProbsArtifactError, ModelLoadError, IdentityFPRError) as error:
            raise FaithfulnessError(f"{technique}: {error}") from error
        payload["shared_population_from"] = shared_from
        payloads.append(payload)
        for row in payload["rows"]:
            if row["population"] == "shared":
                print(f"{technique} k={row['k']}: drop top-k {row['mean_drop_topk']} vs random "
                      f"{row['mean_drop_random']} (n={row['n_posts']}); flips {row['flip_share_topk']}")
        print(f"{technique}: verified (max |diff| {payload['verification']['max_abs_diff']})")
    return payloads


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--technique", action="append",
                        help="repeatable; run one model at a time on a small machine")
    parser.add_argument("--rebuild-table", action="store_true",
                        help="rewrite the cross-technique table from the committed JSONs; loads no model")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sequential-checkpoints", action="store_true",
                        help="load pooled-ensemble checkpoints one at a time (resident=False)")
    args = parser.parse_args(argv)
    if args.rebuild_table:
        if args.technique:
            parser.error("--rebuild-table takes no --technique")
        try:
            table, n = rebuild_table()
        except FaithfulnessError as error:
            print(f"ERROR: {error}; nothing was written", file=sys.stderr)
            return 1
        print(f"rebuilt {table} from {n} technique JSON(s)")
        return 0
    if not args.technique:
        parser.error("--technique is required unless --rebuild-table is given")

    from split_protocol import build_processed_dataset

    processed = build_processed_dataset()
    try:
        payloads = run(args.technique, processed, args)
        for payload in payloads:
            check_text_free(payload)
        for payload in payloads:
            write_outputs(payload)
        table = write_table(payloads)
    except FaithfulnessError as error:
        print(f"ERROR: {error}; nothing was written", file=sys.stderr)
        return 1
    print(f"wrote {len(payloads)} technique JSON(s) and {table}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
