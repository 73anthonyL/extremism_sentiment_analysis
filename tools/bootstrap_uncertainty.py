"""Post-level bootstrap intervals for the descriptive RQ2/RQ3 numbers and the RQ2-RQ4 relation.

The RQ2/RQ3 tables (`rq2_category_shares.csv`, `rq2_same_post_*.csv`,
`rq3_same_post_split_agreement.csv`) and the RQ2-RQ4 rank correlation are
each one number per technique from one attribution run per split. This tool
gives them a sampling interval. It does not test anything: every row it
writes is descriptive, no Holm family is added, and section 5 of
docs/RESEARCH_PLAN.md still decides what may be called a difference.

Method
    Nonparametric bootstrap over explained posts. A replicate draws, with
    replacement, as many posts as the post set holds and re-aggregates the
    per-post attributions exactly as the committed word tables are built
    (`notebook_kit.aggregate_word_attributions`: repeated words summed within
    a post, means over the posts containing the word, support = number of
    posts containing it). A post drawn twice counts twice, so a replicate is
    the committed aggregation applied to a list of posts with repeats. Every
    statistic is then recomputed with the committed definitions:

    category shares   `categorize_attributions.py`: top-K words by mean
                      absolute attribution, each word's positive mass
                      max(mean_attribution, 0) x support, share per lexicon
                      category (the four categories only).
    same-post         `same_post_stability.py`: share of the top-K words by
                      positive mean attribution that occur in one post, and
                      the fewest words holding half the positive mass of the
                      whole vocabulary. For the single-post share, "one post"
                      means one distinct original post (a post drawn twice is
                      still one post); the half-mass count uses the weighted
                      support, as the mass does.
    split agreement   test-run versus validation-run Spearman of mean
                      absolute attribution over shared words and top-K
                      Jaccard (absolute and positive); the two splits are
                      resampled independently.
    relation          Spearman of the identity-term share of positive mass
                      (test, own posts as in the primary row, and common
                      posts as in the sensitivity row) against the RQ4 FPR
                      ratio over the comparison techniques. The shares are
                      resampled over explained posts as above; the FPR ratio
                      is resampled over the 450 test rows of the committed
                      probability artifacts at each technique's locked
                      threshold, with identity matches found as
                      `identity_fpr.py` finds them. The two resamples are
                      independent, so the explained posts are not nested in
                      the resampled test rows; the coupling would change
                      little, since the replicate spread comes mainly from
                      the FPR ratios. A replicate with no false positive on
                      either side for some technique (ratio 0/0), or with an
                      empty group, has no ranking and is dropped
                      (n_reps_valid); one with false positives only on
                      identity-term posts ranks as +inf.

    Within a split, one draw over each distinct post set is shared by every
    technique that explained that set (the common posts are one draw for all
    techniques; the 450-post classical runs share one draw), and one draw of
    the test rows is shared by every technique's FPR ratio. That pairing is
    what makes the paired differences meaningful.

    The bootstrap covers post sampling only: not the SHAP seed, the
    explainer budget, the training seed or another split.

Identity check
    Before any replicate the tool rebuilds every statistic with weight one per
    post and refuses to go on unless it reproduces the committed tables:
    each per-post file must re-aggregate to its committed word table
    (`same_post_stability.load_whole_runs`), and the identity statistics must
    match `rq2_category_shares.csv` (own posts), `rq2_same_post_category_shares.csv`,
    `rq2_same_post_support.csv`, `rq3_same_post_split_agreement.csv`,
    `rq4_identity_fpr.csv` and the primary and common-post rows of
    `rq2_rq4_relation.csv`. The `point` column of every output is that
    identity value, so it is the committed number.

Outputs (results_summary/rq/, text-free: technique names, measures and numbers; no words, no row ids)
    rq2_bootstrap_category_shares.csv     technique x split x post_set x category
    rq2_bootstrap_same_post.csv           technique x split x post_set x measure
    rq3_bootstrap_split_agreement.csv     technique x post_set x measure
    rq2_rq4_bootstrap_relation.csv        relation rho (resampling shares and rows,
                                          shares only, rows only) and each
                                          technique's FPR ratio
    rq2_rq3_bootstrap_paired_differences.csv
                                          contrast technique minus each other
                                          technique on the common posts: identity
                                          share, single-post share, split Spearman.
                                          role descriptive_post_sampling_only: not a
                                          test, and not the section 5 difference
                                          criterion (which uses the across-run SD
                                          of rq3_stability.csv).
    Each row has point, interval_kind, boot_median, boot_mean, boot_sd,
    pct_low, pct_high (percentiles of the replicates), pct_high_censored,
    pct_level, point_inside_interval, share_reps_below_point, n_reps,
    n_reps_valid, n_reps_infinite, seed.

Reading the intervals (interval_kind)
    percentile_ci      Smooth statistics whose replicates centre on the
                       point: category shares, Spearman agreement, the
                       identity-share and Spearman differences, each
                       technique's FPR ratio. The percentile bounds are read
                       as an approximate confidence interval.
    replicate_spread   Statistics whose replicates do not centre on the point,
                       so the percentile bounds are NOT a confidence interval,
                       only the spread of the replicates. Top-K sets and
                       counts (Jaccard, the single-post share and its
                       difference, the half-mass count) shift under
                       resampling: a post drawn twice raises support, a post
                       left out removes rare words, the top-K reshuffles. The
                       relation rho is attenuated: resampling eight noisy FPR
                       ratios scrambles their ranking, so most replicates
                       fall below the point (share_reps_below_point). Its rows
                       are a sensitivity of the ranking to test-row and post
                       resampling; report the replicate median, the share
                       below the point and share_reps_rho_above_zero_not_a_p_value
                       (the share of replicates with rho > 0, which is not a
                       one-sided p-value), never the bounds as a 95% CI.

Refusals: everything `same_post_stability.load_whole_runs` refuses; a missing
committed table; identity statistics that do not reproduce a committed table;
probability artifacts that do not cover the same test rows; fewer than two
replicates; and any output cell that holds a row id or a text column.

USAGE
-----
    python3 tools/bootstrap_uncertainty.py                 # 2000 replicates, seed 30
    python3 tools/bootstrap_uncertainty.py --reps 500 --seed 7
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attributions import AttributionArtifactError
from categorize_attributions import build_word_index, categorize_word
from lexicons import load_lexicons
from notebook_kit import aggregate_word_attributions
from repo_paths import (
    CATEGORIES,
    POSITIVE_LABEL,
    RANDOM_SEED,
    RQ2_BOOTSTRAP_CATEGORY_SHARES_CSV,
    RQ2_BOOTSTRAP_SAME_POST_CSV,
    RQ2_CATEGORY_SHARES_CSV,
    RQ2_RQ3_BOOTSTRAP_PAIRED_DIFFERENCES_CSV,
    RQ2_RQ4_BOOTSTRAP_RELATION_CSV,
    RQ2_RQ4_RELATION_CSV,
    RQ2_SAME_POST_CATEGORY_SHARES_CSV,
    RQ2_SAME_POST_SUPPORT_CSV,
    RQ3_BOOTSTRAP_SPLIT_AGREEMENT_CSV,
    RQ3_SAME_POST_SPLIT_AGREEMENT_CSV,
    RQ4_IDENTITY_FPR_CSV,
    SPLIT_ASSIGNMENTS_CSV,
)
from same_post_stability import SamePostError, common_posts, load_whole_runs
from scan_text_leakage import ROW_ID_RE

DEFAULT_REPS = 2000
DEFAULT_TOP_K = 50
DEFAULT_CI = 0.95
SPLITS = ("test", "validation")
POST_SETS = ("own", "common")
IDENTITY_CATEGORY = "identity_term"
# The two transformer lines whose descriptive contrasts the plan cites.
DEFAULT_CONTRASTS = ("07_TWITTER-ROBERTA_FINE-TUNE", "11_MULTI-CHECKPOINT_LOGIT-POOL")

# Committed tables are written with %.8g (category shares) or %.6g (the rest).
SHARE_ATOL = 1e-6
SIX_G_RTOL = 2e-5
SIX_G_ATOL = 1e-9
WORD_RTOL = 1e-6
WORD_ATOL = 1e-8

SAME_POST_MEASURES = ("share_single_post_top_k_positive", "n_words_half_positive_mass")
AGREEMENT_MEASURES = ("spearman_mean_abs_attribution", "jaccard_top_k_abs", "jaccard_top_k_positive")
INTERVAL_KINDS = ("percentile_ci", "replicate_spread")
# Statistics whose replicates do not centre on the point: discrete top-K sets and counts.
SPREAD_MEASURES = ("share_single_post_top_k_positive", "n_words_half_positive_mass",
                   "jaccard_top_k_abs", "jaccard_top_k_positive")
DIFFERENCE_ROLE = "descriptive_post_sampling_only"
INTERVAL_COLUMNS = ("point", "interval_kind", "boot_median", "boot_mean", "boot_sd", "pct_low", "pct_high",
                    "pct_high_censored", "pct_level", "point_inside_interval", "share_reps_below_point",
                    "n_reps", "n_reps_valid", "n_reps_infinite", "seed")
FORBIDDEN_COLUMNS = ("row_id", "text", "text_hash", "token", "word", "position")


class BootstrapError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Per-post matrices
# ---------------------------------------------------------------------------
class PostMatrix:
    """One run's per-post word totals as sparse (words x posts) matrices.

    Each post is aggregated alone with the kit's own function, so word
    normalization and dropping are the kit's. `vocab` is sorted, so a word's
    index is its rank in the committed tie-break (word ascending).
    """

    def __init__(self, posts):
        self.row_ids = sorted(posts)
        self.position = {r: i for i, r in enumerate(self.row_ids)}
        per_post = []
        vocab = set()
        for row_id in self.row_ids:
            tokens, values = posts[row_id]
            frame = aggregate_word_attributions([tokens], [values])
            per_post.append(frame)
            vocab.update(frame["word"])
        self.vocab = np.array(sorted(vocab), dtype=object)
        index = {w: i for i, w in enumerate(self.vocab)}
        rows, cols, signed = [], [], []
        for j, frame in enumerate(per_post):
            rows.extend(index[w] for w in frame["word"])
            cols.extend([j] * len(frame))
            signed.extend(frame["mean_attribution"].tolist())
        shape = (len(self.vocab), len(self.row_ids))
        signed = np.asarray(signed, dtype=float)
        self.signed = sparse.csr_matrix((signed, (rows, cols)), shape=shape)
        self.absolute = sparse.csr_matrix((np.abs(signed), (rows, cols)), shape=shape)
        self.present = sparse.csr_matrix((np.ones(len(signed)), (rows, cols)), shape=shape)

    def weights(self, row_ids, counts):
        """A weight vector over this matrix's posts from counts over `row_ids` (others weight 0)."""
        w = np.zeros(len(self.row_ids))
        w[[self.position[r] for r in row_ids]] = counts
        return w

    def table(self, w):
        """The weighted word table: index into vocab, means, weighted and distinct support."""
        support = self.present @ w
        keep = np.flatnonzero(support > 0)
        support = support[keep]
        return WordTable(
            keep,
            (self.signed @ w)[keep] / support,
            (self.absolute @ w)[keep] / support,
            support,
            (self.present @ (w > 0).astype(float))[keep],
        )


class WordTable:
    def __init__(self, idx, mean, mean_abs, support, distinct):
        self.idx, self.mean, self.mean_abs, self.support, self.distinct = idx, mean, mean_abs, support, distinct

    def top_abs(self, k):
        # Committed order: mean_abs descending, word ascending; nlargest keeps the first.
        return np.lexsort((self.idx, -self.mean_abs))[:k]

    def top_positive(self, k):
        # nlargest(k, "mean_attribution") on the frame in committed order.
        return np.lexsort((self.idx, -self.mean_abs, -self.mean))[:k]

    def positive_mass(self):
        return np.clip(self.mean, 0.0, None) * self.support


def category_shares(table, category_of, top_k):
    """Share of the top-K positive mass per category, in CATEGORIES order (categorize_attributions)."""
    top = table.top_abs(top_k)
    mass = table.positive_mass()[top]
    total = float(mass.sum()) or 1.0
    cats = category_of[table.idx[top]]
    return np.array([mass[cats == c].sum() / total for c in range(len(CATEGORIES))])


def single_post_share(table, top_k):
    top = table.top_positive(top_k)
    return float((table.distinct[top] == 1).mean())


def half_mass_count(table):
    mass = table.positive_mass()
    mass = mass[mass > 0]
    total = float(mass.sum())
    if total <= 0:
        return 0.0
    share = np.sort(mass)[::-1] / total
    return float(np.searchsorted(np.cumsum(share), 0.5 - 1e-12) + 1)


def _spearman(a, b):
    if len(a) < 3:
        return np.nan
    ra, rb = rankdata(a), rankdata(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def agreement(matrix_t, table_t, matrix_v, table_v, top_k, vocab_map):
    """(Spearman of mean_abs over shared words, top-K Jaccard abs, top-K Jaccard positive)."""
    words_t, words_v = matrix_t.vocab[table_t.idx], matrix_v.vocab[table_v.idx]
    # Shared words: map the test vocabulary onto validation table positions.
    position_v = np.full(len(matrix_v.vocab), -1)
    position_v[table_v.idx] = np.arange(len(table_v.idx))
    mapped = position_v[np.where(vocab_map >= 0, vocab_map, 0)[table_t.idx]]
    mapped = np.where(vocab_map[table_t.idx] >= 0, mapped, -1)
    it = np.flatnonzero(mapped >= 0)
    rho = _spearman(table_t.mean_abs[it], table_v.mean_abs[mapped[it]])

    def jaccard(a, b):
        a, b = set(a), set(b)
        return len(a & b) / len(a | b)

    return (rho,
            jaccard(words_t[table_t.top_abs(top_k)], words_v[table_v.top_abs(top_k)]),
            jaccard(words_t[table_t.top_positive(top_k)], words_v[table_v.top_positive(top_k)]))


# ---------------------------------------------------------------------------
# FPR ratio inputs
# ---------------------------------------------------------------------------
def fpr_inputs(techniques, processed, terms, probs, thresholds):
    """Per technique (negative_with, negative_without, pred) arrays over one sorted test-row order.

    `probs` maps technique to its committed test artifact, `thresholds` to its
    locked threshold. Text stays in memory and is only used for matching.
    """
    from identity_fpr import identity_matches, merge_with_dataset

    order, out = None, {}
    for technique in techniques:
        merged = merge_with_dataset(technique, probs[technique], processed).sort_values("row_id")
        ids = merged["row_id"].astype(str).tolist()
        if order is None:
            order = ids
        elif ids != order:
            raise BootstrapError(f"{technique}: probability artifact covers other test rows than "
                                 f"{techniques[0]}; the shared row resample needs the same rows")
        has_identity = np.array([bool(m) for m in identity_matches(merged["text"].tolist(), terms)])
        negative = merged["y_true"].to_numpy() != POSITIVE_LABEL
        pred = (merged["y_prob"].to_numpy() >= float(thresholds[technique])).astype(float)
        out[technique] = ((negative & has_identity).astype(float), (negative & ~has_identity).astype(float), pred)
    return out, (len(order) if order else 0)


def fpr_ratios(inputs, counts):
    """FPR ratio per technique for each row of `counts` (reps x rows).

    No false positive without identity terms but some with them -> +inf; none on
    either side -> nan; a group with no negatives at all -> nan (never inf).
    """
    result = {}
    for technique, (with_id, without_id, pred) in inputs.items():
        n_with, n_without = counts @ with_id, counts @ without_id
        fp_with, fp_without = counts @ (with_id * pred), counts @ (without_id * pred)
        # (fp_with x n_without) / (n_with x fp_without): integer products divided once, so two
        # techniques with the same count ratio tie exactly, as they do in the committed table.
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (fp_with * n_without) / (n_with * fp_without)
        ratio = np.where((fp_without == 0) & (fp_with > 0), np.inf, ratio)
        ratio = np.where((fp_without == 0) & (fp_with == 0), np.nan, ratio)
        # An empty group has no rate: nan wins over inf.
        ratio = np.where((n_with == 0) | (n_without == 0), np.nan, ratio)
        result[technique] = ratio
    return result


def rank_rho(x, y):
    """Spearman over techniques; nan when either side has a nan."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.isnan(x).any() or np.isnan(y).any():
        return np.nan
    return _spearman(x, y)


# ---------------------------------------------------------------------------
# Identity check against the committed tables
# ---------------------------------------------------------------------------
def _close(a, b, rtol, atol):
    a, b = np.asarray(a, float), np.asarray(b, float)
    both_nan = np.isnan(a) & np.isnan(b)
    return bool(np.all(both_nan | np.isclose(a, b, rtol=rtol, atol=atol)))


def _require_table(committed, name):
    frame = None if committed is None else committed.get(name)
    if frame is None:
        raise BootstrapError(f"committed table {name} is missing; run its tool first")
    return frame


def verify_word_tables(runs, matrices, committed_frames):
    """The weight-one matrix table must be the committed word table, word for word."""
    for (technique, split), matrix in matrices.items():
        label = f"{technique}/{runs[technique][split][0]}"
        table = matrix.table(np.ones(len(matrix.row_ids)))
        rebuilt = pd.DataFrame({"word": matrix.vocab[table.idx], "mean_attribution": table.mean,
                                "mean_abs_attribution": table.mean_abs, "support": table.support})
        merged = rebuilt.merge(committed_frames[(technique, split)], on="word", how="outer",
                               suffixes=("_b", "_c"), indicator=True)
        if (merged["_merge"] != "both").any() or (merged["support_b"] != merged["support_c"]).any():
            raise BootstrapError(f"{label}: the per-post matrix does not reproduce the committed word table")
        for column in ("mean_attribution", "mean_abs_attribution"):
            if not _close(merged[f"{column}_b"], merged[f"{column}_c"], WORD_RTOL, WORD_ATOL):
                raise BootstrapError(f"{label}: the per-post matrix does not reproduce the committed {column}")


def verify_point(point, committed):
    """Refuse identity statistics that do not reproduce the committed RQ2/RQ3 tables."""
    shares = _require_table(committed, "rq2_category_shares")
    whole = shares["member"].isna() | shares["member"].astype(str).str.strip().eq("")
    shares = shares[whole]
    for (technique, split, post_set), values in point["categories"].items():
        if post_set != "own":
            continue
        rows = shares[(shares["technique"] == technique) & (shares["split"] == split)]
        rows = rows.set_index("category").reindex(list(CATEGORIES))
        if rows["share_of_positive_mass"].isna().any() or not np.allclose(
                values, rows["share_of_positive_mass"].to_numpy(float), atol=SHARE_ATOL, rtol=0):
            raise BootstrapError(f"{technique}/{split}: own-post category shares do not reproduce "
                                 "the committed rq2_category_shares.csv")

    same = _require_table(committed, "rq2_same_post_category_shares")
    for (technique, split, post_set), values in point["categories"].items():
        rows = same[(same["technique"] == technique) & (same["split"] == split) & (same["post_set"] == post_set)]
        rows = rows.set_index("category").reindex(list(CATEGORIES))
        if rows["share_of_positive_mass"].isna().any() or not np.allclose(
                values, rows["share_of_positive_mass"].to_numpy(float), atol=SHARE_ATOL, rtol=0):
            raise BootstrapError(f"{technique}/{split}/{post_set}: category shares do not reproduce "
                                 "the committed rq2_same_post_category_shares.csv")

    support = _require_table(committed, "rq2_same_post_support")
    for (technique, split, post_set), values in point["same_post"].items():
        rows = support[(support["technique"] == technique) & (support["split"] == split)
                       & (support["post_set"] == post_set)]
        if len(rows) != 1 or not _close(values, rows[list(SAME_POST_MEASURES)].to_numpy(float)[0],
                                        SIX_G_RTOL, SIX_G_ATOL):
            raise BootstrapError(f"{technique}/{split}/{post_set}: same-post measures do not reproduce "
                                 "the committed rq2_same_post_support.csv")

    agree = _require_table(committed, "rq3_same_post_split_agreement")
    for (technique, post_set), values in point["agreement"].items():
        rows = agree[(agree["technique"] == technique) & (agree["post_set"] == post_set)]
        if len(rows) != 1 or not _close(values, rows[list(AGREEMENT_MEASURES)].to_numpy(float)[0],
                                        SIX_G_RTOL, SIX_G_ATOL):
            raise BootstrapError(f"{technique}/{post_set}: split agreement does not reproduce "
                                 "the committed rq3_same_post_split_agreement.csv")


def verify_relation_point(ratios, rhos, committed, techniques):
    fpr = _require_table(committed, "rq4_identity_fpr")
    fpr = fpr[fpr["split"] == "test"].set_index("technique")
    for technique in techniques:
        if technique not in fpr.index or not _close([ratios[technique]], [fpr.loc[technique, "fpr_ratio"]],
                                                    SIX_G_RTOL, SIX_G_ATOL):
            raise BootstrapError(f"{technique}: the FPR ratio from the probability artifact does not reproduce "
                                 "the committed rq4_identity_fpr.csv")
    relation = _require_table(committed, "rq2_rq4_relation")
    for x_measure, rho in rhos.items():
        rows = relation[(relation["x_measure"] == x_measure) & (relation["y_measure"] == "fpr_ratio")
                        & (relation["subset"] == "all")]
        if len(rows) != 1 or not _close([rho], [rows["spearman_rho"].iloc[0]], SIX_G_RTOL, SIX_G_ATOL):
            raise BootstrapError(f"{x_measure} vs fpr_ratio: rho does not reproduce the committed "
                                 "rq2_rq4_relation.csv")


# ---------------------------------------------------------------------------
# The bootstrap
# ---------------------------------------------------------------------------
class Design:
    """Runs, matrices and the post sets each replicate resamples."""

    def __init__(self, runs, top_k, lexicons):
        self.runs = runs
        self.top_k = top_k
        self.techniques = sorted(runs)
        word_index, _ = build_word_index(lexicons)
        self.matrices = {(t, s): PostMatrix(runs[t][s][1]) for t in self.techniques for s in SPLITS
                         if s in runs[t]}
        self.category_of = {}
        for key, matrix in self.matrices.items():
            self.category_of[key] = np.array(
                [CATEGORIES.index(categorize_word(w, word_index)[0]) for w in matrix.vocab])
        # Test vocabulary index -> validation vocabulary index (-1 when absent).
        self.vocab_map = {}
        for t in self.techniques:
            if (t, "test") in self.matrices and (t, "validation") in self.matrices:
                index_v = {w: i for i, w in enumerate(self.matrices[(t, "validation")].vocab)}
                self.vocab_map[t] = np.array([index_v.get(w, -1) for w in self.matrices[(t, "test")].vocab])
        # Distinct post sets per split, in a fixed order: common first, then by size and ids.
        self.post_sets = {}
        self.common = {}
        for split in SPLITS:
            common = common_posts(runs, split)
            self.common[split] = None if common is None else tuple(sorted(common))
            sets = {tuple(sorted(runs[t][split][1])) for t in self.techniques if split in runs[t]}
            ordered = sorted(sets - {self.common[split]}, key=lambda s: (len(s), s))
            self.post_sets[split] = ([self.common[split]] if self.common[split] else []) + ordered

    def own_set(self, technique, split):
        return tuple(sorted(self.runs[technique][split][1]))

    def draw(self, rng, identity=False):
        """{(split, post_set_tuple): counts} for one replicate."""
        draws = {}
        for split in SPLITS:
            for ids in self.post_sets[split]:
                n = len(ids)
                draws[(split, ids)] = np.ones(n) if identity else rng.multinomial(n, np.full(n, 1.0 / n)).astype(float)
        return draws

    def statistics(self, draws):
        """Every statistic for one replicate."""
        tables = {}
        out = {"categories": {}, "same_post": {}, "agreement": {}}
        for (technique, split), matrix in self.matrices.items():
            for post_set in POST_SETS:
                ids = self.own_set(technique, split) if post_set == "own" else self.common[split]
                if ids is None:
                    continue
                table = matrix.table(matrix.weights(ids, draws[(split, ids)]))
                tables[(technique, split, post_set)] = table
                out["categories"][(technique, split, post_set)] = category_shares(
                    table, self.category_of[(technique, split)], self.top_k)
                out["same_post"][(technique, split, post_set)] = np.array(
                    [single_post_share(table, self.top_k), half_mass_count(table)])
        for technique in self.techniques:
            for post_set in POST_SETS:
                kt, kv = (technique, "test", post_set), (technique, "validation", post_set)
                if kt in tables and kv in tables:
                    out["agreement"][(technique, post_set)] = np.array(agreement(
                        self.matrices[(technique, "test")], tables[kt],
                        self.matrices[(technique, "validation")], tables[kv], self.top_k,
                        self.vocab_map[technique]))
        return out

    def n_posts(self, technique, split, post_set):
        return len(self.own_set(technique, split) if post_set == "own" else self.common[split])


def interval(point, samples, ci, seed, kind):
    """Summary of one statistic's replicates.

    `kind` is `percentile_ci` (a smooth statistic whose percentile interval is
    read as a confidence interval) or `replicate_spread` (a discrete or
    attenuated statistic whose replicates do not centre on the point; the
    bounds are the spread of the replicates, not a confidence interval).
    nan replicates are dropped (n_reps_valid). +inf replicates are kept and
    counted (n_reps_infinite); the bounds are order statistics, never
    interpolated across inf, and a bound that would be inf is replaced by the
    largest finite replicate with pct_high_censored true. boot_mean and
    boot_sd are empty when any replicate is infinite.
    """
    if kind not in INTERVAL_KINDS:
        raise BootstrapError(f"unknown interval kind {kind}")
    samples = np.asarray(samples, float)
    valid = np.sort(samples[~np.isnan(samples)])
    infinite = int(np.isinf(valid).sum())
    alpha = (1.0 - ci) / 2.0
    low = high = mean = sd = median = below = np.nan
    censored = False
    if len(valid):
        method = "linear" if infinite == 0 else "inverted_cdf"
        low, median, high = np.percentile(valid, [100 * alpha, 50.0, 100 * (1 - alpha)], method=method)
        finite = valid[np.isfinite(valid)]
        if np.isinf(high):
            high, censored = (float(finite.max()) if len(finite) else np.nan), True
        if np.isinf(low):
            low = float(finite.min()) if len(finite) else np.nan
        if infinite == 0:
            mean = float(np.mean(valid))
            sd = float(np.std(valid, ddof=1)) if len(valid) > 1 else np.nan
        below = float(np.mean(valid < point - 1e-12))
    # A censored upper bound stands for +inf, so anything above the low bound is inside.
    inside = bool(len(valid)) and bool(low - 1e-12 <= point and (censored or point <= high + 1e-12))
    return {"point": float(point), "interval_kind": kind, "boot_median": float(median), "boot_mean": mean,
            "boot_sd": sd, "pct_low": float(low), "pct_high": float(high), "pct_high_censored": censored,
            "pct_level": ci, "point_inside_interval": inside, "share_reps_below_point": below,
            "n_reps": int(len(samples)), "n_reps_valid": int(len(valid)), "n_reps_infinite": infinite,
            "seed": int(seed)}


def build(runs, lexicons, committed, top_k=DEFAULT_TOP_K, reps=DEFAULT_REPS, seed=RANDOM_SEED,
          ci=DEFAULT_CI, fpr=None, contrasts=DEFAULT_CONTRASTS, progress=None):
    """All five output tables. `fpr` is (inputs, n_rows) from fpr_inputs, or None to skip the relation."""
    if reps < 2:
        raise BootstrapError("need at least two replicates")
    if top_k < 1:
        raise BootstrapError("top_k must be positive")
    design = Design(runs, top_k, lexicons)
    verify_word_tables(runs, design.matrices, {
        key: committed["word_tables"][key] for key in design.matrices})
    point = design.statistics(design.draw(None, identity=True))
    verify_point(point, committed)

    attribution_rng, row_rng = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2))
    samples = {"categories": {}, "same_post": {}, "agreement": {}}
    start = time.time()
    for rep in range(reps):
        stats = design.statistics(design.draw(attribution_rng))
        for kind, values in stats.items():
            for key, value in values.items():
                samples[kind].setdefault(key, []).append(value)
        if progress and (rep + 1) % progress == 0:
            print(f"  {rep + 1}/{reps} replicates ({time.time() - start:.0f}s)", flush=True)
    samples = {kind: {k: np.vstack(v) for k, v in values.items()} for kind, values in samples.items()}

    category_rows, same_rows, agreement_rows = [], [], []
    for (technique, split, post_set), values in point["categories"].items():
        for c, category in enumerate(CATEGORIES):
            category_rows.append({"technique": technique, "split": split, "post_set": post_set,
                                  "n_posts": design.n_posts(technique, split, post_set), "top_k": top_k,
                                  "category": category, "measure": "share_of_positive_mass",
                                  **interval(values[c], samples["categories"][(technique, split, post_set)][:, c],
                                             ci, seed, "percentile_ci")})
    for (technique, split, post_set), values in point["same_post"].items():
        for m, measure in enumerate(SAME_POST_MEASURES):
            same_rows.append({"technique": technique, "split": split, "post_set": post_set,
                              "n_posts": design.n_posts(technique, split, post_set), "top_k": top_k,
                              "measure": measure,
                              **interval(values[m], samples["same_post"][(technique, split, post_set)][:, m],
                                         ci, seed, _kind(measure))})
    for (technique, post_set), values in point["agreement"].items():
        for m, measure in enumerate(AGREEMENT_MEASURES):
            agreement_rows.append({"technique": technique, "post_set": post_set,
                                   "n_posts_test": design.n_posts(technique, "test", post_set),
                                   "n_posts_validation": design.n_posts(technique, "validation", post_set),
                                   "top_k": top_k, "measure": measure,
                                   **interval(values[m], samples["agreement"][(technique, post_set)][:, m],
                                              ci, seed, _kind(measure))})

    differences = paired_differences(design, point, samples, contrasts, ci, seed)
    relation = relation_rows(design, point, samples, fpr, committed, reps, row_rng, ci, seed)
    return {
        "categories": pd.DataFrame(category_rows),
        "same_post": pd.DataFrame(same_rows),
        "agreement": pd.DataFrame(agreement_rows),
        "relation": relation,
        "differences": differences,
    }


def _kind(measure):
    return "replicate_spread" if measure in SPREAD_MEASURES else "percentile_ci"


def paired_differences(design, point, samples, contrasts, ci, seed):
    """Contrast technique minus each other technique on the common posts, same replicate draws."""
    identity = CATEGORIES.index(IDENTITY_CATEGORY)
    measures = (
        ("identity_share_of_positive_mass", "test", "categories", lambda k: (k[0], "test", "common"), identity),
        ("share_single_post_top_k_positive", "test", "same_post", lambda k: (k[0], "test", "common"), 0),
        ("spearman_mean_abs_attribution", "test_vs_validation", "agreement", lambda k: (k[0], "common"), 0),
    )
    rows = []
    contrasts = [c for c in contrasts if c in design.techniques]
    for contrast in contrasts:
        for reference in design.techniques:
            if reference == contrast or reference in contrasts and reference < contrast:
                continue
            for measure, split, kind, key, column in measures:
                ka, kb = key((contrast,)), key((reference,))
                if ka not in point[kind] or kb not in point[kind]:
                    continue
                diff = samples[kind][ka][:, column] - samples[kind][kb][:, column]
                rows.append({"measure": measure, "split": split, "post_set": "common",
                             "n_posts": design.n_posts(contrast, "test", "common"),
                             "top_k": design.top_k, "technique": contrast, "reference_technique": reference,
                             "role": DIFFERENCE_ROLE,
                             **interval(point[kind][ka][column] - point[kind][kb][column], diff, ci, seed,
                                        _kind(measure))})
    return pd.DataFrame(rows)


def relation_rows(design, point, samples, fpr, committed, reps, row_rng, ci, seed):
    """Relation rho under three resampling schemes, and each technique's FPR ratio."""
    if fpr is None:
        return pd.DataFrame()
    inputs, n_rows = fpr
    techniques = design.techniques
    missing = [t for t in techniques if t not in inputs]
    if missing:
        raise BootstrapError(f"no probability artifact inputs for {missing}")
    point_ratio = {t: float(v[0]) for t, v in fpr_ratios(inputs, np.ones((1, n_rows))).items()}
    identity = CATEGORIES.index(IDENTITY_CATEGORY)
    share_point = {
        post_set: np.array([point["categories"][(t, "test", post_set)][identity] for t in techniques])
        for post_set in POST_SETS}
    share_samples = {
        post_set: np.column_stack([samples["categories"][(t, "test", post_set)][:, identity] for t in techniques])
        for post_set in POST_SETS}
    x_names = {"own": "identity_share_positive_mass", "common": "identity_share_positive_mass_common_posts"}
    ratio_vector = np.array([point_ratio[t] for t in techniques])
    rhos = {x_names[p]: rank_rho(share_point[p], ratio_vector) for p in POST_SETS}
    verify_relation_point(point_ratio, rhos, committed, techniques)

    counts = row_rng.multinomial(n_rows, np.full(n_rows, 1.0 / n_rows), size=reps).astype(float)
    ratios = fpr_ratios(inputs, counts)
    ratio_samples = np.column_stack([ratios[t] for t in techniques])

    rows = []
    for post_set in POST_SETS:
        schemes = {
            "posts_and_rows": (share_samples[post_set], ratio_samples),
            "posts_only": (share_samples[post_set], np.tile(ratio_vector, (reps, 1))),
            "rows_only": (np.tile(share_point[post_set], (reps, 1)), ratio_samples),
        }
        for scheme, (xs, ys) in schemes.items():
            rho = np.array([rank_rho(x, y) for x, y in zip(xs, ys)])
            rows.append({"quantity": "spearman_rho", "x_measure": x_names[post_set], "y_measure": "fpr_ratio",
                         "technique": "", "resampled": scheme, "n_techniques": len(techniques),
                         # The share of replicates with rho > 0: a description of the spread, not a p-value.
                         "share_reps_rho_above_zero_not_a_p_value": float(np.mean(rho[~np.isnan(rho)] > 0)),
                         **interval(rhos[x_names[post_set]], rho, ci, seed, "replicate_spread")})
    for t, technique in enumerate(techniques):
        rows.append({"quantity": "fpr_ratio", "x_measure": "", "y_measure": "fpr_ratio", "technique": technique,
                     "resampled": "rows_only", "n_techniques": 1,
                     "share_reps_rho_above_zero_not_a_p_value": np.nan,
                     **interval(point_ratio[technique], ratio_samples[:, t], ci, seed, "percentile_ci")})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def check_text_free(frame, label):
    """Refuse a payload with a row id, a per-post or text column, or a word column."""
    bad = [c for c in frame.columns if str(c).lower() in FORBIDDEN_COLUMNS]
    if bad:
        raise BootstrapError(f"{label}: refuses column(s) {bad}; the bootstrap tables are numbers only")
    for column in frame.columns:
        if frame[column].dtype == object:
            hits = frame[column].astype(str).str.contains(ROW_ID_RE, regex=True)
            if hits.any():
                raise BootstrapError(f"{label}: column {column} holds a row id; refusing to write")


OUTPUT_PATHS = {
    "categories": RQ2_BOOTSTRAP_CATEGORY_SHARES_CSV,
    "same_post": RQ2_BOOTSTRAP_SAME_POST_CSV,
    "agreement": RQ3_BOOTSTRAP_SPLIT_AGREEMENT_CSV,
    "relation": RQ2_RQ4_BOOTSTRAP_RELATION_CSV,
    "differences": RQ2_RQ3_BOOTSTRAP_PAIRED_DIFFERENCES_CSV,
}


def write_tables(tables, paths=None):
    paths = OUTPUT_PATHS if paths is None else paths
    for name, frame in tables.items():
        check_text_free(frame, name)
    written = []
    for name, frame in tables.items():
        if frame.empty:
            continue
        path = Path(paths[name])
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, float_format="%.6g")
        written.append(path)
    return written


def load_committed(runs, results_dir=None):
    """The committed tables the identity check compares against."""
    from attributions import load_run

    committed = {"word_tables": {}}
    for technique, by_split in runs.items():
        for split, (run_id, _) in by_split.items():
            committed["word_tables"][(technique, split)] = load_run(technique, run_id, results_dir)[0]
    for name, path in (("rq2_category_shares", RQ2_CATEGORY_SHARES_CSV),
                       ("rq2_same_post_category_shares", RQ2_SAME_POST_CATEGORY_SHARES_CSV),
                       ("rq2_same_post_support", RQ2_SAME_POST_SUPPORT_CSV),
                       ("rq3_same_post_split_agreement", RQ3_SAME_POST_SPLIT_AGREEMENT_CSV),
                       ("rq4_identity_fpr", RQ4_IDENTITY_FPR_CSV),
                       ("rq2_rq4_relation", RQ2_RQ4_RELATION_CSV)):
        if Path(path).is_file():
            committed[name] = pd.read_csv(path)
    return committed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--ci", type=float, default=DEFAULT_CI)
    parser.add_argument("--contrasts", nargs="*", default=list(DEFAULT_CONTRASTS))
    args = parser.parse_args()
    try:
        from identity_fpr import load_threshold
        from probs_artifact import load_technique_probs
        from split_protocol import build_processed_dataset

        assignments = pd.read_csv(SPLIT_ASSIGNMENTS_CSV, usecols=["row_id", "split"], dtype=str)
        runs = load_whole_runs(assignments=assignments)
        committed = load_committed(runs)
        techniques = sorted(runs)
        lexicons = load_lexicons()
        fpr = fpr_inputs(techniques, build_processed_dataset(), lexicons[IDENTITY_CATEGORY],
                         {t: load_technique_probs(t, "test") for t in techniques},
                         {t: load_threshold(t) for t in techniques})
        print(f"{len(techniques)} techniques; {args.reps} replicates, seed {args.seed}")
        tables = build(runs, lexicons, committed, top_k=args.top_k, reps=args.reps, seed=args.seed,
                       ci=args.ci, fpr=fpr, contrasts=args.contrasts, progress=max(args.reps // 10, 1))
    except (AttributionArtifactError, SamePostError, BootstrapError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    for path in write_tables(tables):
        print(f"wrote {path}")
    relation = tables["relation"]
    for _, row in relation[relation["quantity"] == "spearman_rho"].iterrows():
        print(f"{row['x_measure']} vs fpr_ratio [{row['resampled']}]: rho {row['point']:.3f}, "
              f"replicate median {row['boot_median']:.3f}, {row['share_reps_below_point']:.0%} of replicates "
              f"below the point, rho > 0 in {row['share_reps_rho_above_zero_not_a_p_value']:.0%}; "
              f"replicate spread {row['pct_low']:.3f} to {row['pct_high']:.3f} (not a confidence interval) "
              f"({row['n_reps_valid']}/{row['n_reps']} valid replicates)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
