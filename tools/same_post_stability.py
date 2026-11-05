"""RQ2/RQ3: explanation stability on the same posts.

The classical techniques' attribution runs explain all 450 posts of a split;
the transformer runs explain a 200-post stratified sample of it. The
lexicon-free support table (`rq2_top_word_support.csv`) therefore compares
top-word support over different post sets: a word explained in 200 posts is
more likely to be seen in only one of them. This tool removes that
difference. It reads each technique's per-post attribution file from
`external/<TECHNIQUE>/attributions_local/<run_id>.csv` (row_id, position,
token, attribution; never committed), re-aggregates it with the notebook kit's
own aggregation, and refuses to go on unless the re-aggregate reproduces the
committed word table. It then restricts every technique to the common post
set, the posts every technique in the comparison explained (for the current
runs, the transformers' 200-post samples), and recomputes:

rq2_same_post_support.csv
    One row per technique x split x post_set (`own`: every post that
    technique explained; `common`: the shared post set). Vocabulary size, the
    median support of the top-K words by positive mean attribution and the
    share of them seen in one post (the same definitions as
    `word_agreement.py`), and the concentration of positive attribution mass
    over the whole vocabulary: a word's positive mass is
    max(mean_attribution, 0) x support, as in `categorize_attributions.py`;
    `n_words_half_positive_mass` is the fewest words that hold half of it,
    `positive_mass_entropy_normalized` its Shannon entropy divided by
    log(number of words with positive mass), `effective_n_words_positive_mass`
    exp(entropy), and `share_positive_mass_top_k` the share held by the top-K.

rq3_same_post_split_agreement.csv
    One row per technique x post_set: agreement of the test-split and
    validation-split word tables (different posts, same model), top-K
    Jaccard by positive and by absolute mean attribution and Spearman of mean
    absolute attribution over the shared words. With `common`, both splits are
    restricted to the common post set of that split, so every technique is
    compared on the same number of posts.

rq2_same_post_category_shares.csv
    One row per technique x split x post_set x category: the RQ2 category
    shares of the top-K words by mean absolute attribution, computed with
    `categorize_attributions.py`'s own functions and the committed lexicons.
    The `own` rows must reproduce the committed `rq2_category_shares.csv`
    (refused otherwise); the `common` rows give every technique's shares on
    the same posts, so the RQ2-RQ4 relation can be checked without mixing
    450-post and 200-post runs.

All three tables are text-free: counts, shares and correlations only, no
words, no row ids. The tool prints numbers only.

Refusals: a whole-model run without its per-post file; a per-post file with
other columns, non-finite attributions, duplicate or gapped positions, a
post count different from the sidecar's n_posts_explained, row ids outside
the run's split, or a re-aggregate that does not match the committed table;
more than one whole-model run for a technique and split; own-post category
shares that do not reproduce the committed RQ2 table; and post samples that
are not nested (the common set must be the whole of the
smallest technique's sample, so no technique is trimmed below what it
explained without that being visible). Member runs of an ensemble are not
read.

USAGE
-----
    python3 tools/same_post_stability.py
    python3 tools/same_post_stability.py --top-k 50 --techniques 01_LOG-REG_TF-IDF 07_TWITTER-ROBERTA_FINE-TUNE
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attributions import AttributionArtifactError, list_runs, load_run
from categorize_attributions import build_word_index, categorize_frame, category_shares
from lexicons import load_lexicons
from notebook_kit import aggregate_word_attributions
from repo_paths import (
    EXTERNAL_DIR,
    RQ2_CATEGORY_SHARES_CSV,
    RQ2_SAME_POST_CATEGORY_SHARES_CSV,
    RQ2_SAME_POST_SUPPORT_CSV,
    RQ3_SAME_POST_SPLIT_AGREEMENT_CSV,
    SPLIT_ASSIGNMENTS_CSV,
    technique_dirs,
    in_comparison,
)

DEFAULT_TOP_K = 50
SPLITS = ("test", "validation")
LOCAL_SUBDIR = "attributions_local"
LOCAL_COLUMNS = ("row_id", "position", "token", "attribution")
POST_SETS = ("own", "common")
# The committed table is written with float_format="%.8g".
MATCH_RTOL = 1e-6
MATCH_ATOL = 1e-8

SUPPORT_COLUMNS = (
    "technique",
    "run_id",
    "split",
    "post_set",
    "n_posts",
    "top_k",
    "n_words",
    "median_support_top_k_positive",
    "share_single_post_top_k_positive",
    "share_single_post_vocabulary",
    "n_words_positive_mass",
    "n_words_half_positive_mass",
    "positive_mass_entropy_normalized",
    "effective_n_words_positive_mass",
    "share_positive_mass_top_k",
)

CATEGORY_COLUMNS = (
    "technique",
    "run_id",
    "split",
    "post_set",
    "n_posts",
    "top_k",
    "category",
    "n_words",
    "share_of_words",
    "abs_mass",
    "share_of_abs_mass",
    "positive_mass",
    "share_of_positive_mass",
)
# Own-post category rows are checked against the committed RQ2 table, written with "%.8g".
CATEGORY_MATCH_RTOL = 1e-6
CATEGORY_MATCH_ATOL = 1e-9

AGREEMENT_COLUMNS = (
    "technique",
    "post_set",
    "run_id_test",
    "run_id_validation",
    "n_posts_test",
    "n_posts_validation",
    "top_k",
    "n_shared_words",
    "jaccard_top_k_positive",
    "jaccard_top_k_abs",
    "spearman_mean_abs_attribution",
)


class SamePostError(RuntimeError):
    pass


def _is_member(value):
    return not (value is None or (isinstance(value, float) and np.isnan(value)) or value == "")


def local_path(technique, run_id, external_dir=None):
    external_dir = EXTERNAL_DIR if external_dir is None else Path(external_dir)
    return external_dir / technique / LOCAL_SUBDIR / f"{run_id}.csv"


def read_local(path, label):
    """Read and validate one per-post attribution file. Returns {row_id: (tokens, values)}."""
    path = Path(path)
    if not path.is_file():
        raise SamePostError(f"{label}: per-post file not found at {path}; download it to external/ first")
    # keep_default_na=False: a token such as "nan" or "null" is a word, not a missing value.
    frame = pd.read_csv(path, dtype={"row_id": str, "token": str}, keep_default_na=False)
    if tuple(frame.columns) != LOCAL_COLUMNS:
        raise SamePostError(f"{label}: columns are {list(frame.columns)}, expected {list(LOCAL_COLUMNS)}")
    values = pd.to_numeric(frame["attribution"], errors="coerce")
    if not np.isfinite(values).all():
        raise SamePostError(f"{label}: non-finite or non-numeric attribution values")
    positions = pd.to_numeric(frame["position"], errors="coerce")
    if positions.isna().any() or (positions != positions.round()).any():
        raise SamePostError(f"{label}: non-integer positions")
    frame = frame.assign(attribution=values.astype(float), position=positions.astype(int))
    posts = {}
    for row_id, group in frame.groupby("row_id", sort=False):
        group = group.sort_values("position")
        if group["position"].tolist() != list(range(len(group))):
            raise SamePostError(f"{label}: a post has duplicate or missing token positions")
        posts[row_id] = (group["token"].tolist(), group["attribution"].to_numpy())
    return posts


def aggregate(posts, row_ids=None):
    """The committed word-level table over the given posts (all when row_ids is None)."""
    keys = list(posts) if row_ids is None else [r for r in posts if r in row_ids]
    return aggregate_word_attributions([posts[k][0] for k in keys], [posts[k][1] for k in keys])


def verify_against_committed(posts, committed, label):
    """Refuse a per-post file whose re-aggregate is not the committed word table."""
    rebuilt = aggregate(posts)
    merged = rebuilt.merge(committed, on="word", how="outer", suffixes=("_local", "_committed"), indicator=True)
    if (merged["_merge"] != "both").any():
        raise SamePostError(
            f"{label}: per-post file and committed table have different vocabularies "
            f"({int((merged['_merge'] != 'both').sum())} words differ)"
        )
    if (merged["support_local"] != merged["support_committed"]).any():
        raise SamePostError(f"{label}: per-post file and committed table disagree on word support")
    for column in ("mean_attribution", "mean_abs_attribution"):
        if not np.allclose(merged[f"{column}_local"], merged[f"{column}_committed"], rtol=MATCH_RTOL, atol=MATCH_ATOL):
            raise SamePostError(f"{label}: per-post file does not reproduce the committed {column}")


def load_whole_runs(techniques=None, results_dir=None, external_dir=None, assignments=None):
    """Return {technique: {split: (run_id, posts)}} for the whole-model runs, verified.

    `assignments` is the split assignment table (row_id, split); when given,
    every per-post row id must belong to the run's split.
    """
    folders = technique_dirs(results_dir)
    if techniques:
        wanted = set(techniques)
        unknown = wanted - {f.name for f in folders}
        if unknown:
            raise SamePostError(f"unknown techniques {sorted(unknown)}")
        folders = [f for f in folders if f.name in wanted]
    else:
        folders = [f for f in folders if in_comparison(f.name, results_dir)]
    split_of = None
    if assignments is not None:
        split_of = dict(zip(assignments["row_id"].astype(str), assignments["split"].astype(str)))
    runs = {}
    for folder in folders:
        technique = folder.name
        for run_id in list_runs(technique, results_dir):
            committed, meta = load_run(technique, run_id, results_dir)
            if _is_member(meta.get("member")) or meta["split"] not in SPLITS:
                continue
            split = meta["split"]
            if split in runs.get(technique, {}):
                raise SamePostError(
                    f"{technique}: more than one whole-model {split} run "
                    f"({runs[technique][split][0]}, {run_id}); the same-post tables use exactly one"
                )
            label = f"{technique}/{run_id}"
            posts = read_local(local_path(technique, run_id, external_dir), label)
            if len(posts) != int(meta["n_posts_explained"]):
                raise SamePostError(
                    f"{label}: per-post file has {len(posts)} posts, sidecar says "
                    f"n_posts_explained={meta['n_posts_explained']}"
                )
            if split_of is not None:
                outside = sum(1 for r in posts if split_of.get(r) != split)
                if outside:
                    raise SamePostError(f"{label}: {outside} per-post row ids are not in the {split} split")
            verify_against_committed(posts, committed, label)
            runs.setdefault(technique, {})[split] = (run_id, posts)
    if len(runs) < 2:
        raise SamePostError("fewer than two techniques have a whole-model run with a per-post file")
    return runs


def common_posts(runs, split):
    """The posts every technique explained for `split`; refused unless the samples are nested."""
    sets = {t: set(by_split[split][1]) for t, by_split in runs.items() if split in by_split}
    if len(sets) < 2:
        return None
    common = set.intersection(*sets.values())
    smallest = min(len(s) for s in sets.values())
    if len(common) != smallest:
        raise SamePostError(
            f"{split}: post samples are not nested; {len(common)} posts are shared but the "
            f"smallest sample has {smallest}. The same-post comparison needs every sample to "
            "contain the smallest one."
        )
    return common


def _top_positive(frame, k):
    return frame.nlargest(k, "mean_attribution")


def _top_abs(frame, k):
    return frame.nlargest(k, "mean_abs_attribution")


def concentration(frame, top_k):
    """Positive-mass concentration over the whole vocabulary."""
    mass = frame["mean_attribution"].clip(lower=0.0).to_numpy() * frame["support"].to_numpy()
    mass = mass[mass > 0]
    total = float(mass.sum())
    if total <= 0:
        return {"n_words_positive_mass": 0, "n_words_half_positive_mass": 0,
                "positive_mass_entropy_normalized": np.nan, "effective_n_words_positive_mass": np.nan,
                "share_positive_mass_top_k": np.nan}
    ordered = np.sort(mass)[::-1]
    share = ordered / total
    half = int(np.searchsorted(np.cumsum(share), 0.5 - 1e-12) + 1)
    entropy = float(-(share * np.log(share)).sum())
    normalized = entropy / np.log(len(share)) if len(share) > 1 else np.nan
    top = _top_positive(frame, top_k)
    top_mass = float((top["mean_attribution"].clip(lower=0.0) * top["support"]).sum())
    return {
        "n_words_positive_mass": int(len(share)),
        "n_words_half_positive_mass": half,
        "positive_mass_entropy_normalized": normalized,
        "effective_n_words_positive_mass": float(np.exp(entropy)),
        "share_positive_mass_top_k": top_mass / total,
    }


def support_row(technique, run_id, split, post_set, n_posts, frame, top_k):
    top = _top_positive(frame, top_k)
    row = {
        "technique": technique,
        "run_id": run_id,
        "split": split,
        "post_set": post_set,
        "n_posts": int(n_posts),
        "top_k": int(top_k),
        "n_words": int(len(frame)),
        "median_support_top_k_positive": float(top["support"].median()),
        "share_single_post_top_k_positive": float((top["support"] == 1).mean()),
        "share_single_post_vocabulary": float((frame["support"] == 1).mean()),
    }
    row.update(concentration(frame, top_k))
    return row


def agreement_row(technique, post_set, run_test, run_val, frame_test, frame_val, n_test, n_val, top_k):
    pos_t, pos_v = set(_top_positive(frame_test, top_k)["word"]), set(_top_positive(frame_val, top_k)["word"])
    abs_t, abs_v = set(_top_abs(frame_test, top_k)["word"]), set(_top_abs(frame_val, top_k)["word"])
    merged = frame_test.merge(frame_val, on="word", suffixes=("_t", "_v"))
    rho = np.nan
    if len(merged) >= 3:
        rho = float(spearmanr(merged["mean_abs_attribution_t"], merged["mean_abs_attribution_v"])[0])
    return {
        "technique": technique,
        "post_set": post_set,
        "run_id_test": run_test,
        "run_id_validation": run_val,
        "n_posts_test": int(n_test),
        "n_posts_validation": int(n_val),
        "top_k": int(top_k),
        "n_shared_words": int(len(merged)),
        "jaccard_top_k_positive": len(pos_t & pos_v) / len(pos_t | pos_v),
        "jaccard_top_k_abs": len(abs_t & abs_v) / len(abs_t | abs_v),
        "spearman_mean_abs_attribution": rho,
    }


def category_rows(technique, run_id, split, post_set, n_posts, frame, word_index, top_k):
    shares = category_shares(categorize_frame(frame, word_index, top_k))
    shares.insert(0, "top_k", int(top_k))
    shares.insert(0, "n_posts", int(n_posts))
    shares.insert(0, "post_set", post_set)
    shares.insert(0, "split", split)
    shares.insert(0, "run_id", run_id)
    shares.insert(0, "technique", technique)
    return shares[list(CATEGORY_COLUMNS)]


def verify_own_categories(categories, committed):
    """Refuse own-post category shares that differ from the committed RQ2 table for the same run."""
    if committed is None:
        return
    keys = ["technique", "run_id", "top_k", "category"]
    needed = keys + ["n_words", "share_of_abs_mass", "share_of_positive_mass"]
    missing = [c for c in needed if c not in committed.columns]
    if missing:
        raise SamePostError(f"rq2_category_shares: missing columns {missing}")
    own = categories[categories["post_set"] == "own"]
    merged = own.merge(committed[keys + ["share_of_positive_mass", "share_of_abs_mass", "n_words"]],
                       on=keys, how="left", suffixes=("", "_committed"), indicator=True)
    present = merged[merged["_merge"] == "both"]
    for column in ("share_of_positive_mass", "share_of_abs_mass"):
        if not np.allclose(present[column], present[f"{column}_committed"],
                           rtol=CATEGORY_MATCH_RTOL, atol=CATEGORY_MATCH_ATOL):
            raise SamePostError(f"own-post {column} does not reproduce the committed rq2_category_shares.csv")
    if (present["n_words"] != present["n_words_committed"]).any():
        raise SamePostError("own-post category word counts do not reproduce the committed rq2_category_shares.csv")


def build(top_k=DEFAULT_TOP_K, techniques=None, results_dir=None, external_dir=None, assignments=None):
    """The support and split-agreement tables (see build_all for the category table too)."""
    support, agreement, _ = build_all(top_k, techniques, results_dir, external_dir, assignments)
    return support, agreement


def build_all(top_k=DEFAULT_TOP_K, techniques=None, results_dir=None, external_dir=None, assignments=None,
              lexicons=None, committed_shares=None):
    """Support, split-agreement and category-share tables.

    `committed_shares` is the committed rq2_category_shares table; when given,
    the own-post category rows must reproduce it.
    """
    if top_k < 1:
        raise SamePostError("top_k must be positive")
    word_index, _ = build_word_index(load_lexicons() if lexicons is None else lexicons)
    runs = load_whole_runs(techniques, results_dir, external_dir, assignments)
    common = {split: common_posts(runs, split) for split in SPLITS}
    support_rows, agreement_rows, category_tables = [], [], []
    for technique in sorted(runs):
        frames = {}
        for split in SPLITS:
            if split not in runs[technique]:
                continue
            run_id, posts = runs[technique][split]
            frames[("own", split)] = (run_id, len(posts), aggregate(posts))
            if common[split] is not None:
                frames[("common", split)] = (run_id, len(common[split]), aggregate(posts, common[split]))
        for post_set in POST_SETS:
            for split in SPLITS:
                if (post_set, split) in frames:
                    run_id, n_posts, frame = frames[(post_set, split)]
                    support_rows.append(support_row(technique, run_id, split, post_set, n_posts, frame, top_k))
                    category_tables.append(
                        category_rows(technique, run_id, split, post_set, n_posts, frame, word_index, top_k))
            if (post_set, "test") in frames and (post_set, "validation") in frames:
                rt, nt, ft = frames[(post_set, "test")]
                rv, nv, fv = frames[(post_set, "validation")]
                agreement_rows.append(agreement_row(technique, post_set, rt, rv, ft, fv, nt, nv, top_k))
    support = pd.DataFrame(support_rows, columns=list(SUPPORT_COLUMNS))
    agreement = pd.DataFrame(agreement_rows, columns=list(AGREEMENT_COLUMNS))
    categories = (pd.concat(category_tables, ignore_index=True) if category_tables
                  else pd.DataFrame(columns=list(CATEGORY_COLUMNS)))
    verify_own_categories(categories, committed_shares)
    return support, agreement, categories


def write_tables(support, agreement, categories=None, support_path=RQ2_SAME_POST_SUPPORT_CSV,
                 agreement_path=RQ3_SAME_POST_SPLIT_AGREEMENT_CSV,
                 categories_path=RQ2_SAME_POST_CATEGORY_SHARES_CSV):
    tables = [(support, support_path), (agreement, agreement_path)]
    if categories is not None:
        tables.append((categories, categories_path))
    for table, path in tables:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False, float_format="%.6g" if table is not categories else "%.8g")
    return [Path(path) for _, path in tables]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--techniques", nargs="*", default=None, help="restrict to these technique names")
    args = parser.parse_args()
    assignments = pd.read_csv(SPLIT_ASSIGNMENTS_CSV, usecols=["row_id", "split"], dtype=str)
    committed = pd.read_csv(RQ2_CATEGORY_SHARES_CSV) if Path(RQ2_CATEGORY_SHARES_CSV).is_file() else None
    try:
        support, agreement, categories = build_all(args.top_k, args.techniques, assignments=assignments,
                                                   committed_shares=committed)
    except (AttributionArtifactError, SamePostError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    paths = write_tables(support, agreement, categories)
    test = support[support["split"] == "test"]
    for _, row in test.iterrows():
        print(f"{row['technique']} [{row['post_set']}, {row['n_posts']} test posts]: "
              f"{row['share_single_post_top_k_positive']:.0%} of top-{args.top_k} positive words in one post; "
              f"{row['n_words_half_positive_mass']} words hold half the positive mass "
              f"(normalized entropy {row['positive_mass_entropy_normalized']:.3f})")
    for _, row in agreement.iterrows():
        print(f"{row['technique']} [{row['post_set']}]: test vs validation top-{args.top_k} Jaccard "
              f"positive {row['jaccard_top_k_positive']:.3f}, abs {row['jaccard_top_k_abs']:.3f}")
    identity = categories[(categories["category"] == "identity_term") & (categories["split"] == "test")]
    for _, row in identity.iterrows():
        print(f"{row['technique']} [{row['post_set']}, {row['n_posts']} test posts]: identity-term share of "
              f"top-{args.top_k} positive mass {row['share_of_positive_mass']:.3f}")
    for path in paths:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
