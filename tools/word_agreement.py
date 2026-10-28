"""RQ2/RQ3: do the models rely on the same words?

Compares the committed whole-model attribution runs across techniques
directly, word by word, with no lexicon. For one split (default test) it
takes each technique's whole-model run and writes three text-free tables to
results_summary/rq/:

rq2_word_agreement_pairs.csv
    One row per technique pair: the number of words both tables contain, the
    Spearman correlation of mean attribution and of mean absolute attribution
    over those shared words, and the Jaccard overlap of the two top-K word
    sets, once ranked by positive mean attribution (words pushing toward
    EXTREMIST) and once by mean absolute attribution. Each run's
    n_posts_explained is carried along: classical runs explain all 450 posts
    of the split, transformer runs a 200-post sample, so vocabularies differ.

rq2_shared_top_words.csv
    Every word in at least one technique's top-K by positive mean
    attribution, with how many techniques' top-K contain it, which ones, its
    best and mean rank, and its minimum and maximum support. These words are
    already committed in the per-technique tables; no post text, no row ids.

rq2_top_word_support.csv
    Per technique: vocabulary size, n_posts_explained, the median support of
    its top-K positive words, and the share of them seen in a single
    explained post. A top list made of single-post words is a top list that
    will not survive a different post sample.

Member runs of an ensemble are left out unless --include-members is given,
in which case each member is compared as its own column. Two tables whose
shared vocabulary is smaller than --min-shared-words are refused, because a
rank correlation over a handful of words is noise, and the tool refuses to
run when fewer than two techniques have a whole-model run for the split.

USAGE
-----
    python3 tools/word_agreement.py
    python3 tools/word_agreement.py --split validation --top-k 50
    python3 tools/word_agreement.py --include-members
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attributions import AttributionArtifactError, list_runs, load_run, top_words
from repo_paths import (
    RQ2_SHARED_TOP_WORDS_CSV,
    RQ2_TOP_WORD_SUPPORT_CSV,
    RQ2_WORD_AGREEMENT_CSV,
    technique_dirs,
)

DEFAULT_SPLIT = "test"
DEFAULT_TOP_K = 50
DEFAULT_MIN_SHARED_WORDS = 30
WHOLE = "whole"

PAIR_COLUMNS = (
    "technique_a",
    "member_a",
    "technique_b",
    "member_b",
    "split",
    "top_k",
    "n_words_a",
    "n_words_b",
    "n_posts_explained_a",
    "n_posts_explained_b",
    "n_shared_words",
    "spearman_mean_attribution",
    "spearman_mean_abs_attribution",
    "jaccard_top_k_positive",
    "jaccard_top_k_abs",
    "n_shared_top_k_positive",
)

SHARED_COLUMNS = (
    "word",
    "split",
    "top_k",
    "n_columns",
    "n_in_top_k",
    "columns_in_top_k",
    "best_rank",
    "mean_rank",
    "min_support",
    "max_support",
)

SUPPORT_COLUMNS = (
    "technique",
    "member",
    "run_id",
    "split",
    "top_k",
    "n_words",
    "n_posts_explained",
    "median_support_top_k_positive",
    "share_single_post_top_k_positive",
    "share_single_post_vocabulary",
)


class WordAgreementError(RuntimeError):
    pass


def _is_member(value):
    return not (value is None or (isinstance(value, float) and np.isnan(value)) or value == "")


def select_runs(split=DEFAULT_SPLIT, include_members=False, results_dir=None, techniques=None):
    """Return [(technique, member, run_id, frame, meta)] for the split, one per column.

    One whole-model run per technique (the lowest run id if several exist for
    the split) and, with include_members, one run per member as well.
    """
    folders = technique_dirs(results_dir)
    if techniques:
        folders = [f for f in folders if f.name in set(techniques)]
    columns = []
    for folder in folders:
        technique = folder.name
        by_member = {}
        for run_id in list_runs(technique, results_dir):
            frame, meta = load_run(technique, run_id, results_dir)
            if meta["split"] != split:
                continue
            member = meta.get("member") if _is_member(meta.get("member")) else WHOLE
            if member != WHOLE and not include_members:
                continue
            by_member.setdefault(member, (run_id, frame, meta))
        for member in sorted(by_member, key=lambda m: (m != WHOLE, m)):
            run_id, frame, meta = by_member[member]
            columns.append((technique, member, run_id, frame, meta))
    if len(columns) < 2:
        raise WordAgreementError(
            f"fewer than two whole-model runs for split '{split}'; nothing to compare"
        )
    return columns


def _top_positive(frame, k):
    return frame.nlargest(k, "mean_attribution").reset_index(drop=True)


def _column_label(technique, member):
    return technique if member == WHOLE else f"{technique}:{member}"


def pairwise_table(columns, split, top_k, min_shared_words):
    rows = []
    for (tech_a, mem_a, _, frame_a, meta_a), (tech_b, mem_b, _, frame_b, meta_b) in itertools.combinations(columns, 2):
        merged = frame_a.merge(frame_b, on="word", suffixes=("_a", "_b"))
        if len(merged) < min_shared_words:
            raise WordAgreementError(
                f"{_column_label(tech_a, mem_a)} and {_column_label(tech_b, mem_b)} share only "
                f"{len(merged)} words (minimum {min_shared_words}); a rank correlation would be noise"
            )
        rho_signed, _ = spearmanr(merged["mean_attribution_a"], merged["mean_attribution_b"])
        rho_abs, _ = spearmanr(merged["mean_abs_attribution_a"], merged["mean_abs_attribution_b"])
        pos_a, pos_b = set(_top_positive(frame_a, top_k)["word"]), set(_top_positive(frame_b, top_k)["word"])
        abs_a, abs_b = set(top_words(frame_a, top_k)["word"]), set(top_words(frame_b, top_k)["word"])
        rows.append(
            {
                "technique_a": tech_a,
                "member_a": mem_a,
                "technique_b": tech_b,
                "member_b": mem_b,
                "split": split,
                "top_k": top_k,
                "n_words_a": len(frame_a),
                "n_words_b": len(frame_b),
                "n_posts_explained_a": meta_a["n_posts_explained"],
                "n_posts_explained_b": meta_b["n_posts_explained"],
                "n_shared_words": len(merged),
                "spearman_mean_attribution": float(rho_signed),
                "spearman_mean_abs_attribution": float(rho_abs),
                "jaccard_top_k_positive": len(pos_a & pos_b) / len(pos_a | pos_b),
                "jaccard_top_k_abs": len(abs_a & abs_b) / len(abs_a | abs_b),
                "n_shared_top_k_positive": len(pos_a & pos_b),
            }
        )
    return pd.DataFrame(rows, columns=list(PAIR_COLUMNS))


def shared_top_words_table(columns, split, top_k):
    records = {}
    for technique, member, _, frame, _ in columns:
        label = _column_label(technique, member)
        top = _top_positive(frame, top_k)
        for rank, row in enumerate(top.itertuples(index=False), start=1):
            entry = records.setdefault(row.word, {"labels": [], "ranks": [], "supports": []})
            entry["labels"].append(label)
            entry["ranks"].append(rank)
            entry["supports"].append(int(row.support))
    rows = []
    for word, entry in records.items():
        rows.append(
            {
                "word": word,
                "split": split,
                "top_k": top_k,
                "n_columns": len(columns),
                "n_in_top_k": len(entry["labels"]),
                "columns_in_top_k": ";".join(entry["labels"]),
                "best_rank": min(entry["ranks"]),
                "mean_rank": float(np.mean(entry["ranks"])),
                "min_support": min(entry["supports"]),
                "max_support": max(entry["supports"]),
            }
        )
    table = pd.DataFrame(rows, columns=list(SHARED_COLUMNS))
    return table.sort_values(["n_in_top_k", "mean_rank", "word"], ascending=[False, True, True]).reset_index(drop=True)


def support_table(columns, split, top_k):
    rows = []
    for technique, member, run_id, frame, meta in columns:
        top = _top_positive(frame, top_k)
        rows.append(
            {
                "technique": technique,
                "member": member,
                "run_id": run_id,
                "split": split,
                "top_k": top_k,
                "n_words": len(frame),
                "n_posts_explained": meta["n_posts_explained"],
                "median_support_top_k_positive": float(top["support"].median()),
                "share_single_post_top_k_positive": float((top["support"] == 1).mean()),
                "share_single_post_vocabulary": float((frame["support"] == 1).mean()),
            }
        )
    return pd.DataFrame(rows, columns=list(SUPPORT_COLUMNS))


def build(split=DEFAULT_SPLIT, top_k=DEFAULT_TOP_K, include_members=False,
          min_shared_words=DEFAULT_MIN_SHARED_WORDS, results_dir=None, techniques=None):
    columns = select_runs(split, include_members, results_dir, techniques)
    pairs = pairwise_table(columns, split, top_k, min_shared_words)
    shared = shared_top_words_table(columns, split, top_k)
    support = support_table(columns, split, top_k)
    return pairs, shared, support


def write_tables(pairs, shared, support, pairs_path=RQ2_WORD_AGREEMENT_CSV,
                 shared_path=RQ2_SHARED_TOP_WORDS_CSV, support_path=RQ2_TOP_WORD_SUPPORT_CSV):
    for table, path in ((pairs, pairs_path), (shared, shared_path), (support, support_path)):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False, float_format="%.6g")
    return Path(pairs_path), Path(shared_path), Path(support_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--include-members", action="store_true", help="compare ensemble members as their own columns")
    parser.add_argument("--min-shared-words", type=int, default=DEFAULT_MIN_SHARED_WORDS)
    parser.add_argument("--techniques", nargs="*", default=None, help="restrict to these technique names")
    args = parser.parse_args()

    try:
        pairs, shared, support = build(
            args.split, args.top_k, args.include_members, args.min_shared_words, techniques=args.techniques
        )
    except (AttributionArtifactError, WordAgreementError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    paths = write_tables(pairs, shared, support)
    for _, row in support.iterrows():
        print(f"{_column_label(row['technique'], row['member'])}: {row['n_words']} words over "
              f"{row['n_posts_explained']} posts; {row['share_single_post_top_k_positive']:.0%} of the "
              f"top-{args.top_k} positive words occur in one post")
    core = shared[shared["n_in_top_k"] == shared["n_columns"]]
    print(f"{len(core)} words in every column's top-{args.top_k} by positive attribution: "
          + ", ".join(core["word"].tolist()))
    print(f"{len(pairs)} pairs; median Spearman (mean attribution) {pairs['spearman_mean_attribution'].median():.3f}, "
          f"median top-{args.top_k} positive Jaccard {pairs['jaccard_top_k_positive'].median():.3f}")
    for path in paths:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
