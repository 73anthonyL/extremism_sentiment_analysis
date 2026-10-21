"""Word beeswarms over the dataset posts each notebook explained, per technique and split.

Each notebook explains its test and validation posts (all of them, or a
seeded sample) with the kit's one explainer, commits the word table
`results_summary/<TECHNIQUE>/attributions/<run_id>.csv`, and writes the
per-post token attributions behind it to
`external/<TECHNIQUE>/attributions_local/<run_id>.csv` (row_id, position,
token, attribution; a per-post file, kept outside the repository). This tool
draws, for the whole model of every technique and each split, one beeswarm:
the words with the largest total |SHAP| over the explained posts (the order
of SHAP's own beeswarm, mean |SHAP| over all posts), one dot per post
containing the word, coloured by the post's true label from the committed
probability artifact.

Before drawing anything, it rebuilds every committed word table from its
per-post file with the kit's own aggregation
(`notebook_kit.aggregate_word_attributions`) and refuses a file that does not
reproduce it, so every dot is an attribution the committed RQ2 tables
summarise. It also refuses a missing per-post file, row ids outside the
split, and token positions that are not 0 to n-1 within a post.

The outputs carry words and values only, never post text or row ids:

    <out>/<TECHNIQUE>/beeswarm_<split>.png
    <out>/checks.csv     technique, split, n_posts, n_words, max_table_diff

Usage:

    python3 tools/split_beeswarms.py --out external/split_beeswarms
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import notebook_kit as nk  # noqa: E402
from curated_plots import DEFAULT_TOP_WORDS, draw_beeswarm  # noqa: E402
from probs_artifact import load_technique_probs  # noqa: E402
from repo_paths import (  # noqa: E402
    ATTRIBUTIONS_SUBDIR,
    NEGATIVE_CLASS_NAME,
    POSITIVE_CLASS_NAME,
    POSITIVE_LABEL,
    RANDOM_SEED,
    REPO_ROOT,
    RESULTS_DIR,
)

DEFAULT_EXTERNAL_ROOT = REPO_ROOT / "external"
DEFAULT_OUT = DEFAULT_EXTERNAL_ROOT / "split_beeswarms"
LOCAL_SUBDIR = "attributions_local"
SPLITS = ("test", "validation")
LOCAL_COLUMNS = ("row_id", "position", "token", "attribution")
CHECKS_FILENAME = "checks.csv"
CHECK_COLUMNS = ("technique", "split", "n_posts", "n_words", "max_table_diff")
DEFAULT_TABLE_TOLERANCE = 1e-6
DOT_SIZE = 9


class SplitBeeswarmsError(ValueError):
    """The per-post attributions cannot be drawn faithfully."""


def run_id(split, seed=RANDOM_SEED):
    return f"shap-partition_{split}_seed{seed}"


def beeswarm_filename(split):
    return f"beeswarm_{split}.png"


def read_posts(path, label):
    """[(row_id, tokens, values)] from one per-post file, tokens in position order."""
    if not path.is_file():
        raise SplitBeeswarmsError(
            f"{label}: no per-post attributions at {path}; download the technique's external folder"
        )
    frame = pd.read_csv(path, keep_default_na=False)
    missing = [c for c in LOCAL_COLUMNS if c not in frame.columns]
    if missing:
        raise SplitBeeswarmsError(f"{label}: {path.name} lacks columns {missing}")
    frame = frame.sort_values(["row_id", "position"], kind="mergesort")
    posts = []
    for row_id, block in frame.groupby("row_id", sort=False):
        if not np.array_equal(block["position"].to_numpy(), np.arange(len(block))):
            raise SplitBeeswarmsError(f"{label}: a post's token positions are not 0 to n-1")
        posts.append((row_id, block["token"].astype(str).tolist(), block["attribution"].to_numpy(dtype=float)))
    if not posts:
        raise SplitBeeswarmsError(f"{label}: {path.name} holds no posts")
    return posts


def table_difference(posts, committed, label):
    """Largest |difference| between the committed word table and the one rebuilt from `posts`."""
    rebuilt = nk.aggregate_word_attributions([p[1] for p in posts], [p[2] for p in posts])
    merged = committed.merge(rebuilt, on="word", how="outer", suffixes=("_committed", "_rebuilt"),
                             indicator=True)
    unmatched = int((merged["_merge"] != "both").sum())
    if unmatched or not (merged["support_committed"] == merged["support_rebuilt"]).all():
        raise SplitBeeswarmsError(
            f"{label}: the per-post file does not reproduce the committed word table "
            f"({unmatched} word(s) in only one of them, or a different support)"
        )
    return float(max(
        (merged[f"{column}_committed"] - merged[f"{column}_rebuilt"]).abs().max()
        for column in ("mean_abs_attribution", "mean_attribution")
    ))


def post_contributions(technique, posts):
    """One row per (post, word): the word's summed attribution in that post, by the kit's rule."""
    rows = []
    for row_id, tokens, values in posts:
        per_post = nk.aggregate_word_attributions([tokens], [values])
        rows += [
            {"technique": technique, "example_id": row_id, "word": word, "attribution": float(value)}
            for word, value in zip(per_post["word"], per_post["mean_attribution"])
        ]
    return pd.DataFrame(rows, columns=["technique", "example_id", "word", "attribution"])


def available_techniques(external_root=DEFAULT_EXTERNAL_ROOT, results_dir=RESULTS_DIR):
    """Techniques with both a downloaded per-post folder and committed attribution runs."""
    external_root, results_dir = Path(external_root), Path(results_dir)
    return sorted(
        p.name for p in external_root.iterdir()
        if (p / LOCAL_SUBDIR).is_dir() and (results_dir / p.name / ATTRIBUTIONS_SUBDIR).is_dir()
    )


def run_split_beeswarms(out_dir=DEFAULT_OUT, external_root=DEFAULT_EXTERNAL_ROOT, results_dir=RESULTS_DIR,
                        techniques=None, splits=SPLITS, n_words=DEFAULT_TOP_WORDS,
                        tolerance=DEFAULT_TABLE_TOLERANCE, probs_loader=load_technique_probs, log=print):
    """Check every per-post file against its committed table, then draw. Returns the checks frame."""
    out_dir, external_root, results_dir = Path(out_dir), Path(external_root), Path(results_dir)
    techniques = list(techniques) if techniques else available_techniques(external_root, results_dir)
    if not techniques:
        raise SplitBeeswarmsError(f"{external_root}: no technique folder holds {LOCAL_SUBDIR}/")

    checked = []
    for technique in techniques:
        for split in splits:
            label = f"{technique} {split}"
            name = f"{run_id(split)}.csv"
            posts = read_posts(external_root / technique / LOCAL_SUBDIR / name, label)
            committed_path = results_dir / technique / ATTRIBUTIONS_SUBDIR / name
            if not committed_path.is_file():
                raise SplitBeeswarmsError(f"{label}: no committed word table at {committed_path}")
            committed = pd.read_csv(committed_path, keep_default_na=False)
            difference = table_difference(posts, committed, label)
            if difference > tolerance:
                raise SplitBeeswarmsError(
                    f"{label}: the per-post file does not reproduce the committed word table "
                    f"(max |diff| {difference:.2g}, tolerance {tolerance:g})"
                )
            probs = probs_loader(technique, split)
            truth = dict(zip(probs["row_id"], probs["y_true"]))
            outside = sum(1 for row_id, _, _ in posts if row_id not in truth)
            if outside:
                raise SplitBeeswarmsError(
                    f"{label}: {outside} explained post(s) are not rows of the committed {split} artifact"
                )
            labels = {row_id: POSITIVE_CLASS_NAME if int(truth[row_id]) == POSITIVE_LABEL else NEGATIVE_CLASS_NAME
                      for row_id, _, _ in posts}
            checked.append((technique, split, posts, labels, len(committed), difference))

    rows = []
    for technique, split, posts, labels, n_table_words, difference in checked:
        draw_beeswarm(out_dir / technique / beeswarm_filename(split), post_contributions(technique, posts),
                      technique, labels, len(posts), n_words, collection=f"{split} posts", unit="post",
                      label_prefix="labelled", dot_size=DOT_SIZE)
        rows.append({"technique": technique, "split": split, "n_posts": len(posts),
                     "n_words": n_table_words, "max_table_diff": difference})
        log(f"{technique} {split}: {len(posts)} posts, word table reproduced (max |diff| {difference:.1e})")
    checks = pd.DataFrame(rows, columns=list(CHECK_COLUMNS))
    out_dir.mkdir(parents=True, exist_ok=True)
    checks.to_csv(out_dir / CHECKS_FILENAME, index=False)
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="folder for the beeswarms and checks.csv")
    parser.add_argument("--external-root", default=str(DEFAULT_EXTERNAL_ROOT),
                        help="folder holding one downloaded external/<TECHNIQUE>/ per technique")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR), help="committed result folders")
    parser.add_argument("--techniques", nargs="*", default=[], help="technique names (default: every downloaded one)")
    parser.add_argument("--splits", nargs="*", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--top-words", type=int, default=DEFAULT_TOP_WORDS, help="words per beeswarm")
    args = parser.parse_args(argv)
    try:
        run_split_beeswarms(args.out, args.external_root, args.results_dir, args.techniques, args.splits,
                            args.top_words)
    except SplitBeeswarmsError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
