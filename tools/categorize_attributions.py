"""RQ2: what does each model rely on? Categorize top attributed words.

For every attribution run, the top-K words by mean absolute attribution are
labelled `slur`, `extremist_framing`, `identity_term`, or `topical` from the
lexicons in data/lexicons/, and the share of attribution mass each category
carries is computed. Mass is `mean_abs_attribution * support` (approximately
the total absolute attribution the word received across explained posts);
positive mass uses `max(mean_attribution, 0) * support`, i.e. only the push
toward EXTREMIST.

Outputs, per run, next to the artifact:
    <run_id>__categorized.csv      top-K words with category and matches
    <run_id>__category_shares.csv  one row per category
and one cross-technique long table, results_summary/rq/rq2_category_shares.csv,
which compare_reliance.py and render_tables.py read. It also writes
results_summary/rq/rq2_lexicon_coverage.csv: per category, the number of
lexicon entries and how many the word-level masker would split (spaces,
hyphens, other non-word characters), which no word-level artifact can
match, so that count is reported rather than only printed. The column keeps
its name, n_multi_word_unmatchable: such an entry is several masker tokens.

USAGE
-----
    python3 tools/categorize_attributions.py --all
    python3 tools/categorize_attributions.py --technique 01_LOG-REG_TF-IDF --top-k 50
    python3 tools/categorize_attributions.py --check-coverage
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attributions import attributions_dir, list_runs, load_run, top_words
from lexicons import load_lexicons, normalize_entry, split_by_arity
from repo_paths import (
    CATEGORIES,
    CATEGORY_PRECEDENCE,
    DEFAULT_CATEGORY,
    RQ2_CATEGORY_SHARES_CSV,
    RQ2_LEXICON_COVERAGE_CSV,
    RQ_DIR,
    technique_dirs,
)

DEFAULT_TOP_K = 50

SHARE_COLUMNS = (
    "technique",
    "run_id",
    "member",
    "split",
    "seed",
    "top_k",
    "category",
    "n_words",
    "share_of_words",
    "abs_mass",
    "share_of_abs_mass",
    "positive_mass",
    "share_of_positive_mass",
)


def build_word_index(lexicons):
    """{category: set(single words)} plus the count of skipped phrases."""
    index = {}
    skipped = 0
    for category, entries in lexicons.items():
        singles, phrases = split_by_arity(entries)
        index[category] = set(singles)
        skipped += len(phrases)
    return index, skipped


COVERAGE_COLUMNS = ("category", "n_entries", "n_single_word", "n_multi_word_unmatchable")


def lexicon_coverage(lexicons):
    """One row per category: entries, word-level entries, and entries the masker splits.

    Phrases cannot match a word-level attribution artifact, so they never
    contribute to a category share; the decision rules require the count.
    """
    rows = []
    for category in CATEGORIES:
        if category == DEFAULT_CATEGORY:
            continue
        singles, phrases = split_by_arity(lexicons.get(category, []))
        rows.append(
            {
                "category": category,
                "n_entries": len(singles) + len(phrases),
                "n_single_word": len(singles),
                "n_multi_word_unmatchable": len(phrases),
            }
        )
    return pd.DataFrame(rows, columns=COVERAGE_COLUMNS)


def write_coverage(lexicons, path=RQ2_LEXICON_COVERAGE_CSV):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lexicon_coverage(lexicons).to_csv(path, index=False)
    return path


def check_coverage(lexicons, path=RQ2_LEXICON_COVERAGE_CSV):
    """Refuse a coverage table that does not match the current lexicons."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"{path} is missing; run categorize_attributions.py --all")
    committed = pd.read_csv(path)
    expected = lexicon_coverage(lexicons)
    if list(committed.columns) != list(COVERAGE_COLUMNS) or not committed.equals(expected):
        raise ValueError(
            f"{path} does not match data/lexicons/; a lexicon changed without "
            "re-running categorize_attributions.py (and RQ2/RQ4)"
        )
    return path


def categorize_word(word, word_index):
    """Return (category, matched_categories) for one word.

    Matching is exact on the normalized word. The first category in
    CATEGORY_PRECEDENCE wins; every match is also returned so overlaps are
    reported rather than silently resolved.
    """
    key = normalize_entry(word)
    matched = [c for c in CATEGORY_PRECEDENCE if key in word_index.get(c, ())]
    return (matched[0] if matched else DEFAULT_CATEGORY), matched


def categorize_frame(frame, word_index, top_k=DEFAULT_TOP_K):
    """Top-K words with category, matched categories, mass columns, and rank."""
    top = top_words(frame, top_k).copy()
    labels = [categorize_word(w, word_index) for w in top["word"]]
    top["category"] = [c for c, _ in labels]
    top["matched_categories"] = ["|".join(m) for _, m in labels]
    top["abs_mass"] = top["mean_abs_attribution"] * top["support"]
    top["positive_mass"] = top["mean_attribution"].clip(lower=0.0) * top["support"]
    top["rank"] = range(1, len(top) + 1)
    return top


def category_shares(categorized):
    """One row per category (all four, zero-filled) with word and mass shares."""
    total_words = max(len(categorized), 1)
    total_abs = float(categorized["abs_mass"].sum()) or 1.0
    total_pos = float(categorized["positive_mass"].sum()) or 1.0
    rows = []
    for category in CATEGORIES:
        part = categorized[categorized["category"] == category]
        abs_mass = float(part["abs_mass"].sum())
        pos_mass = float(part["positive_mass"].sum())
        rows.append(
            {
                "category": category,
                "n_words": int(len(part)),
                "share_of_words": len(part) / total_words,
                "abs_mass": abs_mass,
                "share_of_abs_mass": abs_mass / total_abs,
                "positive_mass": pos_mass,
                "share_of_positive_mass": pos_mass / total_pos,
            }
        )
    return pd.DataFrame(rows)


def categorize_run(technique, run_id, word_index, top_k=DEFAULT_TOP_K, results_dir=None):
    """Categorize one run, write its two derived files, return the share rows."""
    frame, meta = load_run(technique, run_id, results_dir)
    categorized = categorize_frame(frame, word_index, top_k)
    shares = category_shares(categorized)

    folder = attributions_dir(technique, results_dir)
    categorized.to_csv(folder / f"{run_id}__categorized.csv", index=False, float_format="%.8g")
    shares.to_csv(folder / f"{run_id}__category_shares.csv", index=False, float_format="%.8g")

    shares = shares.copy()
    shares.insert(0, "top_k", int(top_k))
    shares.insert(0, "seed", meta.get("seed"))
    shares.insert(0, "split", meta["split"])
    shares.insert(0, "member", meta.get("member"))
    shares.insert(0, "run_id", run_id)
    shares.insert(0, "technique", technique)
    return shares[list(SHARE_COLUMNS)]


def categorize_all(techniques, top_k=DEFAULT_TOP_K, results_dir=None, lexicons=None):
    """Categorize every run of the given techniques; return the long share table."""
    lexicons = load_lexicons() if lexicons is None else lexicons
    word_index, skipped = build_word_index(lexicons)
    tables = []
    for technique in techniques:
        for run_id in list_runs(technique, results_dir):
            tables.append(categorize_run(technique, run_id, word_index, top_k, results_dir))
            print(f"categorized {technique}/{run_id} (top {top_k})")
    if not tables:
        return pd.DataFrame(columns=SHARE_COLUMNS), skipped
    return pd.concat(tables, ignore_index=True), skipped


def write_rq2_table(table, path=RQ2_CATEGORY_SHARES_CSV):
    """Merge new rows into the cross-technique table, replacing rows for the same runs."""
    path = Path(path)
    if path.exists():
        existing = pd.read_csv(path)
        keep = ~existing.set_index(["technique", "run_id"]).index.isin(
            table.set_index(["technique", "run_id"]).index
        )
        table = pd.concat([existing[keep], table], ignore_index=True)
    table = table.sort_values(["technique", "run_id", "category"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, float_format="%.8g")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--technique")
    group.add_argument("--all", action="store_true")
    group.add_argument(
        "--check-coverage",
        action="store_true",
        help="verify rq2_lexicon_coverage.csv against data/lexicons/ and exit",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    if args.check_coverage:
        try:
            path = check_coverage(load_lexicons())
        except ValueError as error:
            print(f"FAIL: {error}", file=sys.stderr)
            return 1
        print(f"OK: {path} matches data/lexicons/")
        return 0

    techniques = [args.technique] if args.technique else [d.name for d in technique_dirs()]
    table, skipped = categorize_all(techniques, top_k=args.top_k)
    if table.empty:
        print("no attribution runs found; nothing written", file=sys.stderr)
        return 1
    path = write_rq2_table(table)
    print(f"wrote {path} ({len(table)} rows)")
    lexicons = load_lexicons()
    coverage_path = write_coverage(lexicons)
    print(f"wrote {coverage_path}")
    if skipped:
        print(f"note: {skipped} lexicon entries split by the masker (spaces, hyphens, punctuation) "
              "cannot match word-level artifacts")
    for category, entries in lexicons.items():
        if not entries:
            print(f"note: lexicon for '{category}' is empty; no word can take that category")
    return 0


if __name__ == "__main__":
    sys.exit(main())
