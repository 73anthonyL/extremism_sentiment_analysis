"""RQ4 / WP7 sensitivity to ambiguous identity terms, and identity-lexicon gaps (WP3).

Some entries of data/lexicons/identity_terms.txt are often not references to a
group: "white" and "black" are colours, "man" is an interjection, "english"
is a language, "native" is "native speaker", "asylum" is a hospital. A post
whose only identity matches are such words may not mention a community at
all, which would inflate the "with identity terms" group of RQ4 and the WP7
counterfactual population with posts that do not belong there.

This tool does NOT edit the lexicon. It declares the ambiguous entries once,
in AMBIGUOUS_IDENTITY_TERMS below, each with its reason, and recomputes:

fpr             RQ4 identity FPR per technique under three variants:
                  full_lexicon            the committed RQ4 numbers, recomputed
                                          (a self-check against rq4_identity_fpr.csv)
                  exclude_ambiguous_only  posts whose ONLY identity matches are
                                          ambiguous are dropped from the
                                          population entirely (primary)
                  reduced_lexicon         the ambiguous entries are removed from
                                          the term list, so those posts count as
                                          "without identity terms"
counterfactual  the WP7 remove counterfactual on the restricted population
                (test posts with at least one UNAMBIGUOUS identity match),
                re-scored by the restored predictor after the same verification
                as identity_counterfactual.py:
                  remove              every identity match is deleted (as WP7)
                  remove_unambiguous  only the unambiguous matches are deleted
                Holm is reported over three families, because the choice is
                a judgement and no single one is "the WP7 family":
                  mcnemar_p_holm                    every row (techniques x
                      populations x both operations, 32 rows). Same SIZE as
                      WP7, different makeup: WP7 is remove + substitute, here
                      remove and remove_unambiguous are nearly the same edit,
                      so their p values are correlated and this family is
                      conservative for the remove question.
                  mcnemar_p_holm_remove             remove rows only (16)
                  mcnemar_p_holm_remove_extremist   extremist remove rows (8)
                flagged_flip_rate (n_flip_pos_to_neg / n_flagged_original) is
                the per-flagged-post effect size, to read next to the p value:
                a smaller population lowers power without changing it.
lexicon-gaps    for each technique's whole-model test attribution run, the
                top-K attributed words that look like identity references but
                are not in the identity lexicon, as a candidate list for WP3.
                A word qualifies by one of three declared rules:
                  inflection  a plural or singular of a lexicon entry (whites)
                  compound    contains a lexicon entry of >= COMPOUND_MIN_ENTRY_LENGTH
                              characters as part of a longer token (hashtags)
                  group_noun  in GROUP_NOUN_CANDIDATES, generic group nouns
                It also counts how many top-K identity words are ambiguous.
                The candidates are for human review; nothing is added.

Text is held in memory only. Outputs carry counts, rates, p values and
vocabulary words (which the committed attribution artifacts already carry);
no post text, row id or hash. A counterfactual run writes nothing unless every
requested technique passes.

Outputs (results_summary/rq/):
    rq4_identity_fpr_sensitivity.csv
    rq4_identity_counterfactual_sensitivity.csv
    rq2_identity_lexicon_gaps_sensitivity.csv
    rq2_identity_top_k_sensitivity.csv

USAGE
-----
    python3 tools/identity_sensitivity.py fpr
    python3 tools/identity_sensitivity.py lexicon-gaps
    python3 tools/identity_sensitivity.py counterfactual --dry-run-population --all
    python3 tools/identity_sensitivity.py counterfactual --technique 01_LOG-REG_TF-IDF
    python3 tools/identity_sensitivity.py counterfactual \\
        --technique 11_MULTI-CHECKPOINT_LOGIT-POOL --device mps --sequential-checkpoints
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lexicons import load_lexicons, normalize_entry, phrase_pattern  # noqa: E402
from repo_paths import (  # noqa: E402
    POSITIVE_LABEL,
    RQ2_IDENTITY_LEXICON_GAPS_SENSITIVITY_CSV,
    RQ2_IDENTITY_TOP_K_SENSITIVITY_CSV,
    RQ4_IDENTITY_COUNTERFACTUAL_SENSITIVITY_CSV,
    RQ4_IDENTITY_FPR_SENSITIVITY_CSV,
    comparison_technique_dirs,
    in_comparison,
)

# ---------------------------------------------------------------------------
# The declared ambiguous list (the only place it is defined)
# ---------------------------------------------------------------------------
# An entry is ambiguous when, in ordinary social-media English, it is often NOT
# a reference to a group of people. Every entry must be in the identity
# lexicon; the tool refuses otherwise, so the list cannot drift from it.
AMBIGUOUS_IDENTITY_TERMS = {
    "white": "colour word (white house, white car) as often as a racial group",
    "black": "colour word (black friday, black hole) as often as a racial group",
    "man": "interjection and address ('man, ...', 'come on man') and any single person",
    "men": "any group of male persons, not a protected identity reference by itself",
    "boy": "interjection ('oh boy') and any single child",
    "boys": "address ('the boys') and any group of children",
    "girl": "address and interjection ('girl, ...') and any single person",
    "girls": "address and any group of children or women",
    "english": "the language (in english, english class) more often than the nationality",
    "native": "'native speaker', 'native plant', 'native app'",
    "trans": "prefix and abbreviation (trans fat, trans-atlantic) besides transgender",
    "asylum": "a psychiatric hospital and the legal process, not only asylum seekers",
    "indian": "conflates India and Indigenous Americans, and the cuisine (indian food)",
}

# Generic group-category nouns that are not lexicon entries. Used only to
# propose WP3 candidates; they never enter RQ4.
GROUP_NOUN_CANDIDATES = (
    "minority", "minorities", "race", "races", "ethnic", "ethnicity", "ethnicities",
    "caucasian", "caucasians", "mulatto", "mulattos", "afro", "nationality",
    "nationalities", "religion", "religions", "tribe", "tribes", "alien", "aliens",
)
COMPOUND_MIN_ENTRY_LENGTH = 5
DEFAULT_TOP_K = 50
TEST_SPLIT = "test"

FPR_VARIANTS = ("full_lexicon", "exclude_ambiguous_only", "reduced_lexicon")
FPR_COLUMNS = (
    "technique",
    "split",
    "variant",
    "threshold",
    "n_negative",
    "n_negative_with_identity_terms",
    "n_negative_excluded",
    "fpr_with_identity_terms",
    "fpr_without_identity_terms",
    "fpr_ratio",
    "fisher_odds_ratio",
    "fisher_p_value",
    "n_positive_with_identity_terms",
    "n_positive_excluded",
    "recall_with_identity_terms",
    "recall_without_identity_terms",
)

CF_OPERATIONS = ("remove", "remove_unambiguous")
CF_ROW_COLUMNS = (
    "technique",
    "split",
    "population",
    "y_true",
    "operation",
    "threshold",
    "n_posts",
    "n_posts_full_population",
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
CF_TABLE_COLUMNS = CF_ROW_COLUMNS + (
    "flagged_flip_rate",
    "wilcoxon_p_holm",
    "mcnemar_p_holm",
    "mcnemar_p_holm_remove",
    "mcnemar_p_holm_remove_extremist",
)

GAP_COLUMNS = ("technique", "run_id", "top_k", "rank", "word", "rule", "related_entry",
               "other_lexicon_category", "mean_abs_attribution", "support")
TOP_K_COLUMNS = ("technique", "run_id", "top_k", "n_identity_term", "n_identity_term_ambiguous",
                 "ambiguous_words", "n_gap_candidates", "gap_words")


class IdentitySensitivityError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Ambiguous list and match classes
# ---------------------------------------------------------------------------
def check_ambiguous_terms(ambiguous, terms):
    """Refuse an ambiguous list that is empty, strays outside the lexicon, or covers all of it."""
    ambiguous = [normalize_entry(t) for t in ambiguous]
    if not ambiguous:
        raise IdentitySensitivityError("the ambiguous identity list is empty")
    lexicon = {normalize_entry(t) for t in terms}
    stray = sorted(set(ambiguous) - lexicon)
    if stray:
        raise IdentitySensitivityError(
            f"ambiguous terms not in the identity lexicon: {stray}; the list must be a subset"
        )
    if lexicon <= set(ambiguous):
        raise IdentitySensitivityError("the ambiguous list covers the whole identity lexicon")
    return frozenset(ambiguous)


def unambiguous_terms(terms, ambiguous):
    return [t for t in terms if normalize_entry(t) not in ambiguous]


def match_classes(texts, terms, ambiguous):
    """Per text: 'none', 'ambiguous_only' or 'unambiguous' (at least one non-ambiguous match)."""
    from identity_fpr import identity_matches

    classes = []
    for matches in identity_matches(list(texts), terms):
        if not matches:
            classes.append("none")
        elif all(normalize_entry(m) in ambiguous for m in matches):
            classes.append("ambiguous_only")
        else:
            classes.append("unambiguous")
    return classes


# ---------------------------------------------------------------------------
# RQ4 FPR sensitivity
# ---------------------------------------------------------------------------
def fpr_sensitivity(technique, processed, probs, terms, ambiguous, threshold):
    """The three FPR variants for one technique, as summary rows."""
    from identity_fpr import analyze, merge_with_dataset

    merged = merge_with_dataset(technique, probs, processed)
    merged["match_class"] = match_classes(merged["text"], terms, ambiguous)
    excluded = merged["match_class"] == "ambiguous_only"
    n_neg_excluded = int((excluded & (merged["y_true"] != POSITIVE_LABEL)).sum())
    n_pos_excluded = int((excluded & (merged["y_true"] == POSITIVE_LABEL)).sum())

    inputs = {
        "full_lexicon": (probs, terms, 0, 0),
        "exclude_ambiguous_only": (
            probs[~probs["row_id"].isin(merged.loc[excluded, "row_id"])].reset_index(drop=True),
            terms, n_neg_excluded, n_pos_excluded,
        ),
        "reduced_lexicon": (probs, unambiguous_terms(terms, ambiguous), 0, 0),
    }
    rows = []
    for variant in FPR_VARIANTS:
        variant_probs, variant_terms, n_neg_x, n_pos_x = inputs[variant]
        summary, _ = analyze(technique, TEST_SPLIT, threshold, processed=processed,
                             probs=variant_probs, terms=variant_terms)
        summary.update(variant=variant, n_negative_excluded=n_neg_x, n_positive_excluded=n_pos_x)
        rows.append({c: summary[c] for c in FPR_COLUMNS})
    return rows


def write_fpr_table(rows, path=None):
    path = Path(RQ4_IDENTITY_FPR_SENSITIVITY_CSV if path is None else path)
    frame = pd.DataFrame(rows)[list(FPR_COLUMNS)]
    if path.exists():
        existing = pd.read_csv(path)
        if list(existing.columns) != list(FPR_COLUMNS):
            raise IdentitySensitivityError(f"{path.name}: columns differ from this version of the tool")
        frame = pd.concat([existing[~existing["technique"].isin(frame["technique"])], frame],
                          ignore_index=True)
    order = {v: i for i, v in enumerate(FPR_VARIANTS)}
    frame = frame.sort_values(["technique", "variant"], key=lambda s: s.map(order) if s.name == "variant" else s)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.reset_index(drop=True).to_csv(path, index=False, float_format="%.6g")
    return path


# ---------------------------------------------------------------------------
# WP7 counterfactual sensitivity
# ---------------------------------------------------------------------------
def build_restricted_population(technique, processed, probs, terms, ambiguous):
    """WP7 population minus posts whose only identity matches are ambiguous (in memory only)."""
    from identity_counterfactual import build_population

    population, pattern = build_population(technique, processed, probs, terms)
    classes = match_classes(population["text"], terms, ambiguous)
    keep = np.array([c == "unambiguous" for c in classes], dtype=bool)
    full_counts = {name: int((population["y_true"] == label).sum())
                   for label, name in ((0, "non_extremist"), (POSITIVE_LABEL, "extremist"))}
    restricted = population.loc[keep].reset_index(drop=True)
    unambiguous_pattern = phrase_pattern(unambiguous_terms(terms, ambiguous))
    return restricted, pattern, unambiguous_pattern, full_counts


def counterfactual_sensitivity(technique, model, population, pattern, unambiguous_pattern,
                               threshold, full_counts, tolerance=None):
    """Verify the predictor on the restricted originals, then score both remove operations."""
    from curated_examples import DEFAULT_TOLERANCE, CuratedExamplesError, verify_model
    from identity_counterfactual import POPULATIONS, _round, counterfactual_text, paired_metrics

    tolerance = DEFAULT_TOLERANCE if tolerance is None else tolerance
    restored = getattr(model, "threshold", None)
    if restored is not None and not math.isclose(float(restored), float(threshold)):
        raise IdentitySensitivityError(f"{technique}: restored threshold {restored} != committed {threshold}")
    if population.empty:
        raise IdentitySensitivityError(f"{technique}: the restricted population is empty")
    try:
        verification = verify_model(model, population[["row_id", "text"]],
                                    population[["row_id", "y_prob"]], tolerance)
    except CuratedExamplesError as error:
        raise IdentitySensitivityError(str(error)) from error

    texts = population["text"].tolist()
    n = len(texts)
    batches = [
        [counterfactual_text(t, pattern, "remove") for t in texts],
        [counterfactual_text(t, unambiguous_pattern, "remove") for t in texts],
    ]
    scored = np.asarray(model.predict(batches[0] + batches[1]), dtype=float)
    p_cf = {"remove": scored[:n], "remove_unambiguous": scored[n:]}
    p_original = population["y_prob"].to_numpy(dtype=float)
    y_true = population["y_true"].to_numpy()

    rows = []
    for label, name in POPULATIONS:
        mask = y_true == label
        for op in CF_OPERATIONS:
            row = {"technique": technique, "split": TEST_SPLIT, "population": name,
                   "y_true": int(label), "operation": op, "threshold": float(threshold),
                   "n_posts_full_population": int(full_counts[name])}
            row.update(paired_metrics(p_original[mask], p_cf[op][mask], threshold, label))
            row["verification_max_abs_diff"] = verification["max_abs_diff"]
            rows.append({k: _round(v) for k, v in row.items()})
    return rows


def write_cf_table(rows, path=None, results_dir=None):
    """Merge rows by technique into the sensitivity table; derive the flip rate and the Holm columns.

    Every derived column is recomputed from CF_ROW_COLUMNS, so the table can be
    rebuilt from its own row columns without re-scoring any model.
    """
    from identity_counterfactual import _holm_column

    path = Path(RQ4_IDENTITY_COUNTERFACTUAL_SENSITIVITY_CSV if path is None else path)
    new = pd.DataFrame(rows)[list(CF_ROW_COLUMNS)]
    if path.exists():
        existing = pd.read_csv(path)
        if list(existing.columns) != list(CF_TABLE_COLUMNS):
            raise IdentitySensitivityError(f"{path.name}: columns differ from this version of the tool")
        existing = existing[~existing["technique"].isin(new["technique"])][list(CF_ROW_COLUMNS)]
        new = pd.concat([existing, new], ignore_index=True)
    new = new.sort_values(["technique", "y_true", "operation"]).reset_index(drop=True)
    family = new["technique"].map(lambda t: "" if in_comparison(t, results_dir) else t)
    flagged = new["n_flagged_original"].astype(float)
    new["flagged_flip_rate"] = (new["n_flip_pos_to_neg"] / flagged.where(flagged > 0)).round(6)
    for column in ("wilcoxon_p_holm", "mcnemar_p_holm", "mcnemar_p_holm_remove",
                   "mcnemar_p_holm_remove_extremist"):
        new[column] = np.nan
    is_remove = new["operation"] == "remove"
    is_extremist = new["y_true"] == POSITIVE_LABEL
    for _, index in new.groupby(family, sort=False).groups.items():
        new.loc[index, "wilcoxon_p_holm"] = _holm_column(new.loc[index, "wilcoxon_p"])
        new.loc[index, "mcnemar_p_holm"] = _holm_column(new.loc[index, "mcnemar_exact_p"])
        for column, mask in (("mcnemar_p_holm_remove", is_remove),
                             ("mcnemar_p_holm_remove_extremist", is_remove & is_extremist)):
            sub = new.index.isin(index) & mask.to_numpy()
            new.loc[sub, column] = _holm_column(new.loc[sub, "mcnemar_exact_p"])
    path.parent.mkdir(parents=True, exist_ok=True)
    new[list(CF_TABLE_COLUMNS)].to_csv(path, index=False, float_format="%.6g")
    return path


# ---------------------------------------------------------------------------
# Lexicon gaps among top attributed words
# ---------------------------------------------------------------------------
def _inflections(entry):
    forms = {entry + "s", entry + "es"}
    if entry.endswith("s"):
        forms.add(entry[:-1])
    if entry.endswith("es"):
        forms.add(entry[:-2])
    if entry.endswith("man"):
        forms.add(entry[:-3] + "men")
    if entry.endswith("men"):
        forms.add(entry[:-3] + "man")
    forms.discard(entry)
    return forms


def gap_rule(word, identity_terms, group_nouns=GROUP_NOUN_CANDIDATES):
    """(rule, related_entry) when `word` looks like an identity reference missing from the lexicon."""
    word = normalize_entry(word)
    singles = [t for t in identity_terms if " " not in t]
    if not word or word in singles:
        return None
    for entry in singles:
        if word in _inflections(entry):
            return "inflection", entry
    long_entries = sorted((e for e in singles if len(e) >= COMPOUND_MIN_ENTRY_LENGTH), key=len, reverse=True)
    for entry in long_entries:
        if entry in word and len(word) > len(entry):
            return "compound", entry
    if word in group_nouns:
        return "group_noun", ""
    return None


def lexicon_gaps(technique, frame, run_id, lexicons, ambiguous, top_k=DEFAULT_TOP_K):
    """Candidate rows and the per-run summary row for one attribution run."""
    from attributions import top_words

    identity = lexicons["identity_term"]
    identity_set = set(identity)
    top = top_words(frame, top_k)
    other_patterns = {c: phrase_pattern(e) for c, e in lexicons.items() if c != "identity_term"}
    candidates, ambiguous_hits, n_identity = [], [], 0
    for rank, row in enumerate(top.itertuples(index=False), start=1):
        word = normalize_entry(str(row.word))
        if word in identity_set:
            n_identity += 1
            if word in ambiguous:
                ambiguous_hits.append(word)
            continue
        rule = gap_rule(word, identity)
        if rule is None:
            continue
        other = [c for c, p in other_patterns.items() if p is not None and p.fullmatch(word)]
        candidates.append({
            "technique": technique, "run_id": run_id, "top_k": int(top_k), "rank": rank,
            "word": word, "rule": rule[0], "related_entry": rule[1],
            "other_lexicon_category": ";".join(other),
            "mean_abs_attribution": float(row.mean_abs_attribution), "support": int(row.support),
        })
    summary = {
        "technique": technique, "run_id": run_id, "top_k": int(top_k),
        "n_identity_term": n_identity, "n_identity_term_ambiguous": len(ambiguous_hits),
        "ambiguous_words": ";".join(ambiguous_hits), "n_gap_candidates": len(candidates),
        "gap_words": ";".join(c["word"] for c in candidates),
    }
    return candidates, summary


def whole_model_test_run(technique, results_dir=None):
    from attributions import list_runs, load_run

    for run_id in list_runs(technique, results_dir):
        frame, meta = load_run(technique, run_id, results_dir)
        if meta.get("split") == TEST_SPLIT and meta.get("member") is None:
            return run_id, frame
    raise IdentitySensitivityError(f"{technique}: no whole-model test attribution run")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _techniques(args):
    return args.technique if getattr(args, "technique", None) else [d.name for d in comparison_technique_dirs()]


def cmd_fpr(args, lexicons, ambiguous):
    from identity_fpr import load_threshold
    from probs_artifact import load_technique_probs
    from split_protocol import build_processed_dataset

    processed = build_processed_dataset()
    rows = []
    for technique in _techniques(args):
        probs = load_technique_probs(technique, TEST_SPLIT)
        rows += fpr_sensitivity(technique, processed, probs, lexicons["identity_term"], ambiguous,
                                load_threshold(technique))
        primary = rows[-2]
        print(f"{technique}: exclude_ambiguous_only n_neg_with={primary['n_negative_with_identity_terms']} "
              f"ratio={primary['fpr_ratio']:.3f} Fisher p={primary['fisher_p_value']:.4f}")
    print(f"wrote {write_fpr_table(rows)}")
    return 0


def cmd_gaps(args, lexicons, ambiguous):
    candidates, summaries = [], []
    for technique in _techniques(args):
        run_id, frame = whole_model_test_run(technique)
        found, summary = lexicon_gaps(technique, frame, run_id, lexicons, ambiguous, args.top_k)
        candidates += found
        summaries.append(summary)
        print(f"{technique}: {summary['n_identity_term']} identity words in top {args.top_k} "
              f"({summary['n_identity_term_ambiguous']} ambiguous), {summary['n_gap_candidates']} gap candidates")
    RQ2_IDENTITY_LEXICON_GAPS_SENSITIVITY_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(candidates, columns=list(GAP_COLUMNS)).to_csv(
        RQ2_IDENTITY_LEXICON_GAPS_SENSITIVITY_CSV, index=False, float_format="%.6g")
    pd.DataFrame(summaries, columns=list(TOP_K_COLUMNS)).to_csv(RQ2_IDENTITY_TOP_K_SENSITIVITY_CSV, index=False)
    print(f"wrote {RQ2_IDENTITY_LEXICON_GAPS_SENSITIVITY_CSV} and {RQ2_IDENTITY_TOP_K_SENSITIVITY_CSV}")
    return 0


def cmd_counterfactual(args, lexicons, ambiguous):
    from identity_counterfactual import EXTERNAL_ROOT
    from identity_fpr import load_threshold
    from model_loaders import load_technique
    from probs_artifact import load_technique_probs
    from split_protocol import build_processed_dataset

    processed = build_processed_dataset()
    terms = lexicons["identity_term"]
    rows = []
    for technique in _techniques(args):
        probs = load_technique_probs(technique, TEST_SPLIT)
        population, pattern, unamb_pattern, full_counts = build_restricted_population(
            technique, processed, probs, terms, ambiguous)
        counts = {name: int((population["y_true"] == label).sum())
                  for label, name in ((0, "non_extremist"), (POSITIVE_LABEL, "extremist"))}
        if args.dry_run_population:
            print(f"{technique}: restricted population non_extremist {counts['non_extremist']} of "
                  f"{full_counts['non_extremist']}, extremist {counts['extremist']} of {full_counts['extremist']}")
            continue
        model = load_technique(technique, EXTERNAL_ROOT / technique, device=args.device,
                               resident=not args.sequential_checkpoints)
        found = counterfactual_sensitivity(technique, model, population, pattern, unamb_pattern,
                                           load_threshold(technique), full_counts)
        del model
        rows += found
        for r in found:
            print(f"{technique} {r['population']} {r['operation']}: n={r['n_posts']} flagged "
                  f"{r['n_flagged_original']} -> {r['n_flagged_counterfactual']}, McNemar p={r['mcnemar_exact_p']}")
    if args.dry_run_population:
        return 0
    print(f"wrote {write_cf_table(rows)}")
    return 0


def _tool_errors():
    from attributions import AttributionArtifactError
    from identity_counterfactual import IdentityCounterfactualError
    from identity_fpr import IdentityFPRError
    from model_loaders import ModelLoadError
    from probs_artifact import ProbsArtifactError

    return (IdentitySensitivityError, IdentityFPRError, IdentityCounterfactualError,
            ProbsArtifactError, ModelLoadError, AttributionArtifactError)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    fpr = sub.add_parser("fpr", help="RQ4 FPR under the three variants")
    fpr.add_argument("--technique", action="append")
    gaps = sub.add_parser("lexicon-gaps", help="identity-like top-K words missing from the lexicon")
    gaps.add_argument("--technique", action="append")
    gaps.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    cf = sub.add_parser("counterfactual", help="WP7 remove counterfactual on the restricted population")
    group = cf.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true")
    group.add_argument("--technique", action="append")
    cf.add_argument("--device", default="auto")
    cf.add_argument("--sequential-checkpoints", action="store_true")
    cf.add_argument("--dry-run-population", action="store_true")
    args = parser.parse_args(argv)

    lexicons = load_lexicons()
    try:
        ambiguous = check_ambiguous_terms(AMBIGUOUS_IDENTITY_TERMS, lexicons["identity_term"])
        handler = {"fpr": cmd_fpr, "lexicon-gaps": cmd_gaps, "counterfactual": cmd_counterfactual}
        return handler[args.command](args, lexicons, ambiguous)
    except _tool_errors() as error:
        print(f"ERROR: {error}; nothing more was written", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
