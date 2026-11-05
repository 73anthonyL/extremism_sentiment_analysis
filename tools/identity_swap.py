"""RQ4 / WP7b: does the model respond to WHICH group is named, or only to a group?

WP7 (identity_counterfactual.py) found that deleting identity terms lowers
p(extremist) a lot for the transformers, while a neutral group noun in their
place mostly does not. This tool replaces each identity term with OTHER
identity terms of the same lexicon category and form, and re-scores the posts.
The design is preregistered in docs/preregistrations/WP7b_identity_swap.md;
the swap map is data/lexicons/identity_swap_pairs.csv.

The swap map is refused at startup unless every identity-lexicon entry has
exactly one row, every source and target is an identity-lexicon entry in the
same lexicon section (category) as the row, every target is itself a swapped
row of the same category and form and another group, no source or target is
ambiguous (identity_sensitivity.AMBIGUOUS_IDENTITY_TERMS) or excluded, no word
is matched by the slur or extremist-framing lexicon, and each swapped row has
exactly three distinct targets.

Population: test posts with at least one match of a swapped source. Every
operation edits the same spans (the swapped-source matches); matches of
excluded entries stay in place. Case follows the matched text.

* swap        variant k replaces every span by target_k of its source; p is
              the mean over the three variants. The primary operation.
* remove      the spans are deleted and whitespace collapsed.
* substitute  the spans are replaced by each word of WP7's vetted neutral
              panel (SUBSTITUTE_WORDS), probabilities averaged.

delta = p_original - p_counterfactual; p_original is the committed artifact
probability, and the restored predictor must reproduce it on the original
texts first (curated_examples.verify_model). Thresholds are the locked ones
from best_config.json. Wilcoxon and exact McNemar p-values are Holm-adjusted
over every technique x population x operation row of the comparison (one
family per test); a secondary pair of columns adjusts over the swap rows
only. A technique outside the comparison is its own family.

Descriptive extras: mean |delta| over posts and variants, the mean within-post
spread across variants, and a directions table that swaps one source at a
time to each of its targets (rq4_identity_swap_directions.csv). Each map
term's training-split extremist share and 01_LOG-REG_TF-IDF coefficient are
recorded so a reader can see whether a target group is itself label-skewed.

Re-scoring frozen models at locked thresholds is a perturbation analysis,
not a test evaluation. Text is held in memory only; every payload is checked
for row ids, text hashes and text-bearing keys before it is written, and
nothing is written unless every requested technique passes.

Outputs:
    results_summary/<TECHNIQUE>/identity_swap_test.json
    results_summary/rq/rq4_identity_swap.csv
    results_summary/rq/rq4_identity_swap_directions.csv

USAGE
-----
    python3 tools/identity_swap.py --dry-run-population --all
    python3 tools/identity_swap.py --technique 01_LOG-REG_TF-IDF
    python3 tools/identity_swap.py --technique 11_MULTI-CHECKPOINT_LOGIT-POOL \\
        --device mps --sequential-checkpoints
"""

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attributions import AttributionArtifactError  # noqa: E402
from curated_examples import DEFAULT_TOLERANCE, CuratedExamplesError, verify_model  # noqa: E402
from identity_counterfactual import (  # noqa: E402
    EXTERNAL_ROOT,
    POPULATIONS,
    IdentityCounterfactualError,
    SUBSTITUTE_WORDS,
    WHITESPACE_RE,
    _holm_column,
    _round,
    check_substitute_word,
    load_substitute_profile,
    paired_metrics,
    recorded_device,
)
from identity_fpr import IdentityFPRError, load_threshold, merge_with_dataset  # noqa: E402
from identity_sensitivity import AMBIGUOUS_IDENTITY_TERMS  # noqa: E402
from lexicons import load_lexicons, normalize_entry, phrase_pattern  # noqa: E402
from model_loaders import ModelLoadError, load_technique  # noqa: E402
from probs_artifact import ProbsArtifactError, load_technique_probs  # noqa: E402
from repo_paths import (  # noqa: E402
    IDENTITY_SWAP_FILENAME,
    IDENTITY_SWAP_MAP_CSV,
    LEXICON_FILES,
    REPO_ROOT,
    RESULTS_DIR,
    RQ4_IDENTITY_SWAP_CSV,
    RQ4_IDENTITY_SWAP_DIRECTIONS_CSV,
    in_comparison,
    technique_dirs,
)

SPLIT = "test"
OPERATIONS = ("remove", "substitute", "swap")
PRIMARY_OPERATION = "swap"
N_TARGETS = 3
EXCLUDED_FORM = "excluded"
MAP_COLUMNS = ("category", "form", "group", "source", "target_1", "target_2", "target_3", "note")
SIG_DIGITS = 6
PREREGISTRATION = "docs/preregistrations/WP7b_identity_swap.md"

# Section headers of data/lexicons/identity_terms.txt -> category name used in the map.
SECTION_CATEGORIES = {
    "religion": "religion",
    "ethnicity, race, nationality": "ethnicity_nationality",
    "migration status": "migration",
    "gender and sexuality": "gender_sexuality",
}

ROW_ID_RE = re.compile(r"\bex_\d{6}\b")
TEXT_HASH_RE = re.compile(r"(?<![0-9a-f])[0-9a-f]{16}(?![0-9a-f])")
FORBIDDEN_KEYS = {"row_id", "row_ids", "text", "texts", "text_hash", "post", "posts"}

ROW_COLUMNS = (
    "technique",
    "split",
    "population",
    "y_true",
    "operation",
    "threshold",
    "n_posts",
    "n_variants",
    "mean_p_original",
    "mean_p_counterfactual",
    "mean_delta",
    "median_delta",
    "share_delta_positive",
    "mean_abs_delta",
    "mean_variant_spread",
    "n_flagged_original",
    "n_flagged_counterfactual",
    "n_flip_pos_to_neg",
    "n_flip_neg_to_pos",
    "fp_share_cleared",
    "wilcoxon_p",
    "mcnemar_exact_p",
    "verification_max_abs_diff",
)
HOLM_COLUMNS = ("wilcoxon_p_holm", "mcnemar_p_holm", "wilcoxon_p_holm_swap", "mcnemar_p_holm_swap")
TABLE_COLUMNS = ROW_COLUMNS + HOLM_COLUMNS
DIRECTION_COLUMNS = ("technique", "split", "population", "category", "source", "target",
                     "n_posts", "mean_delta", "mean_abs_delta")


class IdentitySwapError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Swap map
# ---------------------------------------------------------------------------
def lexicon_sections(path=None):
    """{entry: category} from the section headers of the identity lexicon."""
    path = Path(LEXICON_FILES["identity_term"] if path is None else path)
    sections, current = {}, None
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped.startswith("#"):
            header = normalize_entry(stripped.lstrip("#"))
            if header in SECTION_CATEGORIES:
                current = SECTION_CATEGORIES[header]
            continue
        entry = normalize_entry(raw.split("#", 1)[0])
        if entry and current is not None:
            sections[entry] = current
    return sections


def read_swap_map(path=None):
    path = Path(IDENTITY_SWAP_MAP_CSV if path is None else path)
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MAP_COLUMNS:
            raise IdentitySwapError(f"{path.name}: columns must be {list(MAP_COLUMNS)}")
        return [{k: normalize_entry(v or "") if k != "note" else (v or "") for k, v in row.items()}
                for row in reader]


def check_swap_map(rows, lexicons, sections, ambiguous=AMBIGUOUS_IDENTITY_TERMS):
    """Validate the preregistered map; return ({source: spec}, {excluded: note})."""
    identity = list(lexicons["identity_term"])
    identity_set = set(identity)
    ambiguous = {normalize_entry(t) for t in ambiguous}
    other_patterns = {c: phrase_pattern(e) for c, e in lexicons.items() if c != "identity_term"}

    swapped, excluded = {}, {}
    for row in rows:
        source = row["source"]
        if source in swapped or source in excluded:
            raise IdentitySwapError(f"swap map: '{source}' has more than one row")
        if source not in identity_set:
            raise IdentitySwapError(f"swap map: source '{source}' is not an identity-lexicon entry")
        if sections.get(source) != row["category"]:
            raise IdentitySwapError(
                f"swap map: '{source}' is in lexicon section {sections.get(source)!r}, "
                f"not {row['category']!r}; a swap must not cross categories")
        targets = [row[f"target_{i}"] for i in range(1, N_TARGETS + 1)]
        if row["form"] == EXCLUDED_FORM:
            if any(targets):
                raise IdentitySwapError(f"swap map: excluded '{source}' must have no targets")
            excluded[source] = row["note"]
            continue
        if source in ambiguous:
            raise IdentitySwapError(f"swap map: '{source}' is ambiguous and may not be swapped")
        if not all(targets) or len(set(targets)) != N_TARGETS:
            raise IdentitySwapError(f"swap map: '{source}' needs exactly {N_TARGETS} distinct targets")
        if not row["group"]:
            raise IdentitySwapError(f"swap map: '{source}' has no group")
        swapped[source] = {"category": row["category"], "form": row["form"],
                           "group": row["group"], "targets": targets}

    missing = [t for t in identity if t not in swapped and t not in excluded]
    if missing:
        raise IdentitySwapError(f"swap map: identity entries without a row: {missing}")
    not_excluded = sorted(ambiguous & set(swapped))
    if not_excluded:
        raise IdentitySwapError(f"swap map: ambiguous entries must be excluded: {not_excluded}")

    for source, spec in swapped.items():
        for target in spec["targets"]:
            if target not in identity_set:
                raise IdentitySwapError(
                    f"swap map: target '{target}' of '{source}' is not an identity-lexicon entry")
            if sections.get(target) != spec["category"]:
                raise IdentitySwapError(
                    f"swap map: target '{target}' of '{source}' is in section {sections.get(target)!r}, "
                    f"not {spec['category']!r}; a swap must not cross categories")
            if target in ambiguous or target in excluded:
                raise IdentitySwapError(f"swap map: target '{target}' of '{source}' is ambiguous or excluded")
            if target not in swapped:
                raise IdentitySwapError(f"swap map: target '{target}' of '{source}' has no swapped row")
            if swapped[target]["form"] != spec["form"]:
                raise IdentitySwapError(
                    f"swap map: target '{target}' ({swapped[target]['form']}) changes the form of "
                    f"'{source}' ({spec['form']})")
            if target == source or swapped[target]["group"] == spec["group"]:
                raise IdentitySwapError(f"swap map: target '{target}' names the same group as '{source}'")
            for category, pattern in other_patterns.items():
                if pattern is not None and pattern.search(target):
                    raise IdentitySwapError(
                        f"swap map: target '{target}' is matched by the {category} lexicon")
    return swapped, excluded


def map_digest(path=None):
    path = Path(IDENTITY_SWAP_MAP_CSV if path is None else path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Edited text
# ---------------------------------------------------------------------------
def match_case(matched, word):
    """Give `word` the casing of the text it replaces."""
    if len(matched) > 1 and matched.isupper():
        return word.upper()
    if matched[:1].isupper():
        return word[:1].upper() + word[1:]
    return word


def edit_text(text, pattern, replace):
    """Replace every match by replace(lower-cased match) (None = keep), case-matched; collapse spaces."""
    def substitute(match):
        word = replace(match.group(0).lower())
        if word is None:
            return match.group(0)
        return match_case(match.group(0), word) if word else ""

    return WHITESPACE_RE.sub(" ", pattern.sub(substitute, text)).strip()


def remove_text(text, pattern):
    return edit_text(text, pattern, lambda source: "")


def substitute_text(text, pattern, word):
    return edit_text(text, pattern, lambda source: word)


def swap_text(text, pattern, swap_map, k):
    """Variant k (0-based): every swapped-source match -> target k of its own source."""
    return edit_text(text, pattern, lambda source: swap_map[source]["targets"][k])


def swap_one_source(text, pattern, only_source, target):
    """Swap only `only_source`'s matches to `target`; other matches stay."""
    return edit_text(text, pattern, lambda source: target if source == only_source else None)


def sources_in(text, pattern):
    return sorted({m.group(0).lower() for m in pattern.finditer(text)})


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------
def build_population(technique, processed, probs, swap_map):
    """Test rows with at least one swapped-source match (in memory only)."""
    pattern = phrase_pattern(list(swap_map))
    if pattern is None:
        raise IdentitySwapError("swap map has no swapped source")
    merged = merge_with_dataset(technique, probs, processed)
    has_source = merged["text"].map(lambda t: pattern.search(t) is not None)
    population = merged.loc[has_source, ["row_id", "text", "y_true", "y_prob"]]
    return population.reset_index(drop=True), pattern


def population_counts(population):
    return {name: int((population["y_true"] == label).sum()) for label, name in POPULATIONS}


# ---------------------------------------------------------------------------
# Term profile (reported, never used to refuse)
# ---------------------------------------------------------------------------
def term_profile(processed, swap_map, results_dir=None):
    words = sorted(set(swap_map) | {t for spec in swap_map.values() for t in spec["targets"]})
    profile = load_substitute_profile(processed, words=words, results_dir=results_dir)
    for key in ("max_share_gap", "max_abs_coef"):
        profile.pop(key, None)
    for word, stats in profile["words"].items():
        stats["category"] = swap_map[word]["category"]
        stats["form"] = swap_map[word]["form"]
        stats["used_as_target"] = any(word in s["targets"] for s in swap_map.values())
    return profile


# ---------------------------------------------------------------------------
# Payload guard
# ---------------------------------------------------------------------------
def check_text_free(payload, texts=()):
    """Refuse a payload carrying a row id, a text hash, a text-bearing key or a population text."""
    def keys(value):
        if isinstance(value, dict):
            for k, v in value.items():
                yield k
                yield from keys(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                yield from keys(v)

    bad = sorted({k for k in keys(payload) if str(k).lower() in FORBIDDEN_KEYS})
    if bad:
        raise IdentitySwapError(f"payload carries text-bearing keys {bad}; nothing was written")
    serialized = json.dumps(payload, sort_keys=True)
    if ROW_ID_RE.search(serialized):
        raise IdentitySwapError("payload carries a row id; nothing was written")
    if TEXT_HASH_RE.search(serialized):
        raise IdentitySwapError("payload carries a text hash; nothing was written")
    for text in texts:
        if len(text) >= 20 and text in serialized:
            raise IdentitySwapError("payload carries population text; nothing was written")
    return payload


# ---------------------------------------------------------------------------
# One technique
# ---------------------------------------------------------------------------
def _variant_stats(p_original, variants):
    """Mean |delta| over posts and variants, and mean within-post spread across variants."""
    stack = np.vstack(variants)
    abs_delta = float(np.abs(p_original[None, :] - stack).mean()) if stack.size else float("nan")
    spread = float((stack.max(axis=0) - stack.min(axis=0)).mean()) if len(variants) > 1 and stack.size \
        else float("nan")
    return abs_delta, spread


def analyze(technique, model, population, pattern, swap_map, threshold,
            device=None, tolerance=DEFAULT_TOLERANCE, substitutes=SUBSTITUTE_WORDS,
            profile=None, map_sha256=None, excluded=None):
    """Verify on the originals, then score remove, substitute, swap and the directions."""
    substitutes = tuple(substitutes)
    restored_threshold = getattr(model, "threshold", None)
    if restored_threshold is not None and not math.isclose(float(restored_threshold), float(threshold)):
        raise IdentitySwapError(f"{technique}: restored threshold {restored_threshold} != committed {threshold}")
    if population.empty:
        raise IdentitySwapError(f"{technique}: no test post contains a swapped identity term")
    try:
        verification = verify_model(
            model, population[["row_id", "text"]], population[["row_id", "y_prob"]], tolerance)
    except CuratedExamplesError as error:
        raise IdentitySwapError(str(error)) from error

    texts = population["text"].tolist()
    variants = {
        "remove": [[remove_text(t, pattern) for t in texts]],
        "substitute": [[substitute_text(t, pattern, w) for t in texts] for w in substitutes],
        "swap": [[swap_text(t, pattern, swap_map, k) for t in texts] for k in range(N_TARGETS)],
    }
    directions = []  # (post index, source, target, text)
    for i, text in enumerate(texts):
        for source in sources_in(text, pattern):
            for target in swap_map[source]["targets"]:
                directions.append((i, source, target, swap_one_source(text, pattern, source, target)))

    unique = list(dict.fromkeys([t for op in variants.values() for v in op for t in v]
                                + [d[3] for d in directions]))
    scored = dict(zip(unique, np.asarray(model.predict(unique), dtype=float)))
    if len(scored) != len(unique):
        raise IdentitySwapError(f"{technique}: predictor returned the wrong number of scores")
    p_variants = {op: [np.array([scored[t] for t in v]) for v in vs] for op, vs in variants.items()}

    p_original = population["y_prob"].to_numpy(dtype=float)
    y_true = population["y_true"].to_numpy()
    rows, per_variant = [], []
    for label, name in POPULATIONS:
        mask = y_true == label
        for op in OPERATIONS:
            vs = [v[mask] for v in p_variants[op]]
            p_cf = np.mean(vs, axis=0) if vs[0].size else vs[0]
            row = {"technique": technique, "split": SPLIT, "population": name, "y_true": int(label),
                   "operation": op, "threshold": float(threshold), "n_variants": len(vs)}
            row.update(paired_metrics(p_original[mask], p_cf, threshold, label))
            row["mean_abs_delta"], row["mean_variant_spread"] = _variant_stats(p_original[mask], vs)
            row["verification_max_abs_diff"] = verification["max_abs_diff"]
            rows.append({k: _round(row[k]) for k in ROW_COLUMNS})
            if op != "remove":
                for k, v in enumerate(vs):
                    word = substitutes[k] if op == "substitute" else f"target_{k + 1}"
                    per_variant.append(_round({
                        "population": name, "operation": op, "variant": word,
                        "mean_delta": float((p_original[mask] - v).mean()) if v.size else float("nan"),
                        "n_flagged_counterfactual": int((v >= threshold).sum()),
                    }))

    grouped = defaultdict(list)
    for i, source, target, text in directions:
        name = dict(POPULATIONS)[int(y_true[i])]
        grouped[(name, swap_map[source]["category"], source, target)].append(p_original[i] - scored[text])
    direction_rows = [
        _round({"technique": technique, "split": SPLIT, "population": pop, "category": cat,
                "source": src, "target": tgt, "n_posts": len(d),
                "mean_delta": float(np.mean(d)), "mean_abs_delta": float(np.mean(np.abs(d)))})
        for (pop, cat, src, tgt), d in sorted(grouped.items())
    ]
    source_counts = defaultdict(int)
    for text in texts:
        for source in sources_in(text, pattern):
            source_counts[swap_map[source]["category"]] += 1

    payload = {
        "technique": technique,
        "split": SPLIT,
        "threshold": float(threshold),
        "device": device,
        "preregistration": PREREGISTRATION,
        "swap_map": str(IDENTITY_SWAP_MAP_CSV.relative_to(REPO_ROOT)),
        "swap_map_sha256": map_sha256,
        "n_swapped_sources": len(swap_map),
        "excluded_entries": sorted(excluded or ()),
        "operations": list(OPERATIONS),
        "primary_operation": PRIMARY_OPERATION,
        "aggregation": "mean probability over the variants (3 targets for swap, the WP7 panel for substitute)",
        "substitute_words": list(substitutes),
        "case_rule": "all-caps -> upper, capital first letter -> capitalised, otherwise lower",
        "n_population": len(texts),
        "n_population_by_label": population_counts(population),
        "n_source_post_pairs_by_category": dict(sorted(source_counts.items())),
        "n_texts_scored": len(unique),
        "p_original_source": "committed probability artifact",
        "holm_adjusted_p": f"see {RQ4_IDENTITY_SWAP_CSV.name}",
        "term_profile": _round(profile) if profile else None,
        "per_variant": per_variant,
        "verification": {k: _round(v) for k, v in verification.items()},
        "rows": rows,
        "directions": direction_rows,
    }
    return check_text_free(payload, texts)


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def write_outputs(payload, results_dir=None):
    check_text_free(payload)
    base = RESULTS_DIR if results_dir is None else Path(results_dir)
    folder = base / payload["technique"]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / IDENTITY_SWAP_FILENAME.format(split=payload["split"])
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def _merge(path, new, columns, key_columns):
    if path.exists():
        existing = pd.read_csv(path)
        if list(existing.columns) != list(columns):
            raise IdentitySwapError(f"{path.name}: columns differ from this version of the tool; rerun --all")
        existing = existing[~existing["technique"].isin(new["technique"])][list(key_columns)]
        new = pd.concat([existing, new], ignore_index=True)
    return new


def write_table(payloads, path=None, results_dir=None):
    """Merge rows into rq4_identity_swap.csv and recompute Holm per family."""
    path = Path(RQ4_IDENTITY_SWAP_CSV if path is None else path)
    new = pd.DataFrame([row for p in payloads for row in p["rows"]])[list(ROW_COLUMNS)]
    new = _merge(path, new, TABLE_COLUMNS, ROW_COLUMNS)
    new = new.sort_values(["technique", "y_true", "operation"]).reset_index(drop=True)
    family = new["technique"].map(lambda t: "" if in_comparison(t, results_dir) else t)
    for column in HOLM_COLUMNS:
        new[column] = np.nan
    is_swap = (new["operation"] == PRIMARY_OPERATION).to_numpy()
    for _, index in new.groupby(family, sort=False).groups.items():
        new.loc[index, "wilcoxon_p_holm"] = _holm_column(new.loc[index, "wilcoxon_p"])
        new.loc[index, "mcnemar_p_holm"] = _holm_column(new.loc[index, "mcnemar_exact_p"])
        sub = new.index.isin(index) & is_swap
        new.loc[sub, "wilcoxon_p_holm_swap"] = _holm_column(new.loc[sub, "wilcoxon_p"])
        new.loc[sub, "mcnemar_p_holm_swap"] = _holm_column(new.loc[sub, "mcnemar_exact_p"])
    path.parent.mkdir(parents=True, exist_ok=True)
    new[list(TABLE_COLUMNS)].to_csv(path, index=False, float_format=f"%.{SIG_DIGITS}g")
    return path


def write_directions(payloads, path=None):
    path = Path(RQ4_IDENTITY_SWAP_DIRECTIONS_CSV if path is None else path)
    new = pd.DataFrame([row for p in payloads for row in p["directions"]], columns=list(DIRECTION_COLUMNS))
    new = _merge(path, new, DIRECTION_COLUMNS, DIRECTION_COLUMNS)
    new = new.sort_values(["technique", "population", "category", "source", "target"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new[list(DIRECTION_COLUMNS)].to_csv(path, index=False, float_format=f"%.{SIG_DIGITS}g")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run(techniques, processed, swap_map, args, profile=None, map_sha256=None, excluded=None):
    """Analyse every technique in memory. Raises on the first fatal error; writes nothing."""
    payloads = []
    for technique in techniques:
        try:
            probs = load_technique_probs(technique, SPLIT)
            population, pattern = build_population(technique, processed, probs, swap_map)
            counts = population_counts(population)
            if args.dry_run_population:
                print(f"{technique}: swap population {len(population)} "
                      f"(non_extremist {counts['non_extremist']}, extremist {counts['extremist']})")
                continue
            threshold = load_threshold(technique)
            model = load_technique(technique, EXTERNAL_ROOT / technique, device=args.device,
                                   resident=not args.sequential_checkpoints)
            payload = analyze(technique, model, population, pattern, swap_map, threshold,
                              device=recorded_device(model, args.device), profile=profile,
                              map_sha256=map_sha256, excluded=excluded)
            del model
        except (ProbsArtifactError, ModelLoadError) as error:
            if args.all:
                print(f"skip {technique}: {error}")
                continue
            raise IdentitySwapError(str(error)) from error
        except IdentityFPRError as error:
            raise IdentitySwapError(str(error)) from error
        payloads.append(payload)
        for r in payload["rows"]:
            print(f"{technique} {r['population']} {r['operation']}: n={r['n_posts']} mean delta "
                  f"{r['mean_delta']} flagged {r['n_flagged_original']} -> {r['n_flagged_counterfactual']}")
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
    try:
        swap_map, excluded = check_swap_map(read_swap_map(), lexicons, lexicon_sections())
        check_substitute_word(lexicons)
        processed = build_processed_dataset()
        profile = term_profile(processed, swap_map)
        techniques = args.technique if args.technique else [d.name for d in technique_dirs()]
        payloads = run(techniques, processed, swap_map, args, profile=profile,
                       map_sha256=map_digest(), excluded=excluded)
    except (IdentitySwapError, IdentityCounterfactualError, AttributionArtifactError) as error:
        print(f"ERROR: {error}; nothing was written", file=sys.stderr)
        return 1

    if args.dry_run_population:
        return 0
    if not payloads:
        print("no technique could be evaluated", file=sys.stderr)
        return 1
    for payload in payloads:
        write_outputs(payload)
    print(f"wrote {len(payloads)} technique JSON(s), {write_table(payloads)} and {write_directions(payloads)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
