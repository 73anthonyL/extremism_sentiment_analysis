"""A static copy of the curated SHAP gallery that GitHub Pages can serve as is.

Reads the gallery `tools/curated_plots.py` wrote under `<out>/plots/` and
writes a site folder with `index.html` as the front page:

    index.html, examples/, models/   the gallery pages, each linking to the method page
    by_model/                        the beeswarm and waterfall images (the by_example/
                                     copies are left out; no page links to them)
    method.html                      how the plots were made, in plain English and technically
    README.md, .nojekyll             what the site is; serve the files without Jekyll
    .curated_site                    marks the folder as this tool's output

Every page gains a robots `noindex` meta, since the pages hold the curated
texts. Nothing is redrawn: the pages and images are the gallery's own. The
method page takes every number it states from the run's records (each
technique's run.json and committed best_config.json, the gallery's
checks.csv) and from the committed dataset manifest, so it describes the run
it sits beside.

The site holds the curated texts, so it belongs in its own folder under
`external/` (untracked here) and is published from there as a separate,
deliberate step. A rebuild replaces everything in the folder except `.git`,
so the folder can be its own repository.

The tool refuses, before writing anything: a plots folder without the
gallery, a gallery older than the explain outputs, techniques explained with
different explainer settings, a technique without its committed
best_config.json, a page that links to a file the site would not contain,
and a destination that holds files this tool did not write.

Usage:

    python3 tools/curated_site.py --out external/curated_examples_34 --site external/curated_shap_site
"""

import argparse
import html
import json
import posixpath
import re
import shutil
import sys
from pathlib import Path, PurePosixPath

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from curated_examples import DEFAULT_OUT, _expected_labels, load_outputs  # noqa: E402
from curated_plots import (  # noqa: E402
    CHECKS_FILENAME,
    DEFAULT_RAW_GAP_NOTE,
    DEFAULT_TOLERANCE as PLOT_TOLERANCE,
    LABEL_NAMES,
    PLOTS_DIRNAME,
)
from repo_paths import (  # noqa: E402
    DATASET_MANIFEST_JSON,
    NEGATIVE_CLASS_NAME,
    POSITIVE_CLASS_NAME,
    REPO_ROOT,
    RESULTS_DIR,
)

DEFAULT_SITE = REPO_ROOT / "external" / "curated_shap_site"
MARKER_FILENAME = ".curated_site"
MARKER_TEXT = "Written by tools/curated_site.py. A rebuild replaces everything here except .git.\n"
KEPT_ON_REBUILD = frozenset({".git"})
PAGE_FOLDERS = ("examples", "models")
IMAGE_FOLDER = "by_model"
METHOD_FILENAME = "method.html"
METHOD_TITLE = "How these plots were made"
VIEWPORT_META = '<meta name="viewport" content="width=device-width, initial-scale=1">'
ROBOTS_META = '<meta name="robots" content="noindex, nofollow">'
LINK_PATTERN = re.compile(r'(?:href|src)="([^"]*)"')
SCHEME_PATTERN = re.compile(r"^[a-z][a-z0-9+.-]*:", re.I)
STYLE_PATTERN = re.compile(r"<style>.*?</style>", re.S)
METHOD_STYLE = (
    ".method { max-width:820px; } .method li { margin-bottom:8px; } "
    "code { font-size:13px; background:var(--card); padding:0 4px; border-radius:3px; }"
)
PROJECT_URL = "https://github.com/73anthonyL/extremism_sentiment_analysis"
SEQUENTIAL_SERVING = "sequential_checkpoints"

# best_config.json family codes in words; an unknown code is shown as written.
MODEL_WORDS = {
    "logistic_regression": "Logistic regression",
    "calibrated_linear_svm": "Calibrated linear SVM",
    "single_layer_perceptron": "Single-layer perceptron",
    "fine_tuned_transformer_roberta": "Fine-tuned RoBERTa",
    "heterogeneous_checkpoint_logit_pooled_transformer_ensemble":
        "Fine-tuned transformer checkpoints, pooled in log-odds",
}
FEATURE_WORDS = {
    "tfidf_word": "Word TF-IDF",
    "tfidf_char": "Character TF-IDF",
    "tfidf_word_char": "Word and character TF-IDF",
    "fasttext_document_embeddings": "FastText document embeddings",
    "contextual_transformer_sequence_representation": "Contextual token representations",
    "contextual_transformer_logit_probability_pooling": "Each checkpoint's contextual representations",
}


class CuratedSiteError(ValueError):
    """The gallery cannot be turned into a faithful static site."""


def adapt_page(page, name):
    """One gallery page with the robots meta and a link to the method page."""
    prefix = "../" * name.count("/")
    nav = f'<nav><a href="{prefix}index.html">Overview</a></nav>'
    if page.count(VIEWPORT_META) != 1 or page.count(nav) != 1:
        raise CuratedSiteError(f"{name}: not a curated_plots.py page (expected one viewport meta and one nav)")
    page = page.replace(VIEWPORT_META, VIEWPORT_META + ROBOTS_META)
    return page.replace(nav, f'<nav><a href="{prefix}index.html">Overview</a>'
                             f'<a href="{prefix}{METHOD_FILENAME}">{METHOD_TITLE}</a></nav>')


def check_links(pages, files):
    """Refuse a relative href/src that leaves the site or names a file it will not contain.

    `pages` maps a site-relative page path to its HTML; `files` is every
    site-relative path the site will hold.
    """
    broken = []
    for name, text in pages.items():
        base = PurePosixPath(name).parent
        for target in LINK_PATTERN.findall(text):
            target = target.split("#", 1)[0]
            if not target or SCHEME_PATTERN.match(target):
                continue
            resolved = posixpath.normpath((base / target).as_posix())
            if resolved.startswith("..") or resolved not in files:
                broken.append(f"{name} -> {target}")
    if broken:
        raise CuratedSiteError(
            f"{len(broken)} link(s) leave the site or point to a file it would not contain, "
            f"e.g. {', '.join(broken[:3])}"
        )


def readme(n_texts, n_techniques):
    return (
        "# Curated SHAP gallery\n\n"
        f"SHAP waterfall and beeswarm plots for {n_texts} hand-written curated texts under each of "
        f"the {n_techniques} techniques of [extremism_sentiment_analysis]({PROJECT_URL}). "
        f"Open `index.html`; `{METHOD_FILENAME}` explains how the plots were made.\n\n"
        "Built by `tools/curated_site.py` from the gallery of `tools/curated_plots.py`. The curated "
        "texts are not dataset rows; they illustrate model behaviour and support no accuracy claim. "
        "This is not a moderation tool.\n"
    )


# ---------------------------------------------------------------------------
# Method page
# ---------------------------------------------------------------------------
def method_facts(out_dir, results_dir=RESULTS_DIR, manifest_path=DATASET_MANIFEST_JSON):
    """Every number the method page states, read from the run's records and the committed files."""
    out_dir, results_dir = Path(out_dir), Path(results_dir)
    examples, _, _, records = load_outputs(out_dir)
    techniques = sorted(records)
    checks = pd.read_csv(out_dir / PLOTS_DIRNAME / CHECKS_FILENAME)
    drawn = sorted(checks["technique"].unique())
    if drawn != techniques:
        raise CuratedSiteError(
            f"the gallery covers {drawn} but the explain outputs hold {techniques}; the gallery "
            f"is older than the outputs, so rerun `curated_plots.py --out {out_dir}`"
        )
    explainers = {json.dumps(records[t].get("explainer"), sort_keys=True) for t in techniques}
    if len(explainers) != 1 or records[techniques[0]].get("explainer") is None:
        raise CuratedSiteError(
            "the techniques were explained with different (or unrecorded) explainer settings; "
            "the method page states one explainer"
        )

    rows = []
    for technique in techniques:
        config_path = results_dir / technique / "best_config.json"
        if not config_path.is_file():
            raise CuratedSiteError(f"{technique}: no committed {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        record = records[technique]
        verification = record.get("verification") or {}
        rows.append({
            "technique": technique,
            "model": MODEL_WORDS.get(config.get("model_family"), str(config.get("model_family"))),
            "features": FEATURE_WORDS.get(config.get("feature_family"), str(config.get("feature_family"))),
            "pretraining": config.get("external_pretraining") or "none",
            "threshold_metric": config.get("threshold_metric"),
            "threshold": record.get("threshold"),
            "max_abs_diff": verification.get("max_abs_diff"),
            "verify_rows": verification.get("n_rows"),
            "verify_tolerance": verification.get("tolerance"),
            "serving": record.get("serving") or {},
        })

    labels = _expected_labels(examples)
    noted = checks[checks["raw_vs_explained_gap"].abs() > DEFAULT_RAW_GAP_NOTE]
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return {
        "raw_rows": manifest["raw_rows"],
        "processed_rows": manifest["processed_rows"],
        "rows_removed": manifest["rows_removed_summary"]["rows_removed_total"],
        "n_positive": manifest["label_counts"][str(manifest["positive_class"])],
        "n_negative": manifest["label_counts"][str(manifest["negative_class"])],
        "splits": manifest["split_counts_actual"],
        "split_version": manifest["split_version"],
        "example_ids": examples["example_id"].astype(str).tolist(),
        "expected": {name: sum(1 for v in labels.values() if v == name)
                     for name in (POSITIVE_CLASS_NAME, NEGATIVE_CLASS_NAME)},
        "techniques": rows,
        "explainer": records[techniques[0]]["explainer"],
        "n_waterfalls": int(len(checks)),
        "max_additivity_gap": float(checks["additivity_gap"].abs().max()),
        "raw_gap_notes": {str(t): int(n) for t, n in noted.groupby("technique").size().items()},
    }


def _number(value):
    """A probability-scale number in HTML: 0.0023, or 2.2x10^-6 below a thousandth."""
    if value is None:
        return "not checked"
    if value == 0:
        return "0"
    if abs(value) >= 1e-3:
        return f"{value:.4f}".rstrip("0").rstrip(".")
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa}&times;10<sup>{int(exponent)}</sup>"


def _threshold(value):
    return "" if value is None else f"{value:.3f}"


def _one_of(values, what):
    """The one recorded value, or every distinct value when the techniques disagree."""
    found = sorted({str(v) for v in values if v is not None})
    if not found:
        return "unrecorded"
    if len(found) == 1:
        return found[0]
    return f"{what} " + " / ".join(found)


def method_page(facts, style):
    """The method page: plain English, then technical, every number from `facts`."""
    esc = html.escape
    rows = facts["techniques"]
    n_models = len(rows)
    n_texts = len(facts["example_ids"])
    splits = facts["splits"]
    explainer = facts["explainer"]
    verified = [r for r in rows if r["max_abs_diff"] is not None]
    verify_rows = _one_of([r["verify_rows"] for r in verified], "rows")
    verify_tolerance = _one_of([r["verify_tolerance"] for r in verified], "tolerances")
    metric = _one_of([r["threshold_metric"] for r in rows], "metrics")
    n_pos, n_neg = facts["expected"][POSITIVE_CLASS_NAME], facts["expected"][NEGATIVE_CLASS_NAME]
    share = 100.0 * facts["n_positive"] / facts["processed_rows"]
    n_noted = sum(facts["raw_gap_notes"].values())

    if n_pos + n_neg:
        expected = (f"{n_pos} are meant to be {LABEL_NAMES[POSITIVE_CLASS_NAME]} and {n_neg} "
                    f"{LABEL_NAMES[NEGATIVE_CLASS_NAME]}, and none come from the dataset.")
    else:
        expected = "None come from the dataset, and they carry no expected label."
    unverified = n_models - len(verified)
    checked = (f"{len(verified)} of {n_models} models were checked this way, and all matched."
               + (f" {unverified} were not checked in this run." if unverified else ""))
    sequential = [r for r in rows if r["serving"].get("mode") == SEQUENTIAL_SERVING]
    sequential_text = "".join(
        f" {esc(r['technique'])} was explained one checkpoint at a time, replaying each text "
        f"against a cache of every member's log-odds ({r['serving'].get('rounds', '?')} rounds); a test "
        "shows this reproduces explaining the whole ensemble at once."
        for r in sequential
    )
    if n_noted:
        per_technique = ", ".join(f"{esc(t)}: {n}" for t, n in sorted(facts["raw_gap_notes"].items()))
        raw_note = (f"The {n_noted} of {facts['n_waterfalls']} waterfalls where the two differ by more "
                    f"than {DEFAULT_RAW_GAP_NOTE:g} ({per_technique}) carry a note.")
    else:
        raw_note = f"No waterfall differs from the raw-text probability by more than {DEFAULT_RAW_GAP_NOTE:g}."

    plain = [
        f"<b>We started with a labelled dataset.</b> {facts['processed_rows']:,} social-media posts, "
        f"each labelled by people as extremist or not extremist; {share:.0f}% are extremist.",
        f"<b>We split it once and never changed the split.</b> {splits['train']:,} posts are for "
        f"training, {splits['validation']:,} for tuning and {splits['test']:,} for the final test. "
        "Every model is trained, tuned and tested on the same posts.",
        f"<b>We trained {n_models} models</b>, listed in the table below. Each learns from the training "
        "posts only.",
        "<b>Each model got its own decision line.</b> A model outputs a score from 0 to 1, how "
        "extremist it thinks a post is. Each model's cutoff for \"call it extremist\" was chosen on the "
        "tuning posts and then locked. The test posts were scored once at the end.",
        "<b>We saved each trained model.</b> Training ran on Kaggle. The trained models were "
        "downloaded for this run, and the scores they gave the test posts are saved in the research "
        "repository.",
        f"<b>{n_texts} new posts were written by hand.</b> {expected}",
        f"<b>Each downloaded model was checked before it was trusted.</b> It re-scored {verify_rows} "
        f"test posts, and its scores had to match the saved ones to within {verify_tolerance}. {checked}",
        "<b>SHAP asked each model which words mattered.</b> For each post it hid different groups of "
        f"words (a hidden word becomes \"{esc(explainer.get('mask_token', ''))}\") and watched how the "
        f"model's score moved, with a budget of {explainer.get('max_evals')} model calls per post. Each "
        "word gets a number: positive if it pushed the post toward extremist, negative if it pushed away.",
        "<b>A waterfall shows one post under one model.</b> It starts from the model's score when every "
        "word is hidden, adds each word's push one bar at a time, and ends at the model's actual score. "
        f"The dashed line is that model's locked cutoff. {n_texts} posts &times; {n_models} models = "
        f"{facts['n_waterfalls']} waterfalls.",
        "<b>A beeswarm shows one model across all the posts.</b> It lists the words that mattered most "
        "to that model overall. Each dot is one post containing that word, placed by how hard the word "
        "pushed and coloured by the post's expected label.",
        "<b>This site organises everything:</b> an overview grid, one page per post and one page per model.",
    ]

    table_rows = "".join(
        f"<tr><td>{esc(r['technique'])}</td><td>{esc(r['model'])}</td><td>{esc(r['features'])}</td>"
        f"<td>{esc(str(r['pretraining']))}</td>"
        f"<td class=\"num\">{_threshold(r['threshold'])}</td>"
        f"<td class=\"num\">{_number(r['max_abs_diff'])}</td></tr>"
        for r in rows
    )
    table = (
        '<div class="scroll"><table><thead><tr><th>Technique</th><th>Model</th><th>Features</th>'
        "<th>External pretraining</th><th>Locked threshold</th><th>Restored vs committed, max |diff|</th>"
        f"</tr></thead><tbody>{table_rows}</tbody></table></div>"
    )
    technical = [
        f"<b>Data.</b> <code>data/dataset.csv</code> has {facts['raw_rows']:,} raw rows; processing "
        f"leaves {facts['processed_rows']:,} ({facts['rows_removed']} removed) with post-level binary "
        f"labels: {facts['n_positive']:,} extremist and {facts['n_negative']:,} non-extremist. Notebook 00 "
        f"reproduces the frozen split <code>{esc(facts['split_version'])}</code> ({splits['train']:,} train, "
        f"{splits['validation']:,} validation, {splits['test']:,} test) and verifies it against the "
        "committed split file; the split is never regenerated.",
        "<b>Models and thresholds (Kaggle).</b> Every notebook uses one shared helper, fits on train "
        f"only, chooses its decision threshold on validation only by maximising validation {esc(metric)} "
        "over a fixed grid, locks it, and evaluates the test split once. Test probabilities are "
        "committed; weights live in external storage." + table,
        "<b>Research-side SHAP.</b> Each notebook also explains test and validation posts and exports "
        "aggregated word attributions for the research questions. The curated plots here reuse the same "
        "explainer but not those files.",
        f"<b>Curated inputs.</b> {n_texts} hand-written texts, one per line, become "
        f"<code>{esc(facts['example_ids'][0])}</code> to <code>{esc(facts['example_ids'][-1])}</code>; "
        "expected labels come from a separate file in the same order. They are not dataset rows.",
        "<b>Restoring predictors.</b> A loader rebuilds each notebook's <code>predict_proba_texts</code> "
        "from the downloaded weights and the locked threshold, reproducing each notebook's preprocessing; "
        "tests pin these to the notebooks.",
        f"<b>Verification.</b> {verify_rows} seeded test rows are re-scored and compared with the "
        f"committed test probabilities; a difference above {verify_tolerance} is refused. The table "
        "above lists each model's largest difference.",
        "<b>Attribution.</b> <code>shap.Explainer(predict, shap.maskers.Text(r\""
        f"{esc(explainer.get('masker', ''))}\", mask_token=\"{esc(explainer.get('mask_token', ''))}\"), "
        f"algorithm=\"{esc(explainer.get('algorithm', ''))}\")</code> with "
        f"<code>max_evals={explainer.get('max_evals')}</code> and <code>seed={explainer.get('seed')}</code>. "
        "The text masker splits a post into word tokens and builds a hierarchical partition tree over "
        "them; the partition explainer estimates Owen values (a hierarchy-respecting approximation of "
        "Shapley values) by replacing groups of words with the mask token and querying the model. "
        f"Attributions are in probability units of P(extremist).{sequential_text}",
        "<b>Endpoints and additivity.</b> Per text, the base value is the model on the fully masked text "
        "and the explained output is the model on the masker's full-text reconstruction. Base value + "
        "&Sigma; attributions = explained output; the largest observed gap is "
        f"{_number(facts['max_additivity_gap'])}. The reconstruction drops trailing separators such as a "
        "final \".\", so the explained output can differ from the probability on the raw text. "
        + raw_note,
        "<b>Plot gate.</b> Plotting is refused if an endpoint is missing, a value falls outside [0, 1], "
        "token positions are not exactly 0 to n&minus;1, or base value + attributions misses the "
        f"explained output by more than the plotting tolerance ({PLOT_TOLERANCE:g} by default).",
        "<b>Waterfalls.</b> <code>shap.plots.waterfall</code> on each text's token explanation, the "
        "largest bars shown and the rest grouped as \"other features\". The x axis is P(extremist), the "
        "base is labelled \"masked\" instead of E[f(X)], and a dashed line marks the locked threshold.",
        "<b>Beeswarms.</b> SHAP's built-in beeswarm needs one feature set shared by every sample, which "
        "text lacks, so these are drawn directly. Tokens are normalised to words and a word's occurrences "
        "within one text are summed; words are ranked by total |SHAP| across the texts, one dot per "
        "(word, text), x = SHAP value, colour = expected label, placed with a swarm layout.",
    ]

    items = lambda values: "".join(f"<li>{v}</li>" for v in values)  # noqa: E731
    style = style.replace("</style>", f"{METHOD_STYLE}</style>")
    return (
        f"<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">{VIEWPORT_META}{ROBOTS_META}"
        f"<title>{METHOD_TITLE}</title>{style}</head><body><main class=\"method\">"
        f"<nav><a href=\"index.html\">Overview</a><a href=\"{METHOD_FILENAME}\">{METHOD_TITLE}</a></nav>"
        f"<h1>{METHOD_TITLE}</h1>"
        '<p class="muted">From data to model to explanation, in plain English and in technical terms.</p>'
        f"<h2>Plain English</h2><ol>{items(plain)}</ol>"
        f"<h2>Technical</h2><ol>{items(technical)}</ol>"
        '<p class="muted">The curated texts are illustrations: a handful of hand-picked posts supports '
        "no accuracy or model-ranking claim. Labels are post-level and features text-only; these models "
        "detect extremist content, not accounts or radicalisation over time. Source: "
        f'<a href="{PROJECT_URL}">extremism_sentiment_analysis</a>.</p>'
        "</main></body></html>\n"
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _prepare_destination(site_dir):
    """Empty the destination except .git, refusing a folder this tool did not write."""
    if site_dir.exists():
        leftovers = [p for p in site_dir.iterdir() if p.name not in KEPT_ON_REBUILD]
        if leftovers and not (site_dir / MARKER_FILENAME).is_file():
            raise CuratedSiteError(
                f"{site_dir} holds {len(leftovers)} entries this tool did not write (no "
                f"{MARKER_FILENAME}); choose a new or empty --site folder"
            )
        for path in leftovers:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    site_dir.mkdir(parents=True, exist_ok=True)


def run_site(out_dir=DEFAULT_OUT, site_dir=DEFAULT_SITE, results_dir=RESULTS_DIR,
             manifest_path=DATASET_MANIFEST_JSON, log=print):
    """Check the gallery and the records, then write the site folder. Returns the site folder."""
    out_dir, site_dir = Path(out_dir), Path(site_dir)
    plots_dir = out_dir / PLOTS_DIRNAME
    if not (plots_dir / "index.html").is_file():
        raise CuratedSiteError(f"{plots_dir}: no gallery; run `curated_plots.py --out {out_dir}` first")

    sources = [plots_dir / "index.html"]
    for folder in PAGE_FOLDERS:
        sources += sorted((plots_dir / folder).glob("*.html"))
    pages = {}
    for source in sources:
        name = source.relative_to(plots_dir).as_posix()
        pages[name] = adapt_page(source.read_text(encoding="utf-8"), name)
    facts = method_facts(out_dir, results_dir, manifest_path)
    style = STYLE_PATTERN.search((plots_dir / "index.html").read_text(encoding="utf-8"))
    if style is None:
        raise CuratedSiteError(f"{plots_dir / 'index.html'}: no <style> to share with the method page")
    pages[METHOD_FILENAME] = method_page(facts, style.group(0))
    images = sorted(p for p in (plots_dir / IMAGE_FOLDER).rglob("*") if p.is_file())
    extras = {
        "README.md": readme(len(facts["example_ids"]), len(facts["techniques"])),
        ".nojekyll": "",
        MARKER_FILENAME: MARKER_TEXT,
    }
    files = set(pages) | set(extras) | {p.relative_to(plots_dir).as_posix() for p in images}
    check_links(pages, files)

    _prepare_destination(site_dir)
    for name, text in {**pages, **extras}.items():
        target = site_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    for image in images:
        target = site_dir / image.relative_to(plots_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(image, target)
    log(f"wrote {len(pages)} pages and {len(images)} images to {site_dir}")
    return site_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="an output folder of curated_examples.py explain")
    parser.add_argument("--site", default=str(DEFAULT_SITE), help="the site folder to (re)write")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR),
                        help="committed result folders holding each technique's best_config.json")
    args = parser.parse_args(argv)
    try:
        run_site(args.out, args.site, args.results_dir)
    except CuratedSiteError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
