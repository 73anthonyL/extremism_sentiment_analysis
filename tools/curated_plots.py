"""SHAP waterfall and beeswarm plots for the curated examples, with an HTML gallery.

Reads an output folder of `tools/curated_examples.py explain` and draws, for
the whole model of every technique (ensemble members are not plotted):

* one waterfall per curated text: from the model on the fully masked text
  (the explainer's base value) through each word's SHAP value to the model on
  the text as the masker rebuilt it, with the locked threshold marked;
* one beeswarm per technique: the words with the largest total |SHAP| over
  the curated texts, one dot per text containing the word, coloured by the
  text's expected label. SHAP's own beeswarm needs one shared feature set
  across samples, which text does not have, so this is drawn here.

Every plot shows the attributions `explain` computed with the kit's one
explainer; nothing is re-explained. Before drawing, the tool refuses a run
whose base value plus attributions does not reproduce the explained output,
since such a waterfall would misstate the model.

The masker's rebuilt text drops trailing separators (a final '.' or '?'), so
the explained output can differ from the model on the raw text, which decides
the predicted class. Waterfalls note the raw-text probability when the two
differ, and `checks.csv` records both.

Outputs go under `<out>/plots/` and hold the curated texts, so they stay
untracked like the rest of the folder:

    index.html                     overview: every text x technique, links to all pages
    examples/<example_id>.html     one text: its predictions and all waterfalls
    models/<TECHNIQUE>.html        one technique: its beeswarm and all waterfalls
    by_model/<TECHNIQUE>/          beeswarm.png, waterfall_<example_id>.png
    by_example/<example_id>/       <TECHNIQUE>.png (copies, for slides)
    checks.csv                     text-free: endpoints and gaps per text x technique

Usage:

    python3 tools/curated_plots.py --out external/curated_examples_34
"""

import argparse
import html
import shutil
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from curated_examples import (  # noqa: E402
    BASE_VALUE_COLUMN,
    DEFAULT_OUT,
    EXPECTED_LABEL_COLUMN,
    EXPLAINED_OUTPUT_COLUMN,
    WHOLE_MODEL,
    CuratedExamplesError,
    _expected_labels,
    load_outputs,
)
from notebook_kit import normalize_word  # noqa: E402
from repo_paths import NEGATIVE_CLASS_NAME, POSITIVE_CLASS_NAME  # noqa: E402

PLOTS_DIRNAME = "plots"
CHECKS_FILENAME = "checks.csv"
DEFAULT_TOLERANCE = 1e-3
DEFAULT_RAW_GAP_NOTE = 5e-3
DEFAULT_TOP_WORDS = 20
DEFAULT_MAX_DISPLAY = 15
DPI = 150
LABEL_COLOURS = {POSITIVE_CLASS_NAME: "#c8414b", NEGATIVE_CLASS_NAME: "#2f6db5"}
LABEL_NAMES = {POSITIVE_CLASS_NAME: "extremist", NEGATIVE_CLASS_NAME: "non-extremist"}


class CuratedPlotsError(ValueError):
    """The explain outputs cannot be plotted faithfully."""


# ---------------------------------------------------------------------------
# Loading and checks
# ---------------------------------------------------------------------------
def load_whole_model_outputs(out_dir, tolerance=DEFAULT_TOLERANCE):
    """(examples, predictions, attributions, checks) for the whole model of every technique.

    Refuses outputs that cannot carry a faithful waterfall: written before the
    explainer endpoints were recorded, with gaps or repeats in a text's token
    positions, or whose base value plus attributions misses the explained
    output by more than `tolerance`.
    """
    try:
        examples, predictions, attributions, _ = load_outputs(out_dir)
    except CuratedExamplesError as error:
        raise CuratedPlotsError(str(error)) from error
    predictions = predictions[predictions["member"] == WHOLE_MODEL].copy()
    attributions = attributions[attributions["member"] == WHOLE_MODEL].copy()

    for column in (BASE_VALUE_COLUMN, EXPLAINED_OUTPUT_COLUMN):
        if column not in predictions.columns:
            raise CuratedPlotsError(
                f"{out_dir}: predictions lack '{column}'; they were written before the "
                "explainer endpoints were recorded, so rerun `curated_examples.py explain`"
            )
        values = pd.to_numeric(predictions[column], errors="coerce")
        # A technique folder written before the endpoints existed leaves blanks after concat.
        missing = predictions.loc[values.isna(), ["technique", "example_id"]]
        if len(missing):
            raise CuratedPlotsError(
                f"{missing.iloc[0]['technique']}: predictions lack '{column}' for {len(missing)} "
                "text(s); they were written before the explainer endpoints were recorded, so "
                "rerun `curated_examples.py explain`"
            )
        bad = predictions.loc[~np.isfinite(values) | (values < 0) | (values > 1), ["technique", "example_id"]]
        if len(bad):
            raise CuratedPlotsError(
                f"'{column}' is not a probability for {len(bad)} text(s), e.g. "
                f"{bad.iloc[0]['technique']} {bad.iloc[0]['example_id']}"
            )
        predictions[column] = values

    keys = ["technique", "example_id"]
    if predictions.duplicated(keys).any():
        raise CuratedPlotsError("a technique has more than one whole-model prediction for a text")
    for (technique, example_id), block in attributions.groupby(keys):
        positions = block["position"].to_numpy()
        if sorted(positions.tolist()) != list(range(len(positions))):
            raise CuratedPlotsError(
                f"{technique} {example_id}: token positions are not 0..{len(positions) - 1} "
                "exactly once each"
            )
    attributions = attributions.sort_values(keys + ["position"]).reset_index(drop=True)

    sums = attributions.groupby(keys)["attribution"].sum().rename("sum_attribution").reset_index()
    checks = predictions.merge(sums, on=keys, how="left")
    unexplained = checks[checks["sum_attribution"].isna()]
    if len(unexplained):
        row = unexplained.iloc[0]
        raise CuratedPlotsError(
            f"{len(unexplained)} text(s) have a prediction but no attributions, e.g. "
            f"{row['technique']} {row['example_id']}"
        )
    checks["additivity_gap"] = (
        checks[BASE_VALUE_COLUMN] + checks["sum_attribution"] - checks[EXPLAINED_OUTPUT_COLUMN]
    )
    checks["raw_vs_explained_gap"] = checks["y_prob"] - checks[EXPLAINED_OUTPUT_COLUMN]
    broken = checks[checks["additivity_gap"].abs() > tolerance]
    if len(broken):
        worst = broken.loc[broken["additivity_gap"].abs().idxmax()]
        raise CuratedPlotsError(
            f"{len(broken)} waterfall(s) would not add up: base value + attributions misses the "
            f"explained output by up to {abs(worst['additivity_gap']):.4f} "
            f"({worst['technique']} {worst['example_id']}; tolerance {tolerance})"
        )
    checks = checks[
        keys
        + [
            "threshold",
            BASE_VALUE_COLUMN,
            "sum_attribution",
            EXPLAINED_OUTPUT_COLUMN,
            "y_prob",
            "y_pred",
            "additivity_gap",
            "raw_vs_explained_gap",
        ]
    ].sort_values(keys)
    return examples, predictions, attributions, checks.reset_index(drop=True)


def word_contributions(attributions):
    """Per technique, text and word: the summed attribution of the word's occurrences."""
    frame = attributions.copy()
    frame["word"] = frame["token"].map(normalize_word)
    frame = frame[frame["word"] != ""]
    return (
        frame.groupby(["technique", "example_id", "word"])["attribution"].sum().reset_index()
    )


def top_words(contributions, technique, n_words):
    """The technique's n words with the largest total |SHAP| over the texts, most first."""
    block = contributions[contributions["technique"] == technique]
    totals = block.assign(abs_value=block["attribution"].abs()).groupby("word")["abs_value"].sum()
    totals = totals.sort_values(ascending=False, kind="mergesort")
    return totals.index[:n_words].tolist()


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _display_token(token):
    shown = str(token).strip()
    return shown if shown else "(space)"


def _class_name(y_pred):
    return POSITIVE_CLASS_NAME if int(y_pred) == 1 else NEGATIVE_CLASS_NAME


def draw_waterfall(path, tokens, values, base_value, explained, raw_prob, threshold, title,
                   max_display=DEFAULT_MAX_DISPLAY, raw_gap_note=DEFAULT_RAW_GAP_NOTE):
    """SHAP's waterfall for one text, relabelled for a text masker, threshold marked."""
    plt = _pyplot()
    import shap

    explanation = shap.Explanation(
        values=np.asarray(values, dtype=float),
        base_values=float(base_value),
        data=None,
        feature_names=[_display_token(t) for t in tokens],
    )
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    fig = plt.gcf()
    main = fig.axes[0]

    # The base value is the model on the fully masked text, not an expectation
    # over a background set. Edit SHAP's label strings in place: resetting the
    # ticks would drop the offsets SHAP gives the label and its value.
    from matplotlib.ticker import FuncFormatter

    for axis in fig.axes[1:]:
        original = axis.xaxis.get_major_formatter()
        axis.xaxis.set_major_formatter(
            FuncFormatter(lambda x, pos, f=original: f(x, pos).replace("$E[f(X)]$", "masked"))
        )

    low, high = main.get_xlim()
    pad = 0.04 * max(high - low, 1e-6)
    low, high = min(low, threshold - pad), max(high, threshold + pad)
    for axis in fig.axes:
        axis.set_xlim(low, high)
    main.axvline(threshold, color="#555555", linestyle="--", linewidth=1, zorder=0)
    fig.suptitle(f"{title}\nx axis: P(extremist); dashed line: locked threshold {threshold:.3f}",
                 fontsize=10, y=1.02, x=0.02, ha="left")
    if abs(raw_prob - explained) > raw_gap_note:
        fig.text(
            0.02, -0.02,
            f"Model on the raw text: P = {raw_prob:.3f}. SHAP explains the text as its masker "
            f"rebuilt it (trailing punctuation dropped), where P = {explained:.3f}.",
            fontsize=8, color="#555555", ha="left", va="top",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _swarm_offsets(values, n_bins=100, row_height=0.8, global_range=None):
    """SHAP beeswarm's vertical layout: stack points that fall in the same x bin."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    low, high = global_range if global_range is not None else (values.min(), values.max())
    quant = np.round(n_bins * (values - low) / (high - low + 1e-8))
    order = np.argsort(quant, kind="mergesort")
    offsets = np.zeros(len(values))
    layer, last_bin = 0, None
    for index in order:
        if quant[index] != last_bin:
            layer = 0
        offsets[index] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
        layer += 1
        last_bin = quant[index]
    peak = np.max(np.abs(offsets)) + 1
    return offsets * 0.9 * (row_height / 2) / peak


def draw_beeswarm(path, contributions, technique, labels, n_texts, n_words=DEFAULT_TOP_WORDS,
                  collection="curated texts", unit="text", label_prefix="expected", dot_size=26):
    """One dot per (word, text containing it) for the technique's top words.

    `collection`, `unit` and `label_prefix` word the title and legend, so the
    same drawing serves texts with an expected label and dataset posts with a
    true one; `dot_size` keeps hundreds of dots per row legible.
    """
    plt = _pyplot()
    words = top_words(contributions, technique, n_words)
    block = contributions[(contributions["technique"] == technique) & (contributions["word"].isin(words))]
    fig, ax = plt.subplots(figsize=(8, 0.42 * max(len(words), 1) + 1.6))
    if block.empty:
        ax.text(0.5, 0.5, "no attributed words", ha="center", va="center", transform=ax.transAxes)
    else:
        span = (block["attribution"].min(), block["attribution"].max())
        for row, word in enumerate(words):
            y = len(words) - 1 - row
            points = block[block["word"] == word]
            offsets = _swarm_offsets(points["attribution"].to_numpy(), global_range=span)
            ax.axhline(y, color="#dddddd", linewidth=0.6, zorder=0)
            colours = [LABEL_COLOURS.get(labels.get(e), "#888888") for e in points["example_id"]]
            ax.scatter(points["attribution"], y + offsets, c=colours, s=dot_size, linewidths=0, zorder=2)
        ax.set_yticks(range(len(words)))
        support = block.groupby("word")["example_id"].nunique()
        ax.set_yticklabels([f"{w}  ({support[w]})" for w in reversed(words)])
        ax.axvline(0, color="#999999", linewidth=0.8, zorder=1)
        ax.set_ylim(-0.7, len(words) - 0.3)
    for name, colour in ((POSITIVE_CLASS_NAME, LABEL_COLOURS[POSITIVE_CLASS_NAME]),
                         (NEGATIVE_CLASS_NAME, LABEL_COLOURS[NEGATIVE_CLASS_NAME])):
        ax.scatter([], [], c=colour, s=26, label=f"{label_prefix} {LABEL_NAMES[name]}")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_xlabel("SHAP value (change in P(extremist))")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(
        f"{technique}: top {len(words)} words by total |SHAP| over {n_texts} {collection}\n"
        f"one dot per {unit} containing the word; (n) = number of such {unit}s",
        fontsize=10, loc="left",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Gallery
# ---------------------------------------------------------------------------
STYLE = """
:root { --bg:#ffffff; --fg:#1d1f23; --muted:#5f6670; --line:#e2e5ea; --card:#f6f7f9;
        --ok:#1f7a4a; --bad:#b3261e; --pos:#c8414b; --neg:#2f6db5; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#16181c; --fg:#e8eaed; --muted:#a0a6ae; --line:#30343a; --card:#1f2227;
          --ok:#5cc28c; --bad:#f08a80; --pos:#e2737b; --neg:#7aa7e6; }
  img.plot { background:#ffffff; }
}
* { box-sizing:border-box; }
body { margin:0; padding:24px 16px 64px; background:var(--bg); color:var(--fg);
       font:15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
main { max-width:1200px; margin:0 auto; }
a { color:var(--neg); }
h1 { font-size:24px; margin:0 0 4px; } h2 { font-size:18px; margin:32px 0 8px; }
.muted { color:var(--muted); } nav { margin-bottom:16px; font-size:14px; }
nav a { margin-right:14px; }
.text { background:var(--card); border-left:4px solid var(--line); padding:12px 16px;
        border-radius:6px; font-size:17px; margin:12px 0; }
.text.EXTREMIST { border-left-color:var(--pos); } .text.NON_EXTREMIST { border-left-color:var(--neg); }
.tag { display:inline-block; padding:1px 8px; border-radius:10px; font-size:12px; color:#fff; }
.tag.EXTREMIST { background:var(--pos); } .tag.NON_EXTREMIST { background:var(--neg); }
.ok { color:var(--ok); font-weight:600; } .bad { color:var(--bad); font-weight:600; }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; font-size:13px; width:100%; }
th, td { border-bottom:1px solid var(--line); padding:5px 8px; text-align:left; vertical-align:top; }
th { position:sticky; top:0; background:var(--bg); }
td.num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
.grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(460px, 1fr)); gap:16px; }
.grid.small { grid-template-columns:repeat(auto-fill, minmax(340px, 1fr)); }
figure { margin:0; background:var(--card); border-radius:8px; padding:10px; }
figcaption { font-size:13px; margin-bottom:6px; }
img.plot { width:100%; height:auto; display:block; border-radius:4px; }
@media (max-width:520px) { .grid, .grid.small { grid-template-columns:1fr; } }
"""


def _page(title, body, depth):
    prefix = "../" * depth
    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>{html.escape(title)}</title><style>{STYLE}</style></head><body><main>"
        f"<nav><a href=\"{prefix}index.html\">Overview</a></nav>{body}</main></body></html>\n"
    )


def _verdict(expected, y_pred):
    if not expected:
        return ""
    if expected == _class_name(y_pred):
        return '<span class="ok">&#10003;</span>'
    return '<span class="bad">&#10007;</span>'


def _tag(label):
    if not label:
        return ""
    return f'<span class="tag {label}">expected {LABEL_NAMES[label]}</span>'


def _waterfall_path(technique, example_id):
    return Path("by_model") / technique / f"waterfall_{example_id}.png"


def write_gallery(plots_dir, examples, checks, techniques, labels):
    texts = dict(zip(examples["example_id"].astype(str), examples["text"].astype(str)))
    ids = examples["example_id"].astype(str).tolist()
    by_key = {(r.technique, r.example_id): r for r in checks.itertuples()}

    # Overview
    head = "".join(f"<th>{html.escape(t)}</th>" for t in techniques)
    rows = []
    for example_id in ids:
        expected = labels.get(example_id, "")
        cells = []
        for technique in techniques:
            r = by_key[(technique, example_id)]
            cells.append(
                f'<td class="num"><a href="examples/{example_id}.html">{r.y_prob:.2f}</a> '
                f"{_verdict(expected, r.y_pred)}</td>"
            )
        short = textwrap.shorten(texts[example_id], 90, placeholder=" ...")
        rows.append(
            f'<tr><td><a href="examples/{example_id}.html">{example_id}</a></td>'
            f"<td>{_tag(expected)}</td><td title=\"{html.escape(texts[example_id])}\">"
            f"{html.escape(short)}</td>{''.join(cells)}</tr>"
        )
    correct = {
        t: sum(
            1 for e in ids if labels.get(e) and labels[e] == _class_name(by_key[(t, e)].y_pred)
        )
        for t in techniques
    }
    n_labelled = sum(1 for e in ids if labels.get(e))
    totals = "".join(f'<td class="num">{correct[t]}/{n_labelled}</td>' for t in techniques)
    beeswarms = "".join(
        f'<figure><figcaption><a href="models/{t}.html">{html.escape(t)}</a></figcaption>'
        f'<a href="models/{t}.html"><img class="plot" src="by_model/{t}/beeswarm.png" '
        f'alt="Beeswarm for {html.escape(t)}" loading="lazy"></a></figure>'
        for t in techniques
    )
    model_links = " &middot; ".join(f'<a href="models/{t}.html">{html.escape(t)}</a>' for t in techniques)
    body = (
        "<h1>Curated examples: SHAP plots</h1>"
        f'<p class="muted">{len(ids)} curated texts (not dataset rows) &times; {len(techniques)} '
        "techniques, whole models only. Every value comes from the kit's one explainer "
        "(SHAP partition explainer over a word-level text masker, 500 evaluations per text). "
        "Probabilities are P(extremist) on the raw text; &#10003;/&#10007; compares the prediction "
        "at each technique's locked threshold with the expected label. These texts illustrate "
        "RQ2-RQ4 and support no accuracy claim.</p>"
        f"<p>Model pages: {model_links}</p>"
        "<h2>Every text under every model</h2>"
        f'<div class="scroll"><table><thead><tr><th>Text</th><th>Expected</th><th></th>{head}'
        f'</tr></thead><tbody>{"".join(rows)}<tr><td colspan="3"><b>Matches expected label</b>'
        f"</td>{totals}</tr></tbody></table></div>"
        "<h2>Beeswarms</h2>"
        '<p class="muted">Top words by total |SHAP| over the curated texts; each dot is one '
        "text containing the word, coloured by its expected label.</p>"
        f'<div class="grid small">{beeswarms}</div>'
    )
    (plots_dir / "index.html").write_text(_page("Curated SHAP plots", body, 0), encoding="utf-8")

    # One page per text
    (plots_dir / "examples").mkdir(exist_ok=True)
    for position, example_id in enumerate(ids):
        expected = labels.get(example_id, "")
        nav = []
        if position > 0:
            nav.append(f'<a href="{ids[position - 1]}.html">&larr; {ids[position - 1]}</a>')
        if position < len(ids) - 1:
            nav.append(f'<a href="{ids[position + 1]}.html">{ids[position + 1]} &rarr;</a>')
        table_rows = []
        figures = []
        for technique in techniques:
            r = by_key[(technique, example_id)]
            table_rows.append(
                f'<tr><td><a href="../models/{technique}.html">{html.escape(technique)}</a></td>'
                f'<td class="num">{r.y_prob:.3f}</td><td class="num">{r.threshold:.3f}</td>'
                f"<td>{LABEL_NAMES[_class_name(r.y_pred)]} {_verdict(expected, r.y_pred)}</td>"
                f'<td class="num">{r.base_value:.3f}</td><td class="num">{r.explained_output:.3f}</td></tr>'
            )
            image = _waterfall_path(technique, example_id).as_posix()
            figures.append(
                f"<figure><figcaption><b>{html.escape(technique)}</b>: P = {r.y_prob:.3f}, "
                f"{LABEL_NAMES[_class_name(r.y_pred)]} {_verdict(expected, r.y_pred)}</figcaption>"
                f'<a href="../{image}"><img class="plot" src="../{image}" '
                f'alt="Waterfall for {example_id} under {html.escape(technique)}" loading="lazy"></a></figure>'
            )
        body = (
            f"<nav>{' '.join(nav)}</nav><h1>{example_id}</h1>{_tag(expected)}"
            f'<div class="text {expected}">{html.escape(texts[example_id])}</div>'
            '<div class="scroll"><table><thead><tr><th>Technique</th><th>P(extremist)</th>'
            "<th>Threshold</th><th>Predicted</th><th>All words masked</th><th>Explained</th>"
            f'</tr></thead><tbody>{"".join(table_rows)}</tbody></table></div>'
            '<p class="muted">Each waterfall starts from the model on the fully masked text and adds '
            "one word's SHAP value at a time; the dashed line is the locked threshold. "
            '"Explained" is the model on the text as the masker rebuilt it, which drops '
            "trailing punctuation.</p>"
            f'<div class="grid">{"".join(figures)}</div>'
        )
        (plots_dir / "examples" / f"{example_id}.html").write_text(
            _page(example_id, body, 1), encoding="utf-8"
        )

    # One page per technique
    (plots_dir / "models").mkdir(exist_ok=True)
    for technique in techniques:
        sections = []
        for label in (POSITIVE_CLASS_NAME, NEGATIVE_CLASS_NAME, ""):
            group = [e for e in ids if labels.get(e, "") == label]
            if not group:
                continue
            figures = []
            for example_id in group:
                r = by_key[(technique, example_id)]
                image = _waterfall_path(technique, example_id).as_posix()
                short = textwrap.shorten(texts[example_id], 110, placeholder=" ...")
                figures.append(
                    f'<figure><figcaption><a href="../examples/{example_id}.html">{example_id}</a>: '
                    f"P = {r.y_prob:.3f} {_verdict(label, r.y_pred)}<br>"
                    f'<span class="muted">{html.escape(short)}</span></figcaption>'
                    f'<a href="../{image}"><img class="plot" src="../{image}" '
                    f'alt="Waterfall for {example_id}" loading="lazy"></a></figure>'
                )
            heading = f"Expected {LABEL_NAMES[label]}" if label else "No expected label"
            sections.append(f"<h2>{heading} ({len(group)})</h2><div class=\"grid\">{''.join(figures)}</div>")
        body = (
            f"<h1>{html.escape(technique)}</h1>"
            f'<p class="muted">{correct[technique]} of {n_labelled} curated texts match their expected '
            "label at the locked threshold (illustrative only).</p>"
            f'<figure><img class="plot" src="../by_model/{technique}/beeswarm.png" '
            f'alt="Beeswarm for {html.escape(technique)}"></figure>'
            + "".join(sections)
        )
        (plots_dir / "models" / f"{technique}.html").write_text(
            _page(technique, body, 1), encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_plots(out_dir=DEFAULT_OUT, tolerance=DEFAULT_TOLERANCE, n_words=DEFAULT_TOP_WORDS,
              max_display=DEFAULT_MAX_DISPLAY, log=print):
    """Check the outputs, draw every plot and write the gallery. Returns the plots folder."""
    out_dir = Path(out_dir)
    examples, predictions, attributions, checks = load_whole_model_outputs(out_dir, tolerance)
    labels = _expected_labels(examples) if EXPECTED_LABEL_COLUMN in examples.columns else {}
    techniques = sorted(predictions["technique"].unique())
    ids = examples["example_id"].astype(str).tolist()

    plots_dir = out_dir / PLOTS_DIRNAME
    if plots_dir.exists():
        shutil.rmtree(plots_dir)
    plots_dir.mkdir(parents=True)
    checks.to_csv(plots_dir / CHECKS_FILENAME, index=False)

    by_key = {(r.technique, r.example_id): r for r in checks.itertuples()}
    grouped = {key: block for key, block in attributions.groupby(["technique", "example_id"])}
    contributions = word_contributions(attributions)
    for technique in techniques:
        log(f"== {technique}")
        draw_beeswarm(plots_dir / "by_model" / technique / "beeswarm.png", contributions,
                      technique, labels, len(ids), n_words)
        for example_id in ids:
            r = by_key[(technique, example_id)]
            block = grouped[(technique, example_id)]
            expected = labels.get(example_id)
            verdict = ""
            if expected:
                verdict = " (matches)" if expected == _class_name(r.y_pred) else " (does not match)"
            title = (
                f"{example_id} under {technique}\n"
                + (f"expected {LABEL_NAMES[expected]}; " if expected else "")
                + f"predicted {LABEL_NAMES[_class_name(r.y_pred)]}{verdict}, P = {r.y_prob:.3f}"
            )
            path = plots_dir / _waterfall_path(technique, example_id)
            draw_waterfall(path, block["token"].tolist(), block["attribution"].to_numpy(),
                           r.base_value, r.explained_output, r.y_prob, r.threshold, title,
                           max_display=max_display)
            copy = plots_dir / "by_example" / example_id / f"{technique}.png"
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, copy)
        log(f"   {len(ids)} waterfalls and a beeswarm")
    write_gallery(plots_dir, examples, checks, techniques, labels)
    noted = int((checks["raw_vs_explained_gap"].abs() > DEFAULT_RAW_GAP_NOTE).sum())
    log(
        f"wrote {plots_dir / 'index.html'}; {noted} of {len(checks)} waterfalls note a raw-text "
        "probability that differs from the explained output"
    )
    return plots_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="an output folder of curated_examples.py explain")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                        help="largest accepted |base value + attributions - explained output|")
    parser.add_argument("--top-words", type=int, default=DEFAULT_TOP_WORDS, help="words per beeswarm")
    parser.add_argument("--max-display", type=int, default=DEFAULT_MAX_DISPLAY,
                        help="bars per waterfall before the rest are grouped")
    args = parser.parse_args(argv)
    try:
        run_plots(args.out, args.tolerance, args.top_words, args.max_display)
    except CuratedPlotsError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
