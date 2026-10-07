"""Classify and explain curated texts with every technique, side by side.

Input: a plain-text file with one curated text per line (blank lines and
lines starting with `#` are skipped). Each line becomes `curated_NN`.

For every technique the tool

1. restores `predict_proba_texts` from the downloaded external folder and the
   committed locked threshold (`tools/model_loaders.py`),
2. verifies the restored model against the technique's committed test
   probability artifact on a sample of test rows, and refuses to continue
   when they disagree beyond a tolerance,
3. classifies the curated texts at the locked threshold, and
4. explains them with the kit's one explainer (`notebook_kit.explain`: SHAP
   partition explainer over the word-level text masker), whole model and,
   for ensembles, one run per member.

Outputs go to an untracked folder (default `external/curated_examples/`):

    examples.csv                       example_id, text, sha256 (holds the texts)
    <TECHNIQUE>/predictions.csv        example_id, technique, member, y_prob, threshold, y_pred
    <TECHNIQUE>/local_attributions.csv example_id, technique, member, position, token, attribution, category
    <TECHNIQUE>/run.json               what was loaded, verification result, explainer settings
    summary.csv                        text-free: one row per example x technique x member
    report.html                        every example with each model's highlighted words

Only `summary.csv` is text-free. Nothing here is a committed research
artifact: the curated texts are illustration for RQ2-RQ4, not a fifth question.

Usage:

    python3 tools/curated_examples.py explain --texts curated.txt \\
        --external-root external --techniques 01_LOG-REG_TF-IDF 07_TWITTER-ROBERTA_FINE-TUNE
    python3 tools/curated_examples.py explain --texts curated.txt --external-root external --all
    python3 tools/curated_examples.py report --out external/curated_examples
"""

import argparse
import datetime as dt
import hashlib
import html
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from categorize_attributions import build_word_index, categorize_word  # noqa: E402
from lexicons import load_lexicons  # noqa: E402
from model_loaders import ModelLoadError, load_technique  # noqa: E402
from notebook_kit import (  # noqa: E402
    EXPLAINER_ID,
    KIT_VERSION,
    MASK_TOKEN,
    MASKER_PATTERN,
    TEXT_BEARING_COLUMNS,
    explain,
    normalize_word,
)
from repo_paths import (  # noqa: E402
    CATEGORIES,
    NEGATIVE_CLASS_NAME,
    POSITIVE_CLASS_NAME,
    REPO_ROOT,
    RESULTS_DIR,
    SPLIT_ASSIGNMENTS_CSV,
)

DEFAULT_OUT = REPO_ROOT / "external" / "curated_examples"
DEFAULT_EXTERNAL_ROOT = REPO_ROOT / "external"
EXAMPLES_FILENAME = "examples.csv"
PREDICTIONS_FILENAME = "predictions.csv"
ATTRIBUTIONS_FILENAME = "local_attributions.csv"
RUN_FILENAME = "run.json"
SUMMARY_FILENAME = "summary.csv"
REPORT_FILENAME = "report.html"
WHOLE_MODEL = "whole"
CURATED_SPLIT = "curated"

PREDICTION_COLUMNS = ("example_id", "technique", "member", "y_prob", "threshold", "y_pred")
ATTRIBUTION_COLUMNS = (
    "example_id",
    "technique",
    "member",
    "position",
    "token",
    "attribution",
    "category",
)

DEFAULT_MAX_EVALS = 500
DEFAULT_SEED = 30
DEFAULT_BATCH_SIZE = 64
DEFAULT_VERIFY_ROWS = 64
DEFAULT_TOLERANCE = 1e-2


class CuratedExamplesError(ValueError):
    """An input or an intermediate output violates the tool's contract."""


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def read_examples(path):
    """One curated text per non-empty, non-comment line -> frame(example_id, text, sha256)."""
    path = Path(path)
    if not path.is_file():
        raise CuratedExamplesError(f"texts file {path} does not exist")
    texts = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        texts.append(line)
    if not texts:
        raise CuratedExamplesError(f"{path}: no curated texts (every line is blank or a comment)")
    duplicates = sorted({t for t in texts if texts.count(t) > 1})
    if duplicates:
        raise CuratedExamplesError(f"{path}: {len(duplicates)} text(s) appear more than once")
    width = max(2, len(str(len(texts))))
    return pd.DataFrame(
        {
            "example_id": [f"curated_{i + 1:0{width}d}" for i in range(len(texts))],
            "text": texts,
            "sha256": [hashlib.sha256(t.encode("utf-8")).hexdigest()[:16] for t in texts],
        }
    )


def examples_digest(examples):
    """One digest over the ordered texts, so two runs can be checked for the same input."""
    joined = "\n".join(examples["text"].tolist()).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()[:16]


def explainer_frame(examples):
    """The frame `notebook_kit.explain` takes: row_id, text, one split."""
    return pd.DataFrame(
        {
            "row_id": examples["example_id"].astype(str),
            "text": examples["text"].astype(str),
            "split": CURATED_SPLIT,
        }
    )


# ---------------------------------------------------------------------------
# Verification against the committed probability artifact
# ---------------------------------------------------------------------------
def load_reference_rows(n_rows, seed=DEFAULT_SEED, split="test"):
    """A seeded sample of (row_id, text) from the frozen split, rebuilt from data/dataset.csv."""
    from split_protocol import build_processed_dataset

    processed = build_processed_dataset()
    assignments = pd.read_csv(SPLIT_ASSIGNMENTS_CSV)
    rows = processed.merge(assignments[["row_id", "split"]], on="row_id", how="inner")
    rows = rows.loc[rows["split"] == split, ["row_id", "text"]].sort_values("row_id")
    if n_rows and n_rows < len(rows):
        rows = rows.sample(n=int(n_rows), random_state=int(seed)).sort_values("row_id")
    return rows.reset_index(drop=True)


def verify_model(model, reference, committed, tolerance=DEFAULT_TOLERANCE):
    """Compare the restored model's probabilities with the committed artifact on shared rows.

    `reference` has row_id and text; `committed` has row_id and y_prob. Refuses
    when any shared row differs by more than `tolerance`, which is the evidence
    that the restored model is not the model the notebook explained.
    """
    for label, frame, required in (
        ("reference", reference, ("row_id", "text")),
        ("committed", committed, ("row_id", "y_prob")),
    ):
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise CuratedExamplesError(f"verification {label} rows lack columns {missing}")
    merged = reference.merge(committed[["row_id", "y_prob"]], on="row_id", how="inner")
    if merged.empty:
        raise CuratedExamplesError(
            f"{model.technique}: no shared row ids between the reference rows and the "
            "committed probability artifact"
        )
    restored = model.predict(merged["text"].tolist())
    differences = np.abs(restored - merged["y_prob"].to_numpy(dtype=float))
    result = {
        "n_rows": int(len(merged)),
        "max_abs_diff": float(differences.max()),
        "mean_abs_diff": float(differences.mean()),
        "tolerance": float(tolerance),
        "passed": bool(differences.max() <= tolerance),
    }
    if not result["passed"]:
        raise CuratedExamplesError(
            f"{model.technique}: the restored model disagrees with the committed test "
            f"probabilities (max |diff| {result['max_abs_diff']:.4f} over {result['n_rows']} rows, "
            f"tolerance {tolerance}). The weights are not the ones the notebook explained, or "
            "the serving code drifted from the notebook."
        )
    return result


# ---------------------------------------------------------------------------
# Classification and explanation
# ---------------------------------------------------------------------------
def _explainer_ctx(max_evals, seed, batch_size):
    return SimpleNamespace(
        config={"explainer": {"max_evals": int(max_evals), "seed": int(seed), "batch_size": int(batch_size)}}
    )


def _token_category(token, word_index):
    word = normalize_word(token)
    if not word:
        return ""
    return categorize_word(word, word_index)[0]


def explain_model(
    model,
    examples,
    max_evals=DEFAULT_MAX_EVALS,
    seed=DEFAULT_SEED,
    batch_size=DEFAULT_BATCH_SIZE,
    include_members=True,
    word_index=None,
):
    """Predictions and per-token attributions for the curated texts under one technique.

    Returns (predictions, attributions): frames with PREDICTION_COLUMNS and
    ATTRIBUTION_COLUMNS, one block per predictor (the whole model, then each member).
    """
    if word_index is None:
        word_index, _ = build_word_index(load_lexicons())
    frame = explainer_frame(examples)
    # The partition explainer explains posts independently, but the kit asks for
    # at least two; a single curated text is explained next to a copy of itself.
    padded = len(frame) == 1
    if padded:
        frame = pd.concat([frame, frame], ignore_index=True)
    predictors = [(WHOLE_MODEL, model.predict)]
    if include_members:
        predictors.extend(sorted(model.members.items()))
    ctx = _explainer_ctx(max_evals, seed, batch_size)

    prediction_rows = []
    attribution_rows = []
    for member, predict in predictors:
        y_prob = np.asarray(predict(frame["text"].tolist()), dtype=float)
        y_pred = (y_prob >= model.threshold).astype(int)
        for example_id, prob, pred in list(zip(frame["row_id"], y_prob, y_pred))[: len(examples)]:
            prediction_rows.append(
                {
                    "example_id": example_id,
                    "technique": model.technique,
                    "member": member,
                    "y_prob": float(prob),
                    "threshold": float(model.threshold),
                    "y_pred": int(pred),
                }
            )
        bundle = explain(ctx, predict, frame, seed=seed, max_evals=max_evals, batch_size=batch_size)
        explained = list(zip(bundle.row_ids, bundle.tokens_per_post, bundle.values_per_post))
        if padded:
            explained = explained[:1]
        for example_id, tokens, values in explained:
            for position, (token, value) in enumerate(zip(tokens, values)):
                attribution_rows.append(
                    {
                        "example_id": example_id,
                        "technique": model.technique,
                        "member": member,
                        "position": position,
                        "token": str(token),
                        "attribution": float(value),
                        "category": _token_category(token, word_index),
                    }
                )
    predictions = pd.DataFrame(prediction_rows, columns=list(PREDICTION_COLUMNS))
    attributions = pd.DataFrame(attribution_rows, columns=list(ATTRIBUTION_COLUMNS))
    return predictions, attributions


def write_technique_outputs(out_dir, model, predictions, attributions, record):
    folder = Path(out_dir) / model.technique
    folder.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(folder / PREDICTIONS_FILENAME, index=False)
    attributions.to_csv(folder / ATTRIBUTIONS_FILENAME, index=False)
    with open(folder / RUN_FILENAME, "w") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
    return folder


def _write_examples(out_dir, examples, fresh):
    """Record the texts under out_dir; refuse to mix two different curated sets."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / EXAMPLES_FILENAME
    digest = examples_digest(examples)
    if path.exists() and not fresh:
        existing = pd.read_csv(path)
        if "text" not in existing.columns or examples_digest(existing) != digest:
            raise CuratedExamplesError(
                f"{out_dir} already holds outputs for a different curated set; pass --fresh "
                "to replace them or choose another --out folder"
            )
    elif fresh:
        for stale in out_dir.iterdir():
            if stale.is_dir():
                for item in stale.iterdir():
                    item.unlink()
                stale.rmdir()
            else:
                stale.unlink()
    examples.to_csv(path, index=False)
    return digest


def run_explain(
    texts_path,
    external_root,
    techniques,
    out_dir=DEFAULT_OUT,
    results_dir=None,
    max_evals=DEFAULT_MAX_EVALS,
    seed=DEFAULT_SEED,
    batch_size=DEFAULT_BATCH_SIZE,
    device="auto",
    verify_rows=DEFAULT_VERIFY_ROWS,
    tolerance=DEFAULT_TOLERANCE,
    include_members=True,
    fresh=False,
    reference_loader=load_reference_rows,
    committed_loader=None,
):
    """The `explain` subcommand. Returns the list of technique folders written."""
    examples = read_examples(texts_path)
    digest = _write_examples(out_dir, examples, fresh)
    word_index, _ = build_word_index(load_lexicons())
    if committed_loader is None:
        from probs_artifact import load_technique_probs

        committed_loader = lambda technique: load_technique_probs(technique, "test")  # noqa: E731
    reference = reference_loader(verify_rows, seed) if verify_rows else None

    written = []
    for technique in techniques:
        print(f"== {technique}")
        model = load_technique(
            technique,
            Path(external_root) / technique,
            results_dir=results_dir,
            device=device,
            batch_size=batch_size,
        )
        for note in model.notes:
            print(f"   note: {note}")
        verification = None
        if reference is not None:
            verification = verify_model(model, reference, committed_loader(technique), tolerance)
            print(
                f"   verified against committed test probabilities: max |diff| "
                f"{verification['max_abs_diff']:.5f} over {verification['n_rows']} rows"
            )
        else:
            print("   verification skipped (--verify-rows 0)")
        predictions, attributions = explain_model(
            model,
            examples,
            max_evals=max_evals,
            seed=seed,
            batch_size=batch_size,
            include_members=include_members,
            word_index=word_index,
        )
        record = {
            "technique": technique,
            "family": model.family,
            "threshold": model.threshold,
            "members": sorted(model.members) if include_members else [],
            "sources": model.sources,
            "notes": model.notes,
            "verification": verification,
            "examples": {"n": int(len(examples)), "digest": digest, "texts_file": str(texts_path)},
            "explainer": {
                "explainer": EXPLAINER_ID,
                "algorithm": "partition",
                "masker": MASKER_PATTERN,
                "mask_token": MASK_TOKEN,
                "max_evals": int(max_evals),
                "seed": int(seed),
                "batch_size": int(batch_size),
            },
            "device": device,
            "kit_version": KIT_VERSION,
            "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        folder = write_technique_outputs(out_dir, model, predictions, attributions, record)
        written.append(folder)
        print(f"   wrote {folder}")
    return written


# ---------------------------------------------------------------------------
# Report: summary.csv and report.html
# ---------------------------------------------------------------------------
def load_outputs(out_dir):
    """examples, predictions, attributions and run records from every technique folder."""
    out_dir = Path(out_dir)
    examples_path = out_dir / EXAMPLES_FILENAME
    if not examples_path.exists():
        raise CuratedExamplesError(f"{out_dir}: no {EXAMPLES_FILENAME}; run `explain` first")
    examples = pd.read_csv(examples_path)
    expected_ids = set(examples["example_id"].astype(str))

    predictions, attributions, records = [], [], {}
    for folder in sorted(p for p in out_dir.iterdir() if p.is_dir()):
        files = [folder / PREDICTIONS_FILENAME, folder / ATTRIBUTIONS_FILENAME, folder / RUN_FILENAME]
        if not all(f.exists() for f in files):
            continue
        pred = pd.read_csv(files[0])
        attr = pd.read_csv(files[1], keep_default_na=False)
        with open(files[2]) as handle:
            record = json.load(handle)
        for label, frame, required in (
            (files[0].name, pred, PREDICTION_COLUMNS),
            (files[1].name, attr, ATTRIBUTION_COLUMNS),
        ):
            missing = [c for c in required if c not in frame.columns]
            if missing:
                raise CuratedExamplesError(f"{folder.name}/{label}: missing columns {missing}")
        found_ids = set(pred["example_id"].astype(str))
        if found_ids != expected_ids:
            raise CuratedExamplesError(
                f"{folder.name}: predictions cover {len(found_ids)} example ids but "
                f"{EXAMPLES_FILENAME} lists {len(expected_ids)}; the folder was produced "
                "for a different curated set"
            )
        if record.get("examples", {}).get("digest") != examples_digest(examples):
            raise CuratedExamplesError(
                f"{folder.name}/{RUN_FILENAME}: examples digest differs from {EXAMPLES_FILENAME}"
            )
        predictions.append(pred)
        attributions.append(attr)
        records[folder.name] = record
    if not predictions:
        raise CuratedExamplesError(f"{out_dir}: no technique outputs found")
    return (
        examples,
        pd.concat(predictions, ignore_index=True),
        pd.concat(attributions, ignore_index=True),
        records,
    )


def build_summary(predictions, attributions):
    """Text-free summary: one row per example x technique x member with category shares."""
    attributions = attributions.copy()
    attributions["abs_attribution"] = attributions["attribution"].abs()
    attributions["positive_attribution"] = attributions["attribution"].clip(lower=0.0)
    keys = ["example_id", "technique", "member"]
    totals = attributions.groupby(keys).agg(
        n_tokens=("token", "size"),
        abs_mass=("abs_attribution", "sum"),
        positive_mass=("positive_attribution", "sum"),
    )
    by_category = (
        attributions.groupby(keys + ["category"])[["abs_attribution", "positive_attribution"]]
        .sum()
        .unstack("category", fill_value=0.0)
    )
    summary = predictions.merge(totals.reset_index(), on=keys, how="left")
    for category in CATEGORIES:
        abs_col = ("abs_attribution", category)
        pos_col = ("positive_attribution", category)
        abs_values = by_category[abs_col] if abs_col in by_category.columns else 0.0
        pos_values = by_category[pos_col] if pos_col in by_category.columns else 0.0
        part = pd.DataFrame(
            {
                f"share_abs_mass_{category}": abs_values,
                f"share_positive_mass_{category}": pos_values,
            },
            index=by_category.index,
        ).reset_index()
        summary = summary.merge(part, on=keys, how="left")
        summary[f"share_abs_mass_{category}"] = (
            summary[f"share_abs_mass_{category}"].fillna(0.0) / summary["abs_mass"].replace(0.0, np.nan)
        ).fillna(0.0)
        summary[f"share_positive_mass_{category}"] = (
            summary[f"share_positive_mass_{category}"].fillna(0.0)
            / summary["positive_mass"].replace(0.0, np.nan)
        ).fillna(0.0)
    summary["predicted_class"] = np.where(
        summary["y_pred"] == 1, POSITIVE_CLASS_NAME, NEGATIVE_CLASS_NAME
    )
    leaked = [c for c in summary.columns if c in TEXT_BEARING_COLUMNS or c in ("token", "word")]
    if leaked:
        raise CuratedExamplesError(f"summary would carry text-bearing columns {leaked}")
    return summary.sort_values(keys).reset_index(drop=True)


_CSS = """
:root { --ink:#141414; --muted:#5e5d59; --line:#dddcd6; --paper:#fcfcfb; --pos:214,56,42; --neg:42,120,214; }
body { font: 15px/1.5 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color: var(--ink);
       background: var(--paper); margin: 0; padding: 24px 16px; max-width: 1100px; margin: 0 auto; }
h1 { font-size: 1.5rem; margin: 0 0 4px; } h2 { font-size: 1.15rem; margin: 32px 0 8px; border-top: 1px solid var(--line); padding-top: 16px; }
h3 { font-size: 0.95rem; margin: 16px 0 4px; color: var(--muted); font-weight: 600; }
p.meta { color: var(--muted); margin: 0 0 16px; }
blockquote { margin: 8px 0 12px; padding: 10px 14px; border-left: 3px solid var(--line); background: #f4f3ef; white-space: pre-wrap; }
table { border-collapse: collapse; margin: 4px 0 12px; font-size: 0.9rem; width: 100%; max-width: 820px; }
th, td { text-align: left; padding: 4px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--muted); font-weight: 600; } td.num { text-align: right; font-variant-numeric: tabular-nums; }
.tokens { line-height: 2.1; margin: 2px 0 10px; }
.tok { padding: 2px 3px; border-radius: 3px; margin-right: 2px; white-space: pre; }
.tok[data-cat="slur"], .tok[data-cat="extremist_framing"], .tok[data-cat="identity_term"] { text-decoration: underline dotted; text-underline-offset: 3px; }
.legend { display: inline-block; padding: 1px 6px; border-radius: 3px; margin-right: 6px; }
.flag { color: rgb(var(--pos)); font-weight: 600; }
"""


def _rgba(value, scale):
    if scale <= 0 or value == 0:
        return "transparent"
    alpha = min(1.0, abs(value) / scale) * 0.85
    return f"rgba(var(--{'pos' if value > 0 else 'neg'}),{alpha:.2f})"


def _tokens_html(block):
    block = block.sort_values("position")
    scale = float(block["attribution"].abs().max()) if len(block) else 0.0
    spans = []
    for row in block.itertuples():
        title = f"{row.attribution:+.4f}" + (f" · {row.category}" if row.category else "")
        spans.append(
            f'<span class="tok" data-cat="{html.escape(str(row.category))}" '
            f'style="background:{_rgba(row.attribution, scale)}" title="{title}">'
            f"{html.escape(str(row.token))}</span>"
        )
    return f'<div class="tokens">{"".join(spans)}</div>'


def _predictor_label(technique, member):
    return technique if member == WHOLE_MODEL else f"{technique} · member {member}"


def build_report_html(examples, predictions, attributions, summary, records):
    """A self-contained page: every curated text under every model, words highlighted."""
    predictions = predictions.sort_values(["example_id", "technique", "member"])
    techniques = sorted(predictions["technique"].unique())
    whole = predictions[predictions["member"] == WHOLE_MODEL]
    out = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Curated examples</title>", f"<style>{_CSS}</style></head><body>",
        "<h1>Curated examples across models</h1>",
        f"<p class='meta'>{len(examples)} texts · {len(techniques)} techniques · "
        f"explainer: {html.escape(EXPLAINER_ID)}. "
        "Highlight: <span class='legend' style='background:rgba(var(--pos),0.6)'>pushes toward "
        f"{POSITIVE_CLASS_NAME}</span> <span class='legend' style='background:rgba(var(--neg),0.6)'>"
        f"pushes toward {NEGATIVE_CLASS_NAME}</span>, intensity relative to the strongest word of "
        "that model on that text. Dotted underline: word is in a lexicon (hover for the category).</p>",
    ]

    out.append("<h2>Overview</h2><table><tr><th>Example</th>")
    out.extend(f"<th>{html.escape(t)}</th>" for t in techniques)
    out.append("</tr>")
    for example in examples.itertuples():
        out.append(f"<tr><td>{html.escape(example.example_id)}</td>")
        for technique in techniques:
            cell = whole[(whole["example_id"] == example.example_id) & (whole["technique"] == technique)]
            if cell.empty:
                out.append("<td class='num'>–</td>")
            else:
                prob = float(cell["y_prob"].iloc[0])
                flagged = int(cell["y_pred"].iloc[0]) == 1
                out.append(f"<td class='num{' flag' if flagged else ''}'>{prob:.3f}</td>")
        out.append("</tr>")
    out.append(f"</table><p class='meta'>Cell: P({POSITIVE_CLASS_NAME}); bold red when at or above the "
               "technique's locked validation threshold.</p>")

    for example in examples.itertuples():
        out.append(f"<h2>{html.escape(example.example_id)}</h2>")
        out.append(f"<blockquote>{html.escape(str(example.text))}</blockquote>")
        rows = predictions[predictions["example_id"] == example.example_id]
        out.append("<table><tr><th>Model</th><th>P(EXTREMIST)</th><th>Threshold</th><th>Prediction</th>"
                   "<th>Positive mass: framing / identity / slur / topical</th></tr>")
        for row in rows.itertuples():
            shares = summary[
                (summary["example_id"] == row.example_id)
                & (summary["technique"] == row.technique)
                & (summary["member"] == row.member)
            ]
            share_text = "–"
            if not shares.empty:
                s = shares.iloc[0]
                share_text = " / ".join(
                    f"{s[f'share_positive_mass_{c}']:.2f}"
                    for c in ("extremist_framing", "identity_term", "slur", "topical")
                )
            label = POSITIVE_CLASS_NAME if row.y_pred == 1 else NEGATIVE_CLASS_NAME
            out.append(
                f"<tr><td>{html.escape(_predictor_label(row.technique, row.member))}</td>"
                f"<td class='num'>{row.y_prob:.3f}</td><td class='num'>{row.threshold:.3f}</td>"
                f"<td class='{'flag' if row.y_pred == 1 else ''}'>{label}</td><td>{share_text}</td></tr>"
            )
        out.append("</table>")
        for row in rows.itertuples():
            block = attributions[
                (attributions["example_id"] == row.example_id)
                & (attributions["technique"] == row.technique)
                & (attributions["member"] == row.member)
            ]
            out.append(f"<h3>{html.escape(_predictor_label(row.technique, row.member))}</h3>")
            out.append(_tokens_html(block) if len(block) else "<p class='meta'>no word tokens</p>")

    out.append("<h2>Provenance</h2><table><tr><th>Technique</th><th>Family</th><th>Verification</th>"
               "<th>Notes</th></tr>")
    for technique, record in sorted(records.items()):
        verification = record.get("verification")
        if verification:
            v_text = (f"max |diff| {verification['max_abs_diff']:.5f} over {verification['n_rows']} "
                      f"test rows (tol {verification['tolerance']})")
        else:
            v_text = "skipped"
        out.append(
            f"<tr><td>{html.escape(technique)}</td><td>{html.escape(record.get('family', ''))}</td>"
            f"<td>{html.escape(v_text)}</td><td>{html.escape('; '.join(record.get('notes', [])))}</td></tr>"
        )
    out.append("</table></body></html>")
    return "\n".join(out)


def run_report(out_dir=DEFAULT_OUT):
    """The `report` subcommand: summary.csv and report.html from the technique folders."""
    out_dir = Path(out_dir)
    examples, predictions, attributions, records = load_outputs(out_dir)
    summary = build_summary(predictions, attributions)
    summary.to_csv(out_dir / SUMMARY_FILENAME, index=False)
    page = build_report_html(examples, predictions, attributions, summary, records)
    (out_dir / REPORT_FILENAME).write_text(page, encoding="utf-8")
    print(f"wrote {out_dir / SUMMARY_FILENAME} ({len(summary)} rows)")
    print(f"wrote {out_dir / REPORT_FILENAME}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _available_techniques(external_root, results_dir):
    results = Path(results_dir or RESULTS_DIR)
    found = []
    for folder in sorted(Path(external_root).iterdir()):
        if folder.is_dir() and (folder / "weights").is_dir() and (results / folder.name / "best_config.json").exists():
            found.append(folder.name)
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("explain", help="classify and explain the curated texts with each technique")
    p.add_argument("--texts", required=True, help="plain-text file, one curated text per line")
    p.add_argument("--external-root", default=str(DEFAULT_EXTERNAL_ROOT),
                   help="folder holding one downloaded external/<TECHNIQUE>/ per technique")
    p.add_argument("--techniques", nargs="*", default=[], help="technique names (folder stems)")
    p.add_argument("--all", action="store_true", help="every technique under --external-root with a committed best_config.json")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--results-dir", default=None, help=argparse.SUPPRESS)
    p.add_argument("--max-evals", type=int, default=DEFAULT_MAX_EVALS, help="SHAP budget per text (notebooks use 500)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--verify-rows", type=int, default=DEFAULT_VERIFY_ROWS,
                   help="test rows used to verify the restored model against the committed artifact (0 skips)")
    p.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    p.add_argument("--no-members", action="store_true", help="skip per-member runs for ensembles")
    p.add_argument("--fresh", action="store_true", help="replace outputs from a different curated set")
    p.add_argument("--no-report", action="store_true", help="do not rebuild summary.csv and report.html")

    r = sub.add_parser("report", help="rebuild summary.csv and report.html from existing outputs")
    r.add_argument("--out", default=str(DEFAULT_OUT))

    args = parser.parse_args(argv)
    try:
        if args.command == "explain":
            techniques = list(args.techniques)
            if args.all:
                techniques = _available_techniques(args.external_root, args.results_dir)
            if not techniques:
                raise CuratedExamplesError("no techniques given (use --techniques ... or --all)")
            run_explain(
                args.texts,
                args.external_root,
                techniques,
                out_dir=args.out,
                results_dir=args.results_dir,
                max_evals=args.max_evals,
                seed=args.seed,
                batch_size=args.batch_size,
                device=args.device,
                verify_rows=args.verify_rows,
                tolerance=args.tolerance,
                include_members=not args.no_members,
                fresh=args.fresh,
            )
            if not args.no_report:
                run_report(args.out)
        else:
            run_report(args.out)
    except (CuratedExamplesError, ModelLoadError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
