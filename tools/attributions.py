"""Read/write the committed word-level attribution artifact (RQ2-RQ4 input).

A notebook that explains a model with SHAP produces, per explained post, one
attribution per token. That per-post table carries the post text and must not
be committed. What IS committed is its word-level aggregate:

    word, mean_abs_attribution, mean_attribution, support

one row per word, where `support` is the number of explained posts containing
the word and both means are taken over those posts only (not over all posts).
Averaging over posts that contain the word is what keeps the signed mean
meaningful: for a linear model, SHAP for an absent feature has the opposite sign
to its coefficient, so a mean over all posts would cancel to zero.

Subword pieces are aggregated to the word level before export, in the notebook,
so every model family lands in the same vocabulary. Explainer settings live in
the sidecar meta JSON so runs can be compared like for like.

Artifacts live at results_summary/<TECHNIQUE>/attributions/<run_id>.csv with a
sidecar <run_id>.json. A technique may have several runs: different seeds or
background sets (stability, RQ3), and for an ensemble one run per member plus
one for the whole ensemble (member = null).

The loader refuses any per-post column. That refusal is what makes the artifact
safe to commit for a dataset of extremist text.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from repo_paths import ATTRIBUTIONS_SUBDIR, RESULTS_DIR, SPLIT_NAMES

REQUIRED_COLUMNS = ("word", "mean_abs_attribution", "mean_attribution", "support")

# Any of these means per-post data (or dataset text) is travelling with the
# aggregate. Presence is a refusal, never a warning.
FORBIDDEN_COLUMNS = (
    "row_id",
    "text",
    "text_hash",
    "Original_Message",
    "message",
    "post",
    "text_preview",
    "position",
    "token_index",
    "y_true",
    "y_prob",
)

# Meta fields every run must declare so runs are comparable.
REQUIRED_META_FIELDS = (
    "technique",
    "run_id",
    "split",
    "explainer",
    "background_size",
    "seed",
    "aggregation",
    "n_posts_explained",
    "member",
)
WORD_AGGREGATION = "word"

# Words longer than this are almost certainly not single words; a pasted
# fragment of a post would be caught here.
MAX_WORD_LENGTH = 40


class AttributionArtifactError(ValueError):
    """Raised when an attribution artifact violates its contract."""


def attributions_dir(technique, results_dir=None):
    """The folder holding a technique's attribution runs."""
    base = RESULTS_DIR if results_dir is None else Path(results_dir)
    return base / technique / ATTRIBUTIONS_SUBDIR


def attributions_path(technique, run_id, results_dir=None):
    return attributions_dir(technique, results_dir) / f"{run_id}.csv"


def meta_path(technique, run_id, results_dir=None):
    return attributions_dir(technique, results_dir) / f"{run_id}.json"


def list_runs(technique, results_dir=None):
    """Run ids with both a CSV and a sidecar JSON, sorted."""
    folder = attributions_dir(technique, results_dir)
    if not folder.exists():
        return []
    runs = []
    for csv_path in sorted(folder.glob("*.csv")):
        if "__" in csv_path.stem:
            continue  # derived files (e.g. <run>__categorized.csv) are not runs
        if csv_path.with_suffix(".json").exists():
            runs.append(csv_path.stem)
    return runs


def validate_frame(frame, label="attributions"):
    """Enforce the column contract and value ranges. Returns a clean, sorted copy."""
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise AttributionArtifactError(f"{label}: missing required columns {missing}")

    forbidden = [c for c in FORBIDDEN_COLUMNS if c in frame.columns]
    if forbidden:
        raise AttributionArtifactError(
            f"{label}: carries per-post column(s) {forbidden}. Only word-level "
            "aggregates may be committed."
        )

    extra = [c for c in frame.columns if c not in REQUIRED_COLUMNS]
    if extra:
        raise AttributionArtifactError(
            f"{label}: unexpected columns {extra}; the contract is exactly {list(REQUIRED_COLUMNS)}"
        )

    clean = frame.copy()
    clean["word"] = clean["word"].astype(str)
    bad_words = clean["word"][
        clean["word"].str.strip().eq("")
        | clean["word"].str.contains(r"\s", regex=True)
        | (clean["word"].str.len() > MAX_WORD_LENGTH)
    ]
    if len(bad_words):
        raise AttributionArtifactError(
            f"{label}: {len(bad_words)} 'word' value(s) are empty, contain whitespace, or exceed "
            f"{MAX_WORD_LENGTH} characters; the artifact is word-level, not phrase-level"
        )
    if clean["word"].duplicated().any():
        raise AttributionArtifactError(f"{label}: duplicate word values")

    for column in ("mean_abs_attribution", "mean_attribution"):
        values = pd.to_numeric(clean[column], errors="coerce")
        if not np.isfinite(values).all():
            raise AttributionArtifactError(f"{label}: non-finite values in {column}")
        clean[column] = values.astype(float)
    if (clean["mean_abs_attribution"] < 0).any():
        raise AttributionArtifactError(f"{label}: mean_abs_attribution must be non-negative")
    if (clean["mean_attribution"].abs() > clean["mean_abs_attribution"] + 1e-9).any():
        raise AttributionArtifactError(
            f"{label}: |mean_attribution| exceeds mean_abs_attribution for some words; "
            "both must be means over the same posts"
        )

    support = pd.to_numeric(clean["support"], errors="coerce")
    if support.isna().any() or (support < 1).any() or (support != support.round()).any():
        raise AttributionArtifactError(f"{label}: support must be a positive integer")
    clean["support"] = support.astype(int)

    return clean.sort_values(
        ["mean_abs_attribution", "word"], ascending=[False, True]
    ).reset_index(drop=True)


def validate_meta(meta, label="meta"):
    missing = [f for f in REQUIRED_META_FIELDS if f not in meta]
    if missing:
        raise AttributionArtifactError(f"{label}: missing required meta fields {missing}")
    if meta["aggregation"] != WORD_AGGREGATION:
        raise AttributionArtifactError(
            f"{label}: aggregation is '{meta['aggregation']}', must be '{WORD_AGGREGATION}'"
        )
    if meta["split"] not in SPLIT_NAMES:
        raise AttributionArtifactError(f"{label}: split '{meta['split']}' not in {SPLIT_NAMES}")
    if not isinstance(meta["n_posts_explained"], int) or meta["n_posts_explained"] < 1:
        raise AttributionArtifactError(f"{label}: n_posts_explained must be a positive integer")
    return meta


def load_attributions(path):
    path = Path(path)
    if not path.exists():
        raise AttributionArtifactError(f"attribution artifact not found: {path}")
    return validate_frame(pd.read_csv(path), label=path.name)


def read_meta(technique, run_id, results_dir=None):
    path = meta_path(technique, run_id, results_dir)
    if not path.exists():
        raise AttributionArtifactError(f"attribution meta not found: {path}")
    with open(path) as handle:
        return validate_meta(json.load(handle), label=path.name)


def load_run(technique, run_id, results_dir=None):
    """Return (frame, meta) for one run, both validated and cross-checked."""
    frame = load_attributions(attributions_path(technique, run_id, results_dir))
    meta = read_meta(technique, run_id, results_dir)
    if meta["technique"] != technique or meta["run_id"] != run_id:
        raise AttributionArtifactError(
            f"{run_id}.json: technique/run_id do not match the file location"
        )
    if (frame["support"] > meta["n_posts_explained"]).any():
        raise AttributionArtifactError(
            f"{run_id}.csv: a word's support exceeds n_posts_explained={meta['n_posts_explained']}"
        )
    return frame, meta


def write_run(technique, run_id, frame, meta, results_dir=None):
    """Validate and write a run's CSV and sidecar JSON. Returns the CSV path.

    This is the function a notebook's export cell should call (or mirror
    exactly) so an artifact is refused at export time rather than at review.
    """
    meta = dict(meta, technique=technique, run_id=run_id, aggregation=WORD_AGGREGATION)
    validate_meta(meta, label=f"{run_id}.json")
    clean = validate_frame(frame, label=f"{run_id}.csv")
    if (clean["support"] > meta["n_posts_explained"]).any():
        raise AttributionArtifactError("a word's support exceeds n_posts_explained")

    folder = attributions_dir(technique, results_dir)
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / f"{run_id}.csv"
    clean.to_csv(csv_path, index=False, float_format="%.8g")
    with open(folder / f"{run_id}.json", "w") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
    return csv_path


def top_words(frame, k):
    """The k words with the largest mean absolute attribution."""
    return frame.nlargest(k, "mean_abs_attribution").reset_index(drop=True)


def iter_all_runs(results_dir=None):
    """Yield (technique, run_id, frame, meta) for every run in results_summary/."""
    from repo_paths import technique_dirs

    for folder in technique_dirs(results_dir):
        for run_id in list_runs(folder.name, results_dir):
            frame, meta = load_run(folder.name, run_id, results_dir)
            yield folder.name, run_id, frame, meta
