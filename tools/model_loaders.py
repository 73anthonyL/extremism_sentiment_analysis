"""Rebuild a technique's `predict_proba_texts` from its downloaded external folder.

Every model notebook saves its fitted model under `external/<TECHNIQUE>/weights/`
(and `tokenizer/` for transformers) and its locked threshold in the committed
`results_summary/<TECHNIQUE>/best_config.json`. This module turns those two
things back into the same `predict_proba_texts(list_of_str) -> P(EXTREMIST)`
function the notebook explained, so curated texts that were never part of the
dataset can be classified and explained by the kit's one explainer
(`notebook_kit.explain`) without rerunning a notebook.

The family of a folder is recognised from the files it holds:

    weights/pipeline.joblib                        01, 02, 04, 05  (scikit-learn pipeline)
    weights/vectorizer.joblib + slp_state_dict.pt  03              (TF-IDF + one linear layer)
    weights/fasttext.model + classifier.joblib     06              (FastText pooling + LR)
    weights/transformer_model/ + tokenizer/        07              (one fine-tuned transformer)
    weights/member-<component>-seed<N>/            11              (log-odds pooled ensemble)

The inference code for 03, 06, 07 and 11 is reproduced here from the
notebooks, because a pipeline object cannot be pickled for them. Each
reproduction is pinned to its notebook by `tools/tests/test_curated_examples.py`,
which parses the notebook's constants and compares, and every loaded model is
verified against the technique's committed probability artifact before it
explains anything (`tools/curated_examples.py`).

Notebooks 08, 09 and 10 are closed lines outside the comparison and have no
loader.
"""

import html
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from repo_paths import RESULTS_DIR  # noqa: E402


class ModelLoadError(RuntimeError):
    """The external folder or best_config.json cannot be turned into a predictor."""


@dataclass
class LoadedModel:
    """A technique restored from external storage, ready for the explainer."""

    technique: str
    family: str
    threshold: float
    predict: Callable  # predict(list_of_str) -> np.ndarray of P(EXTREMIST)
    members: dict = field(default_factory=dict)  # member id -> predict function
    sources: dict = field(default_factory=dict)  # what was loaded, for the run record
    notes: list = field(default_factory=list)  # deviations from the notebook's serving


# ---------------------------------------------------------------------------
# Committed configuration
# ---------------------------------------------------------------------------
def read_best_config(technique, results_dir=None):
    """The committed best_config.json of a technique, with its locked threshold."""
    path = Path(results_dir or RESULTS_DIR) / technique / "best_config.json"
    if not path.exists():
        raise ModelLoadError(f"{technique}: no committed best_config.json at {path}")
    with open(path) as handle:
        config = json.load(handle)
    for key in ("selected_threshold", "hyperparameters"):
        if key not in config:
            raise ModelLoadError(f"{path}: best_config.json has no '{key}'")
    threshold = float(config["selected_threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ModelLoadError(f"{path}: selected_threshold {threshold} is not in [0, 1]")
    return config


def detect_family(external_dir):
    """Which notebook family saved the weights under `external_dir`."""
    weights = Path(external_dir) / "weights"
    if not weights.is_dir():
        raise ModelLoadError(f"{external_dir}: no weights/ folder")
    if (weights / "pipeline.joblib").is_file():
        return "sklearn_pipeline"
    if (weights / "slp_state_dict.pt").is_file() and (weights / "vectorizer.joblib").is_file():
        return "slp_tfidf"
    if (weights / "fasttext.model").is_file() and (weights / "classifier.joblib").is_file():
        return "fasttext_logreg"
    if (weights / "transformer_model").is_dir():
        return "transformer"
    if any(weights.glob("member-*-seed*")):
        return "pooled_transformer_ensemble"
    raise ModelLoadError(
        f"{weights}: no recognisable weights (expected pipeline.joblib, slp_state_dict.pt, "
        "fasttext.model, transformer_model/ or member-*-seed*/ folders)"
    )


def _as_probabilities(values, label):
    values = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ModelLoadError(f"{label}: predictor returned values outside [0, 1]")
    return values


# ---------------------------------------------------------------------------
# 01, 02, 04, 05: a pickled scikit-learn pipeline
# ---------------------------------------------------------------------------
def _load_sklearn_pipeline(external_dir):
    import joblib

    path = Path(external_dir) / "weights" / "pipeline.joblib"
    pipeline = joblib.load(path)
    if not hasattr(pipeline, "predict_proba"):
        raise ModelLoadError(f"{path}: object has no predict_proba")

    def predict(texts):
        return pipeline.predict_proba([str(t) for t in texts])[:, 1]

    return predict, {"pipeline": str(path)}


# ---------------------------------------------------------------------------
# 03: TF-IDF vectorizer + single linear layer (notebook 03, section 4)
# ---------------------------------------------------------------------------
def _load_slp(external_dir):
    import joblib
    import torch

    weights = Path(external_dir) / "weights"
    vectorizer = joblib.load(weights / "vectorizer.joblib")
    state = torch.load(weights / "slp_state_dict.pt", map_location="cpu")
    if "linear.weight" not in state or "linear.bias" not in state:
        raise ModelLoadError("slp_state_dict.pt: expected keys linear.weight and linear.bias")
    weight = state["linear.weight"].float()
    bias = state["linear.bias"].float()
    n_features = len(vectorizer.get_feature_names_out())
    if weight.shape != (1, n_features):
        raise ModelLoadError(
            f"slp_state_dict.pt: layer shape {tuple(weight.shape)} does not match the "
            f"vectorizer's {n_features} features"
        )

    def predict(texts):
        features = vectorizer.transform([str(t) for t in texts])
        dense = torch.tensor(features.toarray(), dtype=torch.float32)
        with torch.no_grad():
            logits = dense @ weight.T + bias
        return torch.sigmoid(logits).squeeze(1).numpy()

    return predict, {"vectorizer": str(weights / "vectorizer.joblib"), "state_dict": str(weights / "slp_state_dict.pt")}


# ---------------------------------------------------------------------------
# 06: FastText document embeddings + logistic regression (notebook 06, section 4)
# ---------------------------------------------------------------------------
# Pinned to notebook 06's CONFIG["tokenization"] by the test suite.
FASTTEXT_TOKEN_PATTERN = r"(?u)https?://\S+|#\w+|@\w+|\b\w+\b|[^\w\s]"
FASTTEXT_LOWERCASE = True


def fasttext_stable_hash(value):
    """Notebook 06's `stable_hash`; the saved FastText model references it by name."""
    import zlib

    return zlib.crc32(str(value).encode("utf-8"))


def fasttext_tokenize(value, pattern=FASTTEXT_TOKEN_PATTERN, lowercase=FASTTEXT_LOWERCASE):
    value = "" if value is None else str(value)
    if lowercase:
        value = value.lower()
    return re.findall(pattern, value)


def fasttext_pool(tokenized, vector_of, dimension, pooling, idf, normalize):
    """Notebook 06's `build_document_embeddings`."""
    embeddings = np.zeros((len(tokenized), dimension), dtype=np.float32)
    for row, tokens in enumerate(tokenized):
        weighted_sum = np.zeros(dimension, dtype=np.float32)
        total_weight = 0.0
        for token in tokens:
            vector = vector_of(token)
            if vector is None:
                continue
            if pooling == "idf_weighted_mean":
                weight = float(idf.get(token, 1.0))
            elif pooling == "mean":
                weight = 1.0
            else:
                raise ModelLoadError(f"unknown pooling method {pooling!r}")
            weighted_sum += weight * vector
            total_weight += weight
        if total_weight > 0:
            pooled = weighted_sum / total_weight
            if normalize:
                norm = np.linalg.norm(pooled)
                if norm > 0:
                    pooled = pooled / norm
            embeddings[row] = pooled
    return embeddings


def _load_fasttext_logreg(external_dir, hyperparameters):
    import joblib

    try:
        from gensim.models import FastText
    except ImportError as error:
        raise ModelLoadError("notebook 06's weights need the gensim package") from error

    for key in ("pooling", "normalize_embeddings"):
        if key not in hyperparameters:
            raise ModelLoadError(f"best_config.json hyperparameters lack '{key}' (needed by 06)")
    weights = Path(external_dir) / "weights"
    # The pickled model references __main__.stable_hash (notebook 06 defined it there).
    main_module = sys.modules["__main__"]
    if not hasattr(main_module, "stable_hash"):
        setattr(main_module, "stable_hash", fasttext_stable_hash)
    ft_model = FastText.load(str(weights / "fasttext.model"))
    classifier = joblib.load(weights / "classifier.joblib")
    idf_lookup = joblib.load(weights / "idf_lookup.joblib")
    pooling = hyperparameters["pooling"]
    normalize = bool(hyperparameters["normalize_embeddings"])
    cache = {}

    def vector_of(token):
        if token not in cache:
            try:
                cache[token] = ft_model.wv[token]
            except KeyError:
                cache[token] = None
        return cache[token]

    def predict(texts):
        tokenized = [fasttext_tokenize(t) for t in texts]
        embeddings = fasttext_pool(
            tokenized, vector_of, ft_model.vector_size, pooling, idf_lookup, normalize
        )
        return classifier.predict_proba(embeddings)[:, 1]

    sources = {
        "fasttext_model": str(weights / "fasttext.model"),
        "classifier": str(weights / "classifier.joblib"),
        "idf_lookup": str(weights / "idf_lookup.joblib"),
        "pooling": pooling,
        "normalize_embeddings": normalize,
    }
    return predict, sources


# ---------------------------------------------------------------------------
# Transformers: shared inference (notebooks 07 and 11)
# ---------------------------------------------------------------------------
def _torch_device(device):
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ModelLoadError("device 'cuda' requested but CUDA is not available")
    return torch.device(device)


def _transformer_probabilities(model, tokenizer, strings, max_length, batch_size, device):
    """P(EXTREMIST) = softmax(logits)[:, 1], batch by batch, as in both notebooks."""
    import torch

    strings = [str(value) for value in strings]
    if not strings:
        return np.zeros(0)
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(strings), batch_size):
            encoded = tokenizer(
                strings[start : start + batch_size],
                truncation=True,
                padding=True,
                max_length=int(max_length),
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            outputs.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(outputs)


def _load_hf_classifier(folder, device, dtype="float32"):
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(str(folder))
    model = model.half() if dtype == "float16" else model.float()
    model.to(device)
    model.eval()
    return model


def _load_transformer(external_dir, hyperparameters, device, batch_size):
    from transformers import AutoTokenizer

    external_dir = Path(external_dir)
    model_dir = external_dir / "weights" / "transformer_model"
    tokenizer_dir = external_dir / "tokenizer"
    if not tokenizer_dir.is_dir():
        raise ModelLoadError(f"{external_dir}: transformer weights without a tokenizer/ folder")
    if "max_length" not in hyperparameters:
        raise ModelLoadError("best_config.json hyperparameters lack 'max_length' (needed by 07)")
    torch_device = _torch_device(device)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=True)
    model = _load_hf_classifier(model_dir, torch_device)
    max_length = int(hyperparameters["max_length"])

    def predict(texts):
        return _transformer_probabilities(model, tokenizer, texts, max_length, batch_size, torch_device)

    notes = []
    if torch_device.type != "cuda":
        notes.append("served in float32 on CPU; the notebook used float16 autocast on the GPU")
    return predict, {"model": str(model_dir), "tokenizer": str(tokenizer_dir), "max_length": max_length}, notes


# ---------------------------------------------------------------------------
# 11: log-odds pooled multi-checkpoint ensemble (notebook 11, sections 4 and 5)
# ---------------------------------------------------------------------------
# Pinned to notebook 11's regexes and CONFIG["pooling"] by the test suite.
ENSEMBLE_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+|\bhttp\S*")
ENSEMBLE_USER_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
ENSEMBLE_WHITESPACE_RE = re.compile(r"\s+")
ENSEMBLE_LOGIT_CLIP_EPS = 1e-6
ENSEMBLE_SIGMOID_CLIP = 40.0


def normalize_model_text(value, mode):
    """Notebook 11's `normalize_model_text`."""
    value = str(value)
    if mode == "identity":
        return value
    if mode != "minimal_twitter":
        raise ModelLoadError(f"unsupported text_normalization_mode {mode!r}")
    value = html.unescape(value)
    value = unicodedata.normalize("NFKC", value)
    value = ENSEMBLE_USER_RE.sub("@user", value)
    value = ENSEMBLE_URL_RE.sub("http", value)
    return ENSEMBLE_WHITESPACE_RE.sub(" ", value).strip()


def probability_to_log_odds(probability, eps=ENSEMBLE_LOGIT_CLIP_EPS):
    p = np.clip(np.asarray(probability, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def log_odds_to_probability(values, clip=ENSEMBLE_SIGMOID_CLIP):
    values = np.clip(np.asarray(values, dtype=float), -clip, clip)
    return 1.0 / (1.0 + np.exp(-values))


def pool_component_log_odds(log_odds_by_component, names, weights):
    """Weighted mean of component log-odds -> probability (notebook 11's `pool_components`)."""
    weights = np.asarray(weights, dtype=float)
    if len(weights) != len(names) or weights.sum() <= 0:
        raise ModelLoadError("component weights must be positive and one per component")
    weights = weights / weights.sum()
    stacked = np.stack([np.asarray(log_odds_by_component[name], dtype=float) for name in names])
    return log_odds_to_probability((weights[:, None] * stacked).sum(axis=0))


def _member_folders(weights_dir, component):
    pattern = re.compile(rf"^member-{re.escape(component)}-seed(\d+)$")
    found = []
    for path in sorted(weights_dir.iterdir()):
        match = pattern.match(path.name)
        if match and path.is_dir():
            found.append((int(match.group(1)), path))
    return found


def _read_inference_dtypes(external_dir):
    """{(component, seed): inference_dtype} from training_logs/run_summary.csv, if present."""
    path = Path(external_dir) / "training_logs" / "run_summary.csv"
    if not path.exists():
        return {}
    import pandas as pd

    summary = pd.read_csv(path)
    needed = {"component", "seed", "inference_dtype"}
    if not needed.issubset(summary.columns):
        return {}
    return {
        (str(row.component), int(row.seed)): str(row.inference_dtype)
        for row in summary.itertuples()
    }


def _load_pooled_ensemble(external_dir, hyperparameters, device, batch_size):
    from transformers import AutoTokenizer

    external_dir = Path(external_dir)
    for key in ("components", "component_weights", "text_normalization_mode", "component_run_configs"):
        if key not in hyperparameters:
            raise ModelLoadError(f"best_config.json hyperparameters lack '{key}' (needed by 11)")
    components = list(hyperparameters["components"])
    weights = [float(w) for w in hyperparameters["component_weights"]]
    mode = hyperparameters["text_normalization_mode"]
    run_configs = hyperparameters["component_run_configs"]
    torch_device = _torch_device(device)
    dtypes = _read_inference_dtypes(external_dir)
    notes = []

    loaded = {}
    sources = {"components": {}, "text_normalization_mode": mode, "component_weights": weights}
    for component in components:
        folders = _member_folders(external_dir / "weights", component)
        if not folders:
            raise ModelLoadError(f"{external_dir}: no member-{component}-seed* folders")
        tokenizer_dir = external_dir / "tokenizer" / component
        if not tokenizer_dir.is_dir():
            raise ModelLoadError(f"{external_dir}: no tokenizer/{component}/ folder")
        if component not in run_configs or "max_length" not in run_configs[component]:
            raise ModelLoadError(f"component_run_configs[{component!r}] lacks max_length")
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=True)
        runs = []
        for seed, folder in folders:
            dtype = dtypes.get((component, seed), "float32")
            if torch_device.type != "cuda":
                dtype = "float32"
            runs.append({"seed": seed, "model": _load_hf_classifier(folder, torch_device, dtype)})
        loaded[component] = {
            "tokenizer": tokenizer,
            "runs": runs,
            "max_length": int(run_configs[component]["max_length"]),
        }
        sources["components"][component] = {
            "seeds": [seed for seed, _ in folders],
            "folders": [str(folder) for _, folder in folders],
            "tokenizer": str(tokenizer_dir),
        }
    if torch_device.type != "cuda":
        notes.append("members served in float32 on CPU; the notebook served float16 runs in float16")

    def component_log_odds(component, strings):
        entry = loaded[component]
        prepared = [normalize_model_text(value, mode) for value in strings]
        per_seed = []
        for run in entry["runs"]:
            probabilities = _transformer_probabilities(
                run["model"], entry["tokenizer"], prepared, entry["max_length"], batch_size, torch_device
            )
            per_seed.append(probability_to_log_odds(probabilities))
        return np.mean(per_seed, axis=0)

    def predict(texts):
        strings = [str(t) for t in texts]
        log_odds = {name: component_log_odds(name, strings) for name in components}
        return pool_component_log_odds(log_odds, components, weights)

    def make_member(component):
        return lambda texts: log_odds_to_probability(component_log_odds(component, [str(t) for t in texts]))

    members = {component: make_member(component) for component in components}
    return predict, members, sources, notes


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def load_technique(technique, external_dir, results_dir=None, device="auto", batch_size=64):
    """Restore one technique: its predictor, its members, and its locked threshold."""
    external_dir = Path(external_dir)
    if not external_dir.is_dir():
        raise ModelLoadError(f"{technique}: external folder {external_dir} does not exist")
    config = read_best_config(technique, results_dir)
    hyperparameters = config["hyperparameters"]
    family = detect_family(external_dir)
    members = {}
    notes = []

    if family == "sklearn_pipeline":
        predict, sources = _load_sklearn_pipeline(external_dir)
    elif family == "slp_tfidf":
        predict, sources = _load_slp(external_dir)
    elif family == "fasttext_logreg":
        predict, sources = _load_fasttext_logreg(external_dir, hyperparameters)
    elif family == "transformer":
        predict, sources, notes = _load_transformer(external_dir, hyperparameters, device, batch_size)
    elif family == "pooled_transformer_ensemble":
        predict, members, sources, notes = _load_pooled_ensemble(
            external_dir, hyperparameters, device, batch_size
        )
    else:  # pragma: no cover - detect_family refuses anything else
        raise ModelLoadError(f"unhandled family {family}")

    def checked_predict(texts):
        values = _as_probabilities(predict(texts), technique)
        if len(values) != len(list(texts)):
            raise ModelLoadError(f"{technique}: predictor returned {len(values)} values for {len(list(texts))} texts")
        return values

    return LoadedModel(
        technique=technique,
        family=family,
        threshold=float(config["selected_threshold"]),
        predict=checked_predict,
        members=members,
        sources=sources,
        notes=notes,
    )
