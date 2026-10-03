"""End-to-end test of the notebook pipeline on synthetic text.

Notebooks are never executed locally, so this test drives the kit through
the same sequence of calls a classical notebook makes (notebook 01 is the
reference): load the foundation, fit a TF-IDF + logistic regression
pipeline, select the threshold on validation, evaluate test once, export the
probability artifact, run the SHAP partition explainer over the word masker,
export both attribution runs and the coefficients, write the result folder,
and package.

It then hands the outputs to the real tools. The decisive assertion is
`validate_shap.validate`: the plan's premise is that one model-agnostic
explainer can stand in for every family, and that is only credible if it
recovers what a linear model demonstrably relies on. Here the generating
vocabulary is known, so the check has a ground truth.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIR))

import notebook_kit as nk
import probs_artifact
from attributions import list_runs
from validate_results_folder import validate_folder
from validate_shap import validate

pytest.importorskip("shap")

TECHNIQUE = "01_LOG-REG_TF-IDF"
POSITIVE_WORDS = [f"alarm{i}" for i in range(12)]
NEGATIVE_WORDS = [f"calm{i}" for i in range(12)]
NEUTRAL_WORDS = [f"filler{i}" for i in range(30)]


def synthetic_corpus(directory, seed=30):
    """A foundation folder with the frozen shape whose labels follow known words."""
    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    index = 0
    for split, (_, negatives, positives) in nk.EXPECTED_SPLIT_COUNTS.items():
        for label, count in ((0, negatives), (1, positives)):
            signal, noise = (POSITIVE_WORDS, NEGATIVE_WORDS) if label else (NEGATIVE_WORDS, POSITIVE_WORDS)
            for _ in range(count):
                words = list(rng.choice(NEUTRAL_WORDS, size=rng.integers(6, 14)))
                words += list(rng.choice(signal, size=rng.integers(1, 4)))
                if rng.random() < 0.25:
                    words += list(rng.choice(noise, size=1))
                rng.shuffle(words)
                text = " ".join(words) + f" unique{index}"
                rows.append(
                    {
                        "row_id": f"ex_{index:06d}",
                        "text": text,
                        "label": label,
                        "text_hash": hashlib.sha256(text.encode()).hexdigest()[:16],
                        "split": split,
                    }
                )
                index += 1
    frame = pd.DataFrame(rows)
    frame[["row_id", "text", "label", "text_hash"]].to_csv(
        directory / "processed_dataset.csv", index=False
    )
    frame[["row_id", "label", "text_hash", "split"]].to_csv(
        directory / "split_assignments.csv", index=False
    )
    (directory / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": nk.DATASET_VERSION,
                "split_version": nk.SPLIT_VERSION,
                "processed_rows": len(frame),
            }
        )
    )


CONFIG = {
    "project_name": "extremism_text_classification_comparison",
    "technique_name": TECHNIQUE,
    "role": "classical",
    "in_comparison": True,
    "model_family": "logistic_regression",
    "feature_family": "tfidf_word",
    "dataset_version": nk.DATASET_VERSION,
    "split_version": nk.SPLIT_VERSION,
    "random_seed": 30,
    "threshold": {"metric": "accuracy"},
    "explainer": {
        "max_evals": 200,
        "batch_size": 64,
        "seed": 30,
        "n_posts_per_split": 120,
        "per_member_runs": False,
        "top_k_display": 50,
    },
    "export_coefficients": True,
    "external": {"save_weights": True, "save_predictions": True},
    "error_analysis": {"include_text_preview": False},
}


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """One full run of the classical pipeline; every test below reads its outputs."""
    root = tmp_path_factory.mktemp("pipeline")
    synthetic_corpus(root / "input" / "research_foundation")
    ctx = nk.bootstrap(CONFIG, working_root=root / "working", input_roots=[root / "input"])

    frame, _ = nk.load_foundation(ctx)
    train_df, val_df, test_df = nk.split_frames(frame)
    train_texts = train_df["text"].tolist()
    y_train = train_df["label"].to_numpy().astype(int)
    val_texts = val_df["text"].tolist()

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(sublinear_tf=True, lowercase=True)),
            ("classifier", LogisticRegression(C=10.0, solver="liblinear", random_state=30)),
        ]
    ).fit(train_texts, y_train)

    def predict_proba_texts(texts):
        return pipeline.predict_proba(list(texts))[:, 1]

    screen = nk.evaluate(ctx, val_df, predict_proba_texts(val_texts), 0.5, "validation")
    ablation_results = pd.DataFrame([nk.ablation_row(ctx, "config_001", {"C": 10.0}, screen)])
    best_row = nk.select_best_config(ablation_results)

    val_prob = predict_proba_texts(val_texts)
    threshold, threshold_sweep = nk.select_threshold(val_df, val_prob, metric="accuracy")
    validation_metrics = nk.evaluate(ctx, val_df, val_prob, threshold, "validation")
    test_prob = predict_proba_texts(test_df["text"].tolist())
    test_metrics = nk.evaluate(ctx, test_df, test_prob, threshold, "test")

    nk.export_probabilities(
        ctx,
        val_df,
        val_prob,
        test_df,
        test_prob,
        meta={
            "model_family": CONFIG["model_family"],
            "feature_family": CONFIG["feature_family"],
            "hyperparameters": {"C": 10.0},
            "selected_threshold": threshold,
            "threshold_strategy": nk.THRESHOLD_STRATEGY,
            "threshold_metric": "accuracy",
        },
    )

    settings = CONFIG["explainer"]
    tables = {}
    for split, split_df in (("test", test_df), ("validation", val_df)):
        sample = nk.sample_for_explanation(split_df, settings["n_posts_per_split"], settings["seed"])
        bundle = nk.explain(ctx, predict_proba_texts, sample)
        tables[split] = nk.export_attribution_run(
            ctx, bundle, f"shap-partition_{split}_seed{settings['seed']}"
        )

    feature_names = pipeline.named_steps["tfidf"].get_feature_names_out()
    coefficients = pipeline.named_steps["classifier"].coef_.ravel()
    nk.export_coefficients(ctx, feature_names, coefficients)

    nk.export_results_folder(
        ctx,
        {
            "model_family": CONFIG["model_family"],
            "feature_family": CONFIG["feature_family"],
            "hyperparameters": {"C": 10.0},
            "threshold_strategy": nk.THRESHOLD_STRATEGY,
            "threshold_metric": "accuracy",
            "selected_threshold": threshold,
            "selected_config_id": best_row["config_id"],
        },
        validation_metrics,
        test_metrics,
        ablation_results,
        threshold_sweep,
    )
    nk.export_external_predictions(ctx, val_df, val_prob, test_df, test_prob, threshold)
    zip_path = nk.finalize(ctx)
    return {"ctx": ctx, "tables": tables, "test_metrics": test_metrics, "zip": zip_path}


def test_model_learned_the_synthetic_task(run):
    assert run["test_metrics"]["accuracy"] > 0.9


def test_result_folder_passes_the_real_validator(run):
    assert validate_folder(run["ctx"].results_dir) == []


def test_probability_artifacts_load_through_the_real_loader(run):
    ctx = run["ctx"]
    for split in ("validation", "test"):
        probs_artifact.load_probs(ctx.probs_dir / f"{TECHNIQUE}__{split}.csv", expected_split=split)


def test_both_whole_model_runs_are_registered(run):
    results_dir = run["ctx"].results_dir.parent
    assert list_runs(TECHNIQUE, results_dir) == [
        "shap-partition_test_seed30",
        "shap-partition_validation_seed30",
    ]


def test_word_masker_shap_recovers_logistic_regression_coefficients(run):
    """The premise of the cross-model comparison, checked against a known ground truth."""
    results_dir = run["ctx"].results_dir.parent
    for run_id in ("shap-partition_test_seed30", "shap-partition_validation_seed30"):
        stats, _ = validate(TECHNIQUE, run_id=run_id, top_k=24, results_dir=results_dir)
        assert stats["top_k_sign_agreement"] >= 0.9, stats
        assert stats["passed"], stats


def test_attribution_signs_match_the_generating_vocabulary(run):
    table = run["tables"]["test"].set_index("word")
    positive = [w for w in POSITIVE_WORDS if w in table.index]
    negative = [w for w in NEGATIVE_WORDS if w in table.index]
    assert len(positive) >= 10 and len(negative) >= 10
    assert (table.loc[positive, "mean_attribution"] > 0).all()
    assert (table.loc[negative, "mean_attribution"] < 0).all()
    neutral = [w for w in NEUTRAL_WORDS if w in table.index]
    assert table.loc[neutral, "mean_abs_attribution"].mean() < table.loc[
        positive + negative, "mean_abs_attribution"
    ].mean()


def test_test_and_validation_runs_agree(run):
    """Run 2 exists to measure stability across posts; on a clean task the runs must agree."""
    merged = run["tables"]["test"].merge(
        run["tables"]["validation"], on="word", suffixes=("_test", "_validation")
    )
    signal = merged[merged["word"].isin(POSITIVE_WORDS + NEGATIVE_WORDS)]
    correlation = signal["mean_attribution_test"].corr(signal["mean_attribution_validation"])
    assert correlation > 0.9


def test_committed_tree_holds_no_text_or_row_ids(run):
    ctx = run["ctx"]
    for path in ctx.results_dir.rglob("*.csv"):
        columns = set(pd.read_csv(path, nrows=0).columns)
        assert not columns & set(nk.TEXT_BEARING_COLUMNS + nk.ROW_IDENTITY_COLUMNS), path.name
    for path in ctx.probs_dir.glob("*.csv"):
        assert list(pd.read_csv(path, nrows=0).columns) == list(nk.PROBS_REQUIRED_COLUMNS)
    local = ctx.external_dir / "attributions_local" / "shap-partition_test_seed30.csv"
    assert local.exists() and "row_id" in pd.read_csv(local, nrows=0).columns
