"""Tests for tools/curated_examples.py and tools/model_loaders.py.

Synthetic text only. The end-to-end test fits a small TF-IDF + logistic
regression pipeline, saves it the way notebook 01 does, and drives the tool
through verification, explanation and the report. Adversarial tests prove the
tool refuses: weights it cannot recognise, a best_config without a threshold,
a restored model that disagrees with the committed probabilities, outputs
from two different curated sets in one folder, and a technique folder whose
predictions do not cover every example. The pinning tests parse the
notebooks and compare the constants the loaders reproduce.
"""

import json
import re
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

pytest.importorskip("shap")

import curated_examples as ce  # noqa: E402
import model_loaders as ml  # noqa: E402

NOTEBOOKS_DIR = TOOLS_DIR.parent / "notebooks"
TECHNIQUE = "01_LOG-REG_TF-IDF"
POSITIVE_WORDS = [f"alarm{i}" for i in range(8)]
NEGATIVE_WORDS = [f"calm{i}" for i in range(8)]
NEUTRAL_WORDS = [f"filler{i}" for i in range(20)]


def synthetic_rows(n, seed=30):
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(n):
        label = int(index % 2)
        signal = POSITIVE_WORDS if label else NEGATIVE_WORDS
        words = list(rng.choice(NEUTRAL_WORDS, size=rng.integers(4, 9)))
        words += list(rng.choice(signal, size=rng.integers(1, 3)))
        rng.shuffle(words)
        rows.append({"row_id": f"ex_{index:06d}", "text": " ".join(words), "label": label})
    return pd.DataFrame(rows)


def fitted_pipeline(rows):
    pipeline = Pipeline(
        [("tfidf", TfidfVectorizer(lowercase=True)), ("classifier", LogisticRegression(C=3.0))]
    )
    return pipeline.fit(rows["text"], rows["label"])


def write_best_config(results_dir, technique, threshold=0.5, **hyperparameters):
    folder = results_dir / technique
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_family": "logistic_regression",
        "feature_family": "tfidf_word",
        "hyperparameters": hyperparameters or {"C": 3.0},
        "threshold_strategy": "grid",
        "threshold_metric": "accuracy",
        "selected_threshold": threshold,
    }
    (folder / "best_config.json").write_text(json.dumps(payload))
    return folder


@pytest.fixture
def world(tmp_path):
    """A saved pipeline, its committed config and probabilities, and a curated texts file."""
    import joblib

    rows = synthetic_rows(240)
    pipeline = fitted_pipeline(rows)
    external_root = tmp_path / "external"
    weights = external_root / TECHNIQUE / "weights"
    weights.mkdir(parents=True)
    joblib.dump(pipeline, weights / "pipeline.joblib")
    results_dir = tmp_path / "results_summary"
    write_best_config(results_dir, TECHNIQUE, threshold=0.6)

    reference = rows.iloc[:40][["row_id", "text"]].reset_index(drop=True)
    committed = pd.DataFrame(
        {"row_id": reference["row_id"], "y_prob": pipeline.predict_proba(reference["text"])[:, 1]}
    )
    texts = tmp_path / "curated.txt"
    texts.write_text(
        "# curated set\n"
        f"{POSITIVE_WORDS[0]} {NEUTRAL_WORDS[1]} {POSITIVE_WORDS[2]}\n"
        "\n"
        f"{NEGATIVE_WORDS[0]} {NEUTRAL_WORDS[3]}\n"
        f"{NEUTRAL_WORDS[5]} {NEUTRAL_WORDS[6]} {NEGATIVE_WORDS[1]} {POSITIVE_WORDS[1]}\n",
        encoding="utf-8",
    )
    return {
        "rows": rows,
        "pipeline": pipeline,
        "external_root": external_root,
        "results_dir": results_dir,
        "reference": reference,
        "committed": committed,
        "texts": texts,
        "out": tmp_path / "out",
    }


def run_world(world, texts=None, fresh=False, committed=None, max_evals=120):
    committed = world["committed"] if committed is None else committed
    return ce.run_explain(
        texts or world["texts"],
        world["external_root"],
        [TECHNIQUE],
        out_dir=world["out"],
        results_dir=world["results_dir"],
        max_evals=max_evals,
        verify_rows=40,
        fresh=fresh,
        reference_loader=lambda n, seed: world["reference"],
        committed_loader=lambda technique: committed,
    )


class TestReadExamples:
    def test_blank_and_comment_lines_are_skipped_and_ids_are_stable(self, world):
        examples = ce.read_examples(world["texts"])
        assert examples["example_id"].tolist() == ["curated_01", "curated_02", "curated_03"]
        assert examples["text"].str.len().gt(0).all()
        assert examples["sha256"].str.len().eq(16).all()

    def test_empty_file_is_refused(self, tmp_path):
        path = tmp_path / "empty.txt"
        path.write_text("# only a comment\n\n")
        with pytest.raises(ce.CuratedExamplesError, match="no curated texts"):
            ce.read_examples(path)

    def test_duplicate_text_is_refused(self, tmp_path):
        path = tmp_path / "dup.txt"
        path.write_text("same words here\nother words\nsame words here\n")
        with pytest.raises(ce.CuratedExamplesError, match="more than once"):
            ce.read_examples(path)


class TestLoaders:
    def test_sklearn_pipeline_restores_the_notebook_predictor(self, world):
        model = ml.load_technique(
            TECHNIQUE, world["external_root"] / TECHNIQUE, results_dir=world["results_dir"]
        )
        assert model.family == "sklearn_pipeline"
        assert model.threshold == pytest.approx(0.6)
        texts = world["rows"]["text"].iloc[:5].tolist()
        expected = world["pipeline"].predict_proba(texts)[:, 1]
        assert np.allclose(model.predict(texts), expected)

    def test_unrecognised_weights_are_refused(self, tmp_path):
        external = tmp_path / "external" / "09_T"
        (external / "weights").mkdir(parents=True)
        (external / "weights" / "something.bin").write_bytes(b"x")
        write_best_config(tmp_path / "results_summary", "09_T")
        with pytest.raises(ml.ModelLoadError, match="no recognisable weights"):
            ml.load_technique("09_T", external, results_dir=tmp_path / "results_summary")

    def test_best_config_without_threshold_is_refused(self, world):
        config_path = world["results_dir"] / TECHNIQUE / "best_config.json"
        payload = json.loads(config_path.read_text())
        del payload["selected_threshold"]
        config_path.write_text(json.dumps(payload))
        with pytest.raises(ml.ModelLoadError, match="selected_threshold"):
            ml.load_technique(TECHNIQUE, world["external_root"] / TECHNIQUE, results_dir=world["results_dir"])

    def test_slp_loader_matches_a_hand_computed_sigmoid(self, tmp_path):
        import joblib

        torch = pytest.importorskip("torch")
        rows = synthetic_rows(60)
        vectorizer = TfidfVectorizer(lowercase=True).fit(rows["text"])
        n_features = len(vectorizer.get_feature_names_out())
        rng = np.random.default_rng(1)
        weight = torch.tensor(rng.normal(size=(1, n_features)), dtype=torch.float32)
        bias = torch.tensor([0.1], dtype=torch.float32)
        external = tmp_path / "external" / "03_T"
        (external / "weights").mkdir(parents=True)
        joblib.dump(vectorizer, external / "weights" / "vectorizer.joblib")
        torch.save({"linear.weight": weight, "linear.bias": bias}, external / "weights" / "slp_state_dict.pt")
        write_best_config(tmp_path / "results_summary", "03_T")

        model = ml.load_technique("03_T", external, results_dir=tmp_path / "results_summary")
        texts = rows["text"].iloc[:4].tolist()
        features = vectorizer.transform(texts).toarray()
        expected = 1.0 / (1.0 + np.exp(-(features @ weight.numpy().T + bias.numpy()).ravel()))
        assert model.family == "slp_tfidf"
        assert np.allclose(model.predict(texts), expected, atol=1e-6)

    def test_pooling_matches_notebook_11_arithmetic(self):
        log_odds = {"A": np.array([0.0, 2.0]), "B": np.array([2.0, -2.0])}
        pooled = ml.pool_component_log_odds(log_odds, ["A", "B"], [1.0, 1.0])
        expected = 1.0 / (1.0 + np.exp(-np.array([1.0, 0.0])))
        assert np.allclose(pooled, expected)
        weighted = ml.pool_component_log_odds(log_odds, ["A", "B"], [3.0, 1.0])
        assert np.allclose(weighted, 1.0 / (1.0 + np.exp(-np.array([0.5, 1.0]))))
        with pytest.raises(ml.ModelLoadError):
            ml.pool_component_log_odds(log_odds, ["A", "B"], [1.0])

    def test_minimal_twitter_normalization(self):
        text = "  Hello @Someone see https://x.y/z &amp; www.site.org   now "
        assert ml.normalize_model_text(text, "minimal_twitter") == "Hello @user see http & http now"
        assert ml.normalize_model_text(text, "identity") == text
        with pytest.raises(ml.ModelLoadError):
            ml.normalize_model_text(text, "other")


class TestNotebookPins:
    """The inference code reproduced in model_loaders.py must match the notebooks."""

    @staticmethod
    def notebook_source(name):
        path = NOTEBOOKS_DIR / name
        if not path.exists():
            pytest.skip(f"{name} not present")
        notebook = json.loads(path.read_text())
        return "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")

    def test_fasttext_tokenization_matches_notebook_06(self):
        source = self.notebook_source("06_FASTTEXT-EMB_LOG-REG.ipynb")
        pattern = re.search(r'"pattern":\s*r"([^"]+)"', source)
        lowercase = re.search(r'"lowercase":\s*(True|False)', source)
        assert pattern and pattern.group(1) == ml.FASTTEXT_TOKEN_PATTERN
        assert lowercase and (lowercase.group(1) == "True") == ml.FASTTEXT_LOWERCASE
        assert "def stable_hash" in source and "zlib.crc32" in source

    def test_ensemble_constants_match_notebook_11(self):
        source = self.notebook_source("11_MULTI-CHECKPOINT_LOGIT-POOL.ipynb")
        for name, compiled in (
            ("URL_RE", ml.ENSEMBLE_URL_RE),
            ("USER_RE", ml.ENSEMBLE_USER_RE),
            ("WHITESPACE_RE", ml.ENSEMBLE_WHITESPACE_RE),
        ):
            match = re.search(rf'{name} = re\.compile\(r"([^"]+)"\)', source)
            assert match and match.group(1) == compiled.pattern, name
        eps = re.search(r'"logit_clip_eps":\s*([0-9eE.-]+)', source)
        clip = re.search(r'"sigmoid_clip":\s*([0-9eE.-]+)', source)
        assert eps and float(eps.group(1)) == ml.ENSEMBLE_LOGIT_CLIP_EPS
        assert clip and float(clip.group(1)) == ml.ENSEMBLE_SIGMOID_CLIP
        assert 'html.unescape' in source and 'unicodedata.normalize("NFKC"' in source
        assert 'USER_RE.sub("@user"' in source and 'URL_RE.sub("http"' in source

    def test_weight_file_names_match_the_notebooks(self):
        expectations = {
            "01_LOG-REG_TF-IDF.ipynb": ['"pipeline.joblib"'],
            "03_SLP_TF-IDF.ipynb": ['"slp_state_dict.pt"', '"vectorizer.joblib"'],
            "06_FASTTEXT-EMB_LOG-REG.ipynb": ['"fasttext.model"', '"classifier.joblib"', '"idf_lookup.joblib"'],
            "07_TWITTER-ROBERTA_FINE-TUNE.ipynb": ['"transformer_model"', 'ctx.external("tokenizer")'],
            "11_MULTI-CHECKPOINT_LOGIT-POOL.ipynb": ['f"member-{component_name}-seed{seed}"', 'ctx.external("tokenizer") / component_name'],
        }
        for name, needles in expectations.items():
            source = self.notebook_source(name)
            for needle in needles:
                assert needle in source, f"{name} no longer saves {needle}"


class TestExplainAndReport:
    def test_end_to_end_outputs(self, world):
        written = run_world(world)
        folder = written[0]
        predictions = pd.read_csv(folder / ce.PREDICTIONS_FILENAME)
        attributions = pd.read_csv(folder / ce.ATTRIBUTIONS_FILENAME, keep_default_na=False)
        record = json.loads((folder / ce.RUN_FILENAME).read_text())

        assert list(predictions.columns) == list(ce.PREDICTION_COLUMNS)
        assert list(attributions.columns) == list(ce.ATTRIBUTION_COLUMNS)
        assert predictions["member"].eq(ce.WHOLE_MODEL).all()
        assert len(predictions) == 3
        assert ((predictions["y_prob"] >= 0.6).astype(int) == predictions["y_pred"]).all()
        assert record["verification"]["passed"] and record["verification"]["n_rows"] == 40
        assert record["explainer"]["max_evals"] == 120
        # Every curated text has word tokens attributed, and the alarm words push up.
        assert set(attributions["example_id"]) == {"curated_01", "curated_02", "curated_03"}
        first = attributions[attributions["example_id"] == "curated_01"]
        alarm = first[first["token"].str.startswith("alarm")]
        assert len(alarm) == 2 and (alarm["attribution"] > 0).all()
        assert set(attributions["category"]) <= set(ce.CATEGORIES) | {""}

        summary = ce.run_report(world["out"])
        assert len(summary) == 3
        assert not set(summary.columns) & set(ce.TEXT_BEARING_COLUMNS)
        assert "token" not in summary.columns
        share_columns = [c for c in summary.columns if c.startswith("share_positive_mass_")]
        assert len(share_columns) == 4
        assert np.allclose(summary[share_columns].sum(axis=1), 1.0)
        page = (world["out"] / ce.REPORT_FILENAME).read_text(encoding="utf-8")
        assert "curated_01" in page and TECHNIQUE in page and "alarm0" in page

    def test_a_single_curated_text_is_explained(self, world, tmp_path):
        single = tmp_path / "single.txt"
        single.write_text(f"{POSITIVE_WORDS[4]} {NEUTRAL_WORDS[2]}\n")
        folder = run_world(world, texts=single)[0]
        predictions = pd.read_csv(folder / ce.PREDICTIONS_FILENAME)
        attributions = pd.read_csv(folder / ce.ATTRIBUTIONS_FILENAME, keep_default_na=False)
        assert predictions["example_id"].tolist() == ["curated_01"]
        assert attributions["example_id"].eq("curated_01").all() and len(attributions) == 2

    def test_notes_are_aligned_and_shown(self, world, tmp_path):
        notes = tmp_path / "notes.txt"
        notes.write_text("Test 1: probe alpha\n\nTest 3: probe gamma\n")
        ce.run_explain(
            world["texts"], world["external_root"], [TECHNIQUE], out_dir=world["out"],
            notes_path=notes, results_dir=world["results_dir"], max_evals=60, verify_rows=40,
            reference_loader=lambda n, seed: world["reference"],
            committed_loader=lambda technique: world["committed"],
        )
        examples = pd.read_csv(world["out"] / ce.EXAMPLES_FILENAME, keep_default_na=False)
        assert examples["note"].tolist() == ["Test 1: probe alpha", "", "Test 3: probe gamma"]
        ce.run_report(world["out"])
        page = (world["out"] / ce.REPORT_FILENAME).read_text(encoding="utf-8")
        assert "probe gamma" in page
        short = tmp_path / "short.txt"
        short.write_text("only one note\n")
        with pytest.raises(ce.CuratedExamplesError, match="exactly one line per text"):
            ce.read_notes(short, 3)

    def test_disagreeing_committed_probabilities_are_refused(self, world):
        shifted = world["committed"].copy()
        shifted["y_prob"] = (shifted["y_prob"] + 0.2).clip(0, 1)
        with pytest.raises(ce.CuratedExamplesError, match="disagrees with the committed"):
            run_world(world, committed=shifted)
        assert not (world["out"] / TECHNIQUE / ce.PREDICTIONS_FILENAME).exists()

    def test_a_second_curated_set_in_the_same_folder_is_refused(self, world, tmp_path):
        run_world(world)
        other = tmp_path / "other.txt"
        other.write_text(f"{NEUTRAL_WORDS[0]} {POSITIVE_WORDS[3]}\n")
        with pytest.raises(ce.CuratedExamplesError, match="different curated set"):
            run_world(world, texts=other)
        run_world(world, texts=other, fresh=True)
        assert len(pd.read_csv(world["out"] / ce.EXAMPLES_FILENAME)) == 1

    def test_report_refuses_a_folder_missing_an_example(self, world):
        folder = run_world(world)[0]
        path = folder / ce.PREDICTIONS_FILENAME
        pd.read_csv(path).iloc[:-1].to_csv(path, index=False)
        with pytest.raises(ce.CuratedExamplesError, match="different curated set"):
            ce.run_report(world["out"])

    def test_summary_shares_are_text_free_by_construction(self):
        predictions = pd.DataFrame(
            [{"example_id": "e", "technique": "t", "member": "whole", "y_prob": 0.9, "threshold": 0.5, "y_pred": 1}]
        )
        attributions = pd.DataFrame(
            [
                {"example_id": "e", "technique": "t", "member": "whole", "position": 0, "token": "kkk", "attribution": 0.3, "category": "extremist_framing"},
                {"example_id": "e", "technique": "t", "member": "whole", "position": 1, "token": "and", "attribution": -0.1, "category": "topical"},
                {"example_id": "e", "technique": "t", "member": "whole", "position": 2, "token": "muslim", "attribution": 0.1, "category": "identity_term"},
            ]
        )
        summary = ce.build_summary(predictions, attributions)
        row = summary.iloc[0]
        assert row["n_tokens"] == 3
        assert row["share_positive_mass_extremist_framing"] == pytest.approx(0.75)
        assert row["share_positive_mass_identity_term"] == pytest.approx(0.25)
        assert row["share_abs_mass_topical"] == pytest.approx(0.2)
        assert row["predicted_class"] == "EXTREMIST"
