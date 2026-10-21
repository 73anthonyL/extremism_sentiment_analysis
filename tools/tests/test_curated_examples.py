"""Tests for tools/curated_examples.py and tools/model_loaders.py.

Synthetic text only. The end-to-end test fits a small TF-IDF + logistic
regression pipeline, saves it the way notebook 01 does, and drives the tool
through verification, explanation and the report. Adversarial tests prove the
tool refuses: weights it cannot recognise, a best_config without a threshold,
a restored model that disagrees with the committed probabilities, outputs
from two different curated sets in one folder, a technique folder whose
predictions do not cover every example, a labels file that does not line up
with the texts, and a pooling component that returns malformed log-odds.
The end-to-end test also checks that the recorded explainer endpoints add up
with the attributions. The sequential-checkpoint path is checked against the resident explainer on a
synthetic two-component pool. The pinning tests parse the notebooks and
compare the constants the loaders reproduce.
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

    def test_numpy_random_shim_accepts_numpy2_pickle_conventions(self):
        """numpy 2 pickles pass a BitGenerator class or instance and a newer state format."""
        import numpy.random._pickle as np_pickle
        from numpy.random import MT19937, RandomState

        ml._install_numpy_random_unpickle_shim()
        # Dunder names are mangled inside a class body, so fetch them by string.
        bit_ctor = getattr(np_pickle, "__bit_generator_ctor")
        state_ctor = getattr(np_pickle, "__randomstate_ctor")
        by_class = bit_ctor(MT19937)
        assert isinstance(by_class, MT19937)
        assert isinstance(bit_ctor("MT19937"), MT19937)
        state = state_ctor(MT19937())
        assert isinstance(state, RandomState)
        # A state in an unknown format is ignored instead of failing the load.
        by_class.__setstate__({"bit_generator": "MT19937", "state": {"unknown": 1}})
        state.__setstate__({"bit_generator": "MT19937", "state": {"unknown": 1}})
        # A legacy state still round-trips.
        state.set_state(RandomState(7).get_state())
        assert state.randint(0, 1000) == RandomState(7).randint(0, 1000)
        with pytest.raises(ValueError):
            bit_ctor("NoSuchGenerator")

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
        # The recorded endpoints are the explainer's: base value + attributions = explained output.
        sums = attributions.groupby("example_id")["attribution"].sum()
        endpoints = predictions.set_index("example_id")
        assert np.allclose(endpoints[ce.BASE_VALUE_COLUMN] + sums, endpoints[ce.EXPLAINED_OUTPUT_COLUMN], atol=1e-9)
        assert endpoints[ce.BASE_VALUE_COLUMN].nunique() == 1

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


class TestAnnotations:
    def test_labels_and_notes_attach_in_order(self, world, tmp_path):
        labels = tmp_path / "labels.txt"
        labels.write_text("# expected\nExtremist\n\nnon extremist\n0\n")
        notes = tmp_path / "notes.txt"
        notes.write_text("first\nsecond\nthird\n")
        examples = ce.read_examples(world["texts"], labels, notes)
        assert examples[ce.EXPECTED_LABEL_COLUMN].tolist() == [
            ce.POSITIVE_CLASS_NAME, ce.NEGATIVE_CLASS_NAME, ce.NEGATIVE_CLASS_NAME
        ]
        assert examples[ce.NOTE_COLUMN].tolist() == ["first", "second", "third"]

    def test_label_count_mismatch_is_refused(self, world, tmp_path):
        labels = tmp_path / "labels.txt"
        labels.write_text("extremist\nnon-extremist\n")
        with pytest.raises(ce.CuratedExamplesError, match="one line per text"):
            ce.read_examples(world["texts"], labels)

    def test_unknown_label_is_refused(self, world, tmp_path):
        labels = tmp_path / "labels.txt"
        labels.write_text("extremist\nmaybe\nextremist\n")
        with pytest.raises(ce.CuratedExamplesError, match="unrecognised expected label"):
            ce.read_examples(world["texts"], labels)

    def test_existing_annotations_survive_a_rerun_without_them(self, world, tmp_path):
        labels = tmp_path / "labels.txt"
        labels.write_text("extremist\nnon-extremist\nextremist\n")
        ce.run_explain(
            world["texts"], world["external_root"], [TECHNIQUE], out_dir=world["out"],
            results_dir=world["results_dir"], max_evals=60, verify_rows=0, labels_path=labels,
        )
        run_world(world)  # no labels given this time
        examples = pd.read_csv(world["out"] / ce.EXAMPLES_FILENAME)
        assert examples[ce.EXPECTED_LABEL_COLUMN].tolist() == [
            ce.POSITIVE_CLASS_NAME, ce.NEGATIVE_CLASS_NAME, ce.POSITIVE_CLASS_NAME
        ]

    def test_report_records_correctness_without_rerunning(self, world, tmp_path):
        run_world(world)
        labels = tmp_path / "labels.txt"
        labels.write_text("extremist\nnon-extremist\nextremist\n")
        summary = ce.run_report(world["out"], labels_path=labels)
        assert set(summary["correct"]) <= {"yes", "no"}
        expected = summary[ce.EXPECTED_LABEL_COLUMN] == summary["predicted_class"]
        assert (summary["correct"].eq("yes") == expected).all()
        reread = pd.read_csv(world["out"] / ce.SUMMARY_FILENAME)
        assert reread["correct"].dtype == object and set(reread["correct"]) <= {"yes", "no"}
        assert not set(summary.columns) & set(ce.TEXT_BEARING_COLUMNS)

        results = json.loads((world["out"] / ce.RESULTS_FILENAME).read_text(encoding="utf-8"))
        assert results["n_examples"] == 3 and results["n_labelled"] == 3
        assert TECHNIQUE in results["techniques"]
        first = results["examples"][0]
        assert first["expected_label"] == ce.POSITIVE_CLASS_NAME and first["text"]
        classification = first["classifications"][0]
        assert classification["technique"] == TECHNIQUE and classification["member"] == ce.WHOLE_MODEL
        assert classification["correct"] == (classification["predicted_label"] == first["expected_label"])
        assert classification["tokens"] and {"position", "token", "attribution", "category"} <= set(
            classification["tokens"][0]
        )
        page = (world["out"] / ce.REPORT_FILENAME).read_text(encoding="utf-8")
        assert "Expected" in page and "Correct" in page


def synthetic_pool(seed=5):
    """Two components with continuous, deterministic word scores, pooled like notebook 11."""
    rng = np.random.default_rng(seed)
    vocab = POSITIVE_WORDS + NEGATIVE_WORDS + NEUTRAL_WORDS + [ce.MASK_TOKEN]
    tables = [{w: float(rng.normal(scale=1.5)) for w in vocab} for _ in range(2)]

    def score(table, string):
        return sum(table.get(w, 0.37) for w in str(string).split())

    def log_odds(name, strings):
        table = tables[{"A": 0, "B": 1}[name]]
        return np.array([score(table, s) for s in strings], dtype=float)

    names, weights = ["A", "B"], [1.0, 1.0]

    def predict(texts):
        return ml.pool_component_log_odds({n: log_odds(n, texts) for n in names}, names, weights)

    members = {n: (lambda texts, n=n: ml.log_odds_to_probability(log_odds(n, texts))) for n in names}
    return ml.LoadedModel(
        technique="99_POOL", family="pooled_transformer_ensemble", threshold=0.5, predict=predict,
        members=members, pooling={"names": names, "weights": weights, "log_odds": log_odds},
    )


class TestSequentialCheckpoints:
    def test_replay_matches_the_resident_explainer(self, world):
        model = synthetic_pool()
        examples = ce.read_examples(world["texts"])
        word_index, _ = ce.build_word_index(ce.load_lexicons())
        resident_predictions, resident_attributions = ce.explain_model(
            model, examples, max_evals=200, word_index=word_index
        )
        predictions, attributions, serving = ce.explain_model_sequential(
            model, examples, max_evals=200, word_index=word_index, log=lambda *_: None
        )
        assert serving["mode"] == "sequential_checkpoints" and serving["rounds"] >= 2
        keys = ["example_id", "member"]
        assert predictions.sort_values(keys)["y_prob"].to_numpy() == pytest.approx(
            resident_predictions.sort_values(keys)["y_prob"].to_numpy()
        )
        left = resident_attributions.sort_values(keys + ["position"]).reset_index(drop=True)
        right = attributions.sort_values(keys + ["position"]).reset_index(drop=True)
        assert left["token"].tolist() == right["token"].tolist()
        assert np.allclose(left["attribution"], right["attribution"], atol=1e-9)
        for column in (ce.BASE_VALUE_COLUMN, ce.EXPLAINED_OUTPUT_COLUMN):
            assert predictions.sort_values(keys)[column].to_numpy() == pytest.approx(
                resident_predictions.sort_values(keys)[column].to_numpy()
            )
        for name in model.pooling["names"]:
            assert serving["strings_evaluated_per_component"][name] > 0

    def test_malformed_component_output_is_refused(self, world):
        model = synthetic_pool()
        examples = ce.read_examples(world["texts"])
        broken = dict(model.pooling, log_odds=lambda name, strings: np.zeros(len(strings) - 1))
        short = ml.LoadedModel(**{**model.__dict__, "pooling": broken})
        with pytest.raises(ce.CuratedExamplesError, match="log-odds"):
            ce.explain_model_sequential(short, examples, max_evals=60, log=lambda *_: None)
        nan = dict(model.pooling, log_odds=lambda name, strings: np.full(len(strings), np.nan))
        with pytest.raises(ce.CuratedExamplesError, match="log-odds"):
            ce.explain_model_sequential(
                ml.LoadedModel(**{**model.__dict__, "pooling": nan}), examples, max_evals=60,
                log=lambda *_: None,
            )

    def test_a_model_without_pooling_is_refused(self, world):
        model = ml.load_technique(
            TECHNIQUE, world["external_root"] / TECHNIQUE, results_dir=world["results_dir"]
        )
        with pytest.raises(ce.CuratedExamplesError, match="no pooling"):
            ce.explain_model_sequential(model, ce.read_examples(world["texts"]))
