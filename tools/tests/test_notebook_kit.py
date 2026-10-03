"""Tests for tools/notebook_kit.py, the helper every notebook loads.

The kit copies contracts instead of importing them (it has to run on Kaggle
with no repository on the path), so the first job here is to pin each copy to
its source. The second is the round trip: whatever the kit writes must load
through the real loaders. The rest are adversarial: each asserts that a
specific mistake a notebook could make is refused.
"""

import ast
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIR))

import attributions
import metrics_core
import notebook_kit as nk
import probs_artifact
import repo_paths
from validate_results_folder import validate_folder

TECHNIQUE = "01_LOG-REG_TF-IDF"


def base_config(**overrides):
    config = {
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
            "max_evals": 64,
            "batch_size": 16,
            "seed": 30,
            "n_posts_per_split": None,
            "per_member_runs": False,
        },
        "error_analysis": {"include_text_preview": False},
    }
    config.update(overrides)
    return config


def synthetic_foundation(directory, dataset_version=nk.DATASET_VERSION):
    """Write a foundation folder with the frozen split's exact shape and no real text."""
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    index = 0
    for split, (_, negatives, positives) in nk.EXPECTED_SPLIT_COUNTS.items():
        for label, count in ((0, negatives), (1, positives)):
            for _ in range(count):
                text = f"synthetic post number {index} " + ("bad bad" if label else "good")
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
    with open(directory / "dataset_manifest.json", "w") as handle:
        json.dump(
            {
                "dataset_version": dataset_version,
                "split_version": nk.SPLIT_VERSION,
                "processed_rows": len(frame),
            },
            handle,
        )
    return frame


@pytest.fixture
def input_root(tmp_path):
    root = tmp_path / "input"
    synthetic_foundation(root / "research_foundation")
    return root


@pytest.fixture
def ctx(tmp_path, input_root):
    return nk.bootstrap(base_config(), working_root=tmp_path / "working", input_roots=[input_root])


@pytest.fixture
def frames(ctx):
    frame, _ = nk.load_foundation(ctx)
    return nk.split_frames(frame)


def separable_probabilities(frame, seed=30):
    rng = np.random.default_rng(seed)
    logits = frame["label"].to_numpy() * 2.0 - 1.0 + rng.normal(0.0, 0.6, len(frame))
    return 1.0 / (1.0 + np.exp(-3.0 * logits))


def source_of(module_path, name):
    text = Path(module_path).read_text()
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(text, node)
    raise AssertionError(f"{name} not found in {module_path}")


class TestPinnedToTheToolkit:
    @pytest.mark.parametrize("name", ["compute_binary_metrics", "safe_roc_auc", "safe_pr_auc"])
    def test_metric_functions_are_character_identical(self, name):
        kit = source_of(TOOLS_DIR / "notebook_kit.py", name)
        core = source_of(TOOLS_DIR / "metrics_core.py", name)
        assert kit == core, f"{name} in notebook_kit.py has drifted from metrics_core.py"

    def test_protocol_constants_match_repo_paths(self):
        for name in (
            "RANDOM_SEED",
            "DATASET_VERSION",
            "SPLIT_VERSION",
            "POSITIVE_LABEL",
            "POSITIVE_CLASS_NAME",
            "NEGATIVE_CLASS_NAME",
            "PROCESSED_ROW_COUNT",
            "EXPECTED_SPLIT_COUNTS",
            "SPLIT_NAMES",
            "ATTRIBUTIONS_SUBDIR",
            "COEFFICIENTS_FILENAME",
        ):
            assert getattr(nk, name) == getattr(repo_paths, name), name

    def test_contract_tuples_match_their_sources(self):
        assert nk.REQUIRED_METRIC_FIELDS == metrics_core.REQUIRED_METRIC_FIELDS
        assert nk.PROBS_REQUIRED_COLUMNS == probs_artifact.REQUIRED_COLUMNS
        assert set(probs_artifact.FORBIDDEN_COLUMNS) <= set(nk.PROBS_FORBIDDEN_COLUMNS)
        assert nk.ATTR_REQUIRED_COLUMNS == attributions.REQUIRED_COLUMNS
        assert nk.ATTR_FORBIDDEN_COLUMNS == attributions.FORBIDDEN_COLUMNS
        assert nk.ATTR_REQUIRED_META_FIELDS == attributions.REQUIRED_META_FIELDS
        assert nk.MAX_WORD_LENGTH == attributions.MAX_WORD_LENGTH

    def test_external_kinds_are_manifest_kinds(self):
        import run_manifest

        for kind, _ in nk.EXTERNAL_KINDS.values():
            assert kind in run_manifest.ASSET_KINDS

    def test_kit_imports_nothing_from_the_repository(self):
        tree = ast.parse((TOOLS_DIR / "notebook_kit.py").read_text())
        local_modules = {p.stem for p in TOOLS_DIR.glob("*.py")} - {"notebook_kit"}
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not (imported & local_modules), imported & local_modules


class TestConfigIsRefused:
    def test_prefixless_technique_name(self):
        with pytest.raises(nk.ConfigError, match="filename stem"):
            nk.validate_config(base_config(technique_name="LOG-REG_TF-IDF"))

    def test_abbreviated_split_version(self):
        with pytest.raises(nk.ConfigError, match="full frozen string"):
            nk.validate_config(base_config(split_version="split_v1"))

    def test_wrong_dataset_version(self):
        with pytest.raises(nk.ConfigError, match="dataset_version"):
            nk.validate_config(base_config(dataset_version="extremism_dataset_v1"))

    def test_text_preview_switched_on(self):
        with pytest.raises(nk.ConfigError, match="include_text_preview"):
            nk.validate_config(base_config(error_analysis={"include_text_preview": True}))

    def test_text_preview_hidden_in_a_nested_block(self):
        config = base_config(explainability={"local": {"include_text_preview": True}})
        with pytest.raises(nk.ConfigError, match="include_text_preview"):
            nk.validate_config(config)

    def test_claim_in_technique_name(self):
        with pytest.raises(nk.ConfigError, match="claim"):
            nk.validate_config(base_config(technique_name="08_BEST-ROBERTA_SEED-ENSEMBLE"))

    def test_foundation_may_not_overwrite_the_split(self):
        config = {
            "project_name": "p",
            "technique_name": nk.FOUNDATION_TECHNIQUE,
            "role": "foundation",
            "dataset_version": nk.DATASET_VERSION,
            "split_version": nk.SPLIT_VERSION,
            "random_seed": 30,
            "overwrite_existing_split": True,
        }
        with pytest.raises(nk.ConfigError, match="overwrite_existing_split"):
            nk.validate_config(config)
        config["overwrite_existing_split"] = False
        assert nk.validate_config(config) == "foundation"


class TestFrozenSplit:
    def test_foundation_loads_with_frozen_counts(self, ctx):
        frame, info = nk.load_foundation(ctx)
        assert len(frame) == nk.PROCESSED_ROW_COUNT
        assert info["split_counts_total_negative_positive"] == nk.EXPECTED_SPLIT_COUNTS
        train_df, val_df, test_df = nk.split_frames(frame)
        assert (len(train_df), len(val_df), len(test_df)) == (2099, 450, 450)

    def test_moved_row_is_refused(self, ctx, input_root):
        path = input_root / "research_foundation" / "split_assignments.csv"
        assignments = pd.read_csv(path)
        index = assignments.index[assignments["split"] == "test"][0]
        assignments.loc[index, "split"] = "train"
        assignments.to_csv(path, index=False)
        with pytest.raises(nk.SplitIntegrityError, match="frozen protocol"):
            nk.load_foundation(ctx)

    def test_disagreeing_committed_mirror_is_refused(self, ctx, input_root):
        source = input_root / "research_foundation" / "split_assignments.csv"
        mirror = pd.read_csv(source)
        # Swap one train row with one validation row of the same label: the
        # counts stay frozen, so only the mirror comparison can catch it.
        train_row = mirror.index[(mirror["split"] == "train") & (mirror["label"] == 0)][0]
        val_row = mirror.index[(mirror["split"] == "validation") & (mirror["label"] == 0)][0]
        mirror.loc[train_row, "split"], mirror.loc[val_row, "split"] = "validation", "train"
        (input_root / "repo" / "splits").mkdir(parents=True)
        mirror.to_csv(input_root / "repo" / "splits" / "split_assignments.csv", index=False)
        with pytest.raises(nk.SplitIntegrityError, match="attached mirror"):
            nk.load_foundation(ctx)

    def test_matching_committed_mirror_is_counted(self, ctx, input_root):
        source = input_root / "research_foundation" / "split_assignments.csv"
        (input_root / "repo" / "splits").mkdir(parents=True)
        (input_root / "repo" / "splits" / "split_assignments.csv").write_bytes(source.read_bytes())
        _, info = nk.load_foundation(ctx)
        assert info["committed_mirrors_compared"] == 1

    def test_stale_foundation_manifest_is_refused(self, tmp_path):
        root = tmp_path / "stale"
        synthetic_foundation(root / "research_foundation", dataset_version="extremism_dataset_v1")
        stale = nk.bootstrap(base_config(), working_root=tmp_path / "w", input_roots=[root])
        with pytest.raises(nk.SplitIntegrityError, match="notebook 00"):
            nk.load_foundation(stale)

    def test_missing_input_names_the_file(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        bare = nk.bootstrap(base_config(), working_root=tmp_path / "w", input_roots=[empty])
        with pytest.raises(nk.InputNotFoundError, match="processed_dataset.csv"):
            nk.load_foundation(bare)


class TestValidationAndTestDiscipline:
    def test_threshold_cannot_be_selected_on_test(self, frames):
        _, _, test_df = frames
        with pytest.raises(nk.TestSplitError, match="only 'validation'"):
            nk.select_threshold(test_df, separable_probabilities(test_df))

    def test_threshold_selected_on_validation(self, frames):
        _, val_df, _ = frames
        threshold, sweep = nk.select_threshold(val_df, separable_probabilities(val_df))
        assert threshold in nk.THRESHOLD_GRID
        assert len(sweep) == len(nk.THRESHOLD_GRID)
        assert int(sweep["selected"].sum()) == 1
        assert sweep["threshold"].is_monotonic_increasing

    def test_second_test_evaluation_is_refused(self, ctx, frames):
        _, _, test_df = frames
        prob = separable_probabilities(test_df)
        nk.evaluate(ctx, test_df, prob, 0.5, "test")
        with pytest.raises(nk.TestSplitError, match="already evaluated"):
            nk.evaluate(ctx, test_df, prob, 0.5, "test")

    def test_partial_split_is_refused(self, ctx, frames):
        _, val_df, _ = frames
        part = val_df.iloc[:100]
        with pytest.raises(nk.SplitIntegrityError, match="whole split"):
            nk.evaluate(ctx, part, separable_probabilities(part), 0.5, "validation")

    def test_split_label_must_match_the_rows(self, ctx, frames):
        _, val_df, _ = frames
        with pytest.raises(nk.KitError, match="was given rows from"):
            nk.evaluate(ctx, val_df, separable_probabilities(val_df), 0.5, "test")

    def test_ablation_rows_come_from_validation_only(self, ctx, frames):
        _, val_df, test_df = frames
        val_metrics = nk.evaluate(ctx, val_df, separable_probabilities(val_df), 0.5, "validation")
        row = nk.ablation_row(ctx, "config_001", {"C": 1.0}, val_metrics)
        assert row["validation_accuracy"] == val_metrics["accuracy"]
        test_metrics = nk.evaluate(ctx, test_df, separable_probabilities(test_df), 0.5, "test")
        with pytest.raises(nk.TestSplitError):
            nk.ablation_row(ctx, "config_001", {"C": 1.0}, test_metrics)


class TestProbabilityArtifact:
    META = {
        "model_family": "logistic_regression",
        "feature_family": "tfidf_word",
        "hyperparameters": {"C": 1.0},
        "selected_threshold": 0.5,
        "threshold_strategy": nk.THRESHOLD_STRATEGY,
        "threshold_metric": "accuracy",
    }

    def test_round_trip_through_the_real_loader(self, ctx, frames):
        _, val_df, test_df = frames
        paths = nk.export_probabilities(
            ctx,
            val_df,
            separable_probabilities(val_df),
            test_df,
            separable_probabilities(test_df),
            self.META,
        )
        for split in ("validation", "test"):
            loaded = probs_artifact.load_probs(paths[split], expected_split=split)
            assert list(loaded.columns) == list(probs_artifact.REQUIRED_COLUMNS)
        meta = json.loads(paths["meta"].read_text())
        assert meta["contains_text"] is False
        assert meta["test_set_used_for_selection"] is False
        assert meta["technique"] == TECHNIQUE

    def test_probability_outside_unit_interval_is_refused(self, ctx, frames):
        _, val_df, test_df = frames
        bad = separable_probabilities(val_df)
        bad[0] = 1.5
        with pytest.raises(nk.ArtifactContractError, match=r"outside \[0, 1\]"):
            nk.export_probabilities(
                ctx, val_df, bad, test_df, separable_probabilities(test_df), self.META
            )

    def test_swapped_splits_are_refused(self, ctx, frames):
        _, val_df, test_df = frames
        with pytest.raises(nk.ArtifactContractError, match="expected 'validation' rows"):
            nk.export_probabilities(
                ctx,
                test_df,
                separable_probabilities(test_df),
                val_df,
                separable_probabilities(val_df),
                self.META,
            )

    def test_incomplete_meta_is_refused(self, ctx, frames):
        _, val_df, test_df = frames
        with pytest.raises(nk.ArtifactContractError, match="missing keys"):
            nk.export_probabilities(
                ctx,
                val_df,
                separable_probabilities(val_df),
                test_df,
                separable_probabilities(test_df),
                {"model_family": "x"},
            )


def toy_predict(texts):
    """P(EXTREMIST) rises with 'bad' and falls with 'good': a known ground truth."""
    scores = []
    for text in texts:
        words = text.lower().split()
        scores.append(words.count("bad") - words.count("good"))
    return 1.0 / (1.0 + np.exp(-np.asarray(scores, dtype=float)))


class TestAttributionArtifact:
    def test_aggregation_sums_within_a_post_and_averages_over_containing_posts(self):
        tokens = [["Bad ", "bad", "good."], ["bad"], ["neutral "]]
        values = [[0.2, 0.1, -0.3], [-0.1], [0.05]]
        frame = nk.aggregate_word_attributions(tokens, values).set_index("word")
        # "bad": per-post totals 0.3 and -0.1 over the two posts containing it.
        assert frame.loc["bad", "support"] == 2
        assert frame.loc["bad", "mean_attribution"] == pytest.approx(0.1)
        assert frame.loc["bad", "mean_abs_attribution"] == pytest.approx(0.2)
        assert frame.loc["good", "support"] == 1
        assert frame.loc["good", "mean_attribution"] == pytest.approx(-0.3)
        # The third post does not dilute either mean.
        assert frame.loc["neutral", "support"] == 1

    def test_punctuation_only_and_overlong_tokens_are_dropped_and_counted(self):
        tokens = [["!!! ", "word", "x" * 60]]
        values = [[0.5, 0.1, 0.2]]
        frame, stats = nk.aggregate_word_attributions_with_stats(tokens, values)
        assert list(frame["word"]) == ["word"]
        assert stats == {"tokens_dropped_empty": 1, "tokens_dropped_long": 1}

    def bundle(self, split="test"):
        return nk.AttributionBundle(
            row_ids=["ex_000001", "ex_000002"],
            split=split,
            tokens_per_post=[["bad ", "day"], ["good ", "day"]],
            values_per_post=[np.array([0.3, 0.01]), np.array([-0.2, 0.02])],
            seed=30,
            max_evals=64,
            n_posts=2,
        )

    def test_round_trip_through_the_real_loader(self, ctx):
        nk.export_attribution_run(ctx, self.bundle(), "shap-partition_test_seed30")
        frame, meta = attributions.load_run(
            TECHNIQUE, "shap-partition_test_seed30", results_dir=ctx.results_dir.parent
        )
        assert list(frame.columns) == list(attributions.REQUIRED_COLUMNS)
        assert meta["explainer"] == nk.EXPLAINER_ID
        assert meta["member"] is None
        assert attributions.list_runs(TECHNIQUE, ctx.results_dir.parent) == [
            "shap-partition_test_seed30"
        ]

    def test_member_run_records_its_member(self, ctx):
        run_id = "shap-partition_test_seed30_member-A_anchor"
        nk.export_attribution_run(ctx, self.bundle(), run_id, member="A_anchor")
        _, meta = attributions.load_run(TECHNIQUE, run_id, results_dir=ctx.results_dir.parent)
        assert meta["member"] == "A_anchor"

    def test_run_id_with_double_underscore_is_refused(self, ctx):
        """attributions.list_runs skips such stems, so the run would vanish from every RQ tool."""
        with pytest.raises(nk.ArtifactContractError, match="__"):
            nk.export_attribution_run(ctx, self.bundle(), "shap__test")

    def test_per_post_file_goes_to_external_not_the_result_folder(self, ctx):
        nk.export_attribution_run(ctx, self.bundle(), "shap-partition_test_seed30")
        local = ctx.external_dir / "attributions_local" / "shap-partition_test_seed30.csv"
        assert local.exists()
        assert "row_id" in pd.read_csv(local).columns
        committed = pd.read_csv(
            ctx.results_dir / "attributions" / "shap-partition_test_seed30.csv"
        )
        assert "row_id" not in committed.columns

    def test_per_post_frame_is_refused_by_the_validator(self):
        frame = pd.DataFrame(
            {
                "row_id": ["ex_000001"],
                "word": ["bad"],
                "mean_abs_attribution": [0.1],
                "mean_attribution": [0.1],
                "support": [1],
            }
        )
        with pytest.raises(nk.ArtifactContractError, match="per-post"):
            nk.validate_attribution_frame(frame)

    def test_bigram_coefficient_is_refused(self, ctx):
        with pytest.raises(nk.ArtifactContractError, match="single non-empty words"):
            nk.export_coefficients(ctx, ["white power"], [1.0])

    def test_coefficients_are_written_where_validate_shap_reads_them(self, ctx):
        path = nk.export_coefficients(ctx, ["bad", "good"], [1.5, -1.0])
        assert path == ctx.results_dir / "attributions" / "coefficients.csv"
        assert list(pd.read_csv(path).columns) == ["word", "coefficient"]


class TestExplain:
    def frame(self):
        texts = [
            "this is a bad bad day",
            "what a good morning",
            "bad",
            "good news and bad news",
            "nothing to see here",
            "!!!",
        ]
        return pd.DataFrame(
            {
                "row_id": [f"ex_{i:06d}" for i in range(len(texts))],
                "text": texts,
                "label": [1, 0, 1, 0, 0, 0],
                "split": "test",
            }
        )

    def test_partition_explainer_recovers_the_known_signs(self, ctx):
        pytest.importorskip("shap")
        bundle = nk.explain(ctx, toy_predict, self.frame())
        assert bundle.n_posts == 6
        assert bundle.split == "test"
        table = nk.aggregate_word_attributions(
            bundle.tokens_per_post, bundle.values_per_post
        ).set_index("word")
        assert table.loc["bad", "mean_attribution"] > 0
        assert table.loc["good", "mean_attribution"] < 0
        assert abs(table.loc["bad", "mean_attribution"]) > abs(
            table.loc["nothing", "mean_attribution"]
        )
        # The one-word post is attributed, the punctuation-only post is not.
        assert table.loc["bad", "support"] == 3
        assert bundle.tokens_per_post[5] == []

    def test_predictor_returning_two_columns_is_refused(self, ctx):
        def two_columns(texts):
            p = toy_predict(texts)
            return np.column_stack([1 - p, p])

        with pytest.raises(nk.ArtifactContractError, match="one finite probability"):
            nk.explain(ctx, two_columns, self.frame())

    def test_predictor_returning_logits_is_refused(self, ctx):
        with pytest.raises(nk.ArtifactContractError, match=r"in \[0, 1\]"):
            nk.explain(ctx, lambda texts: np.full(len(texts), 3.0), self.frame())

    def test_sample_is_stratified_and_reproducible(self, frames):
        _, _, test_df = frames
        first = nk.sample_for_explanation(test_df, 200, seed=30)
        second = nk.sample_for_explanation(test_df, 200, seed=30)
        assert len(first) == 200
        assert list(first["row_id"]) == list(second["row_id"])
        assert int(first["label"].sum()) == round(200 * 170 / 450)
        assert len(nk.sample_for_explanation(test_df, None, seed=30)) == 450


class TestResultsFolder:
    def build(self, ctx, frames, threshold=None):
        _, val_df, test_df = frames
        val_prob = separable_probabilities(val_df)
        test_prob = separable_probabilities(test_df, seed=31)
        selected, sweep = nk.select_threshold(val_df, val_prob)
        threshold = selected if threshold is None else threshold
        val_metrics = nk.evaluate(ctx, val_df, val_prob, selected, "validation")
        test_metrics = nk.evaluate(ctx, test_df, test_prob, selected, "test")
        ablation = pd.DataFrame([nk.ablation_row(ctx, "config_001", {"C": 1.0}, val_metrics)])
        best_config = {
            "model_family": "logistic_regression",
            "feature_family": "tfidf_word",
            "hyperparameters": {"C": 1.0},
            "threshold_strategy": nk.THRESHOLD_STRATEGY,
            "threshold_metric": "accuracy",
            "selected_threshold": threshold,
        }
        return best_config, val_metrics, test_metrics, ablation, sweep

    def test_folder_passes_the_real_validator(self, ctx, frames):
        folder = nk.export_results_folder(ctx, *self.build(ctx, frames))
        nk.export_attribution_run(
            ctx, TestAttributionArtifact().bundle(), "shap-partition_test_seed30"
        )
        assert validate_folder(folder) == []
        best_config = json.loads((folder / "best_config.json").read_text())
        assert best_config["provenance"] == "derived_from_probs"
        assert best_config["recomputable"] is True
        assert best_config["threshold_selected_on_split"] == "validation"
        confusion = pd.read_csv(folder / "confusion_matrix_test.csv")
        assert list(confusion.columns) == ["actual", "predicted", "count"]
        assert int(confusion["count"].sum()) == 450

    def test_threshold_that_was_not_evaluated_is_refused(self, ctx, frames):
        args = self.build(ctx, frames, threshold=0.123)
        with pytest.raises(nk.ArtifactContractError, match="locked threshold"):
            nk.export_results_folder(ctx, *args)

    def test_ablation_with_a_test_column_is_refused(self, ctx, frames):
        best_config, val_metrics, test_metrics, ablation, sweep = self.build(ctx, frames)
        ablation["test_accuracy"] = 0.9
        with pytest.raises(nk.ArtifactContractError, match="validation only"):
            nk.export_results_folder(ctx, best_config, val_metrics, test_metrics, ablation, sweep)

    def test_error_counts_carry_no_identifiers(self, frames):
        _, _, test_df = frames
        table = nk.error_counts(test_df, separable_probabilities(test_df), 0.5)
        assert list(table.columns) == ["outcome", "confidence", "count"]
        assert int(table["count"].sum()) == 450


class TestFinalize:
    def test_inventory_flags_text_and_zip_mirrors_the_repository(self, ctx, frames):
        nk.export_results_folder(ctx, *TestResultsFolder().build(ctx, frames))
        (ctx.external("weights") / "model.bin").write_bytes(b"weights")
        (ctx.external("predictions") / "predictions_test.csv").write_text("row_id,text\n")
        zip_path = nk.finalize(ctx)

        inventory = json.loads((ctx.external_dir / nk.EXTERNAL_ASSETS_FILENAME).read_text())
        by_path = {a["relative_path"]: a for a in inventory["assets"]}
        assert by_path["weights/model.bin"]["contains_text"] is False
        assert by_path["weights/model.bin"]["kind"] == "weights"
        assert by_path["predictions/predictions_test.csv"]["contains_text"] is True
        assert len(by_path["weights/model.bin"]["sha256"]) == 64

        with zipfile.ZipFile(zip_path) as archive:
            names = archive.namelist()
        assert f"results_summary/{TECHNIQUE}/metrics_test.json" in names
        assert not any(name.startswith("external/") for name in names)

    def test_text_column_in_the_result_folder_is_refused(self, ctx):
        (ctx.results_dir / "leak.csv").write_text("row_id,text\nex_000001,something\n")
        with pytest.raises(nk.ArtifactContractError, match="text-bearing"):
            nk.finalize(ctx)
