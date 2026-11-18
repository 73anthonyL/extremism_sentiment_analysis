"""Tests for tools/probe_kit.py, the probe-role helpers notebook 13 loads next to the kit.

The first job is to pin its copies: the lexicon matching to RQ4's tool on
every dataset post, McNemar and Holm character for character to
tools/mcnemar_pairs.py. The rest are adversarial: each asserts that a
specific mistake the probe notebook could make is refused, above all touching
the test split before it is opened and writing a table that breaks its
contract. `build_probe_tables` makes the same calls notebook 13 makes, on
synthetic scores, and is reused by test_layer_probes_report.py.
"""

import ast
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIR))

import notebook_kit as nk  # noqa: E402
import probe_kit as pk  # noqa: E402
import repo_paths  # noqa: E402
from test_notebook_kit import (  # noqa: E402
    input_root,  # noqa: F401  (fixture)
    ctx,  # noqa: F401  (fixture)
    separable_probabilities,
    source_of,
    synthetic_foundation,
)

PROBE_TECHNIQUE = "13_TRANSFORMER-LAYER-PROBES"
PROBE_SOURCE = "07_TWITTER-ROBERTA_FINE-TUNE"


def probe_config(**overrides):
    config = {
        "project_name": "extremism_text_classification_comparison",
        "technique_name": PROBE_TECHNIQUE,
        "role": "probe",
        "dataset_version": nk.DATASET_VERSION,
        "split_version": nk.SPLIT_VERSION,
        "random_seed": 30,
        "models": [{"model_id": "ft_a"}, {"model_id": "base_a"}],
        "probe": {"pooling": "mean", "c_grid": [0.01, 0.1], "cv_folds": 3, "cv_seed": 30, "max_iter": 200},
        "control_task": {"seed": 30},
        "bootstrap": {"n_resamples": 200, "seed": 30, "ci_level": 0.95},
        "identity_direction": {"enabled": True},
        "error_analysis": {"include_text_preview": False},
    }
    config.update(overrides)
    return config


@pytest.fixture
def probe_ctx(tmp_path, input_root):
    return nk.bootstrap(probe_config(), working_root=tmp_path / "probe_working", input_roots=[input_root])


@pytest.fixture
def probe_frames(probe_ctx):
    frame, _ = nk.load_foundation(probe_ctx)
    return nk.split_frames(frame)


def _noisy_logits(y, strength, rng):
    return (np.asarray(y) * 2.0 - 1.0) * strength + rng.normal(0.0, 1.0, len(y))


def _noisy_scores(y, strength, rng):
    return pk.logit_to_probability(_noisy_logits(y, strength, rng))


def build_probe_tables(ctx, frames, n_layers=2):
    """A complete, conforming set of WP9 tables from synthetic scores. Opens test."""
    train_df, val_df, test_df = frames
    pk.begin_test_evaluation(ctx)
    cutoff, _ = pk.control_word_type_cutoff(train_df, 30)
    targets = {}
    for frame in (val_df, test_df):
        targets[frame["split"].iloc[0]] = {
            "extremist_label": frame["label"].to_numpy(),
            "identity_term_present": pk.identity_term_present(ctx, frame, ["bad"]),
            "control_word_type": pk.control_word_type_labels(ctx, frame, 30, cutoff),
        }
    models = (
        ("ft_a", "fine_tuned", PROBE_SOURCE, "single", 30),
        ("base_a", "pretrained_base", "", "", float("nan")),
    )
    rng = np.random.default_rng(1)
    probes, scores = [], {}
    for model_id, kind, technique, component, seed in models:
        for layer in range(n_layers):
            for task in pk.PROBE_TASKS:
                for frame in (val_df, test_df):
                    split = frame["split"].iloc[0]
                    y = targets[split][task]
                    strength = (layer + 1) * (1.2 if kind == "fine_tuned" else 0.6)
                    s = _noisy_logits(y, strength, rng)
                    scores[(model_id, layer, task, split)] = (y, s)
                    metrics = pk.score_probe(ctx, frame, y, s)
                    metrics.pop("split")
                    probes.append(
                        {
                            "model_id": model_id,
                            "model_kind": kind,
                            "source_technique": technique,
                            "component": component,
                            "checkpoint": "org/checkpoint",
                            "seed": seed,
                            "text_normalization": "identity",
                            "n_layers": n_layers,
                            "hidden_size": 8,
                            "layer": layer,
                            "task": task,
                            "split": split,
                            "probe_c": 0.1,
                            "cv_auc_train": 0.8,
                            **metrics,
                        }
                    )
    probes = pd.DataFrame(probes)[list(pk.PROBE_COLUMNS)]

    contrasts = []
    for frame in (val_df, test_df):
        split = frame["split"].iloc[0]
        for layer in range(n_layers):
            for task in pk.PROBE_TASKS:
                y_a, s_a = scores[("ft_a", layer, task, split)]
                y_b, s_b = scores[("base_a", layer, task, split)]
                row = pk.paired_auc_difference(ctx, frame, y_a, s_a, y_b, s_b)
                row.pop("split")
                contrasts.append(
                    {"contrast": "fine_tuned_minus_base", "model_id": "ft_a", "reference_id": "base_a",
                     "layer": layer, "task": task, "split": split, **row}
                )
                if task == pk.CONTROL_TASK:
                    continue
                for model_id in ("ft_a", "base_a"):
                    y_t, s_t = scores[(model_id, layer, task, split)]
                    y_c, s_c = scores[(model_id, layer, pk.CONTROL_TASK, split)]
                    row = pk.paired_auc_difference(ctx, frame, y_t, s_t, y_c, s_c)
                    row.pop("split")
                    contrasts.append(
                        {"contrast": "selectivity", "model_id": model_id, "reference_id": model_id,
                         "layer": layer, "task": task, "split": split, **row}
                    )
    contrasts = pd.DataFrame(contrasts)[list(pk.CONTRAST_COLUMNS)]

    ablation = []
    for population, label in (("non_extremist", 0), ("extremist", 1)):
        subset = test_df[test_df["label"] == label].reset_index(drop=True)
        before = _noisy_scores(subset["label"].to_numpy(), 1.0, rng)
        for direction, shift in (("identity_probe", 0.08), ("random", 0.002)):
            after = np.clip(before - shift * rng.uniform(0.5, 1.5, len(before)), 0, 1)
            ablation.append(
                pk.direction_ablation_row(
                    ctx, subset, before, after, 0.5, "locked",
                    {"model_id": "ft_a", "source_technique": PROBE_SOURCE, "seed": 30,
                     "population": population, "direction": direction},
                    random_delta_p95=0.003, n_random_directions=10,
                )
            )
    ablation = pd.DataFrame(ablation)[list(pk.ABLATION_COLUMNS)]
    family = (ablation["direction"] == "identity_probe") & (ablation["threshold_kind"] == "locked")
    ablation.loc[family, "wilcoxon_p_holm"] = pk.holm_adjust(ablation.loc[family, "wilcoxon_p"])
    ablation.loc[family, "mcnemar_p_holm"] = pk.holm_adjust(ablation.loc[family, "mcnemar_p"])
    return probes, contrasts, ablation


class TestProbeRole:
    def test_probe_config_needs_its_blocks(self):
        config = probe_config()
        del config["bootstrap"]
        with pytest.raises(nk.ConfigError, match="bootstrap"):
            pk.validate_probe_config(config)

    def test_pooling_other_than_the_preregistered_one_is_refused(self):
        config = probe_config(
            probe={"pooling": "cls", "c_grid": [1.0], "cv_folds": 5, "cv_seed": 30, "max_iter": 10}
        )
        with pytest.raises(nk.ConfigError, match="preregistered"):
            pk.validate_probe_config(config)

    def test_text_preview_is_refused_in_the_probe_role_too(self):
        with pytest.raises(nk.ConfigError, match="include_text_preview"):
            nk.validate_config(probe_config(error_analysis={"include_text_preview": True}))

    def test_probe_run_writes_to_rq_and_has_no_probability_folder(self, probe_ctx):
        assert probe_ctx.results_dir.parts[-2:] == ("results_summary", "rq")
        assert not probe_ctx.probs_dir.exists()

    def test_probe_kit_imports_only_the_kit_from_the_repository(self):
        tree = ast.parse((TOOLS_DIR / "probe_kit.py").read_text())
        local_modules = {p.stem for p in TOOLS_DIR.glob("*.py")} - {"probe_kit"}
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert imported & local_modules == {"notebook_kit"}

    def test_notebook_13_serves_inputs_as_notebook_11_and_the_loaders_do(self):
        """Notebook 13 reproduces 11's text normalisation; pin it to model_loaders.py."""
        import re

        import model_loaders as ml

        path = TOOLS_DIR.parent / "notebooks" / f"{PROBE_TECHNIQUE}.ipynb"
        cells = json.loads(path.read_text())["cells"]
        source = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
        for name, compiled in (
            ("URL_RE", ml.ENSEMBLE_URL_RE),
            ("USER_RE", ml.ENSEMBLE_USER_RE),
            ("WHITESPACE_RE", ml.ENSEMBLE_WHITESPACE_RE),
        ):
            match = re.search(rf'{name} = re\.compile\(r"([^"]+)"\)', source)
            assert match and match.group(1) == compiled.pattern, name
        for needle in (
            'html.unescape',
            'unicodedata.normalize("NFKC"',
            'USER_RE.sub("@user"',
            'URL_RE.sub("http"',
            "truncation=True, padding=True, max_length=int(spec[\"max_length\"])",
            "torch.softmax(output.logits.float(), dim=-1)[:, 1]",
            "torch.softmax(logits.float(), dim=-1)[:, 1]",
        ):
            assert needle in source, needle

    def test_lexicon_copy_reads_the_identity_list_like_the_tool(self):
        import lexicons

        path = repo_paths.LEXICON_FILES["identity_term"]
        assert pk.read_lexicon_entries(path) == lexicons.read_lexicon(path)

    def test_identity_matching_equals_rq4_on_every_dataset_post(self):
        import identity_fpr
        import lexicons
        from split_protocol import build_processed_dataset

        texts = build_processed_dataset()["text"].tolist()
        entries = lexicons.read_lexicon(repo_paths.LEXICON_FILES["identity_term"])
        expected = [bool(m) for m in identity_fpr.identity_matches(texts, entries)]
        pattern = pk.lexicon_pattern(entries)
        assert [bool(pattern.search(t)) for t in texts] == expected
        assert sum(expected) > 0

    @pytest.mark.parametrize("name", ["mcnemar_exact_p", "holm_adjust"])
    def test_test_statistics_are_character_identical_to_mcnemar_pairs(self, name):
        kit = source_of(TOOLS_DIR / "probe_kit.py", name)
        tool = source_of(TOOLS_DIR / "mcnemar_pairs.py", name)
        assert kit == tool, f"{name} in probe_kit.py has drifted from mcnemar_pairs.py"

    def test_test_rows_are_refused_before_the_split_is_opened(self, probe_ctx, probe_frames):
        train_df, _, test_df = probe_frames
        cutoff, _ = pk.control_word_type_cutoff(train_df, 30)
        y = test_df["label"].to_numpy()
        half = np.full(len(y), 0.5)
        attempts = (
            lambda: pk.identity_term_present(probe_ctx, test_df, ["bad"]),
            lambda: pk.control_word_type_labels(probe_ctx, test_df, 30, cutoff),
            lambda: pk.score_probe(probe_ctx, test_df, y, half),
            lambda: pk.committed_probabilities(probe_ctx, PROBE_SOURCE, test_df),
            lambda: pk.paired_auc_difference(probe_ctx, test_df, y, half, y, half),
        )
        for attempt in attempts:
            with pytest.raises(nk.TestSplitError, match="begin_test_evaluation"):
                attempt()

    def test_test_split_opens_once_and_only_for_probes(self, probe_ctx, ctx):
        pk.begin_test_evaluation(probe_ctx)
        with pytest.raises(nk.TestSplitError, match="already opened"):
            pk.begin_test_evaluation(probe_ctx)
        with pytest.raises(nk.KitError, match="probe role"):
            pk.begin_test_evaluation(ctx)

    def test_control_cutoff_is_fitted_on_train_with_the_label_base_rate(self, probe_frames):
        train_df, val_df, _ = probe_frames
        _, record = pk.control_word_type_cutoff(train_df, 30)
        assert 0 < record["train_control_positive_share"] <= record["train_label_positive_share"]
        with pytest.raises(nk.TestSplitError, match="train split only"):
            pk.control_word_type_cutoff(val_df, 30)

    def test_word_type_bits_depend_only_on_the_word_and_seed(self):
        assert pk.word_type_bit("muslim", 30) == pk.word_type_bit("muslim", 30)
        bits = [pk.word_type_bit(f"w{i}", 30) for i in range(400)]
        assert 0.4 < np.mean(bits) < 0.6
        assert bits != [pk.word_type_bit(f"w{i}", 31) for i in range(400)]

    def test_rank_auc_matches_sklearn_with_ties(self):
        from sklearn.metrics import roc_auc_score

        rng = np.random.default_rng(3)
        y = rng.integers(0, 2, size=(20, 60))
        s = np.round(rng.uniform(size=(20, 60)), 1)
        fast = pk.roc_auc_rows(y, s)
        for row in range(20):
            assert fast[row] == pytest.approx(roc_auc_score(y[row], s[row]))
        assert np.isnan(pk.roc_auc_rows(np.zeros((1, 5)), np.ones((1, 5)))[0])

    def test_intervals_are_reproducible_and_paired_identity_is_zero(self, probe_ctx, probe_frames):
        _, val_df, _ = probe_frames
        y = val_df["label"].to_numpy()
        s = _noisy_logits(y, 1.5, np.random.default_rng(2))
        first = pk.score_probe(probe_ctx, val_df, y, s)
        assert first == pk.score_probe(probe_ctx, val_df, y, s)
        assert first["roc_auc_low"] <= first["roc_auc"] <= first["roc_auc_high"]
        same = pk.paired_auc_difference(probe_ctx, val_df, y, s, y, s)
        assert same["difference"] == 0 and same["difference_low"] == 0 and same["difference_high"] == 0

    def test_saturated_probabilities_do_not_become_ties(self, probe_ctx, probe_frames):
        """Logits of 40 and 50 both give p == 1.0 in float64; ranking logits keeps them apart."""
        _, val_df, _ = probe_frames
        y = val_df["label"].to_numpy()
        logits = np.where(y == 1, 40.0 + np.arange(len(y)) * 0.01, -40.0 - np.arange(len(y)) * 0.01)
        logits[:5] = -logits[:5]  # a few wrong-way posts, also saturated
        metrics = pk.score_probe(probe_ctx, val_df, y, logits)
        from sklearn.metrics import roc_auc_score

        assert metrics["roc_auc"] == pytest.approx(roc_auc_score(y, logits))
        assert metrics["n_saturated"] == len(y)
        tied = roc_auc_score(y, pk.logit_to_probability(logits))
        assert tied < metrics["roc_auc"]  # ranking probabilities would have tied them

    def test_non_finite_logits_are_refused(self, probe_ctx, probe_frames):
        _, val_df, _ = probe_frames
        y = val_df["label"].to_numpy()
        bad = np.zeros(len(y))
        bad[0] = np.inf
        with pytest.raises(nk.ArtifactContractError, match="non-finite"):
            pk.score_probe(probe_ctx, val_df, y, bad)

    def test_restored_weights_must_reproduce_the_committed_artifact(self, tmp_path, probe_frames):
        _, val_df, _ = probe_frames
        root = tmp_path / "with_probs"
        synthetic_foundation(root / "research_foundation")
        probs = root / "research_loop" / "probs"
        probs.mkdir(parents=True)
        prob = separable_probabilities(val_df)
        pd.DataFrame(
            {"row_id": val_df["row_id"], "split": "validation", "y_true": val_df["label"], "y_prob": prob}
        ).to_csv(probs / f"{PROBE_SOURCE}__validation.csv", index=False)
        ctx = nk.bootstrap(probe_config(), working_root=tmp_path / "w", input_roots=[root])
        close = np.clip(prob + 0.001, 0, 1)
        assert pk.verify_reproduction(ctx, PROBE_SOURCE, val_df, close)["max_abs_diff"] < 0.01
        with pytest.raises(nk.ArtifactContractError, match="not the weights"):
            pk.verify_reproduction(ctx, PROBE_SOURCE, val_df, np.clip(prob + 0.05, 0, 1))

    def test_conforming_tables_are_written_to_rq_and_zipped(self, probe_ctx, probe_frames):
        probes, contrasts, ablation = build_probe_tables(probe_ctx, probe_frames)
        paths = pk.export_probe_tables(probe_ctx, probes, contrasts, ablation, {"models": []})
        assert sorted(p.name for p in paths.values()) == sorted(pk.PROBE_TABLE_FILES.values())
        zip_path = nk.finalize(probe_ctx)
        with zipfile.ZipFile(zip_path) as archive:
            names = archive.namelist()
        assert "results_summary/rq/rq3_layer_probes.csv" in names
        assert not any(n.startswith(("research_loop/", "external/")) for n in names)

    def test_export_before_test_is_opened_is_refused(self, probe_ctx):
        with pytest.raises(nk.TestSplitError, match="open test first"):
            pk.export_probe_tables(probe_ctx, [], [], None, {})

    def test_per_post_column_is_refused(self, probe_ctx, probe_frames):
        probes, contrasts, ablation = build_probe_tables(probe_ctx, probe_frames)
        probes.insert(0, "row_id", "ex_000001")
        with pytest.raises(nk.ArtifactContractError, match="per-post"):
            pk.export_probe_tables(probe_ctx, probes, contrasts, ablation, {})

    def test_missing_layer_is_refused(self, probe_ctx, probe_frames):
        probes, contrasts, ablation = build_probe_tables(probe_ctx, probe_frames)
        drop = (probes["model_id"] == "ft_a") & (probes["layer"] == 1) & (probes["task"] == "extremist_label")
        with pytest.raises(nk.ArtifactContractError, match="does not cover layers"):
            pk.export_probe_tables(probe_ctx, probes[~drop], contrasts, ablation, {})

    def test_partial_split_is_refused(self, probe_ctx, probe_frames):
        probes, contrasts, ablation = build_probe_tables(probe_ctx, probe_frames)
        probes.loc[probes["split"] == "test", "n_posts"] = 300
        with pytest.raises(nk.ArtifactContractError, match="all 450 posts"):
            pk.export_probe_tables(probe_ctx, probes, contrasts, ablation, {})

    def test_selectivity_of_the_control_against_itself_is_refused(self, probe_ctx, probe_frames):
        probes, contrasts, ablation = build_probe_tables(probe_ctx, probe_frames)
        bad = contrasts[contrasts["contrast"] == "selectivity"].head(1).copy()
        bad["task"] = pk.CONTROL_TASK
        with pytest.raises(nk.ArtifactContractError, match="selectivity rows"):
            pk.export_probe_tables(probe_ctx, probes, pd.concat([contrasts, bad]), ablation, {})

    def test_contrast_naming_an_unknown_model_is_refused(self, probe_ctx, probe_frames):
        probes, contrasts, ablation = build_probe_tables(probe_ctx, probe_frames)
        contrasts.loc[contrasts["contrast"] == "fine_tuned_minus_base", "reference_id"] = "base_b"
        with pytest.raises(nk.ArtifactContractError, match="does not have"):
            pk.export_probe_tables(probe_ctx, probes, contrasts, ablation, {})

    def test_ablation_whose_flips_do_not_add_up_is_refused(self, probe_ctx, probe_frames):
        probes, contrasts, ablation = build_probe_tables(probe_ctx, probe_frames)
        ablation.loc[0, "n_flip_pos_to_neg"] += 1
        with pytest.raises(nk.ArtifactContractError, match="flips disagree"):
            pk.export_probe_tables(probe_ctx, probes, contrasts, ablation, {})

    def test_enabled_ablation_cannot_be_left_out(self, probe_ctx, probe_frames):
        probes, contrasts, _ = build_probe_tables(probe_ctx, probe_frames)
        with pytest.raises(nk.ArtifactContractError, match="no ablation table"):
            pk.export_probe_tables(probe_ctx, probes, contrasts, None, {})

    def test_run_record_carrying_a_row_id_is_refused(self, probe_ctx, probe_frames):
        probes, contrasts, ablation = build_probe_tables(probe_ctx, probe_frames)
        with pytest.raises(nk.ArtifactContractError, match="row id"):
            pk.export_probe_tables(probe_ctx, probes, contrasts, ablation, {"worst_post": "ex_000123"})

    def test_hidden_state_caches_are_inventoried_as_text_bearing(self, probe_ctx):
        (probe_ctx.external("hidden_states") / "07.npz").write_bytes(b"cache")
        nk.finalize(probe_ctx)
        inventory = json.loads((probe_ctx.external_dir / nk.EXTERNAL_ASSETS_FILENAME).read_text())
        assert inventory["assets"][0]["contains_text"] is True
