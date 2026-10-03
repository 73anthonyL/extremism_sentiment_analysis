"""Adversarial tests for tools/check_notebook.py, the scanner's row-id class,
and the manifest's bulk asset import.

`make_notebook` builds a small notebook that follows the template exactly.
Each test then breaks one thing a real notebook in this repository has
actually done (or the direct analogue) and asserts the checker names it.
"""

import json
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIR))

import notebook_kit as nk
from check_notebook import (
    BOOTSTRAP_SNIPPET,
    ROLE_SECTIONS,
    check_all,
    check_notebook,
    undefined_name_problems,
    parse_cells,
)
from scan_text_leakage import count_row_ids, notebook_output_text

STEM = "01_LOG-REG_TF-IDF"

CONFIG_CELL = '''CONFIG = {
    "project_name": "extremism_text_classification_comparison",
    "technique_name": "%(stem)s",
    "role": "%(role)s",
    "in_comparison": True,
    "model_family": "logistic_regression",
    "feature_family": "tfidf_word",
    "dataset_version": "extremism_dataset_clean_v1",
    "split_version": "split_v1_stratified_70_15_15_seed30",
    "random_seed": 30,
    "threshold": {"metric": "accuracy"},
    "explainer": {
        "max_evals": 500,
        "batch_size": 64,
        "seed": 30,
        "n_posts_per_split": None,
        "per_member_runs": False,
    },
    "export_coefficients": False,
    "error_analysis": {"include_text_preview": False},
}
ctx = nk.bootstrap(CONFIG)
'''

MODEL_CODE = {
    "Bootstrap": [BOOTSTRAP_SNIPPET + "\nimport numpy as np\n"],
    "Configuration": [CONFIG_CELL],
    "Load foundation and assert the frozen split": [
        "frame, foundation = nk.load_foundation(ctx)\n"
        "train_df, val_df, test_df = nk.split_frames(frame)\n"
    ],
    "Features and model": [
        "def predict_proba_texts(texts):\n    return np.full(len(texts), 0.5)\n"
    ],
    "Ablation on validation": [
        'val_prob = predict_proba_texts(val_df["text"].tolist())\n'
        'screen = nk.evaluate(ctx, val_df, val_prob, 0.5, "validation")\n'
        'ablation_results = [nk.ablation_row(ctx, "config_001", {}, screen)]\n'
    ],
    "Final fit": ['fitted = "model fitted on train_df"\n'],
    "Threshold on validation": [
        'threshold, sweep = nk.select_threshold(val_df, val_prob, metric="accuracy")\n'
        'validation_metrics = nk.evaluate(ctx, val_df, val_prob, threshold, "validation")\n'
    ],
    "Single test evaluation": [
        'test_prob = predict_proba_texts(test_df["text"].tolist())\n'
        'test_metrics = nk.evaluate(ctx, test_df, test_prob, threshold, "test")\n'
    ],
    "Probability artifacts": [
        "nk.export_probabilities(ctx, val_df, val_prob, test_df, test_prob, meta={})\n"
    ],
    "Word-level SHAP attribution runs": [
        "bundle = nk.explain(ctx, predict_proba_texts, test_df)\n"
        'nk.export_attribution_run(ctx, bundle, "shap-partition_test_seed30")\n',
        "bundle = nk.explain(ctx, predict_proba_texts, val_df)\n"
        'nk.export_attribution_run(ctx, bundle, "shap-partition_validation_seed30")\n',
    ],
    "Result folder": [
        "nk.export_results_folder(ctx, {}, validation_metrics, test_metrics, ablation_results, sweep)\n"
    ],
    "External assets": ["display(nk.error_counts(test_df, test_prob, threshold))\n"],
    "Interpretation": [],
    "Package outputs": ["nk.finalize(ctx)\n"],
}

FOUNDATION_CONFIG = '''CONFIG = {
    "project_name": "extremism_text_classification_comparison",
    "technique_name": "00_create_dataset_and_splits",
    "role": "foundation",
    "dataset_version": "extremism_dataset_clean_v1",
    "split_version": "split_v1_stratified_70_15_15_seed30",
    "random_seed": 30,
    "overwrite_existing_split": False,
}
ctx = nk.bootstrap(CONFIG)
'''

FOUNDATION_CODE = {
    "Bootstrap": [BOOTSTRAP_SNIPPET + "\nimport pandas as pd\n"],
    "Configuration": [FOUNDATION_CONFIG],
    "Load raw dataset": ['raw_df = pd.read_csv(nk.find_input(ctx, "dataset.csv"))\n'],
    "Build processed dataset": ["processed = raw_df.copy()\n"],
    "Create and verify the split": [
        "split_assignments = processed\n"
        "nk.assert_split_counts(split_assignments)\n"
        "nk.compare_with_committed_split(ctx, split_assignments)\n"
    ],
    "Foundation summaries": ["n_rows = len(processed)\n"],
    "Package outputs": ["nk.finalize(ctx)\n"],
}


def code_cell(source, outputs=None, execution_count=None):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source,
        "outputs": outputs or [],
        "execution_count": execution_count,
    }


def markdown_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def make_notebook(directory, stem=STEM, role="classical", mutate=None, metadata=None):
    """Write a conforming notebook; `mutate(cells)` may then break it."""
    code = FOUNDATION_CODE if role == "foundation" else MODEL_CODE
    cells = [markdown_cell(f"# {stem}\n\nRole in the comparison and what this notebook writes.")]
    for number, title in enumerate(ROLE_SECTIONS[role], start=1):
        cells.append(markdown_cell(f"## {number}. {title}\n\nWhat this section does."))
        for source in code[title]:
            cells.append(code_cell(source % {"stem": stem, "role": role} if "%(stem)s" in source else source))
    if mutate is not None:
        cells = mutate(cells) or cells
    notebook = {
        "cells": cells,
        "metadata": metadata or {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = Path(directory) / f"{stem}.ipynb"
    path.write_text(json.dumps(notebook, indent=1))
    return path


def replace_source(old, new):
    """A mutation that rewrites one cell's source."""

    def mutate(cells):
        hits = [c for c in cells if old in c["source"]]
        assert len(hits) == 1, f"expected exactly one cell containing {old!r}"
        hits[0]["source"] = hits[0]["source"].replace(old, new)

    return mutate


def append_to(marker, extra):
    return replace_source(marker, marker + extra)


def problems_for(tmp_path, primed=True, **kwargs):
    return check_notebook(make_notebook(tmp_path, **kwargs), primed=primed)


class TestConformingNotebooks:
    def test_classical_template_passes_primed_and_post_run(self, tmp_path):
        path = make_notebook(tmp_path)
        assert check_notebook(path, primed=True) == []
        assert check_notebook(path, primed=False) == []

    def test_foundation_template_passes(self, tmp_path):
        path = make_notebook(tmp_path, stem=nk.FOUNDATION_TECHNIQUE, role="foundation")
        assert check_notebook(path, primed=True) == []

    def test_template_sections_cover_every_role(self):
        assert set(ROLE_SECTIONS) == set(nk.ROLES)


class TestIdentityIsRefused:
    def test_prefixless_technique_name(self, tmp_path):
        """01-08 and 10 all declared a technique_name without the numeric prefix."""
        problems = problems_for(
            tmp_path, mutate=replace_source(f'"technique_name": "{STEM}"', '"technique_name": "LOG-REG_TF-IDF"')
        )
        assert any("technique_name" in p and "filename stem" in p for p in problems)

    def test_abbreviated_split_version(self, tmp_path):
        problems = problems_for(
            tmp_path,
            mutate=replace_source('"split_v1_stratified_70_15_15_seed30"', '"split_v1"'),
        )
        assert any("split_version" in p for p in problems)

    def test_claim_in_filename(self, tmp_path):
        problems = problems_for(tmp_path, stem="08_BEST-ROBERTA_SEED-ENSEMBLE")
        assert any("claim" in p for p in problems)

    def test_hyphenated_number_prefix(self, tmp_path):
        """The 2026-09-14 commit saved notebook 01 as 01-LOG-REG_TF-IDF.ipynb."""
        problems = problems_for(tmp_path, stem="01-LOG-REG_TF-IDF")
        assert any("NN_NAME" in p for p in problems)

    def test_two_notebooks_sharing_a_technique_name(self, tmp_path):
        make_notebook(tmp_path)
        other = make_notebook(tmp_path, stem="02_LIN-SVM_TF-IDF")
        other.write_text(other.read_text().replace("02_LIN-SVM_TF-IDF", STEM))
        results = check_all(tmp_path, primed=True)
        assert any("is declared by" in p for p in results[None])


class TestStructureIsRefused:
    def test_swapped_sections(self, tmp_path):
        def swap(cells):
            for cell in cells:
                if cell["source"].startswith("## 7. Threshold on validation"):
                    cell["source"] = cell["source"].replace("## 7. Threshold on validation", "## 7. Single test evaluation")
                elif cell["source"].startswith("## 8. Single test evaluation"):
                    cell["source"] = cell["source"].replace("## 8. Single test evaluation", "## 8. Threshold on validation")

        problems = problems_for(tmp_path, mutate=swap)
        assert any("section headers must be exactly" in p for p in problems)

    def test_code_before_the_first_section(self, tmp_path):
        problems = problems_for(tmp_path, mutate=lambda cells: cells.insert(1, code_cell("x = 1\n")))
        assert any("before the first" in p for p in problems)

    def test_noncanonical_bootstrap_cell(self, tmp_path):
        problems = problems_for(
            tmp_path,
            mutate=replace_source('sys.modules["notebook_kit"] = nk\n', 'sys.modules["kit"] = nk\n'),
        )
        assert any("canonical kit-loading cell" in p for p in problems)

    def test_last_cell_must_package(self, tmp_path):
        problems = problems_for(tmp_path, mutate=lambda cells: cells.append(code_cell("extra = 1\n")))
        assert any("last cell" in p for p in problems)

    def test_claim_in_prose(self, tmp_path):
        problems = problems_for(
            tmp_path,
            mutate=lambda cells: cells[0].update(
                source=f"# {STEM}\n\nThis model outperforms the baselines."
            ),
        )
        assert any("outperforms" in p for p in problems)


class TestPipelineDisciplineIsRefused:
    def test_second_test_evaluation(self, tmp_path):
        problems = problems_for(
            tmp_path,
            mutate=append_to(
                'test_metrics = nk.evaluate(ctx, test_df, test_prob, threshold, "test")\n',
                'again = nk.evaluate(ctx, test_df, test_prob, 0.4, "test")\n',
            ),
        )
        assert any("evaluated exactly once" in p for p in problems)

    def test_threshold_selected_on_test_rows(self, tmp_path):
        problems = problems_for(
            tmp_path,
            mutate=replace_source(
                'nk.select_threshold(val_df, val_prob, metric="accuracy")',
                'nk.select_threshold(test_df, val_prob, metric="accuracy")',
            ),
        )
        assert any("selected on test rows" in p for p in problems)
        assert any("not touched before section 8" in p for p in problems)

    def test_test_data_in_the_ablation(self, tmp_path):
        problems = problems_for(
            tmp_path,
            mutate=append_to(
                'ablation_results = [nk.ablation_row(ctx, "config_001", {}, screen)]\n',
                'peek = predict_proba_texts(test_df["text"].tolist())\n',
            ),
        )
        assert any("Ablation on validation" in p and "test_df" in p for p in problems)

    def test_missing_probability_export(self, tmp_path):
        problems = problems_for(
            tmp_path,
            mutate=replace_source(
                "nk.export_probabilities(ctx, val_df, val_prob, test_df, test_prob, meta={})\n",
                "exported = None\n",
            ),
        )
        assert any("nk.export_probabilities must be called exactly once" in p for p in problems)

    def test_single_attribution_run(self, tmp_path):
        problems = problems_for(
            tmp_path,
            mutate=replace_source(
                'nk.export_attribution_run(ctx, bundle, "shap-partition_validation_seed30")\n',
                "skipped = bundle\n",
            ),
        )
        assert any("at least two attribution runs" in p for p in problems)

    def test_member_runs_declared_but_not_exported(self, tmp_path):
        problems = problems_for(
            tmp_path, mutate=replace_source('"per_member_runs": False', '"per_member_runs": True')
        )
        assert any("per_member_runs" in p for p in problems)

    def test_coefficients_declared_but_not_exported(self, tmp_path):
        problems = problems_for(
            tmp_path, mutate=replace_source('"export_coefficients": False', '"export_coefficients": True')
        )
        assert any("export_coefficients" in p for p in problems)

    def test_fast_path_that_skips_training(self, tmp_path):
        """Notebooks 01 and 03 gained a cell that loaded saved weights instead of training."""
        problems = problems_for(
            tmp_path,
            mutate=replace_source('fitted = "model fitted on train_df"\n', "fast_path = True\n"),
        )
        assert any("shortcut" in p for p in problems)

    def test_foundation_must_not_overwrite_the_split(self, tmp_path):
        problems = problems_for(
            tmp_path,
            stem=nk.FOUNDATION_TECHNIQUE,
            role="foundation",
            mutate=replace_source('"overwrite_existing_split": False', '"overwrite_existing_split": True'),
        )
        assert any("overwrite_existing_split" in p for p in problems)


class TestTextExposureIsRefused:
    def test_head_of_a_frame(self, tmp_path):
        """Every notebook showed five posts with display(df.head())."""
        problems = problems_for(
            tmp_path,
            mutate=append_to(
                "train_df, val_df, test_df = nk.split_frames(frame)\n", "display(frame.head())\n"
            ),
        )
        assert any(".head(" in p for p in problems)
        assert any("display(...) references ['frame']" in p for p in problems)

    def test_printing_a_text_column(self, tmp_path):
        problems = problems_for(
            tmp_path,
            mutate=append_to(
                "train_df, val_df, test_df = nk.split_frames(frame)\n",
                'print(train_df["text"].iloc[0])\n',
            ),
        )
        assert any('["text"]' in p for p in problems)

    def test_row_ids_in_an_error_message(self, tmp_path):
        """Notebook 11 raised ValueError(f'... Examples: {missing_ids}')."""
        problems = problems_for(
            tmp_path,
            mutate=append_to(
                "train_df, val_df, test_df = nk.split_frames(frame)\n",
                'missing_ids = []\nif missing_ids:\n    raise ValueError(f"Examples: {missing_ids}")\n',
            ),
        )
        assert any("error message references ['missing_ids']" in p for p in problems)

    def test_counting_text_is_allowed(self, tmp_path):
        problems = problems_for(
            tmp_path,
            mutate=append_to(
                "train_df, val_df, test_df = nk.split_frames(frame)\n",
                'texts = train_df["text"].tolist()\nprint(len(texts))\n',
            ),
        )
        assert problems == []

    def test_text_preview_switched_on(self, tmp_path):
        problems = problems_for(
            tmp_path,
            mutate=replace_source('"include_text_preview": False', '"include_text_preview": True'),
        )
        assert any("include_text_preview" in p for p in problems)

    def test_per_post_shap_plot(self, tmp_path):
        problems = problems_for(
            tmp_path,
            mutate=append_to(
                'nk.export_attribution_run(ctx, bundle, "shap-partition_test_seed30")\n',
                "import shap\nshap.plots.waterfall(bundle)\n",
            ),
        )
        assert any("per-post SHAP plots" in p for p in problems)


class TestSoundnessIsRefused:
    def test_syntax_error(self, tmp_path):
        problems = problems_for(
            tmp_path, mutate=replace_source('fitted = "model fitted on train_df"\n', "fitted = (\n")
        )
        assert any("does not parse" in p for p in problems)

    def test_reference_to_a_removed_helper(self, tmp_path):
        """Deleting an inlined helper but leaving a call to it is the likeliest priming mistake."""
        problems = problems_for(
            tmp_path,
            mutate=append_to(
                'fitted = "model fitted on train_df"\n', "save_json(fitted, OUTPUT_DIR)\n"
            ),
        )
        assert any("name 'save_json' is used but never defined" in p for p in problems)
        assert any("name 'OUTPUT_DIR' is used but never defined" in p for p in problems)

    def test_name_used_before_the_cell_that_defines_it(self, tmp_path):
        problems = problems_for(
            tmp_path,
            mutate=append_to(
                "def predict_proba_texts(texts):\n    return np.full(len(texts), 0.5)\n",
                "early = threshold\n",
            ),
        )
        assert any("name 'threshold' is used but never defined" in p for p in problems)

    def test_function_may_use_a_global_defined_later(self):
        cells = [
            code_cell("def score():\n    return weight * 2\n"),
            code_cell("weight = 3\nresult = score()\n"),
        ]
        trees, _ = parse_cells(cells)
        assert undefined_name_problems(trees) == []

    def test_scopes_are_respected(self):
        cells = [
            code_cell(
                "import os\n"
                "def outer(a, *args, key=None, **kwargs):\n"
                "    local = [item for item in args]\n"
                "    def inner():\n"
                "        return a + len(local) + os.sep.count('/')\n"
                "    try:\n"
                "        pass\n"
                "    except ValueError as error:\n"
                "        return str(error)\n"
                "    return inner, (lambda z: z + a)\n"
                "class Model:\n"
                "    scale = 2\n"
                "    def run(self):\n"
                "        return outer(self.scale)\n"
            )
        ]
        trees, _ = parse_cells(cells)
        assert undefined_name_problems(trees) == []


class TestOutputsAreRefused:
    def test_primed_mode_refuses_saved_outputs(self, tmp_path):
        def add_output(cells):
            target = next(c for c in cells if "nk.finalize" in c["source"])
            target["outputs"] = [{"output_type": "stream", "name": "stdout", "text": ["done\n"]}]
            target["execution_count"] = 14

        path = make_notebook(tmp_path, mutate=add_output)
        assert any("still carry outputs" in p for p in check_notebook(path, primed=True))
        assert check_notebook(path, primed=False) == []

    def test_primed_mode_refuses_kaggle_run_metadata(self, tmp_path):
        path = make_notebook(tmp_path, metadata={"papermill": {}, "widgets": {}})
        assert any("metadata still carries" in p for p in check_notebook(path, primed=True))

    def test_row_id_in_a_saved_output(self, tmp_path):
        def add_output(cells):
            target = next(c for c in cells if "nk.split_frames" in c["source"])
            target["outputs"] = [
                {"output_type": "stream", "name": "stdout", "text": ["missing: ex_000123\n"]}
            ]

        path = make_notebook(tmp_path, mutate=add_output)
        assert any("row id" in p for p in check_notebook(path, primed=False))

    def test_error_traceback_in_a_saved_output(self, tmp_path):
        def add_error(cells):
            target = next(c for c in cells if "nk.finalize" in c["source"])
            target["outputs"] = [
                {"output_type": "error", "ename": "KeyError", "evalue": "x", "traceback": []}
            ]

        path = make_notebook(tmp_path, mutate=add_error)
        assert any("error traceback" in p for p in check_notebook(path, primed=False))


class TestScannerRowIds:
    def notebook(self, source, output_text):
        return json.dumps(
            {
                "cells": [
                    code_cell(
                        source,
                        outputs=[{"output_type": "stream", "name": "stdout", "text": [output_text]}],
                    )
                ]
            }
        )

    def test_row_id_in_notebook_output_is_reported(self):
        raw = self.notebook("print(queue)\n", "ex_000017 ex_000017 ex_002998\n")
        assert count_row_ids(notebook_output_text(raw)) == 2

    def test_row_id_in_notebook_source_is_not_an_output(self):
        raw = self.notebook('row_id = "ex_000017"  # example id in a comment\n', "450 rows\n")
        assert count_row_ids(notebook_output_text(raw)) == 0

    def test_row_id_in_displayed_table_is_reported(self):
        raw = json.dumps(
            {
                "cells": [
                    code_cell(
                        "display(table)\n",
                        outputs=[
                            {
                                "output_type": "display_data",
                                "data": {"text/html": ["<td>ex_000001</td>"], "text/plain": ["ex_000001"]},
                            }
                        ],
                    )
                ]
            }
        )
        assert count_row_ids(notebook_output_text(raw)) == 1


class TestManifestBulkImport:
    @pytest.fixture
    def technique(self, results_dir):
        from run_manifest import init_manifest

        folder = results_dir / "07_T"
        folder.mkdir()
        (folder / "metrics_test.json").write_text('{"accuracy": 0.88}')
        init_manifest("07_T", "07_T_run", "https://www.kaggle.com/code/u/nb", 3, "abc123",
                      results_dir=results_dir)
        return "07_T"

    def inventory(self, tmp_path, technique="07_T", **override):
        asset = {
            "kind": "weights",
            "relative_path": "weights/model.safetensors",
            "sha256": "a" * 64,
            "size_bytes": 498000000,
            "contains_text": False,
        }
        asset.update(override)
        text_asset = {
            "kind": "predictions_with_text",
            "relative_path": "predictions/predictions_test.csv",
            "sha256": "b" * 64,
            "size_bytes": 1200,
            "contains_text": True,
        }
        path = tmp_path / "external_assets.json"
        path.write_text(json.dumps({"technique": technique, "assets": [text_asset, asset]}))
        return path

    def test_inventory_becomes_manifest_entries(self, tmp_path, results_dir, technique):
        from run_manifest import add_assets_from, check, read_manifest

        added = add_assets_from(
            technique, self.inventory(tmp_path), "kaggle://datasets/u/slug/v1/", results_dir
        )
        assert added == 2
        assets = {a["location"]: a for a in read_manifest(technique, results_dir)["external_assets"]}
        weights = assets["kaggle://datasets/u/slug/v1/weights/model.safetensors"]
        assert weights["sha256"] == "a" * 64 and weights["contains_text"] is False
        assert assets["kaggle://datasets/u/slug/v1/predictions/predictions_test.csv"]["contains_text"]
        assert check(technique, results_dir) == []

    def test_repository_location_is_refused_and_nothing_is_recorded(self, tmp_path, results_dir, technique):
        from run_manifest import ManifestError, add_assets_from, read_manifest

        with pytest.raises(ManifestError, match="must use one of"):
            add_assets_from(technique, self.inventory(tmp_path), "results_summary/07_T", results_dir)
        assert read_manifest(technique, results_dir)["external_assets"] == []

    def test_truncated_hash_is_refused_and_nothing_is_recorded(self, tmp_path, results_dir, technique):
        from run_manifest import ManifestError, add_assets_from, read_manifest

        inventory = self.inventory(tmp_path, sha256="abc123")
        with pytest.raises(ManifestError, match="64 hex"):
            add_assets_from(technique, inventory, "kaggle://datasets/u/slug/v1", results_dir)
        assert read_manifest(technique, results_dir)["external_assets"] == []

    def test_inventory_for_another_technique_is_refused(self, tmp_path, results_dir, technique):
        from run_manifest import ManifestError, add_assets_from

        inventory = self.inventory(tmp_path, technique="11_OTHER")
        with pytest.raises(ManifestError, match="not '07_T'"):
            add_assets_from(technique, inventory, "kaggle://datasets/u/slug/v1", results_dir)
