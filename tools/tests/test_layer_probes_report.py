"""Tests for tools/layer_probes_report.py, the local reader of notebook 13's tables.

The tables are built from synthetic scores by the same kit calls notebook 13
makes (`build_probe_tables` in test_probe_kit.py) and written by
`pk.export_probe_tables`, so the reader is tested on exactly what the
notebook writes. The adversarial tests each break one thing and assert the
reader refuses it.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIR))

import layer_probes_report as lpr  # noqa: E402
import notebook_kit as nk  # noqa: E402
import probe_kit as pk  # noqa: E402
from test_probe_kit import (  # noqa: E402
    build_probe_tables,
    probe_config,
    synthetic_foundation,
)


@pytest.fixture
def probe_run(tmp_path):
    """(ctx, frames, tables) for a synthetic probe run whose test split is open."""
    root = tmp_path / "input"
    synthetic_foundation(root / "research_foundation")
    ctx = nk.bootstrap(probe_config(), working_root=tmp_path / "working", input_roots=[root])
    frame, _ = nk.load_foundation(ctx)
    frames = nk.split_frames(frame)
    return ctx, build_probe_tables(ctx, frames)


def export(ctx, probes, contrasts, ablation):
    pk.export_probe_tables(ctx, probes, contrasts, ablation, {"models": ["ft_a", "base_a"]})
    return ctx.results_dir


class TestConformingRun:
    def test_no_tables_is_not_a_failure(self, tmp_path, capsys):
        assert lpr.main(["--check", "--rq-dir", str(tmp_path)]) == lpr.EXIT_OK
        assert "has not been run" in capsys.readouterr().out

    def test_check_and_write(self, probe_run, capsys):
        ctx, tables = probe_run
        rq_dir = export(ctx, *tables)
        assert lpr.main(["--check", "--rq-dir", str(rq_dir)]) == lpr.EXIT_OK
        assert lpr.main(["--write", "--rq-dir", str(rq_dir)]) == lpr.EXIT_OK
        summary = pd.read_csv(rq_dir / lpr.SUMMARY_FILENAME)
        assert list(summary.columns) == list(lpr.SUMMARY_COLUMNS)
        assert len(summary) == 2 * len(pk.PROBE_TASKS)
        fine_tuned = summary[(summary["model_id"] == "ft_a") & (summary["task"] == "extremist_label")].iloc[0]
        assert fine_tuned["base_id"] == "base_a"
        assert np.isfinite(fine_tuned["selectivity_last"])
        reading = pd.read_csv(rq_dir / lpr.READING_FILENAME)
        assert list(reading.columns) == list(lpr.READING_COLUMNS)
        assert len(reading) == 2  # identity_probe rows at the locked threshold only
        assert set(reading["reading"]) <= {
            "identity_direction_carries_label_signal",
            "no_evidence",
            "partial",
        }
        assert "ex_" not in (rq_dir / lpr.SUMMARY_FILENAME).read_text()

    def test_layer_ranges(self):
        assert lpr._ranges([3, 1, 2, 7, 9, 10]) == "1-3,7,9-10"
        assert lpr._ranges([]) == ""


class TestRefusals:
    def test_hand_edited_table_is_refused(self, probe_run):
        ctx, tables = probe_run
        rq_dir = export(ctx, *tables)
        path = rq_dir / pk.PROBE_TABLE_FILES["probes"]
        frame = pd.read_csv(path)
        frame.loc[0, "roc_auc"] = 0.99
        frame.to_csv(path, index=False)
        with pytest.raises(lpr.LayerProbesError, match="hash mismatch"):
            lpr.load_tables(rq_dir)
        assert lpr.main(["--check", "--rq-dir", str(rq_dir)]) == lpr.EXIT_FAILED

    def test_half_a_run_is_refused(self, probe_run):
        ctx, tables = probe_run
        rq_dir = export(ctx, *tables)
        (rq_dir / pk.PROBE_TABLE_FILES["run"]).unlink()
        with pytest.raises(lpr.LayerProbesError, match="incomplete"):
            lpr.tables_present(rq_dir)

    def test_contrast_the_wrong_way_round_is_refused(self, probe_run):
        ctx, (probes, contrasts, ablation) = probe_run
        pairs = contrasts["contrast"] == "fine_tuned_minus_base"
        contrasts.loc[pairs, ["model_id", "reference_id"]] = ["base_a", "ft_a"]
        rq_dir = export(ctx, probes, contrasts, ablation)
        with pytest.raises(lpr.LayerProbesError, match="not a fine-tuned run minus"):
            lpr.load_tables(rq_dir)

    def test_holm_outside_the_family_is_refused(self, probe_run):
        _, (_, _, ablation) = probe_run
        ablation.loc[ablation["direction"] == "random", "wilcoxon_p_holm"] = 0.5
        with pytest.raises(lpr.LayerProbesError, match="outside the preregistered Holm family"):
            lpr.check_holm(ablation)

    def test_holm_that_is_not_holm_is_refused(self, probe_run):
        _, (_, _, ablation) = probe_run
        family = lpr.holm_family_mask(ablation)
        assert family.sum() == 2
        ablation.loc[family, "wilcoxon_p"] = [0.01, 0.04]
        ablation.loc[family, "wilcoxon_p_holm"] = [0.01, 0.04]  # unadjusted; Holm gives 0.02, 0.04
        with pytest.raises(lpr.LayerProbesError, match="is not Holm"):
            lpr.check_holm(ablation)

    def test_reading_follows_the_preregistered_rule(self):
        base = {
            "model_id": "07",
            "source_technique": "07_TWITTER-ROBERTA_FINE-TUNE",
            "seed": 30,
            "direction": "identity_probe",
            "threshold_kind": "locked",
            "n_posts": 60,
            "random_delta_p95": 0.01,
        }
        rows = [
            {**base, "population": "non_extremist", "mean_delta": 0.05, "wilcoxon_p_holm": 0.001},
            {**base, "population": "extremist", "mean_delta": 0.005, "wilcoxon_p_holm": 0.001},
            {**base, "model_id": "11_pooled", "population": "extremist", "mean_delta": 0.05, "wilcoxon_p_holm": 0.2},
        ]
        reading = lpr.read_ablation(pd.DataFrame(rows)).set_index(["model_id", "population"])["reading"]
        assert reading[("07", "non_extremist")] == "identity_direction_carries_label_signal"
        assert reading[("07", "extremist")] == "partial"
        assert reading[("11_pooled", "extremist")] == "no_evidence"
