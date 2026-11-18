#!/usr/bin/env python3
"""Check and summarise the WP9 layer-probe tables written by notebook 13.

Notebook 13 (`13_TRANSFORMER-LAYER-PROBES`) runs on a Kaggle GPU and writes,
through `probe_kit.export_probe_tables`, three text-free tables and a run
record to `results_summary/rq/`:

    rq3_layer_probes.csv                 model run x layer x task x split
    rq3_layer_probe_contrasts.csv        fine-tuned minus base; selectivity
    rq4_identity_direction_ablation.csv  identity-direction ablation rows
    rq3_layer_probes_run.json            configuration, sources, verification

This tool reads them locally, with no model and no dataset text:

* `--check` re-applies the probe kit's contracts, confirms each table still has the
  hash the run record gave it (a hand edit is refused), recomputes the Holm
  family of the ablation (the `identity_probe` rows at a locked threshold,
  `docs/preregistrations/WP9_layer_probes.md`) and refuses any other row with
  a Holm value, and refuses a fine-tuning contrast whose sides are not a
  fine-tuned run and a pretrained base.
* `--summary` prints, per model run and task, the test AUC at layer 0, at
  the last layer and at its peak, the last-layer selectivity, the layer
  ranges where each fine-tuning contrast's interval lies above or below zero,
  and the preregistered reading of each primary ablation row.
* `--write` also writes that summary to
  `results_summary/rq/rq3_layer_probes_summary.csv` and the ablation reading to
  `results_summary/rq/rq4_identity_direction_reading.csv`.

Before notebook 13 has been run none of the tables exist, and every mode says
so and exits 0.

    python3 tools/layer_probes_report.py --check
    python3 tools/layer_probes_report.py --summary
    python3 tools/layer_probes_report.py --write
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import notebook_kit as nk  # noqa: E402
import probe_kit as pk  # noqa: E402
from repo_paths import RQ_DIR  # noqa: E402

EXIT_OK = 0
EXIT_FAILED = 1

SUMMARY_FILENAME = "rq3_layer_probes_summary.csv"
READING_FILENAME = "rq4_identity_direction_reading.csv"
SUMMARY_COLUMNS = (
    "model_id",
    "model_kind",
    "source_technique",
    "seed",
    "task",
    "n_layers",
    "auc_layer0",
    "auc_last",
    "auc_last_low",
    "auc_last_high",
    "peak_layer",
    "auc_peak",
    "auc_peak_low",
    "auc_peak_high",
    "selectivity_last",
    "selectivity_last_low",
    "selectivity_last_high",
    "base_id",
    "layers_above_base",
    "layers_below_base",
)
READING_COLUMNS = (
    "model_id",
    "population",
    "n_posts",
    "mean_delta",
    "random_delta_p95",
    "wilcoxon_p_holm",
    "reading",
)
READING_ALPHA = 0.05
HOLM_TOLERANCE = 1e-9


class LayerProbesError(RuntimeError):
    """The WP9 tables are malformed or inconsistent with their run record."""


def table_paths(rq_dir=None):
    rq_dir = Path(rq_dir) if rq_dir is not None else RQ_DIR
    return {name: rq_dir / filename for name, filename in pk.PROBE_TABLE_FILES.items()}


def tables_present(rq_dir=None):
    """True when the run exists, False when none of its files does; refuses half a run."""
    paths = table_paths(rq_dir)
    required = ("probes", "contrasts", "run")
    present = {name: paths[name].exists() for name in required}
    if not any(present.values()):
        return False
    missing = [paths[name].name for name, ok in present.items() if not ok]
    if missing:
        raise LayerProbesError(f"incomplete WP9 output: missing {missing}")
    return True


def holm_family_mask(ablation):
    """The preregistered Holm family: identity_probe rows at a locked threshold."""
    return (ablation["direction"] == "identity_probe") & (ablation["threshold_kind"] == "locked")


def load_tables(rq_dir=None):
    """Read, validate and cross-check the WP9 tables. Returns (probes, contrasts, ablation, run)."""
    paths = table_paths(rq_dir)
    with open(paths["run"]) as handle:
        run = json.load(handle)
    recorded = run.get("artifact_sha256") or {}
    for name in ("probes", "contrasts", "ablation"):
        path = paths[name]
        if not path.exists():
            if name in recorded:
                raise LayerProbesError(f"{path.name} is recorded in the run record but missing")
            continue
        if name not in recorded:
            raise LayerProbesError(f"{path.name} is not recorded in the run record")
        if nk.sha256_file(path, 16) != recorded[name]:
            raise LayerProbesError(
                f"{path.name} differs from the file the notebook wrote (hash mismatch); "
                "tables are exported by notebook 13, never edited"
            )

    try:
        probes = pk.validate_probe_table(pd.read_csv(paths["probes"]))
        contrasts = pk.validate_contrast_table(pd.read_csv(paths["contrasts"]), probes)
        ablation = (
            pk.validate_ablation_table(pd.read_csv(paths["ablation"]))
            if paths["ablation"].exists()
            else None
        )
    except nk.ArtifactContractError as error:
        raise LayerProbesError(str(error)) from error

    kinds = probes.drop_duplicates("model_id").set_index("model_id")["model_kind"]
    pairs = contrasts[contrasts["contrast"] == "fine_tuned_minus_base"]
    for row in pairs.drop_duplicates(["model_id", "reference_id"]).itertuples():
        if kinds.get(row.model_id) != "fine_tuned" or kinds.get(row.reference_id) != "pretrained_base":
            raise LayerProbesError(
                f"contrast {row.model_id} - {row.reference_id} is not a fine-tuned run minus a "
                "pretrained base"
            )

    if ablation is not None:
        check_holm(ablation)
    return probes, contrasts, ablation, run


def check_holm(ablation):
    """Holm over the preregistered family must match; every other row has none."""
    family = holm_family_mask(ablation)
    for column, raw in (("wilcoxon_p_holm", "wilcoxon_p"), ("mcnemar_p_holm", "mcnemar_p")):
        outside = ablation.loc[~family, column]
        if np.isfinite(pd.to_numeric(outside, errors="coerce")).any():
            raise LayerProbesError(f"{column} is set on a row outside the preregistered Holm family")
        values = pd.to_numeric(ablation.loc[family, raw], errors="coerce").to_numpy()
        stored = pd.to_numeric(ablation.loc[family, column], errors="coerce").to_numpy()
        finite = np.isfinite(values)
        expected = np.full(len(values), np.nan)
        expected[finite] = pk.holm_adjust(values[finite])
        same = (np.isnan(expected) & np.isnan(stored)) | (np.abs(expected - stored) <= HOLM_TOLERANCE)
        if not same.all():
            raise LayerProbesError(f"{column} is not Holm over the {int(family.sum())} family rows")


def _ranges(layers):
    """[1, 2, 3, 7] -> '1-3,7'; empty -> ''."""
    layers = sorted(int(x) for x in layers)
    parts = []
    start = previous = None
    for layer in layers:
        if start is None:
            start = previous = layer
        elif layer == previous + 1:
            previous = layer
        else:
            parts.append(f"{start}-{previous}" if previous != start else f"{start}")
            start = previous = layer
    if start is not None:
        parts.append(f"{start}-{previous}" if previous != start else f"{start}")
    return ",".join(parts)


def summarize(probes, contrasts):
    """One row per model run x task, on the test split."""
    test = probes[probes["split"] == "test"]
    selectivity = contrasts[(contrasts["contrast"] == "selectivity") & (contrasts["split"] == "test")]
    pairs = contrasts[(contrasts["contrast"] == "fine_tuned_minus_base") & (contrasts["split"] == "test")]
    rows = []
    for (model_id, task), group in test.groupby(["model_id", "task"], sort=True):
        group = group.sort_values("layer")
        last = group.iloc[-1]
        peak = group.loc[group["roc_auc"].idxmax()]
        row = {
            "model_id": model_id,
            "model_kind": last["model_kind"],
            "source_technique": last["source_technique"],
            "seed": last["seed"],
            "task": task,
            "n_layers": int(last["n_layers"]),
            "auc_layer0": float(group.iloc[0]["roc_auc"]),
            "auc_last": float(last["roc_auc"]),
            "auc_last_low": float(last["roc_auc_low"]),
            "auc_last_high": float(last["roc_auc_high"]),
            "peak_layer": int(peak["layer"]),
            "auc_peak": float(peak["roc_auc"]),
            "auc_peak_low": float(peak["roc_auc_low"]),
            "auc_peak_high": float(peak["roc_auc_high"]),
            "selectivity_last": np.nan,
            "selectivity_last_low": np.nan,
            "selectivity_last_high": np.nan,
            "base_id": "",
            "layers_above_base": "",
            "layers_below_base": "",
        }
        chosen = selectivity[
            (selectivity["model_id"] == model_id)
            & (selectivity["task"] == task)
            & (selectivity["layer"] == int(last["layer"]))
        ]
        if len(chosen) == 1:
            row["selectivity_last"] = float(chosen.iloc[0]["difference"])
            row["selectivity_last_low"] = float(chosen.iloc[0]["difference_low"])
            row["selectivity_last_high"] = float(chosen.iloc[0]["difference_high"])
        mine = pairs[(pairs["model_id"] == model_id) & (pairs["task"] == task)]
        if len(mine):
            row["base_id"] = str(mine.iloc[0]["reference_id"])
            row["layers_above_base"] = _ranges(mine.loc[mine["difference_low"] > 0, "layer"])
            row["layers_below_base"] = _ranges(mine.loc[mine["difference_high"] < 0, "layer"])
        rows.append(row)
    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))


def read_ablation(ablation):
    """The preregistered reading of every primary ablation row."""
    family = ablation[holm_family_mask(ablation)]
    rows = []
    for row in family.sort_values(["model_id", "population"]).itertuples():
        holm = row.wilcoxon_p_holm
        if not np.isfinite(holm) or holm >= READING_ALPHA:
            reading = "no_evidence"
        elif np.isfinite(row.random_delta_p95) and row.mean_delta > row.random_delta_p95:
            reading = "identity_direction_carries_label_signal"
        else:
            reading = "partial"
        rows.append(
            {
                "model_id": row.model_id,
                "population": row.population,
                "n_posts": int(row.n_posts),
                "mean_delta": float(row.mean_delta),
                "random_delta_p95": float(row.random_delta_p95),
                "wilcoxon_p_holm": float(holm),
                "reading": reading,
            }
        )
    return pd.DataFrame(rows, columns=list(READING_COLUMNS))


def print_summary(summary, reading):
    print("Test split, per model run and task (AUC; selectivity = task AUC - control AUC):")
    for row in summary.itertuples():
        selectivity = (
            f"  selectivity {row.selectivity_last:+.3f} [{row.selectivity_last_low:+.3f}, "
            f"{row.selectivity_last_high:+.3f}]"
            if np.isfinite(row.selectivity_last)
            else ""
        )
        print(
            f"  {row.model_id:<34} {row.task:<22} layer0 {row.auc_layer0:.3f}  "
            f"last {row.auc_last:.3f} [{row.auc_last_low:.3f}, {row.auc_last_high:.3f}]  "
            f"peak {row.auc_peak:.3f} @ {row.peak_layer}{selectivity}"
        )
        if row.base_id:
            print(
                f"  {'':<34} vs {row.base_id}: above base at layers [{row.layers_above_base}], "
                f"below at [{row.layers_below_base}]"
            )
    if reading is not None and len(reading):
        print("\nIdentity-direction ablation, preregistered reading:")
        for row in reading.itertuples():
            print(
                f"  {row.model_id:<12} {row.population:<14} n={row.n_posts:<4} mean delta "
                f"{row.mean_delta:+.4f} (random p95 {row.random_delta_p95:+.4f}), "
                f"Holm p {row.wilcoxon_p_holm:.3g}: {row.reading}"
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="validate the tables")
    group.add_argument("--summary", action="store_true", help="validate and print the summary")
    group.add_argument("--write", action="store_true", help="validate, print and write the summary")
    parser.add_argument("--rq-dir", default=None, help="folder holding the tables (default results_summary/rq)")
    args = parser.parse_args(argv)

    try:
        if not tables_present(args.rq_dir):
            print("No WP9 tables yet: notebook 13 has not been run. Nothing to check.")
            return EXIT_OK
        probes, contrasts, ablation, run = load_tables(args.rq_dir)
    except LayerProbesError as error:
        print(f"FAIL  {error}")
        return EXIT_FAILED

    models = probes["model_id"].nunique()
    print(
        f"PASS  WP9 tables of run {run.get('run_id')}: {len(probes)} probe rows over {models} "
        f"model runs, {len(contrasts)} contrast rows, "
        f"{0 if ablation is None else len(ablation)} ablation rows"
    )
    if args.check:
        return EXIT_OK
    summary = summarize(probes, contrasts)
    reading = read_ablation(ablation) if ablation is not None else None
    print()
    print_summary(summary, reading)
    if args.write:
        rq_dir = Path(args.rq_dir) if args.rq_dir is not None else RQ_DIR
        summary.to_csv(rq_dir / SUMMARY_FILENAME, index=False, float_format="%.6g")
        print(f"\nwrote {rq_dir / SUMMARY_FILENAME}")
        if reading is not None:
            reading.to_csv(rq_dir / READING_FILENAME, index=False, float_format="%.6g")
            print(f"wrote {rq_dir / READING_FILENAME}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
