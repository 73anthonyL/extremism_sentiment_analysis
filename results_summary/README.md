# Results summary folder

This folder stores compact result artifacts for the research notebooks.

The goal is to make the repository reviewable without requiring readers to re-run every notebook before understanding the current findings.

## Expected structure

```text
results_summary/
├── foundation/
├── rq/                       # cross-technique tables, one per research question
├── 01_LOG-REG_TF-IDF/
├── 02_LIN-SVM_TF-IDF/
├── 03_SLP_TF-IDF/
├── 04_CHAR-TF-IDF_LIN-SVM/
├── 05_WORD-CHAR-TF-IDF_LIN-SVM/
├── 06_FASTTEXT-EMB_LOG-REG/
├── 07_TWITTER-ROBERTA_FINE-TUNE/
└── 11_MULTI-CHECKPOINT_LOGIT-POOL/
```

Notebooks `08`, `09`, and `10` have no folder here. For `09` that is correct —
it was never run. For `08` and `10` it is a gap: both evaluated the test split,
but neither exported a probability artifact, so neither folder can be derived
without a rerun. Their numbers therefore appear nowhere in this repository's
result tables.

## How these folders are produced

Results are derived, not transcribed. The notebook computes every metric from
its own probabilities with the shared `compute_binary_metrics` and writes the
files listed in `docs/RESULTS_SCHEMA.md`; it also exports the probability
artifact those numbers came from. Never hand-copy a number out of a notebook
into one of these files.

The primed notebooks write these files through `tools/notebook_kit.py`, in
one layout for every model family, and stamp them `provenance:
derived_from_probs`. The folders here were committed before priming, so some
use older layouts and none carries that stamp yet; each is replaced when its
notebook is rerun.

Validate every folder, including its attribution runs and manifest, with:

```bash
python3 tools/validate_results_folder.py --all
```

## Explainability artifacts

A technique takes part in RQ2-RQ4 once its folder carries:

```text
attributions/<run_id>.csv, <run_id>.json   word-level SHAP aggregate (+ coefficients.csv for LR)
identity_fpr_test.json                     written by tools/identity_fpr.py from the probability artifact
run_manifest.json                          where weights, logs, and per-post files live
```

The `rq/` folder holds the cross-technique tables the documentation renders.

## Foundation artifacts

The `foundation/` folder stores dataset-level artifacts such as label counts, split counts, duplicate checks, removed-row summaries, and the dataset manifest.

## Model result artifacts

Most classical and embedding model folders should contain:

```text
ablation_results.csv
best_config.json
classification_report_test.json
confusion_matrix_test.csv
metrics_validation.json
metrics_test.json
```

The RoBERTa folder currently uses a compact transformer-specific summary:

```text
ablation_results.csv
best_config.json
confusion_matrix_test.png
metrics_validation.json
metrics_test.json
threshold_sweep_validation.csv
```

Raw predictions, per-post attribution files, and trained model weights are never committed; they go to external storage and are recorded in the technique's `run_manifest.json`.

## Current held-out test results

The positive class is `EXTREMIST`.

This table is rendered from the folders above by `tools/render_tables.py`. Do
not edit it by hand — edits inside the marker pair are clobbered on the next
`--write`, and `--check` exits 1 in the meantime.

<!-- RENDERED-TABLE:BEGIN id=test-detail -->
| Technique | Accuracy | Positive F1 | Positive precision | Positive recall | ROC-AUC | PR-AUC | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| `01_LOG-REG_TF-IDF` | 0.8556 | 0.8048 | 0.8221 | 0.7882 | 0.9137 | 0.8914 | 0.48 |
| `02_LIN-SVM_TF-IDF` | 0.8533 | 0.7963 | 0.8377 | 0.7588 | 0.9065 | 0.8887 | 0.485 |
| `03_SLP_TF-IDF` | 0.8400 | 0.7600 | 0.8769 | 0.6706 | 0.9042 | 0.8787 | 0.52 |
| `04_CHAR-TF-IDF_LIN-SVM` | 0.8311 | 0.7500 | 0.8507 | 0.6706 | 0.9147 | 0.8873 | 0.5 |
| `05_WORD-CHAR-TF-IDF_LIN-SVM` | 0.8556 | 0.7782 | 0.9268 | 0.6706 | 0.9198 | 0.9016 | 0.56 |
| `06_FASTTEXT-EMB_LOG-REG` | 0.8333 | 0.7508 | 0.8626 | 0.6647 | 0.8986 | 0.8721 | 0.5 |
| `07_TWITTER-ROBERTA_FINE-TUNE` | 0.8889 | 0.8521 | 0.8571 | 0.8471 | 0.9510 | 0.9320 | 0.555 |
| `11_MULTI-CHECKPOINT_LOGIT-POOL` | 0.9089 | 0.8746 | 0.9108 | 0.8412 | 0.9682 | 0.9553 | 0.5 |

Rendered by tools/render_tables.py from results_summary/ — do not edit by hand.
<!-- RENDERED-TABLE:END id=test-detail -->


## Interpretation rule

Results in this folder are baseline research metrics. They should not be interpreted as deployment readiness or as evidence that the models can safely make automated moderation decisions.

## Interpretation caution: statistical power

The test split is 450 rows. Distinguishing two techniques at McNemar exact
significance needs roughly 14 rows of difference, and most gaps in the table
above are far smaller. The ordering of the classical baselines carries no
statistical weight. A higher number in this table is not, by itself, a better
model; RQ1 is kept brief for that reason.

## Update rule

When a notebook is rerun and results change:

1. Replace the probability artifact, the result folder files, and the
   attribution runs together, from the same run.
2. Re-run `tools/identity_fpr.py` and `tools/categorize_attributions.py` for
   the technique, then `tools/compare_reliance.py`.
3. Re-validate with `tools/validate_results_folder.py --all`.
4. Regenerate every documentation table with `tools/render_tables.py --write`,
   then confirm `--check` exits 0.

Do not update a table by editing it.
