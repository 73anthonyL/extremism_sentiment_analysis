# Results summary folder

This folder stores compact result artifacts for the research notebooks.

The goal is to make the repository reviewable without requiring readers to re-run every notebook before understanding the current findings.

## Expected structure

```text
results_summary/
├── foundation/
├── 01_LOG-REG_TF-IDF/
├── 02_LIN-SVM_TF-IDF/
├── 03_SLP_TF-IDF/
├── 04_CHAR-TF-IDF_LIN-SVM/
├── 05_WORD-CHAR-TF-IDF_LIN-SVM/
├── 06_FASTTEXT-EMB_LOG-REG/
└── 07_TWITTER-ROBERTA_FINE-TUNE/
```

Notebooks `08`–`11` have no folder here. For `09` and `11` that is correct —
they have not been run. For `08` and `10` it is a gap: both evaluated the test
split, but neither exported a probability artifact, so neither folder can be
derived. Their numbers therefore appear nowhere in this repository's result
tables. See `research_loop/STATE.md`.

## How these folders are produced

Results are derived, not transcribed:

```bash
python3 tools/eval_from_probs.py --technique <TECHNIQUE> --threshold <selected>
```

`eval_from_probs.py` builds the whole folder from a committed probability
artifact in `research_loop/probs/`. Never hand-copy a number out of a notebook
into one of these files — a metric that cannot be rebuilt from an artifact
cannot be checked by anyone, including the person who wrote it.

Validate a folder against `docs/RESULTS_SCHEMA.md` with:

```bash
python3 tools/validate_results_folder.py --all
```

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

Additional transformer artifacts, raw predictions, local attribution files, and trained model weights should not be committed to normal Git unless they are intentionally sanitized or stored through Git LFS/releases/external storage.

## Current held-out test results

The positive class is `EXTREMIST`.

This table is rendered from the folders above by `tools/render_tables.py`. Do
not edit it by hand — edits inside the marker pair are clobbered on the next
`--write`, and `--check` exits 1 in the meantime.

<!-- RENDERED-TABLE:BEGIN id=test-detail -->
| Technique | Accuracy | Positive F1 | Positive precision | Positive recall | ROC-AUC | PR-AUC | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| `01_LOG-REG_TF-IDF` | 0.8533 | 0.8024 | 0.8171 | 0.7882 | 0.9111 | 0.8881 | 0.45 |
| `02_LIN-SVM_TF-IDF` | 0.8556 | 0.7962 | 0.8523 | 0.7471 | 0.9037 | 0.8825 | 0.47 |
| `03_SLP_TF-IDF` | 0.8378 | 0.7768 | 0.8089 | 0.7471 | 0.9022 | 0.8777 | 0.5 |
| `04_CHAR-TF-IDF_LIN-SVM` | 0.8333 | 0.7875 | 0.7596 | 0.8176 | 0.9028 | 0.8745 | 0.4 |
| `05_WORD-CHAR-TF-IDF_LIN-SVM` | 0.8356 | 0.7886 | 0.7667 | 0.8118 | 0.9032 | 0.8818 | 0.42 |
| `06_FASTTEXT-EMB_LOG-REG` | 0.8178 | 0.7722 | 0.7316 | 0.8176 | 0.9033 | 0.8763 | 0.47 |
| `07_TWITTER-ROBERTA_FINE-TUNE` | 0.8889 | 0.8555 | 0.8409 | 0.8706 | 0.9496 | 0.9233 | 0.72 |

Rendered by tools/render_tables.py from results_summary/ — do not edit by hand.
<!-- RENDERED-TABLE:END id=test-detail -->


## Interpretation rule

Results in this folder are baseline research metrics. They should not be interpreted as deployment readiness or as evidence that the models can safely make automated moderation decisions.

## Interpretation caution: statistical power

The test split is 450 rows. Distinguishing two techniques at McNemar exact
significance needs roughly 14 rows of difference, and most gaps in the table
above are far smaller. The ordering of the classical baselines carries no
statistical weight, and `INCONCLUSIVE` is the expected verdict for most
comparisons this project can run.

Only `tools/compare_techniques.py` may issue a verdict. A higher number in this
table is not, by itself, a better model.

## Update rule

When a notebook is rerun and results change:

1. Re-derive the folder with `tools/eval_from_probs.py` from the new
   probability artifact.
2. Re-validate with `tools/validate_results_folder.py --all`.
3. Regenerate every documentation table with `tools/render_tables.py --write`,
   then confirm `--check` exits 0.

Do not update a table by editing it.
