# Results schema

This document defines the expected output structure for each experiment in `results_summary/`.

`tools/validate_results_folder.py --all` checks a folder against this document —
both that the required files exist and that they are internally consistent
(confusion-matrix counts agreeing with the reported rates, support totals
matching the split sizes, and so on). A folder that does not validate is not
eligible for the comparison tables.

## Results are derived, not transcribed

The primary artifact of a run is not a metrics file. It is a probability file,
and the metrics are computed from it:

```bash
python3 tools/eval_from_probs.py --technique <TECHNIQUE> --threshold <selected>
```

This builds the entire `results_summary/<TECHNIQUE>/` folder. Hand-copying a
number from a notebook cell output into a JSON file is not an acceptable
substitute: it breaks the guarantee that a published figure can only change when
a committed artifact changes, and it leaves nothing for a replicator to check.

### Probability artifact contract

Sanitized probability files live in `research_loop/probs/` and are named
`<TECHNIQUE>__<split>.csv`. The contract, enforced by `tools/probs_artifact.py`
and asserted at export time by the notebook:

| Column | Meaning |
|---|---|
| `row_id` | Canonical dataset row id, unique within the file. |
| `split` | `validation` or `test`. |
| `y_true` | Ground-truth label, `0` or `1`. |
| `y_prob` | Predicted probability of the positive class, finite and in `[0, 1]`. |

Exactly these four columns, and no others. Any text-bearing column is a leakage
failure, not a formatting problem — the artifact is committed, and the dataset
text is not meant to travel with it.

An accompanying `<TECHNIQUE>__meta.json` records the run's selected threshold
and provenance. The threshold is read from there rather than re-derived, because
`eval_from_probs.py` deliberately will not select one.

### Provenance fields

Where a result folder cannot be derived — for example because the notebook
predates the probability-export cell — it must say so explicitly:

| Field | Meaning |
|---|---|
| `provenance` | How the numbers were obtained, e.g. `derived_from_probs` or `notebook_cell_outputs`. |
| `recomputable` | `true` only if `eval_from_probs.py` can rebuild the folder from a committed artifact. |

A folder marked `recomputable: false` is a documented exception, not a normal
state, and should carry a note about what it would take to fix.

## Folder layout

Each model family should write results to:

```text
results_summary/<TECHNIQUE>/
```

Example classical or embedding folder:

```text
results_summary/05_WORD-CHAR-TF-IDF_LIN-SVM/
├── ablation_results.csv
├── best_config.json
├── classification_report_test.json
├── confusion_matrix_test.csv
├── metrics_validation.json
└── metrics_test.json
```

Example transformer folder:

```text
results_summary/07_TWITTER-ROBERTA_FINE-TUNE/
├── ablation_results.csv
├── best_config.json
├── confusion_matrix_test.png
├── metrics_validation.json
├── metrics_test.json
└── threshold_sweep_validation.csv
```

Some model families may include additional metadata files, plots, or interpretability summaries when useful. Raw predictions, local attribution files containing raw text, and large model artifacts should not be committed to normal Git unless intentionally sanitized or stored with Git LFS/releases/external storage.

## Required compact files

### `best_config.json`

Stores the selected model configuration.

Recommended fields:

| Field | Description |
|---|---|
| `technique` | Technique name, such as `05_WORD-CHAR-TF-IDF_LIN-SVM`. |
| `model_family` | General model type. |
| `feature_family` | Feature representation. |
| `random_seed` | Random seed used where applicable. |
| `split_version` | Split version used for the run. |
| `hyperparameters` | Selected hyperparameters. |
| `threshold_strategy` | Method used to choose the decision threshold. |
| `selected_threshold` | Final threshold used for test evaluation. |

### `metrics_validation.json`

Stores metrics on the validation split used during selection.

Required or recommended fields:

* `technique`
* `split`
* `threshold`
* `support`
* `positive_support`
* `negative_support`
* `accuracy`
* `balanced_accuracy`
* `positive_precision`
* `positive_recall`
* `positive_f1`
* `macro_f1` or `f1_macro`
* `weighted_f1` or `f1_weighted`
* `roc_auc`
* `pr_auc`
* `brier_score`, if probabilities are available
* `tn`, `fp`, `fn`, `tp`
* `false_positive_rate`
* `false_negative_rate`

### `metrics_test.json`

Stores final locked metrics on the held-out test split. This file should not be generated until the model configuration and threshold have been selected.

Its presence means a test unlock was spent. That unlock is once-per-technique,
is recorded in `research_loop/test_ledger.jsonl`, and raises the
multiple-comparison bar for every later candidate. See `docs/RESEARCH_LOOP.md`.

`tools/render_tables.py` skips any technique folder lacking this file, so a
technique whose test split has not been unlocked simply does not appear in the
comparison tables — which is the correct behaviour, not an omission to work
around.

### `confusion_matrix_test.csv` or `confusion_matrix_test.png`

Classical and embedding folders should prefer `confusion_matrix_test.csv` with this format:

```text
actual,predicted,count
NON_EXTREMIST,NON_EXTREMIST,250
NON_EXTREMIST,EXTREMIST,30
EXTREMIST,NON_EXTREMIST,36
EXTREMIST,EXTREMIST,134
```

Transformer folders may save `confusion_matrix_test.png` instead when the compact figure is the reviewed artifact. If only the PNG is saved, the confusion-matrix counts must still appear inside `metrics_test.json`.

### `classification_report_test.json`

Stores class-level precision, recall, F1, and support from the final test evaluation. This is expected for classical and embedding folders. It is optional for transformer folders if `metrics_test.json` already contains the main model-comparison metrics.

### `ablation_results.csv`

Stores the validation results of compared configurations.

This is the one required file `eval_from_probs.py` cannot derive — the
probability artifact records the selected configuration's outputs, not the
alternatives it beat. It must be supplied from the run's own configuration
comparison.

Recommended columns:

* `technique`
* `config_id`
* `model_family`
* `feature_family`
* `hyperparameter_summary`
* `threshold`
* `validation_accuracy`
* `validation_positive_f1`
* `validation_roc_auc`
* `validation_pr_auc`
* `notes`

### `threshold_sweep_validation.csv`

Stores validation-set threshold comparisons. This file is optional for classical notebooks if threshold information is already summarized in metrics/config files. It is recommended for transformer notebooks because threshold choice can strongly affect the apparent precision/recall tradeoff.

## Interpretability artifacts

Interpretability files are optional but encouraged for the XAI focus of this repository.

Safe-to-commit summary files include:

```text
interpretability/global_token_attribution_summary.csv
interpretability/top_positive_tokens_by_gradient.csv
interpretability/top_negative_tokens_by_gradient.csv
```

Use caution with local explanation files because they may contain raw text:

```text
interpretability/local_token_attribution_explanations.csv
interpretability/local_token_attributions_long.csv
error_analysis/manual_review_queue_test.csv
predictions_test.csv
predictions_validation.csv
```

These should only be committed if sanitized or if the dataset text is intended to be public in that form.

## Foundation folder

Dataset and split artifacts should be stored under:

```text
results_summary/foundation/
```

Expected files:

* `dataset_manifest.json`
* `label_distribution.csv`
* `split_label_distribution.csv`
* `text_length_summary.csv`
* `duplicate_text_report.csv`
* `rows_removed_summary.json`

## Rendered documentation tables

Result tables in `README.md`, `docs/EXPERIMENTS.md`, `docs/MODEL_CARD.md`, and
`results_summary/README.md` are generated from these files, not written by hand.
Each lives inside a pair of HTML comments carrying a `BEGIN`/`END` marker and a
table id, which survive Markdown rendering invisibly. The exact marker syntax is
given in the docstring of `tools/render_tables.py`; it is not reproduced here
because the tool scans this document and counts markers literally.

Everything between a region's markers is rewritten wholesale by
`tools/render_tables.py --write`; hand edits inside a region are deliberately
clobbered. `--check` exits 1 if any document has drifted from the artifacts, and
also if a document contains an unpaired marker.

Three table ids are defined:

| Table id | Contents |
|---|---|
| `main-comparison` | Validation accuracy, test accuracy, balanced accuracy, macro F1, ROC-AUC. |
| `test-detail` | Test accuracy, positive-class F1 / precision / recall, ROC-AUC, PR-AUC, threshold. |
| `confusion-test` | Test confusion-matrix counts and error rates. |

Adding a technique to the documentation therefore means adding its result
folder, not editing a table.

## Reproducibility expectations

Every saved result should make clear:

* Which dataset version was used.
* Which split version was used.
* Which model family and feature representation were used.
* Whether the threshold was selected on validation data.
* Whether test data was excluded from model selection.
* Whether external pretraining or transfer learning was used.
