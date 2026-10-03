# Results schema

This document defines the expected output structure for each experiment in `results_summary/`.

`tools/validate_results_folder.py --all` checks a folder against this document —
both that the required files exist and that they are internally consistent
(confusion-matrix counts agreeing with the reported rates, support totals
matching the split sizes, and so on). A folder that does not validate is not
eligible for the comparison tables.

## Results are derived, not transcribed

The primary artifact of a run is not a metrics file. It is a probability file,
and every metric is computed from it with `tools/metrics_core.py`'s
`compute_binary_metrics`, the same function the notebooks use. Hand-copying a
number from a notebook cell output into a JSON file is not acceptable: it
breaks the guarantee that a published figure can only change when a committed
artifact changes, and it leaves nothing for a replicator to check.
`tools/validate_results_folder.py --all` recomputes every threshold-dependent
metric from the stored confusion counts and fails on any disagreement.

Every artifact below is written by `tools/notebook_kit.py`, the helper each
notebook loads, and refused at export time if it breaks its contract. The kit
runs on Kaggle with no repository on the import path, so it carries its own
copy of each contract; `tools/tests/test_notebook_kit.py` pins those copies to
the tools named here and loads every artifact the kit writes back through the
real loaders.

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
failure, not a formatting problem. An accompanying `<TECHNIQUE>__meta.json`
records the run's selected threshold and provenance.

`tools/identity_fpr.py` (RQ4) reads these artifacts; a technique without one
cannot appear in the identity false-positive table.

### Attribution artifact contract

The explainability research questions run on a second committed artifact: the
word-level aggregate of a SHAP run. It lives at
`results_summary/<TECHNIQUE>/attributions/<run_id>.csv` with a sidecar
`<run_id>.json`, enforced by `tools/attributions.py`:

| Column | Meaning |
|---|---|
| `word` | A single word (no whitespace), lowercased. Words are the units of the explainer's word-level text masker, so a model's own subword tokenizer plays no part. |
| `mean_abs_attribution` | Mean absolute SHAP value over the explained posts that contain the word. A word that occurs several times in a post contributes the sum of its occurrences for that post. |
| `mean_attribution` | Mean signed SHAP value toward `EXTREMIST` over those same posts. |
| `support` | Number of explained posts containing the word. |

Exactly these four columns. Means are over posts *containing* the word, never
over all posts: for a linear model the attribution of an absent feature has
the opposite sign to its coefficient, so an all-post mean would cancel. Any
per-post column (`row_id`, `text`, `position`, ...) is refused.

The sidecar must declare `technique`, `run_id`, `split`, `explainer`,
`background_size`, `seed`, `aggregation` (always `word`), `n_posts_explained`,
and `member` (`null` for a whole model or whole ensemble; the member id for one
ensemble member).

Every notebook uses one explainer: SHAP's partition explainer over a
word-level text masker (`shap.maskers.Text(r"\W+")`), applied to the model as
a function from text to P(EXTREMIST). The text masker replaces words with a
mask token and uses no background sample, so `background_size` is `0`; the
sidecar also records the evaluation budget per post in `explainer_settings`.

A technique has these runs, named by convention:

| Run id | What it explains |
|---|---|
| `shap-partition_test_seed30` | The whole model (or whole ensemble) on the test split. |
| `shap-partition_validation_seed30` | The same model on the validation split. The two whole-model runs feed the stability check. |
| `shap-partition_test_seed30_member-<id>` | One ensemble member, on the same test posts. These feed the member-versus-whole comparison. |

Classical notebooks explain every post of a split; transformer and ensemble
notebooks explain a label-stratified sample (200 posts by default), and
`n_posts_explained` says which. A run id never contains `__`: that separator
marks files derived from a run, and such stems are not listed as runs.

Logistic-regression techniques also commit `attributions/coefficients.csv`
(`word, coefficient`) so `tools/validate_shap.py` can confirm the pipeline
recovers the coefficients before any cross-model claim is made.

### Derived RQ files

| File | Written by | Contents |
|---|---|---|
| `attributions/<run_id>__categorized.csv` | `categorize_attributions.py` | Top-K words with `category` (`slur`, `extremist_framing`, `identity_term`, `topical`) and every matched category. |
| `attributions/<run_id>__category_shares.csv` | `categorize_attributions.py` | Share of words, absolute mass, and positive mass per category. |
| `attributions/<run_id>__coefficient_check.json` | `validate_shap.py` | Spearman and sign-agreement statistics against the coefficients. |
| `identity_fpr_<split>.json`, `identity_fpr_terms_<split>.csv` | `identity_fpr.py` | False-positive rate on non-extremist posts with vs without identity terms; per-term counts. No text. |
| `run_manifest.json` | `run_manifest.py` | Kaggle notebook version, git commit, hashes of committed files, and the location and hash of every external asset. |

Cross-technique tables live in `results_summary/rq/`, one per research
question, and are the inputs to the rendered documentation tables.

### Provenance fields

`best_config.json`, `metrics_validation.json` and `metrics_test.json` say how
their numbers were obtained:

| Field | Meaning |
|---|---|
| `provenance` | How the numbers were obtained, e.g. `derived_from_probs` or `notebook_cell_outputs`. |
| `recomputable` | `true` only if the folder can be rebuilt from a committed artifact. |

A folder written by the kit carries `derived_from_probs` and `true`. Folders
committed before the notebooks were primed carry neither field; each is
replaced when its notebook is rerun.

## Folder layout

Each model family should write results to:

```text
results_summary/<TECHNIQUE>/
```

Every primed notebook writes the same folder, whatever the model family:

```text
results_summary/<TECHNIQUE>/
├── ablation_results.csv
├── best_config.json
├── classification_report_test.json
├── confusion_matrix_test.csv
├── metrics_validation.json
├── metrics_test.json
├── threshold_sweep_validation.csv
└── attributions/
```

Folders committed before priming use older layouts (the transformer folder
has a confusion-matrix image instead of the CSV). Raw predictions, per-post
attribution files, and model weights are never committed; they go to external
storage and are recorded in `run_manifest.json`.

## Required compact files

### `best_config.json`

Stores the selected model configuration.

Fields:

| Field | Description |
|---|---|
| `technique` | Technique name, such as `05_WORD-CHAR-TF-IDF_LIN-SVM`. |
| `model_family` | General model type. |
| `feature_family` | Feature representation. |
| `random_seed` | Random seed used where applicable. |
| `dataset_version`, `split_version` | The frozen dataset and split the run used. |
| `hyperparameters` | Selected hyperparameters. |
| `threshold_strategy`, `threshold_metric` | Method and metric used to choose the decision threshold. |
| `selected_threshold` | Final threshold used for test evaluation. |
| `threshold_selected_on_split` | Always `validation`. |
| `test_set_used_for_selection` | Always `false`. |
| `n_validation_candidates` | How many configurations were compared on validation. |
| `validation_uses` | Every use of the validation split: `config_selection`, `threshold`, and where they apply `early_stopping`, admission and strategy selection. |
| `in_comparison` | Whether the technique takes part in the controlled comparison. |

Ensembles also record their members, the combination rule, and any
prespecified gate with its verdict.

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

`tools/render_tables.py` skips any technique folder lacking this file, so a
technique that has not been evaluated on test simply does not appear in the
comparison tables.

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

This is the one required file that cannot be derived from the probability
artifact, which records the selected configuration's outputs, not the
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

Word-level attribution runs (above) are the committed interpretability
artifact and are required for every technique that takes part in RQ2-RQ4.

Per-post files carry dataset text and are never committed. A notebook writes
them under `external/<TECHNIQUE>/`, with an inventory:

```text
external/<TECHNIQUE>/
├── weights/                 model weights (and tokenizer/ for transformers)
├── training_logs/
├── predictions/             predictions_validation.csv, predictions_test.csv (with text)
├── attributions_local/      one per-post token table per attribution run
├── probs_members/           per-member probabilities for ensembles (no text)
├── plots/                   the diagnostic figure
└── external_assets.json     kind, path, sha256, size and text flag of every file
```

The folder is uploaded to external storage, and
`tools/run_manifest.py add-assets-from --file external_assets.json` records
every file in `run_manifest.json`, text-bearing ones with `contains_text: true`.

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

Result tables in `README.md` and `results_summary/README.md` are generated from
these files, not written by hand.
Each lives inside a pair of HTML comments carrying a `BEGIN`/`END` marker and a
table id, which survive Markdown rendering invisibly. The exact marker syntax is
given in the docstring of `tools/render_tables.py`; it is not reproduced here
because the tool scans this document and counts markers literally.

Everything between a region's markers is rewritten wholesale by
`tools/render_tables.py --write`; hand edits inside a region are deliberately
clobbered. `--check` exits 1 if any document has drifted from the artifacts, and
also if a document contains an unpaired marker.

Five table ids are defined:

| Table id | Contents |
|---|---|
| `main-comparison` | RQ1: validation accuracy, test accuracy, balanced accuracy, macro F1, ROC-AUC. |
| `test-detail` | Test accuracy, positive-class F1 / precision / recall, ROC-AUC, PR-AUC, threshold. |
| `confusion-test` | Test confusion-matrix counts and error rates. |
| `rq2-category-shares` | RQ2: share of positive attribution mass per lexicon category, per technique. |
| `rq4-identity-fpr` | RQ4: false-positive rate on non-extremist test posts with vs without identity terms. |

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
