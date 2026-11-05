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
| `attributions/<run_id>__coefficient_check.json` | `validate_shap.py` | Spearman and sign-agreement statistics against the coefficients and `passed` under the preregistered rule (Spearman over all shared words, top-K sign agreement). Also `spearman_support_ge_min` over the `n_shared_words_support_ge_min` shared words with support of at least `secondary_min_support` (2), a diagnostic that does not decide `passed`. One file per whole-model run; `audit.py run` writes them for every run. |
| `identity_fpr_<split>.json`, `identity_fpr_terms_<split>.csv` | `identity_fpr.py` | False-positive rate on non-extremist posts with vs without identity terms; per-term counts. No text. |
| `identity_counterfactual_test.json` | `identity_counterfactual.py` | WP7: probability shift and flips at the locked threshold when identity terms are removed (primary) or replaced by a control panel (sensitivity; mean probability over the panel), per label and operation, after re-scoring the original texts against the committed artifact. Records the panel's training class balance and LR coefficients, per-word substitute deltas, and the device a torch technique ran on (null for techniques that ignore it). No text, no row ids. |
| `run_manifest.json` | `run_manifest.py` | Kaggle notebook version, git commit, hashes of committed files, and the location and hash of every external asset. |
| `faithfulness_test.json` | `faithfulness.py` | Deletion faithfulness of the whole-model test run's per-post SHAP attributions: the deletion rules, k values and random seeds, the word-table and predictor verification, post counts, and the rows written to `rq/rq2_faithfulness.csv`. No text, no row ids, no tokens. |

Cross-technique tables live in `results_summary/rq/`, one per research
question, and are the inputs to the rendered documentation tables. Their
families are the techniques in the comparison: a technique whose
`best_config.json` says `"in_comparison": false` is kept out of every Holm
family, cross-technique count and rendered table (see the two
`_outside_comparison` files).

| File | Written by | Contents |
|---|---|---|
| `rq/rq1_mcnemar_pairs.csv` | `mcnemar_pairs.py` | One row per technique pair on the test split: both-correct, both-wrong and discordant counts, exact McNemar p, Holm-adjusted p, error-set Jaccard. |
| `rq/rq1_mcnemar_pairs_outside_comparison.csv` | `mcnemar_pairs.py` | Only while a technique is outside the comparison: that technique paired with every comparison technique, same columns as `rq1_mcnemar_pairs.csv`, Holm over its own pairs. Removed when no such technique exists. |
| `rq/rq2_word_agreement_pairs_outside_comparison.csv` | `word_agreement.py` | Only while a technique is outside the comparison: its whole-model test run paired with every comparison technique's, same columns as `rq2_word_agreement_pairs.csv`. Removed when no such technique exists. |
| `rq/rq1_error_overlap.csv` | `mcnemar_pairs.py` | Per technique: errors, false positives, false negatives, errors no other technique makes. |
| `rq/rq1_shared_errors.csv` | `mcnemar_pairs.py` | Posts wrong for exactly k techniques, by true label. Counts only; the row-keyed matrix stays under `external/`. |
| `rq/rq2_category_shares.csv` | `categorize_attributions.py` | Share of top-K words and attribution mass per lexicon category, per run. |
| `rq/rq2_word_agreement_pairs.csv`, `rq/rq2_shared_top_words.csv`, `rq/rq2_top_word_support.csv` | `word_agreement.py` | Lexicon-free comparison of the whole-model word tables across techniques: pairwise Spearman and top-K Jaccard, words in several techniques' top-K, and the share of top-K words seen in a single explained post. |
| `rq/rq3_reliance_comparison.csv`, `rq/rq3_stability.csv` | `compare_reliance.py` | Wide per-run shares with member means; across-run stability per technique. |
| `rq/rq4_identity_fpr.csv` | `identity_fpr.py` | False-positive rate on non-extremist posts with vs without identity terms, per technique. |
| `rq/rq4_identity_counterfactual.csv` | `identity_counterfactual.py` | WP7 counterfactual rows (technique x label x operation): mean/median delta, flagged counts before and after, flips, `fp_share_cleared` (share of original false positives that clear), Wilcoxon and exact McNemar p, their Holm adjustments over every comparison row of the table (an outside-comparison technique's rows form their own family), verification max diff. |
| `rq/report_cards/<TECHNIQUE>.md`, `rq/report_cards/<TECHNIQUE>.json` | `audit.py report` | One card per technique (WP8), read from the committed tables above and the technique's own folder only: RQ1 test accuracy and, against every other technique in its family, the row gap, McNemar and Holm p, whether it clears significance, and the Holm family size; the SHAP pipeline check (anchor status, and each whole-model run's coefficient-check figures and PASS / FAIL / calibration point, or "no coefficients"); RQ2, worded according to that check, the top 10 words by positive mean attribution on the whole-model test run with their lexicon category, the empty lexicon categories, and Spearman and top-50 Jaccard agreement with every other technique (most and least agreeing by each metric); RQ2 support (`faithfulness`): the deletion test's `shared` rows per k (n, top-k and random drop, margin, flagged-post margin and n, share of posts where top-k beats random, drop over summed SHAP, Holm-adjusted Wilcoxon p) with the Holm family size, `n_posts_shared` of `n_posts_shared_set` (the shared set's size, read from the comparison JSONs), or "not run"; RQ3 stability across the whole-model runs and their splits; RQ4 FPR with and without identity terms, ratio, Fisher p, n, and `rq4_ambiguous_terms_sensitivity` (the `exclude_ambiguous_only` variant: negative and positive posts excluded, n, identity-term positives, FPRs, ratio, Fisher p); the WP7 remove rows, or "not run", and under `unambiguous_population` the remove rows of `rq4_identity_counterfactual_sensitivity.csv`, or "not run". Each section names its source file. Words only; no row ids, hashes or post text, and the writer refuses a payload carrying either. The cards are regenerated, never edited, and `audit.py report` refuses to write any card while an RQ table lacks a row for a technique, a coefficient check is missing, the sensitivity table's `full_lexicon` row differs from `rq4_identity_fpr.csv`, `faithfulness_test.json` and the technique's `rq2_faithfulness.csv` rows are not both present, with the same (population, k) rows and every numeric field equal, at the locked threshold, from the current comparison's shared population, with Holm over the current family, and present for every comparison technique or none, or a WP7 or restricted counterfactual row is at another threshold or the restricted post counts differ from the sensitivity table's (`docs/FRAMEWORK.md` lists every refusal). |
| `rq/rq2_faithfulness.csv` | `faithfulness.py` | ERASER-style deletion per technique x population (`explained`: every post the technique explained; `shared`: the posts every comparison technique explained) x k in {1, 3, 5, 10}: the post's top-k positive-attribution tokens are deleted and the post re-scored by the restored predictor, against deleting the same number of tokens at random (5 seeds). Mean probability drop for top-k and random (comprehensiveness), their difference, the same on flagged posts, the summed SHAP value of the deleted tokens (top-k and random) and the ratio of the top-k drop to it, the share of flagged posts that fall below the locked threshold, the median paired difference and the share of posts where top-k beats random, the SD of the random drop over seeds, a paired Wilcoxon p (top-k vs mean random drop) and its Holm adjustment over the comparison rows of the same population (`shared` is the primary family, `explained` the secondary one; they overlap and are not corrected jointly; an outside-comparison technique's rows form their own family), verification max diff. Two table-only columns are derived from each row's means: `drop_random_over_attributed_random` (random drop over the summed SHAP of the randomly deleted tokens) and `drop_minus_random_flagged` (the margin on flagged posts, the comparison across families least affected by calibration). `n_posts` varies by technique inside `shared` because posts with no positive token are left out per technique. `faithfulness.py --rebuild-table` rewrites the table from the committed JSONs. Re-scoring test posts with frozen models at locked thresholds is a perturbation analysis; it selects nothing. |
| `rq/rq4_identity_fpr_sensitivity.csv` | `identity_sensitivity.py fpr` | RQ4 sensitivity to the ambiguous identity terms declared in `identity_sensitivity.AMBIGUOUS_IDENTITY_TERMS` (a subset of the lexicon, each with its reason; the lexicon is not edited). Per technique, three variants: `full_lexicon` (equals `rq4_identity_fpr.csv`), `exclude_ambiguous_only` (posts whose only identity matches are ambiguous are dropped; primary) and `reduced_lexicon` (those terms leave the term list, so the posts count as without). Same rate and Fisher columns as RQ4 plus `n_negative_excluded`, `n_positive_excluded`. |
| `rq/rq4_identity_counterfactual_sensitivity.csv` | `identity_sensitivity.py counterfactual` | The WP7 remove counterfactual on the restricted population (test posts with at least one unambiguous identity match): `remove` deletes every identity match as WP7 does, `remove_unambiguous` only the unambiguous ones. Same columns as `rq4_identity_counterfactual.csv` plus `n_posts_full_population` and `flagged_flip_rate` (`n_flip_pos_to_neg / n_flagged_original`, the per-flagged-post effect size to read next to the p value). Holm is a choice of family, reported three ways: `mcnemar_p_holm` over all 32 rows (the same size as WP7 but not the same makeup: WP7 is remove + substitute, here remove and remove_unambiguous are nearly the same edit, so this family is conservative), `mcnemar_p_holm_remove` over the 16 remove rows, `mcnemar_p_holm_remove_extremist` over the 8 extremist remove rows (empty elsewhere). The restored predictor is verified on the restricted originals first. |
| `rq/rq2_identity_lexicon_gaps_sensitivity.csv`, `rq/rq2_identity_top_k_sensitivity.csv` | `identity_sensitivity.py lexicon-gaps` | WP3 candidates: words in each technique's whole-model test top-50 that look like identity references but are not identity-lexicon entries, with the rule that proposed them (`inflection`, `compound`, `group_noun`), the related entry, `related_entry_ambiguous` (the entry is on the ambiguous list; a plural such as `whites` is far less ambiguous than the singular), any other lexicon that matches them, and `low_support` (support of 2 posts or fewer: weak evidence); and per technique the number of top-50 identity-lexicon words, how many are ambiguous, and the candidate count. Vocabulary words only. Nothing is added to the lexicon. |
| `rq/rq2_same_post_support.csv` | `same_post_stability.py` | Per technique x split x post set (`own`: every post the technique explained; `common`: the posts every comparison technique explained, the transformers' 200-post samples): vocabulary size, median support and single-post share of the top-K words by positive mean attribution, and positive-mass concentration over the vocabulary (words with positive mass, fewest words holding half of it, normalized entropy, exp(entropy), top-K share). Re-aggregated from the external per-post files with the kit's aggregation, after checking that each file reproduces its committed word table. Numbers only. |
| `rq/rq3_same_post_split_agreement.csv` | `same_post_stability.py` | Per technique x post set: test-run versus validation-run agreement (top-K Jaccard by positive and by absolute mean attribution, Spearman of mean absolute attribution over shared words); `common` restricts both splits to the common post set so every technique is compared on the same number of posts. Numbers only. |
| `rq/rq2_same_post_category_shares.csv` | `same_post_stability.py` | Per technique x split x post set x category: the RQ2 category shares of the top-K words by mean absolute attribution (word count and share, absolute and positive mass and share), computed with `categorize_attributions.py`'s functions on the re-aggregated word table. `own` rows must reproduce `rq2_category_shares.csv`; `common` rows put every technique on the same posts. Numbers only. |
| `rq/rq2_rq4_relation.csv` | `rq2_rq4_relation.py` | One row per (role, reliance measure, outcome, subset) over the comparison techniques on test. `primary`: test-run identity-term share of positive mass vs RQ4 FPR ratio. `secondary`: WP7 remove mean delta vs FPR ratio. `sensitivity`: the share on the common post set, the validation-run share, the share over only the techniques whose attributions cover the whole test split (`subset` full_split_posts), and the WP7 share of posts whose probability drops, each vs FPR ratio. `descriptive`: identity share (own and common posts) vs FPR with and without identity terms, and identity share vs mean delta. n and the techniques used (non-finite values excluded and named), Spearman rho, two-sided permutation p (exact up to n = 9), the \|rho\| needed for p <= 0.05 at that n, leave-one-out rho range and the technique whose removal gives the minimum, and Holm p over the primary and secondary rows only (empty otherwise). |
| `rq/rq2_rq4_relation_inputs.csv` | `rq2_rq4_relation.py` | One row per comparison technique: test attribution post count, identity-term share (test own posts, common posts, validation), identity-term and other negatives with their false-positive counts and FPRs, FPR ratio, WP7 mean delta and share of posts whose probability drops. Numbers only. |
| `rq/rq2_bootstrap_category_shares.csv` | `bootstrap_uncertainty.py` | Per technique x split x post set (`own`, `common`) x category: the RQ2 share of top-K positive mass with a post-level bootstrap interval. Every row of the five bootstrap tables has `point` (the committed value, reproduced with weight one per post before any replicate; refused otherwise), `interval_kind`, `boot_median`, `boot_mean`, `boot_sd`, percentile bounds `pct_low`/`pct_high` (order statistics when a replicate is infinite; an infinite upper bound is replaced by the largest finite replicate and flagged `pct_high_censored`), `pct_level`, `point_inside_interval`, `share_reps_below_point`, `n_reps`, `n_reps_valid`, `n_reps_infinite`, `seed`. `interval_kind` `percentile_ci`: a smooth statistic, bounds read as an approximate confidence interval; `replicate_spread`: replicates do not centre on the point, bounds are only their spread, never a CI. Replicates resample explained posts with replacement and re-aggregate them with the kit's aggregation; one draw per post set per split is shared by every technique that explained it. The bootstrap covers post sampling only (not SHAP seed, training seed or another split). Descriptive; no test. Numbers only. |
| `rq/rq2_bootstrap_same_post.csv` | `bootstrap_uncertainty.py` | Per technique x split x post set: single-post share of the top-K positive words (one distinct post) and the fewest words holding half the positive mass, under the post bootstrap. Both are discrete top-K statistics that shift under resampling, so every row is `replicate_spread`. Numbers only. |
| `rq/rq3_bootstrap_split_agreement.csv` | `bootstrap_uncertainty.py` | Per technique x post set: test-versus-validation Spearman of mean absolute attribution (`percentile_ci`) and top-K Jaccard, absolute and positive (`replicate_spread`), resampling the two splits' posts independently. Numbers only. |
| `rq/rq2_rq4_bootstrap_relation.csv` | `bootstrap_uncertainty.py` | `spearman_rho` rows: sensitivity of the RQ2-RQ4 ranking (identity share on own and on common test posts vs FPR ratio, n = 8) to resampling posts and test rows (`posts_and_rows`), posts only, or test rows only. `replicate_spread`, not a confidence interval: resampling the noisy FPR ratios attenuates the rank correlation, so most replicates fall below the point; read `boot_median`, `share_reps_below_point` and `share_reps_rho_above_zero_not_a_p_value` (share of replicates with rho > 0, not a one-sided p). Post and row resamples are independent (explained posts are not nested in the resampled rows). `fpr_ratio` rows (`percentile_ci`): each technique's FPR ratio resampled over the 450 test rows of the committed probability artifacts at the locked threshold; `n_reps_infinite` counts replicates with no false positive on posts without identity terms. Descriptive. Numbers only. |
| `rq/rq2_rq3_bootstrap_paired_differences.csv` | `bootstrap_uncertainty.py` | Contrast technique (07, 11) minus each other comparison technique on the common test posts, with the same replicate draws on both sides: identity-term share of positive mass and test-versus-validation Spearman (`percentile_ci`), single-post share (`replicate_spread`). `role` is `descriptive_post_sampling_only`: no Holm family, no significance claim, and not the section 5 difference criterion, which uses the across-run SD of `rq3_stability.csv`. Numbers only. |

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

### Curated examples (local, untracked)

`tools/curated_examples.py` classifies and explains a hand-curated list of
texts (one per line of a `.txt` file) with every technique, from the
downloaded `external/<TECHNIQUE>/` folders and the committed locked
thresholds. It restores each notebook's `predict_proba_texts`
(`tools/model_loaders.py`), verifies the restored model against the
technique's committed test probability artifact, and runs the kit's one
explainer (`notebook_kit.explain`) on the curated texts. Its outputs live
under `external/curated_examples/` and are never committed:

```text
external/curated_examples/
├── examples.csv                       example_id, text, sha256[, expected_label, note]
├── <TECHNIQUE>/predictions.csv        example_id, technique, member, y_prob, threshold, y_pred,
│                                      base_value, explained_output
├── <TECHNIQUE>/local_attributions.csv example_id, technique, member, position, token, attribution, category
├── <TECHNIQUE>/run.json               sources, verification, explainer settings
├── summary.csv                        text-free: per example x technique x member, category shares
│                                      [, expected_label, correct]
├── results.json                       one record per example: text, label, note, every model
├── report.html                        every text under every model, words highlighted
└── plots/                             tools/curated_plots.py: waterfalls, beeswarms, gallery
```

`base_value` is the model on the fully masked text and `explained_output`
the model on the text as SHAP's masker rebuilt it, the two endpoints the
partition explainer anchors on: `base_value` plus a text's attributions
equals `explained_output`. The rebuilt text drops trailing separators (a
final '.' or '?'), so `explained_output` can differ from `y_prob`, which is
the model on the raw text and decides `y_pred`. `tools/curated_plots.py`
refuses outputs where the sum does not hold and draws whole-model waterfalls
and beeswarms from the recorded attributions, without re-explaining; its
`plots/checks.csv` is text-free and records both gaps.

`tools/curated_site.py` copies that gallery into a static site folder
(default `external/curated_shap_site/`, which can be its own repository)
with `index.html` as the front page, a robots `noindex` meta on every page,
and a `method.html` whose numbers are read from the run's `run.json` files,
the committed `best_config.json` files, `plots/checks.csv` and the dataset
manifest. The site holds the curated texts: nothing from it is committed
here, and publishing it is a separate step. The current site is served at
<https://73anthonyl.github.io/curated-shap-gallery/>.

`tools/split_beeswarms.py` draws a word beeswarm per technique and split
from the per-post attributions under `external/<TECHNIQUE>/attributions_local/`
(test and validation; all posts for 01 to 06, the notebooks' sample for 07
and 11), coloured by the true label. It first rebuilds each committed word
table in `results_summary/<TECHNIQUE>/attributions/` from its per-post file
with `notebook_kit.aggregate_word_attributions` and refuses a file that does
not reproduce it. Its outputs (`external/split_beeswarms/`: PNGs and a
`checks.csv` of technique, split, n_posts, n_words, max_table_diff) hold
words and values only; `curated_site.py --dataset-beeswarms` adds them to
the site as `dataset.html`.

The curated texts are illustration for RQ2-RQ4. They are not dataset rows,
carry no dataset label (only an optional hand-assigned expected label), and
produce no committed artifact; nothing in the paper's tables derives from
them.

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
