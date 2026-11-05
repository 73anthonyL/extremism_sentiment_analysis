# Auditing a new classifier

This guide is for researchers who want to compare what a text classifier for
violent-extremist content relies on with the eight techniques already in this
repository. It is not a moderation tool: nothing here decides about a post,
an account or a community, and a report card is not evidence that a model is
safe to deploy. The contribution it serves is contribution C1 in
`docs/POSITIONING.md`, section 2.

A classifier enters the framework by meeting one contract. A single entry
point, `tools/audit.py`, then checks the contract, runs the existing tool chain
in the documented order and writes one text-free report card. It adds no
analysis of its own; every number comes from a tool described in
`docs/RESULTS_SCHEMA.md`.

## The contract

| Item | Where it lives | Required | Checked by |
|---|---|---|---|
| Test and validation probabilities on the frozen split: exactly `row_id, split, y_true, y_prob`, 450 rows each with the frozen class counts | `research_loop/probs/<TECHNIQUE>__{test,validation}.csv` | yes | `probs_artifact.load_probs` |
| A threshold selected on validation only | `results_summary/<TECHNIQUE>/best_config.json`, key `selected_threshold` | yes | `audit.py check-contract`, then `validate_results_folder.py` |
| At least one whole-model word-level SHAP run on test posts | `results_summary/<TECHNIQUE>/attributions/<run_id>.csv` + `<run_id>.json`, `split: test`, `member: null` | yes | `attributions.load_run` |
| At least one whole-model word-level SHAP run on validation posts | as above, `split: validation` | yes | `attributions.load_run` |
| Coefficients over words, for a linear model | `attributions/coefficients.csv` (`word, coefficient`) | no; if present it must be valid | `validate_shap.load_coefficients` |
| A restorable `predict_proba_texts` | `external/<TECHNIQUE>/weights/` (untracked) | no; needed only for `--with-perturbation` (the identity counterfactuals and the faithfulness test) and curated examples | `model_loaders.detect_family`, then verified against the test artifact |
| Per-post attributions of the whole-model runs (`row_id, position, token, attribution`) | `external/<TECHNIQUE>/attributions_local/<run_id>.csv` (untracked; written by `nk.export_attribution_run`) | no; needed only for the faithfulness test and the same-post tables | each tool re-aggregates the file and refuses it unless it reproduces the committed word table |

Without the two attribution runs a technique cannot take part in RQ2 to RQ4,
whatever its accuracy, and `check-contract` says so.

### In or outside the comparison

`best_config.json` carries `in_comparison`, copied from `CONFIG`. Set it to
`false` for a classifier you are auditing. Every tool then keeps it out of the
paper's families: its McNemar pairs go to
`rq/rq1_mcnemar_pairs_outside_comparison.csv` and its word-agreement pairs to
`rq/rq2_word_agreement_pairs_outside_comparison.csv`, each Holm-corrected over
its own pairs only; its counterfactual rows (WP7 and the ambiguous-term
sensitivity) and its faithfulness rows are each a Holm family of their own; and
it is left out of every rendered table in `README.md` and the docs. Its own
rows in the per-technique tables (category shares, RQ3 stability, RQ4 FPR, the
ambiguous-term FPR, the lexicon-gap candidates) are added, which changes no
other technique's numbers. Two analyses have no outside family and skip it:
`same_post_stability.py` defines the common post set over the comparison, and
`rq2_rq4_relation.py` ranks the comparison techniques only. Its card says it
is outside the comparison.

Setting `in_comparison: true` adds the technique to the paper's comparison, and
that changes other techniques' results. The Holm family grows (from 28 pairs
for eight techniques to 36 for nine), so pairs near alpha can flip in the other
cards. The error-overlap and shared-top-word counts change too, and so do the
counterfactual and faithfulness Holm p values, the common post set of the
same-post tables and the RQ2-RQ4 rank relation. Only do it as a decision recorded in
`docs/RESEARCH_PLAN.md`. Then recheck the prose claims that quote these
numbers, because `render_tables.py --check` cannot see them:
`RESEARCH_PLAN.md` section 3 (the McNemar bullet, "Seven of 28 pairs", and
the WP7 bullet), `docs/POSITIONING.md` (the two "7 of 28 pairs" sentences) and
the WP7 work package in the plan.

## From a notebook to the contract

Start from the template and follow `notebooks/README.md`:

```bash
python3 tools/check_notebook.py --print-template classical   # or transformer, ensemble
```

Every export goes through `tools/notebook_kit.py`, loaded as `nk`. Do not copy
kit functions into the notebook.

| Section of the template | Kit call | What it writes |
|---|---|---|
| 2. Configuration | `nk.bootstrap(CONFIG)` | Seeds, output folders, the context `ctx` |
| 3. Load foundation | `nk.load_foundation`, `nk.split_frames` | Nothing; stops unless the class counts are frozen |
| 4. Features and model | (your code) `predict_proba_texts(texts)` | The one function the threshold, the test evaluation and every explainer run use |
| 5. Ablation on validation | `nk.evaluate(..., "validation")` and `nk.ablation_row` per configuration, then `nk.select_best_config` | The ablation table. At least one row is required: `nk.export_results_folder` refuses an empty one, so a single fixed configuration still gets one row. Validation columns only |
| 6. Final fit | (your code) | The fitted model behind `predict_proba_texts` |
| 7. Threshold on validation | `nk.select_threshold` | The locked threshold |
| 8. Single test evaluation | `nk.evaluate(..., "test")`, once | Test metrics |
| 9. Probability artifacts | `nk.export_probabilities` | `research_loop/probs/<TECHNIQUE>__{validation,test}.csv` and `__meta.json` |
| 10. Attribution runs | `nk.sample_for_explanation`, `nk.explain` on `predict_proba_texts`, then `nk.export_attribution_run`: one test run, one validation run, one per member for ensembles. Logistic-regression notebooks also call `nk.export_coefficients` | `attributions/<run_id>.csv` + `.json` (and `coefficients.csv`) |
| 11. Result folder | `nk.export_results_folder` | `best_config.json`, `metrics_*.json`, the confusion matrix, the sweeps |
| 12. External assets | `nk.export_external_predictions`; weights saved under `external/` in section 6 | Per-post files, never committed |
| 13. Interpretation | (prose) | Counts, metrics and word tables only; no post text, row ids or hashes in any output |
| 14. Package outputs | `nk.finalize(ctx)` | `external_assets.json` and `<TECHNIQUE>_repo_files.zip` |

**How many posts to explain.** `CONFIG["explainer"]["n_posts_per_split"]` sets
it. `None` explains every post of the split (the classical notebooks here,
450 per split); an integer explains a label-stratified sample of that size,
drawn by `nk.sample_for_explanation` (200 in the transformer notebooks, the
default in `notebooks/README.md`). The choice changes the support counts in
RQ2, the RQ3 stability numbers and the coefficient check, so report it: every
card gives `n_posts_explained`.

**After the run.** Follow "After the run" in `notebooks/README.md`: unzip
`<TECHNIQUE>_repo_files.zip` at the repository root, then
`run_manifest.py init` and `run_manifest.py add-assets-from` for the external
folder you upload to a Kaggle Dataset. Unzipping puts the probability
artifacts in `research_loop/probs/`; that is the sanctioned way into
`research_loop/`, which is otherwise never edited by hand. Check the notebook
statically with `python3 tools/check_notebook.py` before committing; notebooks
are never executed locally.

### Running the kit outside Kaggle

`nk.load_foundation` reads notebook 00's `research_foundation/` output, which
is not committed because `processed_dataset.csv` carries post text. Rebuild it
into a scratch folder outside the repository (or under the gitignored
`external/`; the tool refuses any other path in the repository):

```bash
python3 tools/local_foundation.py --out /tmp/kit_inputs
```

It rebuilds the processed dataset as notebook 00 does, copies
`splits/split_assignments.csv` and `results_summary/foundation/dataset_manifest.json`
next to it, and checks the three agree before writing. Then point the kit at
it and keep its outputs out of the repository too:

```python
ctx = nk.bootstrap(CONFIG, working_root="/tmp/kit_output", input_roots=["/tmp/kit_inputs"])
```

Without `working_root` the kit writes to `./kit_output`, which inside the
repository would put per-post files in the tree. After the run, unzip
`/tmp/kit_output/<TECHNIQUE>_repo_files.zip` at the repository root as above.

## Running the audit

From the repository root, with `source .venv/bin/activate`:

```bash
python3 tools/audit.py check-contract --technique <TECHNIQUE>
python3 tools/audit.py run --technique <TECHNIQUE>
python3 tools/audit.py report --all
```

`check-contract` prints one line per item: `OK`, `MISSING` or `INVALID` with
the validator's message. It exits 1 if a required item fails.

`run` checks the contract first, then runs these stages and stops at the first
one that fails, naming it:

1. `validate_results_folder.py --technique T`
2. `validate_shap.py` on every whole-model run (test and validation), if
   `coefficients.csv` exists. The anchor `01_LOG-REG_TF-IDF` is the pipeline
   gate: its committed checks must pass before any other technique is
   audited, and a FAIL on the anchor stops the chain. For any other technique
   a FAIL is recorded, not fatal (see below).
3. `categorize_attributions.py --technique T`
4. `compare_reliance.py`
5. `identity_fpr.py --technique T`
6. `identity_sensitivity.py fpr --technique T`: RQ4 again without the
   non-extremist posts whose only identity matches are ambiguous words
   (colour words, interjections; the list is declared in the tool, the
   lexicon is unchanged). Tells you whether the RQ4 ratio rests on posts
   that may not mention a group at all.
7. `mcnemar_pairs.py`
8. `word_agreement.py`
9. `identity_sensitivity.py lexicon-gaps --technique T`: words in the
   technique's top 50 that look like identity references but are not in the
   identity lexicon (plurals, hashtags, group nouns). A list for human review
   before trusting the RQ2 identity share; nothing is added to the lexicon.
10. `same_post_stability.py`: support, category shares and test-versus-
    validation agreement recomputed on the posts every comparison technique
    explained, so a 450-post run and a 200-post run are compared on the same
    posts. Skipped, with the reason, when any comparison technique's per-post
    files are absent, and for a technique outside the comparison.
11. `identity_counterfactual.py --technique T` (WP7)
12. `identity_sensitivity.py counterfactual --technique T`: the WP7 removal on
    the posts that keep an unambiguous identity match.
13. `faithfulness.py --technique T`: deletes each post's top-k words by
    positive attribution and compares the drop in p(extremist) with deleting as
    many random words. Tells you whether the RQ2 word list names words the
    model needs, which every cross-model RQ2 claim assumes. Skipped, with the
    reason, when the technique's or any comparison technique's per-post test
    file is absent (the `shared` population is the posts they all explained),
    or when the whole-model test run is not named `shap-partition_test_seed30`.
14. `rq2_rq4_relation.py`: Spearman between the comparison techniques' RQ2
    identity share and RQ4 FPR ratio, with exact permutation p. Skipped for a
    technique outside the comparison, and while the WP7 or same-post table
    lacks a comparison technique (the relation table is then stale, and the
    stage line says so).
15. `refresh_manifest`: when the chain wrote files into the technique folder
    (coefficient checks, category shares, identity-FPR, counterfactual and
    faithfulness JSONs) and a `run_manifest.json` exists, the manifest is
    refreshed for them. The folder matched its manifest at stage 1, so only
    tool outputs are recorded. When nothing changed the manifest is left alone.
16. `render_tables.py --write`, whose stage line says which regions it rewrote
17. `audit.py report` for every technique, because a technique in the
    comparison changes the others' RQ1 and RQ2 rows.

Stages 11 to 13 restore the predictor from `external/<TECHNIQUE>/` and
re-score edited test posts, so they run only with `--with-perturbation`
(`--with-counterfactual` is an alias); without it they print `SKIP ... not
requested`. With it, `check-contract` must find a restorable predictor. They
run one after another, one model in memory at a time; pass `--device` and
`--sequential-checkpoints` through for large ensembles. Re-scoring frozen
models at their locked thresholds selects nothing, so it is a perturbation
analysis, not a second test evaluation.

Any error in a stage, including an unexpected exception, stops the chain with
`STOPPED at stage <name>`. A stage whose inputs are absent does not fail and
does not pass: it prints `SKIP  <name>: <reason>`, and the last line lists
every skipped stage. A table such a stage would have written is left as it
was; the report card then says "not run" for faithfulness and WP7.

The chain is deterministic: on the committed state, `run --technique
01_LOG-REG_TF-IDF` rewrites every table, coefficient check and card byte for
byte. With `--with-perturbation` every table and card is reproduced too; the
one difference is the `holm_adjusted_p` note in
`identity_counterfactual_test.json`, whose wording the tool changed after the
committed WP7 runs (the numbers are identical). The coefficient checks cover
every whole-model run of 01 and 06 (test and validation); all four are
committed.

### If the SHAP check fails

The coefficient check passes at Spearman of at least 0.80 between mean
attribution and coefficient over all shared words, and sign agreement of at
least 0.90 on the top 50 words by |coefficient| (the preregistered bar in
`RESEARCH_PLAN.md`, RQ2). Words seen in a single explained post have a
one-sample mean, so the check also records the Spearman over words with
support of at least 2. That figure is a diagnostic only; it does not decide
PASS or FAIL.

For a technique other than the anchor, a FAIL does not stop the chain. It is
written to `<run_id>__coefficient_check.json` and shown on the card's "SHAP
pipeline check" section, and the card's RQ2 section then says that no
cross-model claim about the technique should rest on its word list. What to do:

* Report it. Give the all-words and support >= 2 figures. A near-miss with
  full top-50 sign agreement and a high support >= 2 Spearman usually comes from
  single-post words, not from a broken explainer.
* Explain more posts (`n_posts_per_split: None`), which raises support.
* Check the model. Very weak regularisation, or a vocabulary dominated by rare
  words, gives coefficients that SHAP on a few posts cannot track. Retuning
  regularisation on validation is a legitimate fix, provided the threshold is
  locked again on validation.

Do not change the explainer: `nk.explain` is the one attribution method for
every technique. `06_FASTTEXT-EMB_LOG-REG` is labelled a calibration point
rather than a FAIL because its coefficients act on embedding dimensions, not
words (`RESEARCH_PLAN.md`, WP2).

### Before committing

Stage the new files first. `scan_text_leakage.py` reads tracked files only, so
untracked outputs would pass unread:

```bash
git add -N results_summary/<TECHNIQUE>/ research_loop/probs/<TECHNIQUE>__* results_summary/rq/
python3 tools/check_notebook.py --all      # notebooks carry outputs; --primed is for unrun code only
python3 tools/validate_results_folder.py --all
python3 tools/render_tables.py --check
python3 tools/scan_text_leakage.py
python3 -m pytest tools/tests/ -q
```

`report` reads committed files only. It refuses to write any card when an RQ
table has no row for a technique, a pair table does not pair it with every
other technique in the comparison, `coefficients.csv` exists without a check
file for each whole-model run, the ambiguous-term FPR table's `full_lexicon`
row differs from `rq4_identity_fpr.csv`, or `faithfulness_test.json` and the
technique's rows in `rq2_faithfulness.csv` do not both exist and agree (or
were computed at a threshold other than the locked one). Each means the inputs
are stale; rerun the chain rather than reporting a partial card. A technique
with neither faithfulness file reads "not run".

## Reading a report card

Cards are at `results_summary/rq/report_cards/<TECHNIQUE>.md`, with the same
numbers in `<TECHNIQUE>.json`. Each section names the file it was read from.

* **RQ1.** Test accuracy, then one row per other technique: the row gap (this
  technique's correct rows minus the other's), the discordant count, the exact
  McNemar p and its Holm adjustment. The card names the Holm family: the
  comparison's pairs, or the technique's own pairs when it is outside the
  comparison. A gap that does not clear Holm reads "not distinguishable at
  n = 450". A gap that does is a measured difference on this split, not a
  ranking; the card never says "better".
* **SHAP pipeline check.** Whether the anchor passes, and for a technique with
  coefficients, each run's Spearman (all shared words and support >= 2), sign
  agreement, and PASS, FAIL or calibration point. "No coefficients" means the
  technique's attributions rest on the anchor's check.
* **RQ2.** How to read the words given the SHAP check, then up to ten words
  with positive mean attribution in the whole-model test run, with their
  lexicon category and support (the number of explained posts containing the
  word). A word with support 1 comes from one post. Then word-level agreement
  with every other technique: Spearman over shared words, and Jaccard over the
  top-50 positive words. The most and least agreeing techniques are given for
  each metric, since the two can disagree.
* **RQ2 support (faithfulness).** If the deletion test was run: per k, on the
  `shared` posts, the mean drop in p(extremist) when the top-k words are
  deleted, the same for k random words, their difference (the margin), the
  margin on flagged posts, the share of posts where top-k beats random, the
  drop over the summed SHAP of the deleted words, and the Holm-adjusted
  Wilcoxon p; the card names the Holm family. A positive margin with a small
  p means the word list names words the model needs, not that the model is
  right to need them. The all-post means are capped by how close the model's
  probabilities sit to 0, so compare techniques on the flagged-post margin,
  and only descriptively. Otherwise "not run".
* **RQ3.** Agreement between the whole-model runs on different post samples
  (the card names the splits).
  Low overlap means per-word claims about this model are fragile.
* **RQ4.** False-positive rate on non-extremist test posts with and without an
  identity term, the ratio, Fisher's exact p and the group sizes. This is a
  correlation. A second line gives the same figures after dropping the posts
  whose only identity matches are ambiguous words; if the ratio holds there,
  it does not rest on colour words or interjections.
* **WP7.** If the counterfactual was run: the mean drop in p(extremist) when
  identity terms are removed, the flips at the locked threshold, and the Holm
  p values. This is the intervention behind RQ4. Rows marked `unambiguous
  match` repeat it on the posts that keep an unambiguous identity match
  (their Holm family is the sensitivity table's own rows). Otherwise "not
  run".

The lexicon-gap candidates, the same-post tables and the RQ2-RQ4 relation
are not on the card: the first is a review list, the other two compare
techniques rather than describe one. Read them in `results_summary/rq/`
(`docs/RESULTS_SCHEMA.md` gives every column).

The categories are the four in `data/lexicons/`: extremist framing, identity
term, slur and topical. The slur list is empty today, so slurs count as
topical (`RESEARCH_PLAN.md`, WP3). Each card says so in its RQ2 section, read
from the lexicons when the card is written.

## Limits

* Labels are post-level and features are text-only. The framework audits
  detectors of extremist content in single posts, not of accounts or of
  radicalization over time.
* One dataset and one frozen split. Nothing here shows how a model behaves on
  another platform or period.
* The test split has 450 rows. About 14 rows is the rough floor for an
  uncorrected McNemar test; after Holm correction over the family the floor is
  higher, and only the Holm column decides. Most gaps between similar models
  will not clear it.
* Word-level SHAP describes what moves a model's output on masked text. It is
  not a causal claim about language, and it cannot separate quotation,
  reporting or counter-speech from endorsement.
* Category shares depend on lexicon curation; changing a lexicon changes RQ2
  and RQ4.
