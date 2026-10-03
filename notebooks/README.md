# Notebooks folder

This folder contains the research notebooks for dataset preparation, model
training, evaluation, and attribution. Every notebook follows one template,
loads one shared helper (`tools/notebook_kit.py`), and writes its outputs in
one layout, so techniques can be compared like for like.

Notebooks run on Kaggle. They are never executed locally. Before a notebook is
uploaded, and again before its outputs are committed, it is verified
statically with `tools/check_notebook.py`.

## State of the notebooks

All twelve notebooks are **primed**: the code is rewritten on the template and
ready to rerun, and the cell outputs are empty until each one is rerun on
Kaggle. The outputs of the earlier runs are kept in git history (see
[Earlier outputs](#earlier-outputs)).

### Foundation

| Notebook | Purpose |
|---|---|
| `00_create_dataset_and_splits.ipynb` | Builds the processed dataset, reproduces the frozen stratified split, and verifies it against the frozen class counts and the committed `splits/split_assignments.csv`. Every model notebook attaches its output. |

### Controlled comparison

These produce the results the paper reports. Each exports a probability
artifact, two whole-model attribution runs, and a result folder.

| Notebook | Technique | Role in the research questions |
|---|---|---|
| `01_LOG-REG_TF-IDF.ipynb` | Logistic regression over word TF-IDF. | Classical baseline; the reference for the SHAP pipeline check. Also exports `coefficients.csv`. |
| `02_LIN-SVM_TF-IDF.ipynb` | Calibrated linear SVM over word TF-IDF. | Classical baseline; same features as 01, different loss. |
| `03_SLP_TF-IDF.ipynb` | Single-layer perceptron over word TF-IDF. | Classical baseline; same model class as 01, trained by gradient descent with early stopping. |
| `04_CHAR-TF-IDF_LIN-SVM.ipynb` | Calibrated linear SVM over character n-gram TF-IDF. | Tests whether sub-word features change word-level reliance. |
| `05_WORD-CHAR-TF-IDF_LIN-SVM.ipynb` | Calibrated linear SVM over word plus character TF-IDF. | The hybrid between 02 and 04. |
| `06_FASTTEXT-EMB_LOG-REG.ipynb` | Logistic regression over pooled FastText vectors. | Static-embedding control between lexical models and transformers. Exports a word-level projection of its coefficients as a second calibration point. |
| `07_TWITTER-ROBERTA_FINE-TUNE.ipynb` | Fine-tuned Twitter-RoBERTa (hate-speech pretrained). | The single-transformer reference. |
| `11_MULTI-CHECKPOINT_LOGIT-POOL.ipynb` | Mean log-odds pool over heterogeneous fine-tuned checkpoints, three seeds each. | The ensemble. Its per-component attribution runs are how pretraining lineages are compared (RQ3). |

### Closed lines

These explored transformer ensembling for accuracy. They are not part of the
controlled comparison (`docs/RESEARCH_PLAN.md`, section 6) and their earlier
test numbers are not cited. They are primed to the same standard and declare
`"in_comparison": False`, so any of them can be rerun if its question is
reopened. A rerun replaces the earlier test evaluation, which left no
committed artifact.

| Notebook | Technique | Why it is closed |
|---|---|---|
| `08_TWITTER-ROBERTA_SEED-ENSEMBLE.ipynb` | Mean of probabilities over three seeds of the notebook 07 checkpoint. | The ensemble family is represented by 11. Its members share one lineage, so it would isolate the effect of averaging over seeds. |
| `09_MULTI-CHECKPOINT_LOGIT-STACK.ipynb` | Learned stacker over four checkpoints with train-side cross-validation. | Built and preregistered, never run. Superseded by 11, which kept the multi-checkpoint idea and replaced the learned stacker with an equal-weight pool. |
| `10_TWITTER-ROBERTA_LOGIT-POOL-STABLE.ipynb` | Mean log-odds pool over three seeds. A guarded blend with a character TF-IDF model is implemented and switched off in `CONFIG`, as it was in the earlier run. | Its pooling recipe is what 11 adopted. |

## The template

Cell 0 is a markdown title block: the technique (the filename stem), its role,
the research questions it serves, the Kaggle inputs to attach, the
accelerator, and what goes to the repository and what goes to external
storage. Then the numbered sections, each a markdown header followed by code.

| Section | What it does | What it never does |
|---|---|---|
| 1. Bootstrap | Loads `tools/notebook_kit.py` as `nk` with the canonical cell, then imports. | Edit `sys.path`. |
| 2. Configuration | A literal `CONFIG` dict, then `ctx = nk.bootstrap(CONFIG)`, which validates it, seeds every generator and creates the output folders. | Abbreviate `split_version`; compute `technique_name`. |
| 3. Load foundation and assert the frozen split | `nk.load_foundation` and `nk.split_frames`. Stops unless the class counts are the frozen ones. | Show rows. |
| 4. Features and model | Definitions only, including `predict_proba_texts(texts)`: raw strings to P(EXTREMIST) through the fitted model. Ensembles also define `member_predict_fns`. | Fit anything; touch the test split. |
| 5. Ablation on validation | Compares configurations on validation. Every row of `ablation_results` comes from `nk.evaluate(..., "validation")`. | Touch the test split. |
| 6. Final fit | Trains the selected configuration; saves weights under `external/`. | Compute a test metric. |
| 7. Threshold on validation | The one `nk.select_threshold` call, on validation, from the shared grid. | Touch the test split. |
| 8. Single test evaluation | The one `nk.evaluate(..., "test")` call, and the diagnostic figure. | Evaluate test a second time. |
| 9. Probability artifacts | `nk.export_probabilities`: exactly `row_id, split, y_true, y_prob`. | Add a column. |
| 10. Word-level SHAP attribution runs | `nk.explain` and `nk.export_attribution_run`: one run over the test split, one over validation, and one per member for ensembles. Logistic-regression notebooks also export coefficients. | Draw per-post plots. |
| 11. Result folder | `nk.export_results_folder`, from the metrics computed above. | Hold a typed number. |
| 12. External assets | Per-post predictions and the outcome-by-confidence table. | Display a per-post file. |
| 13. Interpretation | How to read the outputs, and the limitations. Written before the run. | Quote results or rank techniques. |
| 14. Package outputs | `nk.finalize(ctx)`. | Anything else. |

Notebook 00 has seven sections: Bootstrap, Configuration, Load raw dataset,
Build processed dataset, Create and verify the split, Foundation summaries,
Package outputs.

Print the section list and the canonical bootstrap cell for a role with:

```bash
python3 tools/check_notebook.py --print-template classical   # or transformer, ensemble, foundation
```

## Conventions

* `CONFIG["technique_name"]` is exactly the filename stem, numeric prefix
  included. `split_version` is the full frozen string
  `split_v1_stratified_70_15_15_seed30`.
* **No cell prints, displays, plots, or raises with dataset text, row ids, or
  text hashes.** Outputs are committed so a reader can follow each notebook.
  Show counts, metrics, configurations, and word-level tables.
  `include_text_preview` stays `False`.
* `predict_proba_texts` is the single entry point for the validation
  threshold, the test evaluation, and every explainer run. The model that is
  explained is the model that predicts.
* Configurations are compared on validation at a screening threshold of 0.5.
  The decision threshold is then selected once on validation, maximizing
  accuracy over the shared grid (0.05 to 0.95 in steps of 0.005).
* The test split is evaluated once per technique. The kit raises on a second
  evaluation in the same run, and the checker refuses a notebook with two.
* Every attribution comes from one method: SHAP's partition explainer over a
  word-level text masker, applied to `predict_proba_texts`. Classical
  notebooks explain every post of a split; transformer and ensemble notebooks
  explain a label-stratified sample of 200 posts per split, recorded in each
  sidecar. No other attribution method produces a committed artifact.
* Every use of the validation split (early stopping, configuration selection,
  admission, strategy selection, threshold) is declared in `best_config.json`.
* Training is seeded and deterministic flags are set by the kit. Loaders use a
  seeded shuffle and no workers.
* Never encode a claim in a notebook name. Name the technique for what it is.

## Running a notebook on Kaggle

1. **Attach the repository.** Create a Kaggle Dataset from this repository (or
   from `tools/` alone) and attach it to the kernel. It provides
   `tools/notebook_kit.py` and, for the split check, the committed
   `splits/split_assignments.csv`. Update the dataset when the kit changes;
   the kit prints its version and hash in section 2 and stamps the version
   into every artifact.
2. **Run notebook 00 first** and save the version. Attach its output (the
   `research_foundation/` folder) to every model notebook.
3. **Load the primed code into the kernel** before running it. Kaggle's
   GitHub sync writes the whole notebook file on each push, so a kernel that
   still holds the old code would overwrite the primed notebook here.
4. Turn the GPU and internet on for notebooks 07 to 11 (they download
   checkpoints from the Hugging Face Hub).
5. Run all cells. Section 14 writes `<TECHNIQUE>_repo_files.zip`.

After the run:

```bash
unzip <TECHNIQUE>_repo_files.zip -d .          # results_summary/ and research_loop/probs/
python3 tools/run_manifest.py init --technique <TECHNIQUE> --run-id <run_id> \
    --kaggle-notebook <url> --kaggle-version <n> --git-commit <sha>
python3 tools/run_manifest.py add-assets-from --technique <TECHNIQUE> \
    --file external_assets.json --location-prefix kaggle://datasets/<user>/<slug>/v1
python3 tools/validate_results_folder.py --all
python3 tools/validate_shap.py --technique 01_LOG-REG_TF-IDF   # after notebook 01
python3 tools/identity_fpr.py --all
python3 tools/categorize_attributions.py --all
python3 tools/compare_reliance.py
python3 tools/render_tables.py --write
```

`external/<TECHNIQUE>/` (weights, per-post predictions, per-post
attributions) is uploaded to a Kaggle Dataset. It is never committed.

Before committing a notebook with its outputs:

```bash
python3 tools/check_notebook.py --notebook notebooks/<file>.ipynb
python3 tools/scan_text_leakage.py
```

Run order: 00, then 01 (and its `validate_shap.py` check, which gates every
cross-model attribution claim), then 02 to 07 and 11. Notebooks 08 to 10 are
optional.

## Earlier outputs

The notebooks as they were before priming, with the cell outputs of their
earlier Kaggle runs, are at commit `5e185d01`, tagged `notebooks-pre-priming`.
To open one:

```bash
git show 5e185d01:notebooks/07_TWITTER-ROBERTA_FINE-TUNE.ipynb > /tmp/07_pre-priming.ipynb
```

Notebook 08 has its earlier name there, `08_BEST-ROBERTA_SEED-ENSEMBLE.ipynb`.
Those outputs contain dataset text, which is why they are kept in history and
not in the working tree.

## Numbering

Numbers reflect the intended order and are never reused, even when a notebook
is superseded, so result folders, artifacts, and the records in
`research_loop/` keep pointing at the same thing.

```text
00  dataset and split preparation
01  logistic regression, word TF-IDF
02  linear SVM, word TF-IDF
03  single-layer perceptron, word TF-IDF
04  linear SVM, character TF-IDF
05  linear SVM, word + character TF-IDF
06  logistic regression, FastText embeddings
07  fine-tuned transformer
08  seed ensemble of the transformer            (closed line)
09  multi-checkpoint stacking                   (closed line, superseded by 11)
10  logit-pooled seed ensemble                  (closed line)
11  heterogeneous multi-checkpoint logit pool
```

New notebooks take the next free number and the pattern
`<NUMBER>_<REPRESENTATION>_<MODEL>.ipynb`, or the model family and training
method when the two are bundled, as in `07_TWITTER-ROBERTA_FINE-TUNE.ipynb`.
