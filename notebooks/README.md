# Notebooks folder

This folder contains the research notebooks for dataset preparation, model training, evaluation, and analysis.

## Controlled comparison notebooks

These produced the registered results in `results_summary/`.

| Notebook | Purpose |
|---|---|
| `00_create_dataset_and_splits.ipynb` | Validates the dataset, creates processed artifacts, and writes fixed split assignments and foundation summaries. |
| `01_LOG-REG_TF-IDF.ipynb` | Logistic Regression baseline using word-level TF-IDF features. |
| `02_LIN-SVM_TF-IDF.ipynb` | Calibrated Linear SVM baseline using word-level TF-IDF features. |
| `03_SLP_TF-IDF.ipynb` | Single-Layer Perceptron baseline using word-level TF-IDF features. |
| `04_CHAR-TF-IDF_LIN-SVM.ipynb` | Calibrated Linear SVM using character-level TF-IDF features. |
| `05_WORD-CHAR-TF-IDF_LIN-SVM.ipynb` | Calibrated Linear SVM using combined word + character TF-IDF features. |
| `06_FASTTEXT-EMB_LOG-REG.ipynb` | Logistic Regression using FastText document embeddings trained from the training split. |
| `07_TWITTER-ROBERTA_FINE-TUNE.ipynb` | Fine-tuned Twitter-RoBERTa transformer with token-attribution outputs for explainability. |

**Warning:** `00_create_dataset_and_splits.ipynb` sets
`overwrite_existing_split: True`. Set it to `False` before running the notebook
against this repository's `splits/` directory. The split *assignment* must never
be regenerated.

## Candidate notebooks

These explore transformer ensembling. Only `11` has a result folder.

| Notebook | Purpose | State |
|---|---|---|
| `08_BEST-ROBERTA_SEED-ENSEMBLE.ipynb` | Probability-averaged seed ensemble of the fine-tuned transformer. | Ran; regressed against `07`. No result folder, no probability export. Should be renamed `08_TWITTER-ROBERTA_SEED-ENSEMBLE.ipynb` — the `BEST` is a claim its own numbers contradict. |
| `09_MULTI-CHECKPOINT_LOGIT-STACK.ipynb` | Learned stacker over several transformer checkpoints. | Built; never run. Superseded by `11`. |
| `10_TWITTER-ROBERTA_LOGIT-POOL-STABLE.ipynb` | Mean-log-odds seed pooling with exact probability-change threshold intervals. | Ran; evaluated test but left no derivable artifact. |
| `11_MULTI-CHECKPOINT_LOGIT-POOL.ipynb` | Mean-log-odds pooling over the admitted subset of five checkpoints spanning fine-tuning lineage, pretraining corpus, architecture/tokenizer, and scale. | Ran; 409/450 on test. Its probability artifacts are committed and its result folder is derived from them, so it appears in the comparison tables. Closest template for the conventions below. |

## Notebook conventions

Notebooks are committed **with** their cell outputs so a reader can follow
each one like a chapter. That makes what a cell prints a publication decision.
Each model notebook should:

* State the technique name near the top, and set `CONFIG["technique_name"]` to
  exactly the filename stem, numeric prefix included.
* Declare the full frozen `split_version` string
  `split_v1_stratified_70_15_15_seed30`, not an abbreviation.
* Load the fixed split assignments from `splits/split_assignments.csv`, and hard-assert
  that the loaded split sizes and label counts match the frozen assignment.
* Select thresholds using validation data only; never use the test split for
  model or threshold selection.
* **Never print, display, or plot dataset text, row ids, or text hashes.**
  Show counts, metrics, configs, and word-level tables instead. Keep
  `include_text_preview` false. `tools/scan_text_leakage.py` fails on any
  committed output that contains a post.
* Export a sanitized probability artifact (`row_id`, `split`, `y_true`,
  `y_prob`, nothing else) per split.
* Export a word-level attribution artifact (`word`, `mean_abs_attribution`,
  `mean_attribution`, `support`) with its sidecar meta, using or mirroring
  `tools/attributions.py::write_run`. Aggregate subwords to words first; take
  means over posts containing the word. Logistic-regression notebooks also
  export `coefficients.csv`.
* Write the result folder files listed in `docs/RESULTS_SCHEMA.md`, computing
  every metric with the shared `compute_binary_metrics`.
* Record where weights, logs, and per-post files were stored, so the run
  manifest can be filled in.
* Include a short interpretation of false positives, false negatives, and limitations.

Notebook `11` is the closest existing template; it still prints row ids in
places and should be brought in line with the text rule before it is rerun.

Never encode a claim in a notebook name. Name the technique for what it is, not
for how well it did.

## Numbering convention

Notebook numbering should reflect the intended execution order.

```text
00_...  dataset and split preparation
01_...  first baseline model
02_...  second baseline model
03_...  third baseline model
04_...  character-level robustness baseline
05_...  word + character hybrid baseline
06_...  dense static embedding baseline
07_...  contextual transformer fine-tuning experiment
08_...  seed ensembling of the transformer
09_...  multi-checkpoint stacking (superseded)
10_...  logit-pooled seed ensembling
11_...  heterogeneous multi-checkpoint logit pooling
```

Numbers `08` and above are candidate experiments. A number is never reused, even
when a notebook is superseded, so that result folders, artifacts, and the
records in `research_loop/` keep pointing at the same thing they always did.

Newer notebooks use the pattern:

```text
<NUMBER>_<REPRESENTATION>_<MODEL>.ipynb
```

For transformer fine-tuning, the representation and classifier are bundled into the transformer architecture, so the file is named by the model family and training method:

```text
07_TWITTER-ROBERTA_FINE-TUNE.ipynb
```
