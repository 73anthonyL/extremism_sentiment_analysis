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

These explore transformer ensembling. None has a registered result; see
`research_loop/STATE.md` for live status.

| Notebook | Purpose | State |
|---|---|---|
| `08_BEST-ROBERTA_SEED-ENSEMBLE.ipynb` | Probability-averaged seed ensemble of the fine-tuned transformer. | Ran; regressed against `07`. No result folder, no probability export. Should be renamed `08_TWITTER-ROBERTA_SEED-ENSEMBLE.ipynb` — the `BEST` is a claim its own numbers contradict. |
| `09_MULTI-CHECKPOINT_LOGIT-STACK.ipynb` | Learned stacker over several transformer checkpoints. | Built; never run. Superseded by `11`. |
| `10_TWITTER-ROBERTA_LOGIT-POOL-STABLE.ipynb` | Mean-log-odds seed pooling with exact probability-change threshold intervals. | Ran; spent a test unlock but left no derivable artifact. |
| `11_MULTI-CHECKPOINT_LOGIT-POOL.ipynb` | Mean-log-odds pooling over five checkpoints spanning fine-tuning lineage, pretraining corpus, architecture/tokenizer, and scale. | Ran; reported 409/450 on test. Probability artifacts not yet retrieved, so no result folder. Still the reference implementation for the conventions below — its saved outputs contain no dataset text. |

## Notebook conventions

Each model notebook should:

* State the technique name near the top, and set `CONFIG["technique_name"]` to
  exactly the filename stem, numeric prefix included.
* Declare the full frozen `split_version` string
  `split_v1_stratified_70_15_15_seed30`, not an abbreviation.
* Load the fixed split assignments from `splits/split_assignments.csv`, and hard-assert
  that the loaded split sizes and label counts match the frozen assignment.
* Avoid using the test split for model or threshold selection.
* Select thresholds using validation data only.
* Export a sanitized probability artifact (`row_id`, `split`, `y_true`,
  `y_prob`, and nothing else) so `tools/eval_from_probs.py` can derive the
  result folder. Do not hand-copy metrics.
* Save the best configuration used for the final test run.
* Include a short interpretation of false positives, false negatives, and limitations.
* **Carry no saved cell outputs when committed.** Outputs in these notebooks
  contain dataset text and row ids, which is why `tools/scan_text_leakage.py`
  currently fails on notebooks `00`–`08` and `10`.
* Avoid committing raw prediction files or large model artifacts unless they are intentionally tracked outside normal Git.

`tools/protocol_check.py --all` checks the first three and the outputs rule.
Notebook `11` is the only notebook that currently satisfies all of them, and is
the best template to copy from.

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
when a notebook is superseded, so that the ledger and the preregistrations keep
pointing at the same thing they always did.

Newer notebooks use the pattern:

```text
<NUMBER>_<REPRESENTATION>_<MODEL>.ipynb
```

For transformer fine-tuning, the representation and classifier are bundled into the transformer architecture, so the file is named by the model family and training method:

```text
07_TWITTER-ROBERTA_FINE-TUNE.ipynb
```
