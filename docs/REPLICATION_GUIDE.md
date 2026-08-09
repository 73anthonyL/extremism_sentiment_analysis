# Replication guide

This guide describes how to reproduce the core research artifacts in this repository.

## 1. Prepare the environment

Clone the repository and install dependencies.

```bash
git clone https://github.com/73anthonyL/extremism_sentiment_analysis.git
cd extremism_sentiment_analysis
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For stricter environment replication, use:

```bash
pip install -r requirements-lock.txt
```

To run the verification toolkit's own test suite, also install:

```bash
pip install -r requirements-dev.txt
```

Transformer runs are easiest on Kaggle or another GPU environment. CPU-only runs may be slow.

## 1a. Read this before cloning and running

**The notebooks are Kaggle-targeted and will not run as-is against a clone of
this repository.** This is the single biggest obstacle to replication, and it is
not something a dependency install fixes. Known gaps:

* Notebooks use absolute `/kaggle/input/...` and `/kaggle/working/...` paths.
  These must be repointed at local directories.
* `processed_dataset.csv` — required by notebooks `01` onward — is **not
  committed** to this repository. It is produced by
  `00_create_dataset_and_splits.ipynb`, which must therefore run first in any
  environment that lacks it.
* `00_create_dataset_and_splits.ipynb` sets `overwrite_existing_split: True`.
  **Set that to `False` before running it against this repository's `splits/`
  directory**, or it will overwrite the frozen split assignment. The split
  assignment must never be regenerated; see step 2a.
* The transformer notebooks download pretrained checkpoints from the Hugging
  Face Hub and need network access at runtime.
* Notebooks `07` onward expect a GPU. On CPU they will complete but slowly
  enough to be impractical.

A local replication that only wants to check the *reported numbers*, rather than
retrain, does not need any of this: the result artifacts under
`results_summary/` are complete, and the verification commands in step 7 run
against them directly.

## 2. Verify source files

Confirm that the repository contains:

```text
data/dataset.csv
data/extremism_lexicon.txt
splits/split_assignments.csv
notebooks/00_create_dataset_and_splits.ipynb
notebooks/01_LOG-REG_TF-IDF.ipynb
notebooks/02_LIN-SVM_TF-IDF.ipynb
notebooks/03_SLP_TF-IDF.ipynb
notebooks/04_CHAR-TF-IDF_LIN-SVM.ipynb
notebooks/05_WORD-CHAR-TF-IDF_LIN-SVM.ipynb
notebooks/06_FASTTEXT-EMB_LOG-REG.ipynb
notebooks/07_TWITTER-ROBERTA_FINE-TUNE.ipynb
notebooks/11_MULTI-CHECKPOINT_LOGIT-POOL.ipynb
```

Notebook `11_MULTI-CHECKPOINT_LOGIT-POOL.ipynb` is also part of the controlled
comparison. Notebooks `08`, `09`, and `10` are candidate experiments outside it
and are not required to reproduce the reported results.

## 2a. Verify the split mirror before anything else

`splits/split_assignments.csv` is a *mirror* of the frozen split assignment, and
it has been wrong before: until 2026-07-29 the committed file was a stale
pre-correction artifact disagreeing with the frozen assignment on 728 labels and
1420 split assignments. Anyone who replicated against that file got results that
could not match.

Confirm the mirror is intact before you trust anything downstream:

```bash
python3 tools/protocol_check.py --all
```

The first invariant reported is `Split mirror matches the frozen assignment`. If
it fails, repair the mirror with:

```bash
python3 tools/repair_split_mirror.py
```

That tool refuses to write unless its reconstruction reproduces
`results_summary/foundation/` exactly. Repairing the mirror is **not** the same
as regenerating the split, which must never happen. The stale pre-repair file is
preserved at `splits/split_assignments.PRE-REPAIR.csv` for provenance; do not
use it.

## 3. Recreate the dataset foundation

Run:

```text
notebooks/00_create_dataset_and_splits.ipynb
```

This notebook should validate the source dataset, create or verify processed rows, assign canonical IDs, and write foundation artifacts.

Expected foundation artifacts:

```text
results_summary/foundation/dataset_manifest.json
results_summary/foundation/label_distribution.csv
results_summary/foundation/split_label_distribution.csv
results_summary/foundation/text_length_summary.csv
results_summary/foundation/duplicate_text_report.csv
results_summary/foundation/rows_removed_summary.json
```

## 4. Run model notebooks

Run the model notebooks in order:

```text
notebooks/01_LOG-REG_TF-IDF.ipynb
notebooks/02_LIN-SVM_TF-IDF.ipynb
notebooks/03_SLP_TF-IDF.ipynb
notebooks/04_CHAR-TF-IDF_LIN-SVM.ipynb
notebooks/05_WORD-CHAR-TF-IDF_LIN-SVM.ipynb
notebooks/06_FASTTEXT-EMB_LOG-REG.ipynb
notebooks/07_TWITTER-ROBERTA_FINE-TUNE.ipynb
```

Each notebook should read the fixed split assignments and write a result folder under `results_summary/`.

For `07_TWITTER-ROBERTA_FINE-TUNE.ipynb`, use a GPU runtime when possible and avoid committing trained model weights to normal Git.

## 4a. Derive result folders from probability artifacts

Results in this repository are derived, not transcribed. A run exports a
sanitized probability artifact — exactly `row_id`, `split`, `y_true`, `y_prob`,
with no text-bearing columns — and the result folder is rebuilt from it:

```bash
python3 tools/eval_from_probs.py --technique <TECHNIQUE> --threshold <selected>
```

The threshold is the one the run selected on validation data;
`eval_from_probs.py` hard-errors on any attempt to select a threshold using
test-split data. `ablation_results.csv` is the one schema-required file it
cannot derive and must be supplied from the run's configuration comparison.

This means a replication can be checked without rerunning anything: given the
same probability artifact, `eval_from_probs.py` must produce byte-comparable
metrics. `results_summary/11_MULTI-CHECKPOINT_LOGIT-POOL/` was produced exactly
this way, and its derived metrics match the run's own reported numbers to four
decimals. Where a technique has no committed probability artifact — notebooks
`08` and `10` — this check is unavailable, which is precisely why those runs are
not registered.

## 5. Check comparability

A replicated result is comparable only if:

* The same dataset version is used.
* The same split assignments are used.
* Hyperparameters are selected without using test labels.
* Thresholds are selected using validation data only.
* The same metric definitions are used.
* The reported test metrics come from the held-out test split.
* External pretraining or transfer learning is disclosed as part of the experimental condition.

## 6. Current ranking by held-out test PR-AUC

1. `11_MULTI-CHECKPOINT_LOGIT-POOL`
2. `07_TWITTER-ROBERTA_FINE-TUNE`
3. `01_LOG-REG_TF-IDF`
4. `02_LIN-SVM_TF-IDF`
5. `05_WORD-CHAR-TF-IDF_LIN-SVM`
6. `03_SLP_TF-IDF`
7. `06_FASTTEXT-EMB_LOG-REG`
8. `04_CHAR-TF-IDF_LIN-SVM`

Small numeric differences may occur across environments, especially for neural or transformer models. Any material difference should be documented.

Treat this ranking as an ordering of measurements, not of models. On a 450-row
test split, positions 3 through 8 are separated by a few rows each and are not
statistically distinguishable, and the gap between positions 1 and 2 is 9 rows —
below the detection floor, and not yet adjudicated. See `docs/EXPERIMENTS.md`
for the power analysis.

## 7. Verify the replication mechanically

The repository ships its own checks. Run them rather than eyeballing tables:

```bash
python3 tools/protocol_check.py --all           # protocol invariants
python3 tools/validate_results_folder.py --all  # schema + internal consistency
python3 tools/render_tables.py --check          # documentation drift
python3 tools/scan_text_leakage.py              # dataset text in committed files
python3 tools/ledger.py verify                  # test-evaluation chain
python3 -m pytest tools/tests/ -q               # the toolkit's own tests
```

`protocol_check.py` and `scan_text_leakage.py` currently report known,
pre-existing failures against notebooks `00`–`08` and `10`, which carry saved
cell outputs containing dataset text and mostly declare an abbreviated
`split_version`. Those are a documented cleanup backlog, tracked in
`research_loop/STATE.md` — not signs that your replication went wrong. The
checks that must pass cleanly for a replication to be trustworthy are
`validate_results_folder.py`, `render_tables.py --check`, the split-mirror
invariant, and the pytest suite.

## 8. When results do not match

Check the following first:

* Python version and dependency versions.
* Whether `requirements-lock.txt` was used.
* Whether the split file was regenerated accidentally — check
  `overwrite_existing_split` in `00_create_dataset_and_splits.ipynb`, and rerun
  `tools/protocol_check.py --all` to confirm the mirror invariant still passes.
* Whether threshold selection used validation data only.
* Whether labels were mapped consistently.
* Whether the model was run with the same random seed.
* Whether a GPU/non-GPU environment changed transformer reproducibility.
* Whether the pretrained transformer checkpoint changed or was unavailable.
