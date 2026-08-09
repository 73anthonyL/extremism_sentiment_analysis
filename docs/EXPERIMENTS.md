# Experiment protocol

This document defines the standard protocol for experiments in this repository.

The goal is to make model comparisons meaningful by controlling the dataset, splits, threshold-selection process, metrics, and saved outputs.

## Core rule

Every model family should be evaluated under the same fixed train/validation/test split assignments unless the repository explicitly introduces a new split version.

The canonical split file is:

```text
splits/split_assignments.csv
```

## Standard workflow

1. Load `data/dataset.csv`.
2. Run or verify the processed dataset created by `00_create_dataset_and_splits.ipynb`.
3. Merge the fixed split assignments from `splits/split_assignments.csv`.
4. Train only on the training split.
5. Tune hyperparameters only using the training and validation splits.
6. Select any probability or decision threshold using validation data only.
7. Evaluate on the test split once the model configuration and threshold are locked — once per technique, ever.
8. Export a sanitized probability artifact and derive compact output artifacts under `results_summary/<TECHNIQUE>/` from it.

## Test-split policy

The test split is not a metric you can recompute at will. It is a consumable
resource, and the protocol rations it:

* **Once per technique, ever.** Every unlock is recorded in the hash-chained
  `research_loop/test_ledger.jsonl`.
* **Every unlock raises the bar.** Each recorded evaluation increases the
  multiple-comparison family size that future candidates must clear under Holm
  correction.
* **A validation gate comes first.** A candidate that cannot clear a
  prespecified validation bar does not get to look at the test split at all.
* **Threshold selection on test data is refused mechanically.**
  `tools/eval_from_probs.py` hard-errors rather than trusting anyone to
  remember.

`docs/RESEARCH_LOOP.md` describes the surrounding cycle in full.

## Statistical power, and what may be claimed

The test split is 450 rows. Detecting a difference at McNemar exact
significance needs roughly +3 accuracy points, about 14 rows. For scale: a 5-row
gap has a best attainable p-value of 0.0625, so two techniques five rows apart
are **not** distinguishable no matter how the comparison is dressed up.

The transformer variants run so far span 395 to 409 correct test rows. Only the
widest pair in that span reaches the detection floor at all, and every
comparison is further penalised by Holm correction over a family that has now
consumed ten test unlocks. Treat none of them as separated without a verdict.

Three rules follow, and they are not negotiable:

1. `INCONCLUSIVE` is a first-class, publishable outcome. Most comparisons this
   project can run will return it.
2. `tools/compare_techniques.py` is the only thing permitted to issue a verdict.
   Everything else reads the verdict string.
3. A higher accuracy number is not, on its own, evidence of a better model. Do
   not write "improves on", "beats", or "best" without a verdict that supports
   it, and do not encode such a claim in a filename.

## Deriving results

Results are derived, not transcribed. A run exports probabilities; the metrics
come from them:

```bash
python3 tools/eval_from_probs.py --technique <TECHNIQUE> --threshold <selected>
```

The artifact contract is exactly `row_id`, `split`, `y_true`, `y_prob`, with no
text-bearing columns. See `docs/RESULTS_SCHEMA.md` for the full contract.

Hand-copying a number from a notebook cell output into a JSON file breaks the
guarantee that every published figure traces to a committed artifact, and the
result is not eligible for the comparison tables.

## Required metrics

Each model should report at least:

* Accuracy
* Balanced accuracy
* Positive-class precision
* Positive-class recall
* Positive-class F1
* Macro F1
* Weighted F1
* ROC-AUC
* PR-AUC
* Brier score when calibrated probabilities are available
* Confusion-matrix counts: `tn`, `fp`, `fn`, `tp`
* False-positive rate
* False-negative rate

The positive class is `EXTREMIST`.

## Threshold policy

Threshold selection must be based on validation data, not test data.

Recommended threshold strategies:

* Default threshold, usually `0.50`, when appropriate.
* Validation F1 maximization.
* Validation PR-AUC-informed operating point.
* Fixed recall or precision target, if motivated by the research question.

The selected strategy and selected threshold must be saved in `metrics_validation.json`, `metrics_test.json`, or `best_config.json`.

## Naming convention

Use descriptive technique names that identify the representation and model family.

The current repository contains a few legacy names where the model appears first, but newer experiments follow this clearer pattern:

```text
<NUMBER>_<REPRESENTATION>_<MODEL>
```

Examples:

```text
04_CHAR-TF-IDF_LIN-SVM
05_WORD-CHAR-TF-IDF_LIN-SVM
06_FASTTEXT-EMB_LOG-REG
07_TWITTER-ROBERTA_FINE-TUNE
11_MULTI-CHECKPOINT_LOGIT-POOL
```

For transformer fine-tuning, the representation and classifier are integrated into the same pretrained model, so the technique is named by the transformer family and fine-tuning method.

Three further naming rules are checked by `tools/protocol_check.py`:

* The notebook filename, the `technique_name` in its `CONFIG`, and the
  `results_summary/` folder name must all agree, including the numeric prefix.
* The `split_version` string must be the full frozen value
  `split_v1_stratified_70_15_15_seed30`, not an abbreviation.
* **A name must not encode a claim.** `08_BEST-ROBERTA_SEED-ENSEMBLE` asserts a
  superiority its own numbers contradict, and is the standing example of what
  not to do. Name a technique for what it is, not for how it did.

## Required output folder

Each technique should write outputs to:

```text
results_summary/<TECHNIQUE>/
```

Required and optional files are defined in `docs/RESULTS_SCHEMA.md`.

## Current controlled techniques

These have complete, schema-valid result folders and appear in the comparison
tables.

| Technique | Model family | Feature / representation family | Status |
|---|---|---|---|
| `01_LOG-REG_TF-IDF` | Logistic Regression | word-level TF-IDF | completed |
| `02_LIN-SVM_TF-IDF` | calibrated Linear SVM | word-level TF-IDF | completed |
| `03_SLP_TF-IDF` | Single-Layer Perceptron | word-level TF-IDF | completed |
| `04_CHAR-TF-IDF_LIN-SVM` | calibrated Linear SVM | character-level TF-IDF | completed |
| `05_WORD-CHAR-TF-IDF_LIN-SVM` | calibrated Linear SVM | combined word + character TF-IDF | completed |
| `06_FASTTEXT-EMB_LOG-REG` | Logistic Regression | FastText document embeddings | completed |
| `07_TWITTER-ROBERTA_FINE-TUNE` | fine-tuned transformer classifier | Twitter-RoBERTa contextual representations | completed, current champion |
| `11_MULTI-CHECKPOINT_LOGIT-POOL` | logit-pooled heterogeneous transformer ensemble | four admitted checkpoints spanning fine-tuning lineage, pretraining corpus, and scale | completed, not yet adjudicated |

## Candidate techniques not in the comparison

These exist as notebooks but have no registered result. They are listed so the
absence is documented rather than silent; their numbers, where any exist, must
not be mixed into the tables below.

| Technique | Model family | Status | Blocker |
|---|---|---|---|
| `08_BEST-ROBERTA_SEED-ENSEMBLE` | seed ensemble of the fine-tuned transformer | ran; test unlock spent | No result folder and no probability export; test metrics live only in cell outputs. Reported test accuracy is below `07`, and the claim-bearing `BEST` should be dropped from the name. |
| `09_MULTI-CHECKPOINT_LOGIT-STACK` | learned stacker over transformer checkpoints | built; never run | Superseded by `11`, which replaces the learned stacker with an equal-weight mean-log-odds pool. |
| `10_TWITTER-ROBERTA_LOGIT-POOL-STABLE` | mean-log-odds seed pool | ran; test unlock spent | No preregistration, no ledger entry, no probability export, no result folder. Registering it requires a rerun. |

`research_loop/STATE.md` holds the live status; `docs/RESEARCH_LOOP.md`
describes what registering one of these would take.

## Current held-out test results

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
| `11_MULTI-CHECKPOINT_LOGIT-POOL` | 0.9089 | 0.8746 | 0.9108 | 0.8412 | 0.9682 | 0.9553 | 0.5 |

Rendered by tools/render_tables.py from results_summary/ — do not edit by hand.
<!-- RENDERED-TABLE:END id=test-detail -->


The table above is rendered from `results_summary/` by `tools/render_tables.py`
and must not be edited by hand. `tools/render_tables.py --check` exits 1 if any
document drifts from the artifacts.

## Reporting rule

Do not report a technique as directly comparable unless it follows the same dataset, split, and evaluation protocol.

If a model uses a different preprocessing pipeline, external pretraining, different split, or competition-only evaluation setup, clearly label it as a separate condition. The Twitter-RoBERTa notebook uses transfer learning from a pretrained transformer, so it is comparable under the same split protocol but should be described as a contextual pretrained-model condition rather than a from-scratch classical baseline.
