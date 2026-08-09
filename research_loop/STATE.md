# Research loop state

Updated: 2026-08-09

## Current cycle: 11 — MULTI-CHECKPOINT_LOGIT-POOL

- Stage: **integrated; blocked on decide (prereg + ledger + adjudication)**
- Merged to `main` via PR #3; the completed run is commit `99d0f8d8`.
- Notebook: `notebooks/11_MULTI-CHECKPOINT_LOGIT-POOL.ipynb`

### What the run did

The validation gate passed at **393/450 correct** (bar: ≥ 392), so the test
split was unlocked. Component admission at the 0.84 pooled-validation floor:

| Component | Pooled val acc @ 0.50 | Admitted |
|---|---:|---|
| A anchor (`twitter-roberta-base-hate-latest`) | 0.8600 | yes |
| B dynabench (`roberta-hate-speech-dynabench-r4-target`) | 0.8689 | yes |
| C hatebert (`GroNLP/hateBERT`) | 0.8467 | yes |
| D deberta (`microsoft/deberta-v3-base`) | 0.7733 | **no** — below floor |
| E large (`twitter-roberta-large-hate-latest`) | 0.8600 | yes |
| F anchor+FGM | 0.8644 | yes (challenger C3 only) |

The prespecified primary was **retained**: the strongest of the four
challengers gained only 2 correct validation examples, below the required 3. So
the shipped rule is the equal-weight mean of the admitted components' seed-mean
logits at the fixed 0.50 cutoff — exactly what was specified before the run.

Reported held-out test metrics: **409/450 correct, accuracy 0.9089**, balanced
accuracy 0.8956, macro F1 0.9015, ROC-AUC 0.9682, PR-AUC 0.9553, positive F1
0.8746, confusion `tn=266 fp=14 fn=27 tp=143` at threshold 0.50.

That clears the notebook's own prespecified ≥ 90% target (405/450). It is 9 test
rows above the registered champion's 400/450 — still short of the ~14 rows this
split needs for a McNemar-detectable difference, and **no verdict has been
issued**, so it is not yet an established improvement. See "Statistical reality".

### Integration: done

The probability artifacts were retrieved from the Kaggle output and committed to
`research_loop/probs/`, and `results_summary/11_MULTI-CHECKPOINT_LOGIT-POOL/`
was derived from them:

```bash
python3 tools/eval_from_probs.py --technique 11_MULTI-CHECKPOINT_LOGIT-POOL --threshold 0.5
```

The derived metrics reproduce the run's own reported numbers exactly
(validation 0.8733, test 0.9089), `tools/validate_results_folder.py --all`
passes on the folder, and `tools/render_tables.py --write` has pulled the
technique into every rendered comparison table. `ablation_results.csv` was
copied from the run's `ablation/` output, since `eval_from_probs.py` does not
derive it.

### What is still blocking a verdict

1. **No preregistration.** See below.
2. **No ledger entry.** The test unlock is already spent — the run read the test
   split — so recording it is documenting a fact, not authorizing one.
3. **No adjudication.** Nothing has compared this against the champion:
   `tools/compare_techniques.py --candidate 11_MULTI-CHECKPOINT_LOGIT-POOL
   --champion 07_TWITTER-ROBERTA_FINE-TUNE --cycle 11 --prereg-sha <sha>
   --family-size 10` (see the Holm caveat below).

Until step 3 runs, `07_TWITTER-ROBERTA_FINE-TUNE` remains the champion in
`registry.json`, and notebook 11's presence in the results tables is a
statement of measurement, not of superiority.

### Preregistration

Still **not written**. The notebook carried its full prespecification in its
header cell and `CONFIG`, and the run honoured it, but no `research_loop/prereg/`
YAML exists, so there is no prereg SHA to pass at adjudication. Extract one from
the notebook and note explicitly that it was recorded after the run.

### Protocol status of the notebook

Before the run, notebook 11 satisfied every invariant. The completed run
reintroduced one: it now carries **21 code cells with saved outputs**, so it
joins the outputs backlog and `protocol_check.py` fails on it.

The sanitization held, though — `tools/scan_text_leakage.py` finds **no dataset
text and no row hashes** in notebook 11, the only run notebook in the repository
of which that is true. Stripping its outputs is therefore a tidiness fix, not a
leakage remediation.

## Cycle 10 — ran outside the loop, unregistered

`notebooks/10_TWITTER-ROBERTA_LOGIT-POOL-STABLE.ipynb` was run on Kaggle and
committed on 2026-08-08 (`7db524a0`). It **evaluated the test split**, so it
consumed a test unlock, but it went through none of the loop machinery:

- no preregistration in `research_loop/prereg/`
- no entry in `research_loop/test_ledger.jsonl`
- no `results_summary/10_TWITTER-ROBERTA_LOGIT-POOL-STABLE/` folder
- no probability-export cell, so the folder **cannot** be derived with
  `tools/eval_from_probs.py` without re-running the notebook

Its notebook cell outputs report validation accuracy 0.8644 (389/450) and test
accuracy 0.8956 (403/450) at a fixed 0.50 threshold, using mean log-odds
pooling over three seeds (17/30/73) of
`cardiffnlp/twitter-roberta-base-hate-latest`. Those numbers are transcription,
not derived artifacts, and they are deliberately absent from every rendered
results table. Against the registered champion's 400/450 the gap is 3 rows —
far below the ~14 rows this 450-row test split needs for a McNemar-detectable
difference, so it is not evidence of an improvement.

To register it properly the notebook needs the sanitized probability-export
cell (notebook 11 has a working implementation to copy) and a rerun.

## Cycle 09 — built, never run, superseded

`notebooks/09_MULTI-CHECKPOINT_LOGIT-STACK.ipynb` and
`research_loop/prereg/09_MULTI-CHECKPOINT_LOGIT-STACK.yaml` remain committed and
unmodified. The notebook was never run. Notebook 11 covers the same
multi-checkpoint axis but replaces 09's learned logit stacker with notebook
10's proven equal-weight mean-log-odds pool, so 09 is treated as superseded.
The prereg is kept for the record, not resurrected.

## IMPORTANT: Holm family-size caveat

`research_loop/test_ledger.jsonl` does not exist yet — `tools/ledger.py verify`
reports 0 entries — so `compare_techniques.py`'s default family size (ledger
length) would badly understate the true multiple-comparison family. Test
unlocks consumed to date:

| Unlocks | Source |
|---:|---|
| 7 | cycles 01–07 (committed `metrics_test.json` in each result folder) |
| 1 | 08 (test metrics exist only in notebook cell outputs) |
| 1 | 10 (test metrics exist only in notebook cell outputs) |
| 1 | 11 (derived result folder committed; ledger entry still pending) |
| **10** | **total consumed** |

**Every comparison must pass `--family-size` explicitly: 10 for cycle 11,
incremented per subsequent unlock, until the ledger has caught back up with
reality.** Notebook 11's `CONFIG` already carries
`"family_size_for_holm": 10`, which counts its own unlock. Do not fabricate
backdated ledger entries; the chain records only evaluations made through the
tool.

## Statistical reality

The test split is 450 rows. A McNemar-detectable difference needs roughly +3
accuracy points, about 14 rows.

Current standings by reported test accuracy, registered and unregistered
together:

| Technique | Correct / 450 | Accuracy | Registered |
|---|---:|---:|---|
| `11_MULTI-CHECKPOINT_LOGIT-POOL` | 409 | 0.9089 | yes, not adjudicated |
| `10_TWITTER-ROBERTA_LOGIT-POOL-STABLE` | 403 | 0.8956 | no |
| `07_TWITTER-ROBERTA_FINE-TUNE` | 400 | 0.8889 | **yes, champion** |
| `08_BEST-ROBERTA_SEED-ENSEMBLE` | 395 | 0.8778 | no |

Notebook 11 leads the champion by 9 rows. That is the largest margin the
transformer line has produced, and it is still below the ~14-row detection
floor, so it may well come back `INCONCLUSIVE` under Holm correction at family
size 10. The honest position until the comparison is actually run: this is the
most promising candidate the program has, and it is not yet an established
improvement.

`tools/compare_techniques.py` is the only thing permitted to issue a verdict —
no agent and no notebook decides, and neither does a table like the one above.

## History

- 2026-08-08: Notebook 11 run on Kaggle (commit `99d0f8d8`) and merged to `main`
  via PR #3. Gate passed at 393/450 validation; prespecified primary retained;
  reported 409/450 on test. Probability artifacts not yet retrieved.
- 2026-08-08: Notebook 10 committed with a completed Kaggle run (unregistered,
  see above). Notebook 11 designed and built on branch
  `nb11-multi-checkpoint-logit-pool`.
- 2026-07-29: `research_loop/` and all `tools/*.py` sources found missing
  (never committed to git; deleted locally). Sources reconstructed from
  surviving `tools/__pycache__/` bytecode with instruction-level fidelity;
  `render_tables.py` and `scan_text_leakage.py` (no surviving bytecode)
  rebuilt fresh from their documented contracts. Test suite (19 tests)
  reconstructed byte-identically and passing.
- 2026-07-29: `splits/split_assignments.csv` mirror repaired — the committed
  file was the stale pre-correction artifact (728 label / 1420 split
  disagreements vs. the frozen assignment). Verified reconstruction applied;
  stale file preserved at `splits/split_assignments.PRE-REPAIR.csv`;
  `tools/repair_split_mirror.py` confirms the mirror matches the frozen
  assignment.
- 2026-07-29: Cycle 09 designed (CPU pilots: transformer+TF-IDF stacking
  +0.22pp — dead; calibration provably 0pp; scale is the one untried axis),
  preregistered, and notebook built. Never run; superseded by cycle 11.

## Known open items (pre-existing, not blocking cycle 11)

`python3 tools/protocol_check.py --all` currently reports **FAIL (4 blocking)**
and `python3 tools/scan_text_leakage.py` reports **FAIL (2 blocking)**:

- Notebooks 00–08 and 10 carry saved cell outputs containing dataset text
  (`scan_text_leakage.py` exits 1 on them by design). Notebook 11 also carries
  saved outputs after its run, but they contain no dataset text or row hashes.
- Notebooks 01–08 and 10 declare `split_version: "split_v1"` instead of the
  full frozen string. 09 and 11 declare it correctly.
- 04/05/08 have `technique_name` ≠ filename; notebook 10's `technique_name`
  (`TWITTER-ROBERTA_LOGIT-POOL-STABLE`) drops the numeric prefix.
- 08 should be renamed `08_TWITTER-ROBERTA_SEED-ENSEMBLE.ipynb` to drop the
  claim-bearing `BEST` — its own numbers (test accuracy 0.8778) are a
  regression against 07.
- Notebooks 08, 09, and 10 have no `results_summary/` folder. For 09 that is
  correct (never run). For 08 and 10 it is a gap: both spent a test unlock and
  neither left a derivable artifact. Notebook 11's folder now exists and
  validates clean.
- `README.md` no longer carries the bare "89.55%" claim; it now describes
  notebook 10's number as unregistered and not distinguishable from the
  champion (2026-08-09 documentation pass).
