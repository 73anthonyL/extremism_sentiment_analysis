# Research loop state

Updated: 2026-07-29

## Current cycle: 09 — MULTI-CHECKPOINT_LOGIT-STACK

- Stage: **build complete, blocked on human gate 2 (GPU run)**
- Preregistration: `research_loop/prereg/09_MULTI-CHECKPOINT_LOGIT-STACK.yaml`
- Notebook: `notebooks/09_MULTI-CHECKPOINT_LOGIT-STACK.ipynb` (Kaggle, T4/P100,
  ~1.5–3 h; attach the notebook-00 `research_foundation/` output, Internet ON)
- Comparator: `07_TWITTER-ROBERTA_FINE-TUNE` (test accuracy 0.8889)
- Validation gate (in-notebook): unlock test only if val accuracy ≥ 0.8733 AND
  val ROC-AUC ≥ 0.9350. Expected outcome per prereg: INCONCLUSIVE.
- After the run: place probability CSVs in `research_loop/probs/`, derive
  `results_summary/09_MULTI-CHECKPOINT_LOGIT-STACK/` with
  `tools/eval_from_probs.py`, then adjudicate with
  `tools/compare_techniques.py --candidate 09_MULTI-CHECKPOINT_LOGIT-STACK
  --champion 07_TWITTER-ROBERTA_FINE-TUNE --cycle 09 --prereg-sha <sha>
  --family-size 9` (see caveat below).

## IMPORTANT: Holm family-size caveat

`research_loop/test_ledger.jsonl` starts EMPTY as of the 2026-07-29
reconstruction (see History). The original ledger was lost, so
`compare_techniques.py`'s default family size (ledger length) would understate
the true multiple-comparison family. Cycles 01–08 consumed 8 test unlocks
(01–07 have committed `metrics_test.json`; 08's test metrics exist in notebook
cell outputs). **Every comparison must pass `--family-size` explicitly:
9 for cycle 09, incremented per subsequent unlock, until the ledger has
caught back up with reality.** Do not fabricate backdated ledger entries; the
chain records only evaluations made through the tool.

## History

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
  preregistered, and notebook built. Awaiting GPU run.

## Known open items (pre-existing, not blocking cycle 09)

- Notebooks 00–08 carry saved cell outputs containing dataset text
  (`tools/scan_text_leakage.py` exits 1 on them by design).
- Notebooks 01–08 declare `split_version: "split_v1"` instead of the full
  frozen string; 04/05/08 have `technique_name` ≠ filename;
  08 should be renamed `08_TWITTER-ROBERTA_SEED-ENSEMBLE.ipynb`.
- `results_summary/08_TWITTER-ROBERTA_SEED-ENSEMBLE/` does not exist; notebooks
  07/08 lack the probability-export cell (09 has it).
- `README.md:20` carries an 89.55% claim awaiting the user's uncommitted
  notebook (user instruction 2026-07-29: leave until that notebook lands).
