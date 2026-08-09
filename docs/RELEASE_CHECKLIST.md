# Release checklist

Use this checklist before publishing a new public research-repository update, major README update, or manuscript-aligned release.

## Run the checks first

Most of the boxes below are verified by tooling rather than by reading. Start here:

```bash
python3 tools/protocol_check.py --all           # invariants
python3 tools/validate_results_folder.py --all  # schema + internal consistency
python3 tools/render_tables.py --check          # documentation drift
python3 tools/scan_text_leakage.py              # dataset text in committed files
python3 tools/ledger.py verify                  # test-evaluation chain
python3 -m pytest tools/tests/ -q               # the toolkit's own tests
```

- [ ] `validate_results_folder.py --all` passes for every reported technique.
- [ ] `render_tables.py --check` exits 0 — no documentation drift.
- [ ] `ledger.py verify` reports the chain intact.
- [ ] `pytest tools/tests/ -q` passes.
- [ ] Any remaining `protocol_check.py` or `scan_text_leakage.py` failures are
      known, listed in `research_loop/STATE.md`, and none of them is new.

## Dataset and splits

- [ ] Confirm whether `data/dataset.csv` changed.
- [ ] Confirm whether `splits/split_assignments.csv` changed, and that the
      split-mirror invariant still passes.
- [ ] If rows or labels changed, update the dataset version.
- [ ] If split assignments changed, update the split version.
- [ ] Re-run the foundation notebook if the dataset or split changed.
- [ ] Verify `results_summary/foundation/dataset_manifest.json` is current.

## Experiments

- [ ] Confirm all reported models use the same dataset version.
- [ ] Confirm all reported models use the same split version.
- [ ] Confirm threshold selection used validation data only.
- [ ] Confirm test metrics were not used during model selection.
- [ ] Confirm each reported technique has a complete result folder derived from
      a committed probability artifact — not transcribed from cell outputs.
- [ ] Confirm README metrics match `results_summary/` files (`render_tables.py --check`).
- [ ] Confirm every test unlock in this release is recorded in
      `research_loop/test_ledger.jsonl`, and that the Holm family size used in
      any comparison accounts for unlocks the ledger does not yet contain.
- [ ] Confirm no technique is described as better than another without a
      `tools/compare_techniques.py` verdict backing it.
- [ ] Confirm any technique whose numbers appear in prose but not in the
      rendered tables is explicitly labelled as unregistered.

## Documentation

- [ ] Regenerate rendered tables with `tools/render_tables.py --write`; do not
      edit a results table by hand.
- [ ] Update `README.md` if headline results changed.
- [ ] Update `docs/DATA_CARD.md` if dataset details changed.
- [ ] Update `docs/EXPERIMENTS.md` if the protocol changed.
- [ ] Update `docs/MODEL_CARD.md` if model results or limitations changed.
- [ ] Update `docs/RESEARCH_LOOP.md` if the cycle, ledger, or gate rules changed.
- [ ] Update `docs/RESULTS_SCHEMA.md` if result-file structure changed.
- [ ] Update `notebooks/README.md` and `results_summary/README.md` if a notebook
      or result folder was added, renamed, or changed status.
- [ ] Update `research_loop/STATE.md` to reflect the current cycle and stage.
- [ ] Update `CHANGELOG.md` with the release summary.

## Repository hygiene

- [ ] Remove `.DS_Store`, `._*`, and `__MACOSX/` files.
- [ ] Remove notebook checkpoint folders.
- [ ] Ensure no private credentials are committed.
- [ ] Ensure no local-only scratch outputs are committed.
- [ ] Ensure large model binaries are intentionally included or intentionally excluded.

## Responsible-use review

- [ ] Confirm the README still states that the models are not deployment-ready.
- [ ] Confirm `docs/RESPONSIBLE_USE.md` is linked from the README.
- [ ] Avoid claims that the model can identify people, beliefs, or real-world intent.
- [ ] Avoid implying moderation or enforcement readiness.
- [ ] Avoid "best", "beats", "improves on", or "state of the art" where the
      underlying comparison came back `INCONCLUSIVE`.
- [ ] Confirm no artifact name — notebook, folder, or technique — encodes a
      claim about performance.

## Citation and license

- [ ] Confirm `CITATION.cff` is accurate for the current repository artifact.
- [ ] Confirm `LICENSE` reflects the intended permissions.
- [ ] If a paper, DOI, or archived release exists, update the citation instructions.
