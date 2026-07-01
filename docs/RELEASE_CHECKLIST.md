# Release checklist

Use this checklist before publishing a new public research-repository update, major README update, or manuscript-aligned release.

## Dataset and splits

- [ ] Confirm whether `data/dataset.csv` changed.
- [ ] Confirm whether `splits/split_assignments.csv` changed.
- [ ] If rows or labels changed, update the dataset version.
- [ ] If split assignments changed, update the split version.
- [ ] Re-run the foundation notebook if the dataset or split changed.
- [ ] Verify `results_summary/foundation/dataset_manifest.json` is current.

## Experiments

- [ ] Confirm all reported models use the same dataset version.
- [ ] Confirm all reported models use the same split version.
- [ ] Confirm threshold selection used validation data only.
- [ ] Confirm test metrics were not used during model selection.
- [ ] Confirm each reported technique has a complete result folder.
- [ ] Confirm README metrics match `results_summary/` files.

## Documentation

- [ ] Update `README.md` if headline results changed.
- [ ] Update `docs/DATA_CARD.md` if dataset details changed.
- [ ] Update `docs/EXPERIMENTS.md` if the protocol changed.
- [ ] Update `docs/MODEL_CARD.md` if model results or limitations changed.
- [ ] Update `docs/RESULTS_SCHEMA.md` if result-file structure changed.
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

## Citation and license

- [ ] Confirm `CITATION.cff` is accurate for the current repository artifact.
- [ ] Confirm `LICENSE` reflects the intended permissions.
- [ ] If a paper, DOI, or archived release exists, update the citation instructions.
