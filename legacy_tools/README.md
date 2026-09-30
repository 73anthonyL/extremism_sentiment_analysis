# Legacy toolkit

This folder holds the previous verification and adjudication toolkit, moved
here on 2026-09-30 when the project refocused on the explainability research
questions (see `CLAUDE.md`). It is kept for reference and for excerpting; it is
not part of the current workflow, and nothing in the repository should depend
on it.

What it contained:

| Script | Purpose then |
|---|---|
| `ledger.py` | Hash-chained record of test-split unlocks; champion promotion. |
| `compare_techniques.py` | McNemar + Holm adjudication of a candidate against the champion. |
| `eval_from_probs.py` | Built a `results_summary/<TECHNIQUE>/` folder from a probability artifact and appended to `research_loop/val_log.jsonl`. |
| `protocol_check.py` | Invariant audit: split mirror, naming, version strings, saved outputs, coverage. |
| `repair_split_mirror.py` | Rebuilt `splits/split_assignments.csv` from the frozen recipe. |
| `render_tables.py`, `validate_results_folder.py`, `scan_text_leakage.py` | Carried forward into the new `tools/` in adapted form. |
| `repo_paths.py`, `metrics_core.py`, `probs_artifact.py`, `split_protocol.py` | Foundation modules, copied into the new `tools/` unchanged. |

The records these scripts wrote live in `research_loop/` and are treated as
data. The scripts still run from this folder (`python3 legacy_tools/<script>`)
because each inserts its own directory on `sys.path`, but do not run the ones
that write to `research_loop/` unless you mean to extend those records.
