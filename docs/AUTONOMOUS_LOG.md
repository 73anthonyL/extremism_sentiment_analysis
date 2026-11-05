# Autonomous work log

A running record of work done without a human in the loop, newest first.
Each entry says what was done, on which branch, what was checked, and what
comes next. Nothing in this log has been pushed; the author reviews and
pushes. Work that needs an author decision, a Kaggle GPU, or sending
dataset text to an external API is listed under "Blocked" and not attempted.

## Blocked on the author

* WP3: the slur lexicon source (`data/lexicons/slurs.txt` is empty) and the
  identity-list review. Candidates are in
  `results_summary/rq/rq2_identity_lexicon_gaps_sensitivity.csv`. RQ2-RQ4
  category results stay provisional until the lists are frozen.
* Running notebook 12 on Kaggle (RQ3 plain RoBERTa control). Note: it grows
  the RQ1 Holm family from 28 to 36 and will very likely cost 01 vs 11 its
  significance.
* WP9 probes (Kaggle GPU) and WP10 zero-shot LLM baseline (data-sharing
  decision).
* Whether section 5's share-difference rule should use the bootstrap SD
  (`rq2_bootstrap_category_shares.csv`) instead of the two-run SD.
* Whether report cards may print a slur verbatim; whether support >= 2
  becomes the SHAP pass rule; dropping `--primed` from the CLAUDE.md
  pre-commit list now that notebooks carry outputs.

## 2026-11-04 — Bootstrap uncertainty for RQ2/RQ3 and the RQ2-RQ4 relation

Branch `research/bootstrap-uncertainty`, merged into `research/integration`.
`tools/bootstrap_uncertainty.py` resamples explained posts (and, for FPR
ratios, test rows) 2000 times with seed 30, re-aggregating exactly as the
kit does; with weight one it must reproduce every committed table or it
refuses. Five text-free tables under `results_summary/rq/`, each row marked
`percentile_ci` or `replicate_spread`.

What it changed in the paper's claims:
* RQ2-RQ4 relation (rho 0.83): the ranking is unstable under resampling
  (replicate median 0.48, positive in ~90%), driven by the FPR ratios.
  Now described as fragile, never with a "95% CI".
* Transformer test-vs-validation rank agreement: lower than every classical
  technique on paired draws (descriptive). Holds.
* Single-post share and top-50 Jaccard: replicate spreads only; 07/11 do not
  separate from 04 or 06.
* Identity-share bootstrap SDs (0.016-0.036) are far larger than the
  two-run SDs that section 5 uses. Flagged as an author decision.

Adversarial review: mechanics and numbers confirmed by an independent
recompute; it caught the mislabelled relation interval and five guards with
no test. Fixed, each guard mutation-tested.

Checks: all five pass (429 tests).

Next: a results-section draft for the paper, organised by RQ, that cites
only committed artifacts, with an adversarial check of every number.

## 2026-11-04 — Audit chain covers the new analyses (WP8 extension)

Branch `research/audit-new-analyses`, merged into `research/integration`.
`tools/audit.py run` gains stages for `identity_sensitivity.py` (fpr,
lexicon-gaps), `same_post_stability.py`, `faithfulness.py` and
`rq2_rq4_relation.py`. The model-loading stages run only with
`--with-perturbation` (`--with-counterfactual` kept as an alias); a skip is
printed and recorded as SKIP, never OK, and a requested perturbation stage
that skips makes the run exit 2. Report cards gain an RQ2-support
(faithfulness) section and ambiguous-term sensitivity rows for RQ4 and WP7.
`lexicon-gaps` now merges by technique instead of overwriting.

Adversarial review found no wrong number on any card but eleven weak
staleness guards (shared population and Holm family not checked, no locked
threshold check on the WP7 tables, a doc claiming "not run" after a skip,
and others). All eleven were fixed with a refusal test each, except that the
ambiguous-term list itself cannot be re-checked without post text.

Checks: all five pass (404 tests). `audit.py run` on 01 without the flag
and `audit.py report --all` leave every committed file byte-identical.

Known leftover: the eight committed `identity_counterfactual_test.json`
files carry an older wording of the `holm_adjusted_p` note; a full
`--with-perturbation` run rewrites them (numbers unchanged).

Next: bootstrap uncertainty for the descriptive RQ2/RQ3 numbers (category
shares, same-post single-post share, split agreement, RQ2-RQ4 rho), which
the plan repeatedly marks "one run per split, no uncertainty estimate".

## 2026-11-04 — Integrate the five swarm lanes

Branch `research/integration`, cut from `framework/audit-entry-point`
(PR #22). Merged `swarm/related-work`, `swarm/notebook12`,
`swarm/faithfulness`, `swarm/identity-sensitivity`, `swarm/stability`.
The only conflicts were appended rows in `docs/RESULTS_SCHEMA.md` and
`tools/repo_paths.py`; both sides kept. Notebook 12's Holm condition was
corrected before the merge (all eight new pairs must themselves be
Holm-significant, about raw p < 0.0017, not 0.0022). The synthesis text was
applied to `docs/RESEARCH_PLAN.md` §3/§5/WP5/WP7 and `docs/POSITIONING.md`
§4 after re-checking its new numbers against the committed CSVs (WP7
substitute rows, RQ4 sensitivity rows).

Checks: check_notebook --all, validate_results_folder --all,
render_tables --check, scan_text_leakage, pytest (363 passed).

Next: a consistency pass over README and FRAMEWORK.md for the new tools,
and adding the new analyses to `tools/audit.py run` and the report cards.
