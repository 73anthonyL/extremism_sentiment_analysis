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

## 2026-11-04 — Decisions brief; lexicon coverage artifact

`docs/DECISIONS_NEEDED.md` (branch `docs/decisions-brief`): the twelve open
author decisions in one table, each with what it blocks, the options and a
recommendation. "Accept all recommendations" unblocks the local ones.

Branch `research/lexicon-coverage`: `categorize_attributions.py` now writes
`results_summary/rq/rq2_lexicon_coverage.csv` (entries, single words and
multi-word phrases per category) and `--check-coverage` refuses a table
that no longer matches `data/lexicons/`. The count section 5 asks for is 0:
every lexicon entry is a single word. The draft reports it (claim C820) and
the TODO is closed. Review: small change, checked by mutation (disabling the
staleness check fails two tests); category shares are byte-identical after
a full re-run.

Checks: all five pass (529 tests); check_claims OK (632).

Next: place the verified related-work citations in the Results draft.

## 2026-11-04 — WP7b: preregistered within-category identity swap

Branch `research/identity-swap`, merged into `research/integration`. The
swap map (`data/lexicons/identity_swap_pairs.csv`) and analysis plan
(`docs/preregistrations/WP7b_identity_swap.md`) were committed before the
tool existed; the map's sha256 is recorded in every result JSON and did not
change afterwards. `tools/identity_swap.py` replaces each identity term with
three same-category, same-form groups (68 test posts) and compares with
remove and substitute on the same spans, all eight techniques, Holm over 48
rows.

Descriptive results under the preregistered rule (interpretation pending
author review): on extremist posts 07 and 11 respond to a group being named
rather than which group (swap delta 0.006 / 0.004 against removal
0.069 / 0.149); for 11 a neutral noun does not restore p, so an identity
term being present matters. On the 23 non-extremist posts swapping raises
11's p (Holm 0.022; 07 not significant), concentrated in three source
terms and not explained by the targets' training-label skew. Classical
models move as much under swap as under removal in absolute terms but in
both directions, tracking the LR coefficient change; 06 meets the
group-specific rule on extremist posts.

Adversarial review confirmed preregistration integrity, Holm and signs, and
corrected three over-readings in the first summary; it also found six
untested map guards and a latent framing-overlap issue (no post affected).
All fixed, mutation-tested.

Checks: all five pass (523 tests); check_claims OK.

Next: a single "decisions needed" brief so the author can unblock WP3,
section 5, notebook 12 and the WP7/WP7b interpretation in one sitting.

## 2026-11-04 — Paper Results draft with a bound claims table

Branch `paper/results-draft`, merged into `research/integration`.
`docs/paper/RESULTS_DRAFT.md` (about 7,600 words): Results by RQ1-RQ4
including WP7, the C3 "why this is hard" section, and a C1 framework
usability section. Every claimed number is written as `value [Cnnn]`;
`docs/paper/CLAIMS.md` (631 claims) gives each one's artifact, selector,
column and precision, and `tools/check_claims.py` refuses the draft if any
bound number differs from its artifact or from its claim (48 tests, 31
guards mutation-tested).

Adversarial review found three correct numbers attached to wrong
descriptions (27 lexicon-gap "words" are 27 pairs over 14 words; 11's
members do not share 07's recipe; two different 01 agreement values under
one name) and a checker that could not catch a changed prose number. All
fixed.

Finding that changes earlier summaries: under substitution with a neutral
panel word, 07's and 11's shift on non-extremist identity posts mostly
disappears (07 mean delta 0.099 removed vs 0.012 substituted, Holm 0.24;
11 0.095 vs 0.038, Holm 0.92), and their recall drops are no longer
Holm-significant, while 01, 04, 05 and 06 lose significant recall under
substitution. So the transformers' large removal effects may partly reflect
deleting a noun, not the identity term. Draft, POSITIONING section 4 item 3
and the plan's WP7 text now use substitution as the stricter control. The
earlier slide deck and professor summary predate this.

Checks: all five pass (492 tests); check_claims OK.

Next: an identity-swap counterfactual (replace one group term with another
group term, a predeclared mapping) to separate "names a group" from "names
this group" for the transformers.

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
