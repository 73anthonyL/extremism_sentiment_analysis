# Autonomous work log

A running record of work done without a human in the loop, newest first.
Each entry says what was done, on which branch, what was checked, and what
comes next. Work is done on local branches and merged into
`research/integration`; the author decides when it is pushed. Work that
needs an author decision, a Kaggle GPU, or sending dataset text to an
external API is listed under "Blocked" and not attempted.

## Blocked on the author

All twelve open decisions were settled on 2026-11-04 (`docs/DECISIONS_NEEDED.md`).
Remaining author actions: run notebook 12, the WP9 probe notebook and the
WP10 open-weight LLM notebook on Kaggle; upload external assets before
submission.

## 2026-11-05 — WP3: lexicons frozen (HurtLex slurs; aliens added then dropped)

Branch `research/wp3-lexicon-freeze`, merged into `research/integration`.
`data/lexicons/slurs.txt` now holds 58 entries from HurtLex EN 1.2
(conservative level, ethnic slurs and sexual-orientation slurs; CC BY-NC-SA
4.0, attributed in the file, the lexicon README and a third-party notice in
LICENSE). 133 of the 191 HurtLex lemmas were removed in five documented
classes; no slur entry overlaps the identity or framing lists. Lexicon
coverage now counts every entry the SHAP masker splits (48 unmatchable,
mostly punctuated framing entries). A guardrail test pins the three lists to
the freeze note's sha256s.

`aliens` (the only gap word with support >= 3) was added and then dropped by
author decision on 2026-11-05: it came from 06's own test attribution list
and always appears as "illegal aliens". Dropping it restores every identity
result byte for byte. Rule added: no lexicon term on test-split evidence.

What changed in the findings: slur reliance is now measurable but thin (one
racial slur from two posts in most top-50 lists; transformers have the
largest slur shares, passing the preregistered two-run rule but not the
bootstrap check, so no ordering is claimed); the RQ2 hypothesis that
classical models lean more on identity terms and slurs is not supported.
RQ4: three techniques have unadjusted Fisher p < 0.05, none after Holm.

Adversarial review re-ran the whole chain and found it byte-identical; it
caught an undercounted coverage figure, an RQ2 sentence at odds with
section 5, and the aliens issue. All fixed.

Checks: all pass (598 tests); check_claims OK (665); coverage OK.

Next: push to PR #23. Remaining work needs the author: Kaggle runs of
notebooks 12, 13 and 14 and the external-asset upload.

## 2026-11-05 — WP9 and WP10 notebooks written (primed, for Kaggle)

Branches `notebooks/wp10-llm` and `notebooks/wp9-probes`, merged into
`research/integration`. Both preregistrations were committed before the
notebooks. Neither notebook has been run; the authors run them on Kaggle.

* Notebook 14 (WP10): zero-shot `Qwen/Qwen3-4B-Instruct-2507` (Apache-2.0),
  fallback `microsoft/Phi-4-mini-instruct` (MIT) under label-blind rules
  only; P(extremist) from label-token log-probabilities; validation-only
  threshold; SHAP at the 07/11 settings; self-explanation overlap and
  deletion tests (one Holm family of 8). `in_comparison: false`. T4 only,
  about 1.5-3.5 h for the SHAP runs.
* Notebook 13 (WP9): linear probes on every layer of 07, the 12 runs of
  11's members and their pretrained bases, for the label, identity-term
  presence and a Hewitt-Liang control; identity-direction ablation on the
  last layer against 10 random directions. Probe code lives in
  `tools/probe_kit.py`; `notebook_kit.py` gained only a gated `probe` role.
  About 2.5-4 h; needs the 07 and 11 weights uploaded to Kaggle first.

Adversarial reviews found no blockers in either; they found ways a long
run could be wasted (a fallback message that would have evaluated a second
model on test, no early padding check, sklearn-version fragility, late
reproduction checks, saturated-sigmoid AUC ties). All fixed and recorded as
dated pre-run amendments.

Checks: all pass (589 tests); check_claims OK.

## 2026-11-05 — Faithfulness replicates on the validation split

Branch `research/faithfulness-validation`, merged into `research/integration`.
`tools/faithfulness.py --split validation` runs the deletion test on each
technique's validation whole-model SHAP run, verified against the committed
validation probabilities (max diff 0.0058, for 07); results in a separate
table, `rq2_faithfulness_validation.csv`, with its own Holm family, so the
test table and every claim are untouched. All 32 shared rows beat random
(Holm p <= 2.6e-25), and 07/11 again concentrate the effect in flagged
posts. Review: the tool change mirrors the already-reviewed test path and
adds 12 refusal tests; I recomputed Holm (max relative diff 4e-6) and
confirmed the test outputs byte-identical.

Checks: all pass (540 tests); check_claims OK; audit report byte-identical.

## 2026-11-04 — Author decisions recorded; PR #22 merged; integration pushed

The authors settled all twelve decisions (interview, 2026-11-04). Recorded
in `docs/DECISIONS_NEEDED.md`, `docs/RESEARCH_PLAN.md` sections 5 and 7 and
WP7b, and `docs/POSITIONING.md` (WP9, WP10). The local CLAUDE.md checklist
drops `--primed` for the all-notebook check and adds `check_claims.py`.
PR #22 was merged into main; `research/integration` was merged with
`origin/main`, checked (529 tests, check_claims OK) and pushed as one PR.

Next: WP3 (HurtLex slurs, `aliens`, freeze, full re-run) and the WP9 and
WP10 notebooks.

## 2026-11-04 — Citations in the Results draft

Branch `paper/citations`, merged into `research/integration`. 55 author-year
citations placed in `docs/paper/RESULTS_DRAFT.md`, all from the verified
`docs/RELATED_WORK.md`, with a References section; no number or claim
binding changed. Missing references (the classical techniques themselves,
Wilcoxon, Fisher, the bootstrap, the dataset source, TimeLMs) are
TODO(author) notes rather than invented entries.

Adversarial review checked each citation against the paper itself and
found five overstated or inverted uses, now rewritten: Hooker et al. argue
against non-retrained deletion (now cited as the limitation, not as an
endorsed shortcut); Ding and Koehn test saliency consistency, not
cross-sample rank stability; Davidson et al. 2019 is dialect bias; the
hate-latest checkpoints are cited to Antypas and Camacho-Collados. It also
caught a wrong description of our own method in the draft and in
RELATED_WORK.md: the SHAP masker replaces masked words with "...", it does
not delete them. Nine minor wordings fixed.

Checks: all five pass (529 tests); check_claims OK (632).

Next: remaining local, decision-free work is thin. Candidates: faithfulness
on the validation split for the classical techniques (CPU), and refreshing
the stale `holm_adjusted_p` notes in the eight WP7 JSONs.

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
