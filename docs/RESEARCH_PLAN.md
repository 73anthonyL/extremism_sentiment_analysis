# Research plan

Status as of 2026-11-04 (WP0, WP1, WP2, WP7 and WP8 closed). This is the working plan for the paper. It says what
the paper claims, which research question each piece of work serves, what
exists today, what is missing, in what order the gaps get closed, and what
rule decides when each question is answered. Update it when a work package
closes or a decision changes; do not let it drift from `README.md`.

`README.md` summarises the project, `docs/RESULTS_SCHEMA.md` defines every
committed artifact, and `notebooks/README.md` holds the notebook conventions.
`docs/POSITIONING.md` (2026-11-04) frames the paper as three contributions: an
auditing framework, explanation-derived findings about detection, and a
section on why the task is hard. It also adds WP7 to WP10. This document does
not repeat them.

## 1. Thesis and contribution

The paper is an **explainability study**, not a classifier paper. Its
contribution is a like-for-like SHAP attribution comparison across the model
spectrum for one binary task, extremist versus non-extremist social-media
text, on one frozen split:

* classical linear models over TF-IDF (word, character, word+character),
* a single-layer perceptron over TF-IDF,
* logistic regression over static FastText embeddings,
* fine-tuned contextual transformers with different pretraining lineages
  (Twitter-RoBERTa, HateBERT, a hate-speech-tuned RoBERTa, a plain RoBERTa
  control), and
* a logit-pooled heterogeneous transformer ensemble.

Every attribution is reduced to the same committed artifact (word-level mean
SHAP over posts containing the word) and categorised with the same four
lexicon categories: **extremist framing, identity term, slur, topical**. The
question is not which model scores highest. It is what each model relies on,
whether pretraining lineage or ensembling changes that reliance, and whether
reliance on identity terms shows up as false positives on benign posts that
mention a community.

Three framing rules follow from this and are non-negotiable in the prose:

1. Extremism detection is **not** described as unexplored. It is a studied
   task; the novelty is the cross-model attribution comparison.
2. Labels are post-level and features are text-only. We detect extremist
   content in a post. We do not detect accounts, intent, or radicalisation
   over time, and the paper says so as a limitation.
3. A higher accuracy is not a better model. The 450-row test split needs
   about 14 rows of difference for McNemar significance. "Better" and
   "outperforms" are reserved for gaps that clear that floor.

## 2. Research questions

The paper and the results folder are organised by research question, never
by model family. Each question has one tool that answers it, one committed
artifact it reads, and one done-criterion.

### RQ1. How accurate are the models, and which gaps are real?

*Question.* Under one split and one protocol, how do the model families
compare on held-out test accuracy, balanced accuracy, macro F1, ROC-AUC, and
PR-AUC, and which pairwise gaps exceed the McNemar detection floor?

*Hypothesis.* Contextual transformers clear the floor against every classical
baseline. Gaps among classical baselines, and gaps among transformer variants,
do not. The ensemble's lead over the single fine-tuned transformer (9 rows)
does not.

*Artifact.* `research_loop/probs/<TECHNIQUE>__{validation,test}.csv` and the
result folder derived from it (`tools/validate_results_folder.py`).

*Rendered as.* The `main-comparison`, `test-detail`, and `confusion-test`
tables, plus one pairwise McNemar table over discordant pairs computed from
the probability artifacts. RQ1 gets one short section in the paper.

*Done when.* Every technique in the comparison has a committed probability
artifact and a result folder derived from it, and the McNemar table is
computed from those artifacts rather than transcribed.

### RQ2. What does each model rely on?

*Question.* For each model, what share of the top-K attributed words, and of
the positive attribution mass toward `EXTREMIST`, falls in each of the four
lexicon categories?

*Hypothesis.* Classical TF-IDF models put a larger share of positive mass on
identity terms and slurs than fine-tuned transformers do, and a smaller share
on extremist framing. The character-level models sit closest to the topical
category because their words are noisier.

*Artifact.* `results_summary/<TECHNIQUE>/attributions/<run_id>.csv` with its
sidecar, produced by the notebook under the contract in
`tools/attributions.py`, categorised by `tools/categorize_attributions.py`
into `rq/rq2_category_shares.csv`.

*Precondition.* `tools/validate_shap.py` passes on the logistic-regression
run (sign agreement at least 0.90 on the top-K coefficients, Spearman at least
0.80 over shared words). No cross-model claim is made before that.

*Rendered as.* The `rq2-category-shares` table, one row per technique, and
per-technique top-word tables with category labels in the appendix.

*Done when.* Every technique in the comparison has at least one whole-model
run categorised, the LR validation check passes, and the stability check in
RQ3 shows the shares are not seed artefacts.

### RQ3. Does pretraining lineage or ensembling change that reliance?

*Question.* Holding the fine-tuning recipe fixed, do transformers with
different pretraining corpora (Twitter, hate-speech, general web) differ in
category shares? Does pooling several such models change the reliance of the
whole ensemble relative to its members?

*Hypothesis.* Hate-speech pretraining raises the share of mass on slurs and
identity terms relative to a plain RoBERTa control; Twitter pretraining
raises the share on topical and framing vocabulary. Pooling moves the
ensemble's shares toward the member mean and reduces reliance variance, but
does not remove identity-term reliance.

*Artifact.* The same attribution runs as RQ2, with per-member runs for the
ensemble (`member` set in the sidecar) and two whole-model runs per
technique over different posts: one explains the test split, one the
validation split. `tools/compare_reliance.py` writes
`rq/rq3_reliance_comparison.csv` and `rq/rq3_stability.csv`.

*Rendered as.* A pretraining-lineage table (one row per transformer, shares
of positive mass per category, with the across-run standard deviation), a
members-versus-whole table for the ensemble, and the stability table.

*Done when.* At least three transformer lineages plus the ensemble have two
or more whole-model runs each, the stability table reports the across-run
standard deviation for every share that appears in the prose, and every
difference claimed in the prose is larger than the pooled standard deviation
of the runs being compared.

### RQ4. Do identity terms drive false positives?

*Question.* Per model, is the false-positive rate on non-extremist test posts
that mention an identity term higher than on those that do not?

*Hypothesis.* Yes for every model, and the ratio is largest for the models
whose RQ2 share on identity terms is largest. Notebook 11 already shows a
ratio of 4.2 (0.119 versus 0.028, Fisher exact p = 0.006, 67 posts with
identity terms among 280 negatives).

*Artifact.* The probability artifact plus the locked threshold, read by
`tools/identity_fpr.py`, which writes `identity_fpr_test.json`, the per-term
counts, and `rq/rq4_identity_fpr.csv`.

*Rendered as.* The `rq4-identity-fpr` table and a scatter of RQ2 identity
share against RQ4 FPR ratio, one point per technique.

*Done when.* Every technique in the comparison has a row in the RQ4 table,
and the RQ2-versus-RQ4 relation is reported with its rank correlation and
the number of techniques behind it.

*Intervention.* WP7 (`tools/identity_counterfactual.py`) removes the
identity terms from the same posts and re-scores them. It reports the false
positives that clear and, for extremist posts, the recall cost of the same
edit. See WP7 for the results.

## 3. Where things stand

Techniques in the controlled comparison and what each one has today.

| Technique | Result folder | Probability artifact | Attribution run | Coefficients | RQ2 | RQ3 | RQ4 |
|---|---|---|---|---|---|---|---|
| `01_LOG-REG_TF-IDF` | yes, derived | yes | 2 whole-model | yes, check PASS | **yes** | **yes** | **yes** |
| `02_LIN-SVM_TF-IDF` | yes, derived | yes | 2 whole-model | n/a | **yes** | **yes** | **yes** |
| `03_SLP_TF-IDF` | yes, derived | yes | 2 whole-model | n/a | **yes** | **yes** | **yes** |
| `04_CHAR-TF-IDF_LIN-SVM` | yes, derived | yes | 2 whole-model | n/a | **yes** | **yes** | **yes** |
| `05_WORD-CHAR-TF-IDF_LIN-SVM` | yes, derived | yes | 2 whole-model | n/a | **yes** | **yes** | **yes** |
| `06_FASTTEXT-EMB_LOG-REG` | yes, derived | yes | 2 whole-model | yes, check FAIL (calibration point) | **yes** | **yes** | **yes** |
| `07_TWITTER-ROBERTA_FINE-TUNE` | yes, derived | yes | 2 whole-model | n/a | **yes** | **yes** | **yes** |
| `11_MULTI-CHECKPOINT_LOGIT-POOL` | yes, derived | yes | 2 whole-model + 4 member (test) | n/a | **yes** | **yes** | **yes** |

Every result folder is now derived from a committed probability artifact
(`provenance: derived_from_probs`, `recomputable: true`), from the Kaggle
reruns of 2026-10-08. The notebook 00 rerun reproduced the frozen split and
the committed mirror is byte-identical. RQ2 and RQ3 are "yes" since the
WP3 lexicon freeze (2026-11-04) and the re-run of the categorisation tools;
RQ3's pretraining-lineage contrast still waits for notebook 12.

Other facts that shape the order of work:

* **The RQ1 significance table exists** (2026-10-28). `tools/mcnemar_pairs.py`
  reads the eight test artifacts and locked thresholds. Seven of 28 pairs
  clear Holm-corrected significance: the ensemble (11) against every
  classical technique, and 07 against 04. No classical-versus-classical
  gap clears it, and neither does 07 versus 11 (9 rows). Twelve of the 450
  test posts are wrong for all eight techniques, eleven of them extremist
  posts; shared errors are mostly false negatives, while false positives
  are mostly made by one or two techniques.
* **Cross-technique word agreement exists** (2026-10-28).
  `tools/word_agreement.py` compares the committed whole-model test tables
  word by word, with no lexicon. Three words are in every technique's
  top-50 by positive attribution (a racial slur, kill, genocide), and 73 of
  the 168 words in any top-50 belong to one technique only. The word
  TF-IDF models agree with each other (Spearman 0.90, top-50 Jaccard 0.67)
  and the two transformers with each other (0.76, 0.54); across families
  agreement is 0.2 to 0.3. On their own runs, 56% (07) and 52% (11) of the
  transformers' top-50 positive words occur in a single explained post,
  against 22 to 40% for the classical models, but on 200 versus 450 posts.
  On the same 200 test posts (`rq2_same_post_support.csv`) the classical
  shares rise to 32 to 52%: char TF-IDF (04) reaches 52% and FastText (06)
  48%, while the word TF-IDF models (01, 02, 03, 05) stay 12 to 24 points
  below 07. Under a post-level bootstrap the single-post share is a
  replicate spread, not an interval: resampling shifts it upward and the
  point for 01 falls outside its own spread (`rq2_bootstrap_same_post.csv`).
  On paired draws 07's excess over the word TF-IDF models has a lower bound
  of only 0.02 to 0.04, and its difference from 04 spans -0.10 to 0.14. The
  comparison stays descriptive.
* **The identity counterfactual exists** (WP7, 2026-11-04).
  `tools/identity_counterfactual.py` re-scores the 152 test posts that
  contain an identity-lexicon match (68 non-extremist, 84 extremist; 150,
  67 and 83 before the WP3 freeze added `aliens`) with
  the matches removed, using each technique's restored predictor at its
  locked threshold. On the original texts every restored predictor matches
  its committed test probabilities to within 0.0016. On the 68
  non-extremist posts, removal lowers p(extremist) for 7 of 8 techniques
  (Holm-adjusted Wilcoxon p <= 8.1e-4). 03_SLP_TF-IDF shows no detectable
  shift (mean delta +0.002, Wilcoxon p = 0.30), which is not evidence that
  the effect is absent. For 07 and 11 the mean shift is 0.097 and 0.093 but
  the median is 0.001 and 0.003, so the shift sits in a minority of posts.
  Removal clears 7 of 9 original false positives for 07 and 7 of 8 for 11,
  against 0 to 5 of 5 to 11 for the classical techniques. These counts are
  small, no false-positive flip test survives Holm over the 32 rows
  (smallest 0.33), and they are descriptive, not a ranking. On the 84
  extremist posts the same edit unflags 15 of 71 flagged posts for 07 and
  22 of 74 for 11 (Holm-adjusted McNemar p = 0.016 and 1.5e-5); no
  classical drop survives Holm. Removal therefore trades false positives for
  recall. Substitution with a control panel is the stricter control: there
  the recall loss is Holm-significant for 01, 04, 05 and 06 and not for 07
  or 11, so the classical models weight the identity term itself as
  evidence of extremism, while the transformers' removal effect may partly
  reflect deleting a noun rather than the term specifically. The
  per-model counts, the panel and the limits are under WP7. On the 60
  extremist posts that keep an unambiguous identity match, only the 11 drop
  stays Holm-significant (see the sensitivity bullet below).
* **The SHAP attributions pass a deletion test** (2026-11-04).
  `tools/faithfulness.py` deletes each test post's top-k words by positive
  attribution toward extremist (k = 1, 3, 5, 10) and compares the drop in
  p(extremist) with deleting as many random words (5 seeds), using each
  restored predictor at its locked threshold (largest difference from the
  committed probabilities 1.9e-3). On the 196 to 200 test posts every
  technique explained, top-k deletion lowers p(extremist) more than random
  deletion for all eight techniques at every k (Holm-adjusted Wilcoxon p <=
  7.6e-21 over the 32 rows). Deletion recovers 0.46 to 0.80 of the summed
  SHAP of the deleted words. For 07 and 11 the effect sits in the flagged
  posts. Their median paired difference is 0.001 to 0.010 and 0.005 to
  0.024, against 0.034 to 0.177 for 01, 02 and 04 to 06. On flagged posts at
  k = 1 their margin over random is 0.231 and 0.204, against 0.016 to 0.091
  for the classical techniques. No cross-technique test was run, so these
  differences are descriptive. The test shows that SHAP finds the words the
  models need, not that the models are right to need them
  (`rq2_faithfulness.csv`).
  On the validation split's whole-model run
  (`rq2_faithfulness_validation.csv`; 195 to 200 shared posts, every
  predictor verified against its validation artifact at the locked
  threshold, Holm over the 32 shared validation rows) the result replicates:
  all eight techniques lose more probability with their top-k words than
  with random words at every k (Holm p <= 2.6e-25), and for 07 and 11 the
  margin again sits in the flagged posts (flagged margin about twice the
  all-post margin; median paired difference 0.002 to 0.055). Validation is
  not the test split, so this spends nothing of the test budget.
* **RQ4 and WP7 are robust to ambiguous identity words, except for 07's
  significance** (2026-11-04). `tools/identity_sensitivity.py` sets aside 13
  ambiguous lexicon entries (white, black, man, men, boy, boys, girl, girls,
  english, native, trans, asylum, indian); the lexicon is unchanged.
  Excluding the 14 non-extremist and 24 extremist test posts whose only
  matches are those words, every FPR ratio stays above 1 (1.57 to 4.91, 54
  identity-term negatives). Three techniques have Fisher p < 0.05 (04
  0.032, 05 0.019, 11 0.0062); 06, at p = 0.049 on the full lexicon, is not
  (2.36, p = 0.10). 06's full-lexicon p was 0.15 before the freeze and fell
  below 0.05 through one false positive that matches only `aliens`; that
  term came from 06's own top-50 gap list and co-occurs with "illegal" in
  all 25 dataset posts that contain it. Holm over the eight full-lexicon
  Fisher tests leaves none below 0.05 (smallest about 0.057). 03 falls to
  1.57 (p = 0.49); at this n that is not evidence that the effect is
  absent. On the 60 extremist posts that keep an unambiguous match, removal unflags
  15 of 51 flagged posts for 11 (Holm p = 0.002 over 32 rows) and 10 of 48
  for 07 (raw p = 0.012, Holm p = 0.33). 07's rate, about 21%, equals its
  full-population rate (15 of 71); with fewer discordant pairs the test
  loses power. 07's result is therefore the same effect size, not
  significant at the reduced n. Removing only the unambiguous matches
  unflags 11 of 51 for 11 (Holm p = 0.030) and 6 of 48 for 07. Only 11's
  recall loss is Holm-significant under every family tried. No non-extremist
  flip survives Holm (`rq4_identity_fpr_sensitivity.csv`,
  `rq4_identity_counterfactual_sensitivity.csv`).
* **The RQ2-to-RQ4 relation is computed** (WP5, 2026-11-04; re-run on the
  frozen WP3 lexicons). `tools/rq2_rq4_relation.py` ranks the eight
  techniques by test identity-term share of positive mass and by RQ4 FPR
  ratio: Spearman 0.87 (exact permutation p = 0.008; |rho| >= 0.73 is needed
  for p <= 0.05; leave-one-out 0.81 to 0.95; 0.83 and p = 0.017 before the
  freeze). Under section 5 this is
  descriptive at n = 8. It does not show that identity reliance goes with
  identity-term false positives. Identity share against FPR on identity-term
  negatives is -0.23, and the ratio ordering follows low FPR on the other
  negatives (05: 4 of 212, 11: 6 of 212; share against that FPR -0.54). The
  WP7 removal shift shows no clear relation to the ratio (rho 0.16). Among
  the six classical techniques the correlation is 0.71 (p = 0.14). The
  ranking is unstable under resampling.
  Resampling explained posts for the shares and test rows for the FPR ratios
  (`rq2_rq4_bootstrap_relation.csv`, 2000 replicates) gives a replicate
  median of 0.50 against the point 0.87; rho is positive in about 91% of
  replicates and below the point in 95%, with replicates spread from -0.26
  to 0.90. The instability comes mainly from the FPR ratios: resampling test
  rows alone gives a median of 0.53, posts alone 0.83. Each ratio rests on 4
  to 18 false positives, and only 04 [1.04, 6.19] and 11 [1.36, 15.5] keep
  their ratio above 1 across 95% of replicates. Resampling noisy ratios also
  pulls a rank correlation toward zero, so these figures describe how
  fragile the ordering is, not a confidence interval for rho. The relation
  stays descriptive at n = 8.
* **Transformer attributions agree less across post samples.** On the same
  200 posts, the test-versus-validation Spearman of mean absolute
  attribution is 0.43 (07) and 0.49 (11), against 0.62 to 0.74 for the
  classical techniques (`rq3_same_post_split_agreement.csv`). A post-level
  bootstrap that resamples each split's posts independently
  (`rq3_bootstrap_split_agreement.csv`) gives 07 0.43 [0.34, 0.48] and 11
  0.49 [0.41, 0.54]; FastText gives 0.62 [0.57, 0.68] and the other
  classical techniques 0.69 to 0.74, each with a lower bound of at least
  0.63. On paired draws (`rq2_rq3_bootstrap_paired_differences.csv`) every
  transformer-minus-classical difference has an interval that excludes zero
  (07: upper bound <= -0.13; 11: <= -0.08); 07 and 11 do not separate from
  each other (-0.06 [-0.13, 0.005]). This is descriptive: it covers post
  sampling only, not the SHAP seed or another split, and it is not the
  section 5 difference criterion. Lower agreement may reflect attributions
  that depend on context, which a word-level mean over different posts reads
  as instability, rather than less reliable explanations. Top-50 Jaccard by
  absolute attribution (07 0.18, 11 0.22, 06 0.19; the others 0.32 to 0.47)
  is a replicate spread under the bootstrap and does not separate the
  transformers from 06 or 04.
* **Identity shares carry more post-sampling spread than the two runs
  show.** Post-level bootstrap SDs of the identity share
  (`rq2_bootstrap_category_shares.csv`) are 0.016 to 0.036, against 0.002 to
  0.023 across the two runs in `rq3_stability.csv`. On the common posts,
  paired draws put 11 (0.17 [0.11, 0.24]) above 01, 02, 03 and 06 but not
  above 04 or 05; 07 (0.07 [0.04, 0.17]) does not separate from any
  classical technique. These intervals cover post sampling only and do not
  replace the section 5 rule; whether section 5 should use them is an author
  decision.
* **The framework has one entry point** (WP8, 2026-11-04).
  `tools/audit.py` checks a technique against the contract
  (`check-contract`), runs the documented tool chain in order and stops at
  the first failing stage (`run`), and writes one text-free report card per
  technique to `results_summary/rq/report_cards/` (`report`). Running it on
  `01_LOG-REG_TF-IDF` against the committed state reproduces every committed
  table, coefficient check and card byte for byte, so the chain is
  deterministic. The coefficient checks cover every whole-model run of 01
  and 06 (test and validation, four files); each now also records the
  Spearman over words with support >= 2 as a diagnostic (01: 0.932 / 0.939,
  06: 0.873 / 0.864), with the preregistered pass rule unchanged, so 01
  passes and 06 stays the WP2 calibration point. A FAIL on the anchor 01
  stops the chain; a FAIL on any other technique is recorded on its card,
  whose RQ2 section then rules out a cross-model claim on its words. A
  technique with `in_comparison: false` gets its own Holm family and is kept
  out of the comparison tables and the rendered docs, so auditing a new
  classifier leaves the eight-technique results unchanged. The user guide
  is `docs/FRAMEWORK.md`. The chain now also runs the ambiguous-term
  sensitivity, the lexicon-gap candidates, the same-post tables and the
  RQ2-RQ4 relation, and, with `--with-perturbation`, the two counterfactuals
  and the deletion faithfulness test; a stage whose inputs are absent prints
  SKIP with its reason. Each card gains a faithfulness section and the
  ambiguous-term rows (2026-11-04).
* **Every technique has its attribution runs.** Two whole-model SHAP runs
  per technique (test and validation samples), and one test run per
  admitted member of 11 (`A_anchor`, `B_dynabench`, `C_hatebert`,
  `E_twitter_large`; `D_deberta` again missed the 0.84 floor). The members
  are explained on the test sample only, so member-level shares have no
  across-run standard deviation; a validation-sample member loop is a
  small notebook edit if RQ3 prose needs it. Before priming, no notebook
  used SHAP for a committed artifact (07, 08, 10 and 11 computed
  gradient-times-embedding at subword level).
* **The SHAP pipeline is validated** (WP2). On notebook 01 the test run
  reaches Spearman 0.873 and the validation run 0.865 against the
  coefficients, with top-50 sign agreement 1.00 on both. Notebook 06 reaches
  Spearman 0.745 and sign agreement 1.00, below the 0.80 bar as expected
  for coefficients over embedding dimensions; it is recorded as the
  calibration point, not a failure.
* **External assets are not yet recorded.** The `external/<TECHNIQUE>/`
  folders (weights, per-post predictions and attributions; 6.2 GB for 11)
  are still in the Kaggle outputs and have not been uploaded to a Kaggle
  Dataset, so no manifest lists them.
* **Every notebook follows the primed template** (WP0). Each loads
  `tools/notebook_kit.py` and passes `tools/check_notebook.py`. 00 to 07 and
  11 carry their Kaggle outputs. 08 to 10 are empty (closed lines). 12 is
  primed and not yet run. The earlier outputs, which contained text, are at
  tag `notebooks-pre-priming`.
* **Notebook 11 already fine-tunes the lineages RQ3 needs.** Its admitted
  members are `cardiffnlp/twitter-roberta-base-hate-latest`,
  `cardiffnlp/twitter-roberta-large-hate-latest`,
  `facebook/roberta-hate-speech-dynabench-r4-target`, and `GroNLP/hateBERT`;
  `microsoft/deberta-v3-base` was excluded by the prespecified 0.84
  validation floor. Per-member attribution runs from notebook 11 cover the
  hate-speech and Twitter lineages. The plain RoBERTa control is written but
  not run. `notebooks/12_ROBERTA-BASE_FINE-TUNE.ipynb` uses 07's recipe with
  only the checkpoint changed; its head starts from the seed, while 07's
  checkpoint brings a pretrained two-class head. It passes
  `check_notebook.py --primed` and has no result folder yet. Adding it grows
  the RQ1 Holm family from 28 to 36 pairs. 01 vs 11 (Holm p 0.049 now) stays
  significant only if all eight new pairs are themselves Holm-significant
  (with equal raw p, below about 0.0017). Otherwise it rises to at most
  0.067 and the significant count falls from 7 to 6.
* **The lexicons are frozen** (WP3, 2026-11-04). `data/lexicons/slurs.txt`
  holds 58 HurtLex entries (ethnic and sexual-orientation slurs,
  conservative level); `aliens` joined the identity list. Every test top-50
  list except 03's now holds one slur (the same racial slur, two explained
  posts), at most 0.026 of positive mass, so the slur-versus-identity
  distinction is measurable but thin.
* **`tools/scan_text_leakage.py` passes.** The outputs that contained dataset
  text, hashes and row ids were cleared by priming, and the scanner now also
  checks notebook outputs and result files for row ids.
* **Notebooks 08, 09 and 10 are closed lines.** 08 and 10 evaluated the test
  split earlier without exporting probabilities, and 09 was never run. They
  stay out of the comparison and their earlier test numbers are not cited.
  They are primed to the same standard as the rest (08 renamed without
  `BEST`) so they can be rerun if their question is reopened.
* The champion-promotion loop (ledger, judge, `research_loop/registry.json`)
  is archived. The registry still names notebook 07 as champion. That field
  is historical data and is not updated; the paper does not use the word
  "champion".

## 4. Work packages

In dependency order. A package is closed when its done-criterion holds and
the four pre-commit checks pass. Kaggle runs are the expensive step, so each
notebook is rerun once with every export cell in place rather than once per
missing artifact.

### WP0. Repository hygiene (local, no GPU). Closed 2026-10-02.

Fixes that blocked everything downstream because they broke the naming
contract or the leakage rule. What was done:

1. `tools/notebook_kit.py`: one helper every notebook loads, which owns the
   frozen-split assertion, the metric function (pinned to
   `tools/metrics_core.py` by a test), threshold selection on validation, the
   single test evaluation, and the writers for the probability artifact, the
   attribution runs, the coefficient file and the result folder.
2. `tools/check_notebook.py`: a static check of a notebook against the
   template, since notebooks are never executed locally.
3. All twelve notebooks rewritten on the template. `technique_name` equals
   the filename stem everywhere; `split_version` is the full frozen string;
   notebook 00 verifies the split against the committed assignment instead of
   being free to overwrite it; no cell prints dataset text, row ids or text
   hashes; `include_text_preview` is false; every model notebook exports the
   probability artifact and its attribution runs, and 01 and 06 export
   `coefficients.csv`.
4. `08_BEST-ROBERTA_SEED-ENSEMBLE.ipynb` renamed to
   `08_TWITTER-ROBERTA_SEED-ENSEMBLE.ipynb`. Notebooks 01, 03 and 05 are back
   under their canonical names.
5. Two rules were unified across techniques, so reruns will not reproduce the
   earlier numbers exactly: configurations are ranked by validation accuracy
   at a screening threshold of 0.5, and the decision threshold is selected
   once on validation, maximizing accuracy over one shared grid. Earlier
   notebooks tuned a threshold per configuration (and some per epoch) for
   positive F1.

*Done.* `tools/check_notebook.py --all --primed` reports no problems, and
`tools/tests/test_notebook_pipeline.py` drives the kit end to end on
synthetic text and passes the real coefficient check.

### WP1. Probability artifacts for 01 to 07 (Kaggle, CPU for 01 to 06, GPU for 07). Closed 2026-10-08.

Rerun each notebook once with the WP0 cells in place. Each run exports
`<TECHNIQUE>__validation.csv`, `<TECHNIQUE>__test.csv`, and the meta sidecar
to `research_loop/probs/`, rebuilds the result folder from those
probabilities with `compute_binary_metrics`, and writes the attribution run
and `run_manifest.json`. Weights, logs and per-post files go to a Kaggle
Dataset and are recorded with `run_manifest.py add-asset`.

The reruns will not reproduce the old numbers exactly: the threshold and
ranking rules were unified in WP0, early-stopping monitors changed in 03, 07
and 08, and GPU training was not deterministic before. That is fine: each old
folder is replaced, the change is stated in the commit, and the rendered
tables are regenerated. Thresholds are selected on validation only. Test is
evaluated once per rerun. Run order: 00 first, then 01 and its coefficient
check, then the rest.

*Done when.* All eight techniques show `provenance: derived_from_probs` and
`recomputable: true`, `validate_results_folder.py --all` passes, and the RQ1
McNemar table is computed from the artifacts.

### WP2. Validate the SHAP pipeline (local). Closed 2026-10-08.

Run `tools/validate_shap.py --technique 01_LOG-REG_TF-IDF` against the
coefficients and attribution run from WP1. If the sign-agreement or Spearman
bar is missed, the explainer configuration (background set, masker,
aggregation) is wrong and is fixed in the notebook before any other technique
is categorised. Repeat for `06_FASTTEXT-EMB_LOG-REG`, where agreement is
expected to be weaker because the coefficients act on embedding dimensions,
and record the weaker result as a calibration point rather than a failure.

*Done when.* The check file for notebook 01 reports both bars met and is
committed.

### WP3. Lexicon curation (local)

1. Populate `data/lexicons/slurs.txt` from a published, citable slur lexicon
   (record the source and licence in `data/lexicons/README.md`). Entries
   of the source that are neutral identity references already in
   `identity_terms.txt` are removed from the slur list and stay identity
   terms. (This step first said the reverse: remove them from the identity
   list, since `slur` takes precedence. Changed at the freeze, 2026-11-04,
   because HurtLex's slur categories contain neutral words such as `gay` and
   `queer`; dropping them from the identity list would remove neutral
   mentions of those groups from RQ4, which measures false positives on
   exactly such mentions.)
2. Review `identity_terms.txt` (96 entries) for coverage of the communities
   the dataset actually mentions. Use `identity_fpr_terms_test.csv` counts,
   which carry no text, to find high-support terms.
3. Review `extremist_framing.txt` (382 entries, seeded from the single-word
   entries of `data/extremism_lexicon.txt`) and remove entries that are
   really identity terms or slurs.
4. Freeze the three lists with a version note. Any later change re-runs RQ2
   and RQ4 and is described in the commit.

*Done when.* The slur list is non-empty and sourced, the overlap report
from `categorize_attributions.py` shows no identity-versus-slur collisions,
and the lists are frozen.

*Status: done 2026-11-04.* Slurs: HurtLex EN 1.2, conservative level,
categories `ps` and `om`, 58 entries after removing generic insults,
everyday-sense words, neutral references and non-lemmas (CC BY-NC-SA 4.0;
source, licence and the removed words in `data/lexicons/README.md` and the
header of `slurs.txt`). Identity list: `aliens` added (97 entries), the 13
ambiguous entries kept. Framing list reviewed and unchanged; no entry is
shared between any two lists. Version note with sha256 in
`data/lexicons/README.md`. The WP7b swap map gained an excluded row for
`aliens` (Amendment 1 of its preregistration; population unchanged). RQ2,
RQ3, RQ4, WP7, WP7b, the relation and the bootstrap were re-run.

### WP4. Attribution runs for RQ2 and RQ3 (Kaggle)

The primed notebooks export every attribution run in the same Kaggle run
that trains the model, so most of this package is delivered by the WP1
reruns:

1. **Two whole-model runs per technique.** Run 1 explains the test split and
   run 2 the validation split, with the same seed. The partition explainer
   over a text masker has no background sample and almost no randomness, so
   a second seed would show artificially small variance; a second set of
   posts shows how stable the result is across posts, which is what
   `rq3_stability.csv` should measure.
2. **Per-member runs from notebook 11**, one per admitted component (its
   three seeds pooled), with `member` set in the sidecar, so
   `compare_reliance.py` can write the `member_mean` row. The excluded
   components are not explained.
3. **Notebook 12, `12_ROBERTA-BASE_FINE-TUNE`** (still to be written):
   `roberta-base` fine-tuned with notebook 07's recipe. This is the
   no-domain-pretraining control RQ3 lacks. It is built from the transformer
   template and exports both artifacts. Its accuracy is reported in RQ1
   without comment unless it clears the McNemar floor against 07.

The explainer for every family is SHAP's partition explainer over a
word-level text masker (`shap.maskers.Text(r"\W+")`), applied to the model's
`predict_proba_texts` function. The masker tokenizes at word level, so no
subword aggregation is needed and every family lands in one vocabulary. No
notebook used this method before WP0; an earlier version of this plan said
notebook 11 did, which was wrong (it used gradient-times-embedding).
Notebook 07 keeps gradient-times-embedding as a supplementary comparison,
aggregated to words and written to external storage, because the paper's
comparison must use one attribution method everywhere.

Classical notebooks explain all 450 posts of each split. Transformer and
ensemble notebooks explain a label-stratified sample of 200 posts per split
(the same sample for the whole model and every member), with 500 model
evaluations per post; both numbers are recorded in each sidecar.

*Done when.* Every technique has at least two whole-model runs, notebook 11
has one run per member, notebook 12 has a result folder and both artifacts,
and `compare_reliance.py` runs without skipping any technique.

### WP5. Answer the questions (local)

In this order, on the committed artifacts, after WP2 and WP3:

```bash
python3 tools/validate_shap.py --technique 01_LOG-REG_TF-IDF
python3 tools/categorize_attributions.py --all
python3 tools/compare_reliance.py
python3 tools/identity_fpr.py --all
python3 tools/identity_counterfactual.py --all
python3 tools/render_tables.py --write
```

Then compute the two things the tools do not yet produce, and add them as
tools with adversarial tests:

* `tools/mcnemar_pairs.py` (done 2026-10-28): pairwise McNemar exact test
  over discordant pairs for every technique pair, from the test probability
  artifacts and locked thresholds, with Holm correction over the family.
  This is the RQ1 significance table. It also writes the error-overlap
  tables (`rq1_error_overlap.csv`, `rq1_shared_errors.csv`).
* `tools/rq2_rq4_relation.py`: the rank correlation between each technique's
  RQ2 identity-term share and its RQ4 FPR ratio, from the two `rq/` tables.
  Done 2026-11-04 (`rq2_rq4_relation.csv`), and re-run on the frozen WP3
  lexicons.

*Done when.* All five rendered regions are populated for every technique,
`render_tables.py --check` passes, and the two new tools have tests that
refuse a malformed input.

### WP5a. Curated examples (local, after the reruns)

A hand-curated list of texts, one per line of a `.txt` file, classified and
explained by every technique side by side, for the qualitative discussion of
RQ2-RQ4. `tools/curated_examples.py explain` restores each technique's
`predict_proba_texts` from its downloaded `external/<TECHNIQUE>/` folder
(`tools/model_loaders.py`), verifies it against the committed test
probability artifact, and runs `notebook_kit.explain` on the curated texts;
`report` writes a text-free `summary.csv` and a `report.html` with every
text under every model. `tools/curated_plots.py` draws a SHAP waterfall per
text and a word beeswarm per technique (whole models) from those
attributions, with an HTML gallery, and `tools/curated_site.py` copies the
gallery into a static site with a method page, published at
<https://73anthonyl.github.io/curated-shap-gallery/>. `tools/split_beeswarms.py`
adds word beeswarms over the test and validation posts each notebook
explained, checked against the committed word tables, as the site's
dataset page. Outputs stay under `external/`.

The curated texts are not dataset rows and carry no dataset label (only an
optional hand-assigned expected label), so they support
no accuracy claim and enter no rendered table. Notebooks 08, 09 and 10 have
no loader, as closed lines.

*Done when.* Every technique in the comparison passes verification and
appears in `report.html` for every curated text.

### WP7. Identity counterfactual test (local). Closed 2026-11-04.

(The counts in this section are from the first run, before the WP3 freeze
added `aliens`; the re-run on 152 posts, 68 non-extremist and 84
extremist, keeps every significance and direction, see section 3 and
`docs/paper/RESULTS_DRAFT.md` section 8.2.)

`tools/identity_counterfactual.py` takes the 150 test posts with at least one
identity-lexicon match (67 non-extremist, 83 extremist). It edits them and
re-scores them with each technique's restored predictor, at the technique's
locked threshold. Before scoring, the predictor re-scores the original texts
and must match the committed artifact. The largest difference is 6e-8 for
the classical models and 1.6e-3 for 07 and 11 served on MPS.
p_original is always the committed probability. The outputs are
`identity_counterfactual_test.json` per technique and
`rq/rq4_identity_counterfactual.csv`. p-values are Holm-adjusted over all 32
rows, as section 5 requires.

*Remove (primary).* Every identity match is deleted.

* Non-extremist posts. Removal lowers p(extremist) for 7 of 8 techniques,
  each with Holm-adjusted Wilcoxon p <= 0.0014. 03_SLP_TF-IDF shows no
  detectable shift at n = 67: mean delta +0.001, median 3e-5, 51% of posts
  positive, Wilcoxon p = 0.48. That is not evidence that the effect is
  absent. For the five other classical techniques the mean delta is 0.020
  to 0.040 and the median 0.010 to 0.034. For 07 and 11 the mean is about
  0.095 to 0.099 but the median is 0.001 to 0.003, so their shift comes from
  a minority of posts.
* Flagged non-extremist posts, original -> after removal, with the false
  positives that clear and the new ones in brackets:
  01 11 -> 8 (3 / 0), 02 10 -> 8 (3 / 1), 03 6 -> 7 (0 / 1), 04 9 -> 4 (5 / 0),
  05 5 -> 4 (1 / 0), 06 7 -> 7 (1 / 1), 07 9 -> 2 (7 / 0), 11 8 -> 2 (7 / 1).
  These counts are descriptive. None of the flip tests survives Holm (the
  smallest is 07 at 0.33). With 5 to 11 original false positives the exact
  McNemar test has little power: 7 cleared and 0 new gives p = 0.016 before
  correction. The counts are also below the 14-row gap, so they do not rank
  the techniques.
* Extremist posts, the recall cost of the same edit. Flagged extremist
  posts go 01 62 -> 56, 02 60 -> 50, 03 51 -> 49, 04 53 -> 45,
  05 53 -> 47, 06 52 -> 43, 07 71 -> 57 and 11 74 -> 52. Only the 07 drop
  (15 lost, 1 gained, Holm p = 0.016) and the 11 drop (22 lost, 0 gained,
  Holm p = 1.5e-5) survive Holm. This matches RQ4: recall for 07 and 11 is
  0.855 and 0.892 on extremist posts with identity terms, against 0.839 and
  0.793 without. Under removal the transformers respond as if the term were
  evidence of extremism whatever the label; the substitution check below
  does not reproduce that for them, so the effect may partly come from
  deleting a noun. The term is not only a source of errors. Every
  false-positive reduction above is reported next to this cost.

*Substitute (sensitivity check).* Every match is replaced by each word of a
control panel, "group", "population" and "humans", and the probabilities are
averaged over the panel. The tool refuses a panel word if any lexicon
matches it, if its share of extremist posts in the training split is more
than 0.10 from the base rate of 0.376, or if its 01 coefficient has
|coef| > 0.5. The panel words occur in 10, 7 and 5 training posts, with
extremist shares of 0.40, 0.43 and 0.40 and coefficients +0.17, -0.18 and
-0.49 (ranks 1510, 3001 and 4225 of 4692). The JSON records all of these.
Under substitution the extremist recall drop is Holm-significant for 01, 04,
05 and 06 (11 lost each, Holm p = 0.029) and not for 07 (1.0) or 11 (0.094),
so the recall cost of an identity-term edit is not specific to the
transformers.

* The first version substituted "people", which is not a neutral control.
  It occurs in 219 extremist and 116 non-extremist dataset posts. 63% of
  its training posts are extremist, and its LR coefficient (+3.20) is the
  10th largest of 4692. With "people", every classical technique moved
  toward extremist, so that run compared an identity term with an
  extremist-associated word. The panel replaces it, and a test refuses a
  word with that profile.
* With the panel, the classical techniques shift about as much as under
  removal: non-extremist mean delta 0.028 to 0.045, and 03 again shows no
  detectable shift (p = 0.31). For 07 and 11 the shift on non-extremist
  posts is small and depends on the word. 07 has mean 0.012 and median
  -0.0004 (Wilcoxon p = 0.06). Its per-word means are +0.073 for "group",
  +0.028 for "population" and -0.065 for "humans". 11 has mean 0.038 and
  median about 0 (p = 0.34).
* For the transformers, the removal shift is therefore not reproduced when a
  neutral group noun takes the term's place. Removal cannot be attributed to
  the identity term specifically: it may come from losing the group referent,
  or from the ungrammatical text that deletion leaves.

*Limitations.* A flip shows that this one perturbation moved the post across
the locked threshold. That is threshold-local necessity, not proof that the
term causes the error. Deletion leaves ungrammatical text. For TF-IDF models
it also changes the L2 normalisation and creates character n-grams across the
join. The panel words are rare in training (5 to 10 posts), so their balance
is estimated from few posts. Words never seen in training act like removal
for the word-level models. The lexicon covers only the identity terms it
lists. All of this is post-level and text-only.

### WP7b. Identity swap (local, sensitivity analysis of WP7). Run 2026-11-04.

WP7's substitution check left one question open: do the transformers respond
to which group is named, or only to a group being named? WP7b replaces each
identity term with three other identity terms of the same lexicon category
and grammatical form (religion, ethnicity/nationality, migration, gender and
sexuality), averages the probabilities over the three, and compares the
shift with removal and with WP7's neutral panel on the same spans of the
same posts.

The swap map (`data/lexicons/identity_swap_pairs.csv`) and the analysis plan
(`docs/preregistrations/WP7b_identity_swap.md`) were committed before any
model was run on swapped text, so the metrics, the Holm family (the 48
technique x population x operation rows) and the reading rule are fixed in
advance. WP7b is still a sensitivity analysis of RQ4, not a primary outcome:
WP7's removal stays the primary operation. The map excludes the 13
ambiguous entries and 8 more (`indians`, `woman`, `women`, `feminist`,
`feminists`, `queer`, `lgbt`, `lgbtq`), so the population is the 68 test
posts with a swapped term (23 non-extremist, 45 extremist), a subset of
WP7's 150. Operations edit only the swapped spans and match their case, so
the remove and substitute rows here differ from WP7's.

`tools/identity_swap.py` refuses a map that leaves the lexicon, crosses a
category, changes form, repeats the source's group or uses an ambiguous
term. It verifies each restored predictor against the committed artifact
before scoring. It writes `identity_swap_test.json` per technique,
`rq/rq4_identity_swap.csv` and the descriptive
`rq/rq4_identity_swap_directions.csv`. The authors adopted the cautious
reading on 2026-11-04 (`docs/DECISIONS_NEEDED.md`, decision 5): substitution
is the stricter control; the classical models weigh identity terms under
every edit; the transformers respond to a group being named rather than
which group, and part of their removal effect may be the loss of a noun. The
paper does not claim the transformers treat identity terms as evidence of
extremism. Folding WP7b into the results draft is pending.

*Results, descriptive, with the preregistered rule applied as written.*
Here delta is p(extremist) before minus after the edit, and p values are
Holm-adjusted Wilcoxon over the 48 rows of `rq4_identity_swap.csv`.

* Labels. The rule gives "group mention" on both populations for 01, 02
  and 04. For 03 and 05 it gives group mention on non-extremist posts and
  "partial" on extremist posts. There the swap shifts are tiny (0.0045 and
  0.022) but still Holm-significant (0.021 and 0.016), and below half of
  removal. 06 is "group-specific" on extremist posts (swap 0.027, Holm
  0.0056, 0.75 of the removal shift). On non-extremist posts 06 is partial
  (swap 0.027, Holm 2.1e-4, 0.47 of removal). 07 and 11 are group mention on
  extremist posts and partial on non-extremist posts.
* Extremist posts. The swap moves 07 and 11 by 0.006 and 0.004 (Holm 1.0
  for both), against 0.069 and 0.149 under removal, and unflags none of
  their 34 and 38 flagged posts. For 11 the neutral substitute also lowers p
  (0.101, Holm 8.1e-6), so for 11 the presence of an identity term matters,
  not which group it names. For 07 the substitute shift (0.043, Holm 0.12)
  is not significant, so group mention and noun deletion cannot be told
  apart.
* Non-extremist posts. The swap raises p: 11 by 0.059 (Holm 0.022) and 07
  by 0.100 (Holm 0.087, not significant). Both are partial under the rule
  because the sign is the reverse of removal's. The medians are about zero
  (-0.006 and -0.002), so the shift sits in few posts. In the
  single-source directions, three sources carry about 72% of the summed
  upward shift for 07 (refugees, muslim, american) and 73% for 11
  (refugees, american, christian). The sign varies by source: swapping away
  christians lowers p for both models, and muslims for 11 (+0.068). The targets' training-label skew
  does not explain this. Across non-extremist directions, the shift's
  Spearman correlation with the source-minus-target 01 coefficient is
  -0.15 (07) and -0.10 (11), and with the difference in training extremist
  share +0.13 and +0.09. On extremist posts the correlations are +0.21 and
  +0.32 (coefficient) and +0.21 and +0.23 (share).
* Classical models. The averaged swap shift is near zero, but the
  per-variant mean |delta| is 0.91 (non-extremist) and 0.75 (extremist) of
  removal's for 01, and 0.75 and 0.77 for 02. Targets that raise p and
  targets that lower it cancel in the mean. Per-direction shifts track the
  change in 01 coefficient: Spearman 0.92 for 01 and 0.88 for 02 over 114
  directions, then 0.72 (03), 0.67 (05), 0.41 (04) and 0.25 (06). Targets
  never seen in training count as coefficient 0. This is direction-dependent
  sensitivity, not an absence of response.
* Magnitude, extremist posts, swap mean |delta| over remove mean |delta|:
  11 0.24, 07 0.46, 01 0.75, 02 0.77. Descriptive only.
* Precision. p_original is the committed artifact, while every
  counterfactual for 07 and 11 was re-scored on MPS. On the original texts
  the two differ by at most 1.4e-3 (07) and 7.7e-4 (11). That is the same
  order as the extremist swap deltas (0.006 and 0.004) and larger than their
  medians (2e-5 and 3.4e-4).

*Limitations.* The panel fixes one set of comparison groups. Swapping two
co-occurring groups can leave a post's set of groups unchanged. The
non-extremist population is small (23 posts, 2 to 7 flagged), so flip
counts there are descriptive. Word-level models score a target unseen in
training like a deletion. Post-level labels, text-only features.

### WP10. Zero-shot LLM with self-explanation (Kaggle GPU). Primed 2026-11-05.

Author decision 6: an open-weight instruction-tuned model on a Kaggle GPU, so
no dataset text leaves the authors' control. Preregistered in
`docs/preregistrations/WP10_zero_shot_llm.md` before the notebook was
written: `Qwen/Qwen3-4B-Instruct-2507` (Apache-2.0) at a pinned revision, with
`microsoft/Phi-4-mini-instruct` (MIT) as a fallback chosen only by
label-blind rules; one fixed zero-shot prompt; P(EXTREMIST) from the two
answer tokens' log-probabilities; threshold on validation; one test
evaluation. The two-artifact contract is met with the settings of 07 and 11
(200 posts per split, `max_evals` 500), so the model enters RQ2 to RQ4 as an
audited technique with `in_comparison: false` (its own Holm family). The
self-explanation analysis asks the model for at most five decisive words on
the 200 test posts and compares them with its own SHAP attributions:
overlap@k against chance, and deletion of the self-named tokens against
SHAP's top tokens of the same count and random tokens, eight Wilcoxon tests
under one Holm family, read by the preregistered rule. Notebook:
`notebooks/14_ZERO-SHOT-LLM_SELF-EXPLAIN.ipynb`, primed, not yet run.

*Done when.* The notebook has run on Kaggle, its outputs pass
`check_notebook.py` and `scan_text_leakage.py`, and `audit.py run` and
`report` have produced its card.

### WP6. Write the paper

Sections map one-to-one onto the research questions:

1. Introduction: the task, why attribution comparison rather than accuracy,
   the three framing rules.
2. Data and protocol: dataset, frozen split, one test evaluation per
   technique, McNemar floor, the derived-not-transcribed rule.
3. Models: one paragraph per family; the recipe shared by 07, 11's members,
   and 12.
4. Attribution method: SHAP, word aggregation, the LR validation check, the
   four lexicon categories and their sources.
5. RQ1 (short). 6. RQ2. 7. RQ3. 8. RQ4.
9. Limitations: post-level labels, text-only features, one dataset, 450-row
   test split, lexicon coverage, SHAP on subword models, no account or
   temporal signal, not a deployment recommendation.
10. Responsible use, as in the README.

*Done when.* Every number in the paper traces to a committed artifact or a
rendered table, and the prose contains none of the banned phrasings in
section 1.

## 5. Decision rules

These decide what the paper may say. They are fixed now so the results do
not choose them.

**Accuracy (RQ1).** A technique is called more accurate than another only if
the McNemar exact two-sided test on discordant test pairs is significant
after Holm correction over all pairs reported. Otherwise the pair is reported
as not distinguishable at n = 450, and the ordering is presented as
descriptive.

**Reliance (RQ2, RQ3).** A category share is reported only with its
across-run standard deviation from `rq3_stability.csv`. A difference in
share between two techniques is described as a difference only if it exceeds
the pooled across-run standard deviation of the two. This preregistered rule
stays the primary criterion (author decision, 2026-11-04); the post-level
bootstrap intervals (`rq2_bootstrap_category_shares.csv`) are reported next
to it as a stricter check, and the paper says which differences pass both. Top-K is 50 throughout;
changing it re-runs everything and is stated. Runs are compared only when
they explain the same number of posts with the same evaluation budget; the
sidecar records both. Lexicon entries the word-level masker splits
(spaces, hyphens, other punctuation) cannot match word-level artifacts and
the count of skipped entries is reported.

**Identity false positives (RQ4).** The Fisher exact test on the 2x2 table
(identity term present or absent, false positive or true negative) is
reported with the FPR ratio and the count of negatives with identity terms.
With 68 such posts in the test split (67 before the WP3 freeze; the count
was updated on 2026-11-04 when `aliens` joined the identity list, and the
rule itself did not change), only large ratios are detectable; a
non-significant result is reported as such, not as absence of effect. The
RQ2-to-RQ4 relation is reported as a rank correlation over techniques with
n stated; with eight or nine points it is descriptive.

**Identity counterfactual (WP7).** Removal is the primary operation and
substitution is a sensitivity check. A substitute word must pass the
neutrality check in `identity_counterfactual.py`. Wilcoxon and exact McNemar
p-values are Holm-adjusted over every technique x population x operation
row of `rq4_identity_counterfactual.csv`, one family per test, as for RQ1.
Flip counts are always given with n_flagged_original, and the counts on
non-extremist posts are descriptive unless they survive Holm. A
false-positive reduction is never reported without the recall change on
extremist posts. Differences between techniques are not ranked below the
14-row gap.

**Threshold.** Every RQ4 rate uses the technique's locked validation-selected
threshold. No threshold is re-chosen to change an RQ4 result.

**Test budget.** One test evaluation per technique, ever. A rerun in WP1
replaces the earlier evaluation rather than adding one. Notebooks 12 and 14
are the only new test unlocks this plan authorises (14: WP10, approved by
author decision 6 on 2026-11-04, a new technique outside the comparison). WP7 re-scores edited test texts
with frozen models at their locked thresholds. That is a perturbation
analysis, not a test evaluation: it selects no threshold, model or
configuration and does not count against this budget. `tools/faithfulness.py`
and `tools/identity_sensitivity.py counterfactual` likewise re-score perturbed
test texts with frozen models; they select nothing.

**Lexicons.** Frozen after WP3. If a reviewer's question forces a change,
the change is made once, RQ2 and RQ4 are re-run in full, and both versions of
the affected table are kept in the appendix.

## 6. Out of scope

Not part of this paper, and not to be started on this branch of work:

* New classifiers chosen for accuracy. Notebooks 08, 09 and 10 are closed
  lines. No seed ensembles, stackers, calibration, or threshold re-derivation.
* Account-level, temporal, network, or multimodal signals.
* Multi-class ideology labels or a new dataset. External validation is a
  limitation, not a work package.
* Deployment, moderation tooling, or any per-user decision. The auditing
  framework in `docs/POSITIONING.md` is a research toolkit for evaluating
  classifiers, not moderation tooling.
* Re-running the champion-promotion loop. `legacy_tools/` is reference only.

## 7. Open decisions for the authors

All open decisions were settled on 2026-11-04; `docs/DECISIONS_NEEDED.md`
records each with its context and the option chosen. In short: slurs from
HurtLex (conservative level, English slur categories, source and licence
recorded in `data/lexicons/README.md`); identity list gains only terms with
support >= 3 (currently `aliens`), ambiguous entries kept and reported via
the sensitivity check; the section 5 share rule stays primary with the
bootstrap reported alongside; notebook 12 will be run by the authors; WP7
and WP7b take the cautious reading; WP10 uses an open-weight model on a
Kaggle GPU, so no dataset text leaves the authors' control; WP9 probes are
done now (notebook written locally, run on Kaggle); report-card slurs are
redacted in figures only; the SHAP check keeps its preregistered rule;
external assets are uploaded before submission. Earlier decisions (SHAP
background, notebook 12 twin, DeBERTa member, RQ1 Holm family) stand as
recorded in the git history of this section.

## 8. Order of execution

```text
WP0 hygiene ──► WP1 reruns 01-07 ──► WP2 SHAP check ──► WP4 runs + nb12 ──► WP5 answers ──► WP6 paper
                                  └─► WP3 lexicons ────────────────────────┘
```

WP3 runs in parallel with WP1 and WP2 because it needs no GPU and no new
artifacts. WP4 cannot start before WP2 passes, because a failed pipeline
check means the explainer is wrong and every attribution run would have to
be redone. WP5 waits on WP3 because the lexicons are frozen there.
