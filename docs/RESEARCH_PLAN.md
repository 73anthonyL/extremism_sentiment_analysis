# Research plan

Status as of 2026-10-02 (WP0 closed). This is the working plan for the paper. It says what
the paper claims, which research question each piece of work serves, what
exists today, what is missing, in what order the gaps get closed, and what
rule decides when each question is answered. Update it when a work package
closes or a decision changes; do not let it drift from `README.md`.

`README.md` summarises the project, `docs/RESULTS_SCHEMA.md` defines every
committed artifact, and `notebooks/README.md` holds the notebook conventions.
This document does not repeat them.

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

## 3. Where things stand

Techniques in the controlled comparison and what each one has today.

| Technique | Result folder | Probability artifact | Attribution run | Coefficients | RQ2 | RQ3 | RQ4 |
|---|---|---|---|---|---|---|---|
| `01_LOG-REG_TF-IDF` | yes, not derivable | none | none | none | no | no | no |
| `02_LIN-SVM_TF-IDF` | yes, not derivable | none | none | n/a | no | no | no |
| `03_SLP_TF-IDF` | yes, not derivable | none | none | n/a | no | no | no |
| `04_CHAR-TF-IDF_LIN-SVM` | yes, not derivable | none | none | n/a | no | no | no |
| `05_WORD-CHAR-TF-IDF_LIN-SVM` | yes, not derivable | none | none | n/a | no | no | no |
| `06_FASTTEXT-EMB_LOG-REG` | yes, not derivable | none | none | none | no | no | no |
| `07_TWITTER-ROBERTA_FINE-TUNE` | yes, not derivable | none | none | n/a | no | no | no |
| `11_MULTI-CHECKPOINT_LOGIT-POOL` | yes, derived | yes | none | n/a | no | no | **yes** |

"Not derivable" means the folder's numbers came from notebook cell outputs
and no probability artifact exists to recompute them. The RQ1 tables are
therefore still transcribed for seven of eight techniques, which is exactly
what `docs/RESULTS_SCHEMA.md` says is not acceptable.

Other facts that shape the order of work:

* **No attribution run exists yet for any technique.** RQ2 and RQ3 are
  empty until the notebooks are rerun. Before priming, no notebook exported
  the contract artifact and no notebook used SHAP for one: 07, 08, 10 and 11
  computed gradient-times-embedding at subword level, 01 and 03 had a
  `LinearExplainer` plot cell added on 2026-09-14, and the others used
  coefficient, margin or embedding-direction contributions. Every primed
  notebook now exports two whole-model SHAP runs, and the ensembles one run
  per member.
* **All twelve notebooks are primed** (WP0). Each follows one template, loads
  `tools/notebook_kit.py`, and passes `tools/check_notebook.py --primed`.
  Their outputs are empty until rerun; the earlier outputs are at tag
  `notebooks-pre-priming`.
* **Notebook 11 already fine-tunes the lineages RQ3 needs.** Its admitted
  members are `cardiffnlp/twitter-roberta-base-hate-latest`,
  `cardiffnlp/twitter-roberta-large-hate-latest`,
  `facebook/roberta-hate-speech-dynabench-r4-target`, and `GroNLP/hateBERT`;
  `microsoft/deberta-v3-base` was excluded by the prespecified 0.84
  validation floor. Per-member attribution runs from notebook 11 cover the
  hate-speech and Twitter lineages. What is missing is a **plain RoBERTa
  control** with no domain pretraining.
* **The slur lexicon is empty** (`data/lexicons/slurs.txt`). Until it is
  populated, every slur is counted as topical or as extremist framing, which
  biases RQ2 and hides the slur-versus-identity distinction RQ3 turns on.
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

### WP1. Probability artifacts for 01 to 07 (Kaggle, CPU for 01 to 06, GPU for 07)

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

### WP2. Validate the SHAP pipeline (local)

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
   that also appear in `identity_terms.txt` are removed from the identity
   list, since the first-match order gives `slur` precedence anyway.
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
python3 tools/render_tables.py --write
```

Then compute the two things the tools do not yet produce, and add them as
tools with adversarial tests:

* `tools/mcnemar_pairs.py`: pairwise McNemar exact test over discordant
  pairs for every technique pair, from the test probability artifacts and
  locked thresholds, with Holm correction over the family. This is the RQ1
  significance table.
* `tools/rq2_rq4_relation.py`: the rank correlation between each technique's
  RQ2 identity-term share and its RQ4 FPR ratio, from the two `rq/` tables.

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
text under every model. Outputs stay under `external/curated_examples/`.

The curated texts are not dataset rows and carry no label, so they support
no accuracy claim and enter no rendered table. Notebooks 08, 09 and 10 have
no loader, as closed lines.

*Done when.* Every technique in the comparison passes verification and
appears in `report.html` for every curated text.

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
the pooled across-run standard deviation of the two. Top-K is 50 throughout;
changing it re-runs everything and is stated. Runs are compared only when
they explain the same number of posts with the same evaluation budget; the
sidecar records both. Multi-word lexicon entries
cannot match word-level artifacts and the count of skipped entries is
reported.

**Identity false positives (RQ4).** The Fisher exact test on the 2x2 table
(identity term present or absent, false positive or true negative) is
reported with the FPR ratio and the count of negatives with identity terms.
With 67 such posts in the test split, only large ratios are detectable; a
non-significant result is reported as such, not as absence of effect. The
RQ2-to-RQ4 relation is reported as a rank correlation over techniques with
n stated; with eight or nine points it is descriptive.

**Threshold.** Every RQ4 rate uses the technique's locked validation-selected
threshold. No threshold is re-chosen to change an RQ4 result.

**Test budget.** One test evaluation per technique, ever. A rerun in WP1
replaces the earlier evaluation rather than adding one. Notebook 12 is the
only new test unlock this plan authorises.

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
* Deployment, moderation tooling, or any per-user decision.
* Re-running the champion-promotion loop. `legacy_tools/` is reference only.

## 7. Open decisions for the authors

Each of these changes an artifact and must be settled before the package
that depends on it closes.

| Decision | Blocks | Default if not decided |
|---|---|---|
| Source lexicon for `slurs.txt` | WP3 | Not populated; RQ2 slur column reported as unavailable and the limitation stated. |
| ~~SHAP background size and seeds for the second runs~~ | WP4 | Decided 2026-10-02: the text masker uses no background sample; run 2 explains the validation split with the same seed; transformer families explain 200 posts per split. |
| Whether notebook 12 also gets a large-scale twin | WP4 | No. One control is enough for the lineage question. |
| Whether per-member runs for 11 include the excluded DeBERTa member | WP4 | No. Excluded members are not part of the ensemble's reliance. |
| Holm family for RQ1: all 36 pairs or only the 8 pairs against 07 | WP5 | All pairs reported; the 07 column highlighted. |

## 8. Order of execution

```text
WP0 hygiene ──► WP1 reruns 01-07 ──► WP2 SHAP check ──► WP4 runs + nb12 ──► WP5 answers ──► WP6 paper
                                  └─► WP3 lexicons ────────────────────────┘
```

WP3 runs in parallel with WP1 and WP2 because it needs no GPU and no new
artifacts. WP4 cannot start before WP2 passes, because a failed pipeline
check means the explainer is wrong and every attribution run would have to
be redone. WP5 waits on WP3 because the lexicons are frozen there.
