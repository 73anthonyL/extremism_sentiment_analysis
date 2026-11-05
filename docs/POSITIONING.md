# Paper positioning

Written 2026-11-04. This document sets out what the paper contributes and how
the work is presented. `docs/RESEARCH_PLAN.md` still holds the research
questions, work packages and decision rules; this document decides how they
are framed and what counts as future work. The framing rules in section 1 of
the plan still apply: extremism detection is not unexplored, labels are
post-level and text-only, and "better" requires a McNemar-significant gap.

## 1. The three contributions

The paper makes three claims, in this order.

**C1. An open auditing framework for text-based extremism classifiers.** It is
a protocol plus a toolkit, not a moderation product. Papers usually present
this kind of contribution as "we release the framework, here is its contract,
here is how a new model is added, here is what it reports". Section 2 below
gives the contract. The usability case is for a researcher who has a new
classifier and wants to know three things before trusting it: whether it is
measurably more accurate than existing models, what it relies on, and whether
that reliance shows up as false positives on benign posts that mention a
community. The framework answers all three from two exported files, with no
need to re-run the other models.

**C2. What explanations tell us about detection.** These are the findings RQ1
to RQ4 produce when the framework is applied to eight techniques, from TF-IDF
linear models to a pooled transformer ensemble. The claims are about
detection, not about SHAP itself:

* which accuracy gaps are real at n = 450 (RQ1: 7 of 28 pairs after Holm),
* which words the families share and which they do not (RQ2: three words are
  in every technique's top 50, and agreement between families is 0.2 to 0.3),
* whether pretraining or pooling changes that reliance (RQ3), and
* whether identity terms cause false positives. RQ4 reports the correlation;
  the identity counterfactual test (WP7 below) checks whether the identity
  term itself is the cause.

**C3. Why the task is hard, and what would move it.** This is a section of the
paper built from evidence the framework produces, not a list of wishes
(section 4).

## 2. The framework contract (C1)

Every model must meet one contract. This is the existing CLAUDE.md rule,
restated as the interface a new user follows:

| A model must provide | Format | Checked by |
|---|---|---|
| Validation and test probabilities on the frozen split | `research_loop/probs/<TECHNIQUE>__{validation,test}.csv` | `probs_artifact.py`, `validate_results_folder.py` |
| A threshold locked on validation | `results_summary/<TECHNIQUE>/best_config.json` | `validate_results_folder.py` |
| Word-level SHAP runs on test and validation posts | `attributions/<run_id>.csv` with its sidecar | `attributions.py` |
| A restorable `predict_proba_texts` (for counterfactual and curated tests) | `external/<TECHNIQUE>/` | `model_loaders.py`, checked against the test artifact |

For a model that meets the contract, the framework produces:

| Stage | Tool | Answers |
|---|---|---|
| Sanity | `validate_shap.py` (linear models) | Does the explainer recover the known coefficients? |
| RQ1 | `mcnemar_pairs.py` | Is this model measurably different from the others? |
| RQ2 | `word_agreement.py`, `categorize_attributions.py` | What does it rely on, and does anything else rely on it too? |
| RQ3 | `compare_reliance.py` | Are those words stable across post samples? |
| RQ4 | `identity_fpr.py`, `identity_counterfactual.py` | Do identity terms produce false positives, and do they cause them? |
| Inspection | `curated_examples.py`, `curated_site.py` | How does the model behave on hand-picked texts? |
| Hygiene | `scan_text_leakage.py` | Can the outputs be published without dataset text? |

The framework has three properties to stress in the paper:

* **Derived, not transcribed.** Every number is recomputed from probabilities.
* **Text-free outputs.** Every committed output passes `scan_text_leakage.py`, so results can be shared even where the posts cannot.
* **Adversarial tests.** Each tool refuses at least one specific malformed input.

How to audit a new classifier (WP8, closed 2026-11-04): `docs/FRAMEWORK.md`.

## 3. The four candidate end goals, assessed

| Goal | Verdict | Reason |
|---|---|---|
| 1. Higher accuracy from a 5k dataset | **Future work, not an end goal of this paper** | This contradicts the thesis (an explainability study, not a better classifier) and plan section 6 (no new dataset). Its real value is statistical: a test split of about 750 rows would lower the McNemar floor and let more of RQ1's gaps resolve. State that in C3. |
| 2. Internal neural-network analysis | **Adopt as one bounded package (WP9) if a GPU run is available** | The likely meaning is mechanistic or representation-level interpretability: linear probes on each layer's hidden states for "extremist" and "identity term present", and identity-direction ablation. It complements SHAP, which only sees inputs and outputs, and it asks a sharper RQ3 question: at which layer does identity information become predictive of the label? It needs the transformer weights (already in `external/`) and a frozen-split probe protocol. |
| 3. Zero-shot or few-shot LLMs with self-explanation | **Adopt as a comparison point (WP10), with a data caveat** | An instruction-tuned LLM classifies the test split zero-shot and names the words it relied on. Two questions follow: does its accuracy clear the McNemar floor against 07, and do its self-explanations agree with its own SHAP attributions (faithfulness)? A probability artifact is needed (use label-token log-probabilities), and so is a word-attribution artifact, or it cannot enter RQ2 to RQ4. Sending dataset text to a hosted API is a data-sharing decision for the authors. An open-weight local model avoids it. |
| 4. A framework for others | **Adopt as C1** | Most of it exists already (section 2). It needs packaging (WP8), not new research. |

## 4. Why the task is hard (outline for the C3 section)

Each item cites the artifact that shows it.

1. **Small test splits hide real differences.** At 450 test rows a gap needs
   about 14 rows to clear McNemar; only 7 of 28 pairs do
   (`rq1_mcnemar_pairs.csv`). More labelled data mainly buys statistical
   resolution.
2. **Some errors are shared by every model.** Twelve test posts are wrong for
   all eight techniques, eleven of them extremist (`rq1_shared_errors.csv`).
   This points at label ambiguity or signal the text does not contain. That
   fits the post-level, text-only limitation.
3. **Identity terms are a shortcut.** Every technique has a higher FPR on
   benign posts that mention a community (`rq4_identity_fpr.csv`, ratios 1.9
   to 4.2). In the counterfactual test (WP7), removing the term lowers
   p(extremist) on those posts for 7 of 8 techniques. The same edit also
   unflags extremist posts (Holm-significant for 07 and 11), so the models
   treat identity terms as evidence of extremism whatever the label. The
   false-positive flips (5 to 11 posts per model) are descriptive.
4. **Transformer explanations are unstable across post samples.** Test and
   validation top-50 overlap is 0.18 to 0.22 for the transformers, against
   0.3 to 0.6 for the classical models (`rq3_stability.csv`). Per-word claims
   about transformers are therefore fragile, and the paper says so.
5. **Lexicons do not cover the vocabulary.** The "topical" category absorbs
   80 to 100 percent of positive mass while the slur list is empty (WP3), so
   category-level claims depend on curation choices.
6. **Context matters.** Quotation, reporting and counter-speech use the same
   words as extremist posts, which no word-level method separates.

Avenues for progress, each tied to the item it addresses: a larger test split
(1); multi-annotator labels with agreement (2); counterfactual data
augmentation or identity-term-balanced training (3); explanation stability as
a reported metric (4); hand-annotated top words instead of lexicons (5);
probing and LLM self-explanation (WP9, WP10) for context (6).

## 5. New work packages

* **WP7. Identity counterfactual test (local).** For every test post that
  contains an identity term, remove the term (primary) or replace it with a
  label-balanced, near-zero-coefficient control panel (sensitivity). Re-score the post with each technique's restored predictor and
  report the probability shift, the number of predictions that flip at the
  locked threshold, and paired tests. This turns RQ4's correlation into an
  intervention. Tool: `tools/identity_counterfactual.py`. Closed
  2026-11-04: on the 67 non-extremist test posts with identity terms,
  removal lowers p(extremist) for 7 of 8 techniques and also unflags 15 of
  71 (07) and 22 of 74 (11) flagged extremist posts (Holm p = 0.016 and
  1.5e-5); details in `RESEARCH_PLAN.md`, WP7.
* **WP8. Framework entry point and report card (local).** Closed
  2026-11-04: `tools/audit.py` and `docs/FRAMEWORK.md`.
* **WP9. Layer-wise probes on the transformer members (Kaggle GPU).**
* **WP10. Zero-shot LLM comparison with self-explanation (author decision on
  data sharing first).**
