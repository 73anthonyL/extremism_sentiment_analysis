# Results, challenges and framework: first draft

Status: first draft, 2026-11-04, written from the committed artifacts on
branch `research/integration`. It covers the paper's Results (RQ1 to RQ4),
the "Why this is hard" section (C3) and a short framework-usability section
(C1). The Introduction, Data, Models and Attribution-method sections are not
drafted here. Every number is copied from a committed CSV or JSON file and
is listed with its artifact, row selector and column in
`docs/paper/CLAIMS.md`; footnotes give the artifact path and the claim ids.
`python3 tools/check_claims.py` verifies every claim against its artifact
and checks that every claim id below is listed there.

Items that need an author decision or a result that does not exist yet are
marked **TODO(author)**. All category-level RQ2 and RQ3 results are
provisional until WP3 freezes the lexicons: the slur list is empty today, so
slurs are counted as topical.

Scope, stated once for every section below: labels are post-level and
features are text-only. The classifiers detect extremist content in a single
post. They do not detect accounts, intent or radicalization over time, and
nothing here supports a claim about those.

Technique names used throughout: 01 word TF-IDF + logistic regression (LR);
02 word TF-IDF + linear SVM; 03 word TF-IDF + single-layer perceptron (SLP);
04 character TF-IDF + linear SVM; 05 word+character TF-IDF + linear SVM;
06 FastText embeddings + LR; 07 fine-tuned Twitter-RoBERTa; 11 a logit-pooled
ensemble of four fine-tuned transformer checkpoints. 01 to 06 are "classical",
07 and 11 "transformer". The data are 2999 posts on one frozen split
(2099 training posts); the test split has 450 posts (280 non-extremist, 170
extremist) and is evaluated once per technique.[^data]

[^data]: `results_summary/foundation/dataset_manifest.json`,
    `results_summary/foundation/split_label_distribution.csv`. Claims C814,
    C815, C816, C817, C818.

---

## 5. RQ1: How accurate are the models, and which gaps are real?

Table 1 gives the test metrics at each technique's validation-locked
threshold. Accuracy runs from 0.831 (04) to 0.909 (11). Because the test
split has 450 rows, we call one technique more accurate than another only
when the exact McNemar test on discordant posts is significant after Holm
correction over all 28 pairs.[^rq1]

**Table 1.** Test metrics (450 posts, one evaluation per technique).[^t1]

| Technique | Accuracy | Balanced acc. | Macro F1 | ROC-AUC | PR-AUC | FP | FN |
|---|---|---|---|---|---|---|---|
| 01 LR word TF-IDF | 0.856 | 0.842 | 0.845 | 0.914 | 0.891 | 29 | 36 |
| 02 SVM word TF-IDF | 0.853 | 0.835 | 0.841 | 0.907 | 0.889 | 25 | 41 |
| 03 SLP word TF-IDF | 0.840 | 0.807 | 0.820 | 0.904 | 0.879 | 16 | 56 |
| 04 SVM char TF-IDF | 0.831 | 0.800 | 0.811 | 0.915 | 0.887 | 20 | 56 |
| 05 SVM word+char TF-IDF | 0.856 | 0.819 | 0.836 | 0.920 | 0.902 | 9 | 56 |
| 06 LR FastText | 0.833 | 0.800 | 0.813 | 0.899 | 0.872 | 18 | 57 |
| 07 Twitter-RoBERTa | 0.889 | 0.881 | 0.882 | 0.951 | 0.932 | 24 | 26 |
| 11 Pooled ensemble | 0.909 | 0.896 | 0.902 | 0.968 | 0.955 | 14 | 27 |

[^t1]: Metrics: `results_summary/<TECHNIQUE>/metrics_test.json`, keys
    `accuracy`, `balanced_accuracy`, `f1_macro`, `roc_auc`, `pr_auc`.
    FP and FN: `results_summary/rq/rq1_error_overlap.csv`. Claims C001–C056.

Seven of the 28 pairs clear Holm-corrected significance (Table 2): the
ensemble (11) against each of the six classical techniques, and 07 against
04. No pair of classical techniques is distinguishable at n = 450; the
largest classical gap is 11 rows (01 versus 04). The ensemble and the single
fine-tuned transformer are not distinguishable either: 11 is right on 9 more
posts, with 27 discordant posts and Holm p = 1.00. Against every classical
technique except 04, 07 has more correct posts but the gap does not survive
Holm (against 03, for example, row gap -22 and Holm p = 0.21). What decides
is the discordant count, not the row gap alone: 07 leads 06 by 25 rows (83 discordant, Holm p = 0.17) while it leads
04 by 26 rows (62 discordant) at Holm p = 0.031.[^rq1]

**Table 2.** Holm-significant McNemar pairs, and the transformer pair.
Row gap is the first technique's correct posts minus the second's.[^t2]

| Pair | Row gap | Discordant | Holm p |
|---|---|---|---|
| 01 vs 11 | -24 | 58 | 0.049 |
| 02 vs 11 | -25 | 51 | 0.016 |
| 03 vs 11 | -31 | 61 | 0.0023 |
| 04 vs 07 | -26 | 62 | 0.031 |
| 04 vs 11 | -35 | 57 | 9.3e-5 |
| 05 vs 11 | -24 | 54 | 0.034 |
| 06 vs 11 | -34 | 72 | 0.0020 |
| 07 vs 11 (not significant) | -9 | 27 | 1.00 |

[^t2]: `results_summary/rq/rq1_mcnemar_pairs.csv`, columns
    `accuracy_gap_rows`, `n_discordant`, `holm_p`. Claims C059–C082.

[^rq1]: `results_summary/rq/rq1_mcnemar_pairs.csv`: 28 pairs (C057), 7 with
    `significant_holm` true (C058); 03 vs 07 gap and Holm p (C084, C083);
    06 vs 07 (C085, C086, C087); 01 vs 04 gap (C088). Accuracy range from
    Table 1 (C016, C036).

308 test posts are classified correctly by all eight techniques and 12 are
misclassified by all eight, 11 of them extremist: the errors every technique
shares are almost all false negatives. 06
has the most errors of its own (17 posts no other technique gets
wrong).[^err]

[^err]: `results_summary/rq/rq1_shared_errors.csv` (C089, C090, C091) and
    `results_summary/rq/rq1_error_overlap.csv` (C092).

RQ1 reading. The ensemble is more accurate than every classical technique on
this split; 07 is more accurate than 04 and than no other technique. Every
other ordering in
Table 1 is descriptive. **TODO(author):** adding notebook 12 (plain RoBERTa
control) adds 12's pairs to the Holm family. 01 vs 11 (Holm p 0.049) then
stays significant only if every new pair is itself Holm-significant
(RESEARCH_PLAN section 3). Rerun `mcnemar_pairs.py` and update Table 2 and
this paragraph after 12 runs.

---

## 6. RQ2: What does each model rely on?

### 6.1 The attribution pipeline recovers known coefficients

Before any cross-model claim, the SHAP pipeline (partition explainer over a
word-level text masker, one method for every technique) was checked against
the coefficients of the word-level logistic regression (01). Spearman
correlation between mean attribution and coefficient is 0.873 on the test
run and 0.865 on the validation run, with top-50 sign agreement 1.00 on
both, above the preregistered bars of 0.80 and 0.90. Restricted to words
seen in at least two explained posts the correlations are 0.932 and 0.939
(a diagnostic, not the pass rule). The FastText LR (06), whose coefficients
act on embedding dimensions rather than words, reaches 0.745 and 0.757 with
sign agreement 1.00; it does not pass and is reported as the calibration
point, as planned.[^wp2]

[^wp2]: `results_summary/01_LOG-REG_TF-IDF/attributions/shap-partition_{test,validation}_seed30__coefficient_check.json`
    (C100–C105); `results_summary/06_FASTTEXT-EMB_LOG-REG/attributions/shap-partition_{test,validation}_seed30__coefficient_check.json`
    (C106–C109).

### 6.2 The attributions name words the models need

A deletion test checks that each technique's top-attributed words matter to
that technique. For each test post, the k words with the highest positive
attribution toward extremist are deleted and the drop in p(extremist) is
compared with deleting k random words of the same post (5 seeds, C257). On the
posts all eight techniques explained, top-k deletion lowers p(extremist)
more than random deletion for every technique at every k in {1, 3, 5, 10}
(Holm-adjusted Wilcoxon p at most 7.6e-21 over the 32 rows; restored
predictors match the committed probabilities within 0.0019). Deletion
recovers 0.46 to 0.80 of the summed SHAP value of the deleted words. Table 3
gives k = 1.[^faith]

**Table 3.** Deletion faithfulness at k = 1 on the shared test posts.
Margin is mean drop (top-1) minus mean drop (random). These are descriptive
per technique; no test compares techniques.[^t3]

| Technique | Posts | Margin | Median paired diff. | Margin, flagged posts (n) | Share top-1 > random |
|---|---|---|---|---|---|
| 01 | 199 | 0.065 | 0.045 | 0.091 (74) | 0.98 |
| 02 | 200 | 0.055 | 0.049 | 0.067 (70) | 0.99 |
| 03 | 198 | 0.018 | 0.016 | 0.016 (60) | 0.98 |
| 04 | 199 | 0.050 | 0.041 | 0.064 (61) | 0.98 |
| 05 | 199 | 0.052 | 0.043 | 0.064 (55) | 0.99 |
| 06 | 200 | 0.057 | 0.034 | 0.081 (59) | 0.99 |
| 07 | 199 | 0.109 | 0.001 | 0.231 (75) | 0.80 |
| 11 | 196 | 0.094 | 0.005 | 0.204 (72) | 0.87 |

[^t3]: `results_summary/rq/rq2_faithfulness.csv`, rows `population=shared`,
    `k=1`; columns `n_posts`, `drop_minus_random`, `median_paired_diff`,
    `drop_minus_random_flagged`, `n_flagged`, `share_topk_drop_gt_random`.
    Claims C200–C247.

[^faith]: `results_summary/rq/rq2_faithfulness.csv`, `population=shared`:
    32 rows (C251), largest Holm p (C248), range of `drop_over_attributed_topk`
    (C249, C250), largest difference between restored predictor and committed
    probabilities (C252); median paired difference range over k for 07
    (C253, C254) and 11 (C255, C256).

For the transformers the effect is concentrated in the posts they flag. Their
median paired difference is 0.001 to 0.010 (07) and 0.005 to 0.024 (11)
across k, while their margin on flagged posts at k = 1 (0.231 and 0.204) is
larger than any classical technique's (0.016 to 0.091). The test shows that
the attributions find words the models use, not that the models are right
to use them.[^faith]

### 6.3 Category shares

Table 4 gives, for each technique's whole-model test run, the share of
positive attribution mass in the top 50 words that falls in each lexicon
category, with the across-run standard deviation the decision rules require
(test run versus validation run) and a post-level bootstrap SD.

**Table 4.** Share of positive attribution mass in the top 50 words, test
run. Provisional until WP3: the slur lexicon is empty, so the slur column is
unavailable and slurs are counted as topical.[^t4]

| Technique | Posts explained | Extremist framing | Identity term | Slur | Topical | Identity SD (2 runs) | Identity SD (bootstrap) |
|---|---|---|---|---|---|---|---|
| 01 | 450 | 0.000 | 0.094 | n/a | 0.906 | 0.006 | 0.016 |
| 02 | 450 | 0.000 | 0.116 | n/a | 0.884 | 0.015 | 0.025 |
| 03 | 450 | 0.000 | 0.000 | n/a | 1.000 | 0.023 | 0.031 |
| 04 | 450 | 0.000 | 0.104 | n/a | 0.896 | 0.019 | 0.036 |
| 05 | 450 | 0.000 | 0.130 | n/a | 0.870 | 0.003 | 0.020 |
| 06 | 450 | 0.000 | 0.132 | n/a | 0.868 | 0.014 | 0.023 |
| 07 | 200 | 0.030 | 0.070 | n/a | 0.900 | 0.003 | 0.034 |
| 11 | 200 | 0.016 | 0.174 | n/a | 0.810 | 0.002 | 0.033 |

[^t4]: Shares: `results_summary/rq/rq3_reliance_comparison.csv`, rows
    `role=whole`, `split=test`. Posts explained:
    `results_summary/rq/rq2_top_word_support.csv`. Run SD:
    `results_summary/rq/rq3_stability.csv`. Bootstrap SD:
    `results_summary/rq/rq2_bootstrap_category_shares.csv`, rows
    `category=identity_term`, `split=test`, `post_set=own`. Claims C110–C157;
    topical range C158, C159.

The topical category takes 0.81 to 1.00 of the positive mass. The extremist
framing lexicon (seeded from single-word entries of an extremism lexicon)
matches almost none of the top words, and the identity-term share is 0.000
to 0.174. With the slur list empty, Table 4 cannot yet answer the RQ2
hypothesis about slurs versus identity terms. The bootstrap SDs (0.016 to
0.036) are larger than the across-run SDs (0.002 to 0.023), so the two-run
SD the decision rule names understates post-sampling spread.
**TODO(author):** decide whether the section 5 difference rule uses the
two-run SD (as preregistered) or the bootstrap intervals. **TODO(author):**
WP3 (populate `slurs.txt` from a citable source, review the identity and
framing lists, freeze), then rerun `categorize_attributions.py`,
`compare_reliance.py`, `identity_fpr.py` and this table.

The lexicon also misses identity references. Across the eight top-50 lists,
27 words look like identity references (plurals, hashtags, group nouns) but
are not in the identity lexicon, 6 of them for 11.[^gaps] They are a review
list for WP3, not additions.

[^gaps]: `results_summary/rq/rq2_identity_top_k_sensitivity.csv`, sum of
    `n_gap_candidates` (C700) and the 11 row (C702);
    `results_summary/rq/rq2_identity_lexicon_gaps_sensitivity.csv`, row count
    (C701).

### 6.4 Word-level agreement, without lexicons

Comparing the word tables directly avoids the lexicons. Three words are in
the top 50 by positive attribution of all eight techniques: a racial slur,
"kill" and "genocide". Of the 168 words that appear in any top-50 list, 73
appear in one technique's list only.[^agree]

Agreement follows feature family. The two word TF-IDF linear models agree
closely (Spearman over shared words 0.90, top-50 Jaccard 0.67), as do char
and word+char TF-IDF (Spearman 0.85) and the two transformers (0.76 and
0.54). Across families agreement is low: 07 against the six classical
techniques gives Spearman 0.21 to 0.28, and 11 gives 0.26 (03) to 0.32
(05). FastText agrees little even with the word TF-IDF LR (0.17).[^agree]

[^agree]: `results_summary/rq/rq2_shared_top_words.csv`: words in all eight
    lists (C160), in one list (C161), total (C162).
    `results_summary/rq/rq2_word_agreement_pairs.csv`,
    `spearman_mean_attribution` and `jaccard_top_k_positive`: C163–C172.

The transformers' top words rest on fewer posts. In their own runs, 0.56
(07) and 0.52 (11) of the top-50 positive words occur in a single explained
post, against 0.22 (01) to 0.40 (04) for the classical runs. The runs differ
in size (200 versus 450 posts), so the same measure was recomputed on the
200 test posts every technique explained. There the classical shares rise
to 0.32 to 0.52: 04 reaches 0.52 and 06 0.48, while the word TF-IDF models
stay at 0.32 to 0.44 (01 0.32, 02 0.34, 05 0.38, 03 0.44).[^support]

Under a post-level bootstrap this share is a replicate spread, not an
interval: resampling shifts it upward (01's own-run point is 0.22, its
replicate median 0.32), and the point for 01 falls outside its own spread.
On paired draws, 07's excess over the word TF-IDF models has a lower bound
of only 0.02 to 0.04, and its difference from 04 spans -0.10 to 0.14. The
comparison is descriptive.[^support]

[^support]: `results_summary/rq/rq2_top_word_support.csv` (C173–C176);
    `results_summary/rq/rq2_same_post_support.csv`, `post_set=common`,
    `split=test` (C177–C182); `results_summary/rq/rq2_bootstrap_same_post.csv`
    (C183, C184); `results_summary/rq/rq2_rq3_bootstrap_paired_differences.csv`,
    `measure=share_single_post_top_k_positive` (C185–C189).

RQ2 reading. The deletion test supports reading the word tables as what each
model uses. The models share a small core of words (a racial slur, "kill",
"genocide") and otherwise rely on different words, and the split runs along
feature family. Category-level conclusions wait for WP3.

---

## 7. RQ3: Does pretraining or ensembling change that reliance?

### 7.1 Pretraining lineage and pooling

Table 5 compares the single Twitter-RoBERTa (07), the four admitted members
of the ensemble, their mean, and the pooled ensemble (11). The members share
07's fine-tuning recipe and differ in the checkpoint they start from.

**Table 5.** Share of positive attribution mass in the top 50 words, by
transformer. Members are explained on the test sample only (one run each),
so they have no across-run SD. Provisional until WP3.[^t5]

| Model | Checkpoint (pretraining) | Split | Extremist framing | Identity term | Topical |
|---|---|---|---|---|---|
| 07 | Twitter-RoBERTa-base, hate-tuned | test | 0.030 | 0.070 | 0.900 |
| 07 | (same) | validation | 0.000 | 0.064 | 0.936 |
| 11 member A | Twitter-RoBERTa-base, hate-tuned | test | 0.027 | 0.089 | 0.885 |
| 11 member B | RoBERTa, Dynabench hate speech | test | 0.000 | 0.222 | 0.778 |
| 11 member C | HateBERT (Reddit abusive language) | test | 0.014 | 0.089 | 0.897 |
| 11 member E | Twitter-RoBERTa-large, hate-tuned | test | 0.000 | 0.197 | 0.803 |
| 11 member mean | | test | 0.010 | 0.149 | 0.841 |
| 11 pooled | | test | 0.016 | 0.174 | 0.810 |
| 11 pooled | | validation | 0.030 | 0.178 | 0.792 |

[^t5]: `results_summary/rq/rq3_reliance_comparison.csv`, columns
    `share_of_positive_mass__*`; checkpoints from
    `results_summary/11_MULTI-CHECKPOINT_LOGIT-POOL/best_config.json` and
    `results_summary/07_TWITTER-ROBERTA_FINE-TUNE/best_config.json`.
    Claims C300–C326.

The pooled ensemble puts a larger identity-term share on its top words than
07 (0.174 against 0.070 on test; 0.178 against 0.064 on validation). The
difference exceeds the pooled across-run SD of the two (0.002 and 0.003),
which is the preregistered criterion (provisional until WP3). On paired
bootstrap draws over the same 200 posts it is -0.104 (07 minus 11) with a percentile interval of
-0.140 to -0.025.[^rq3a] The pooled share sits above the member mean (0.149)
and between the members, which range from 0.089 (A and C) to 0.222 (B). So
on these runs pooling did not move the ensemble toward the lower-reliance
members; its share is closer to B's and E's. Per-member differences cannot be
claimed under the decision rule, because each member has one run.

[^rq3a]: Table 4 SDs (C150, C156);
    `results_summary/rq/rq2_rq3_bootstrap_paired_differences.csv`,
    `measure=identity_share_of_positive_mass`, 07 vs 11 (C327–C329).

Against the classical techniques the picture is weaker. On paired draws, 11's
identity share is above 01, 02, 03 and 06 (lower bounds 0.026, 0.003, 0.064,
0.032) but not above 04 or 05 (lower bounds -0.052 and -0.017). None of 07's
six differences from a classical technique excludes zero (the bounds nearest
zero are -0.020 against 03 and 0.028 against 04). These intervals cover post
sampling only.[^rq3b]

[^rq3b]: `results_summary/rq/rq2_rq3_bootstrap_paired_differences.csv`,
    `measure=identity_share_of_positive_mass`, `pct_low` / `pct_high`:
    C330–C337.

The lineage question itself is not answered yet. The hypothesis contrasts
hate-speech pretraining with a plain RoBERTa control, and that control has
not run. **TODO(author):** run notebook 12 (`12_ROBERTA-BASE_FINE-TUNE`,
07's recipe with only the checkpoint changed), then rerun the RQ3 chain and
replace Table 5. **TODO(author):** decide whether the member loop also
explains the validation split, which would give members an across-run SD
(a small notebook edit, RESEARCH_PLAN section 3).

### 7.2 Stability across post samples

Each technique has two whole-model runs over different posts (test and
validation). Table 6 gives their agreement on the same post count (200 per
split).

**Table 6.** Test-versus-validation agreement of word attributions on 200
posts per split. The Spearman interval is a post-level bootstrap percentile
interval; Jaccard is shown as a point only, because under the bootstrap it
is a replicate spread.[^t6]

| Technique | Spearman, mean abs. attribution | Bootstrap interval | Top-50 Jaccard (abs.) |
|---|---|---|---|
| 01 | 0.704 | 0.644 to 0.752 | 0.333 |
| 02 | 0.711 | 0.650 to 0.758 | 0.333 |
| 03 | 0.695 | 0.656 to 0.760 | 0.316 |
| 04 | 0.705 | 0.632 to 0.747 | 0.333 |
| 05 | 0.743 | 0.706 to 0.799 | 0.471 |
| 06 | 0.616 | 0.566 to 0.681 | 0.190 |
| 07 | 0.427 | 0.344 to 0.480 | 0.176 |
| 11 | 0.488 | 0.405 to 0.538 | 0.220 |

[^t6]: `results_summary/rq/rq3_bootstrap_split_agreement.csv`, rows
    `post_set=common`; `point`, `pct_low`, `pct_high` for
    `measure=spearman_mean_abs_attribution`, `point` for
    `measure=jaccard_top_k_abs`. Claims C340–C371.

The transformers' attributions agree less across post samples than the
classical models'. On paired draws every transformer-minus-classical
difference in Spearman has an interval that excludes zero; the upper bounds
nearest zero, both against 06, are -0.129 (07) and -0.078 (11). 07 and 11 do
not separate from each other (-0.061, interval -0.126 to 0.005).[^t6b] Top-50
Jaccard does not separate the transformers from FastText (06, 0.190). This
is descriptive. It covers post sampling, not the SHAP seed or another split,
and lower agreement may reflect attributions that depend on context, which a
word-level mean over different posts reads as instability, rather than less
reliable explanations.

[^t6b]: `results_summary/rq/rq2_rq3_bootstrap_paired_differences.csv`,
    `measure=spearman_mean_abs_attribution`: C372–C376.

RQ3 reading. Pooling four transformers gave a whole model with a larger
identity-term share than the single transformer, on both splits. Per-word
claims about either transformer are fragile across post samples. The
pretraining-lineage comparison waits for notebook 12.

---

## 8. RQ4: Do identity terms drive false positives?

### 8.1 Correlation: false-positive rates with and without identity terms

Of the 280 non-extremist test posts, 67 contain at least one identity-lexicon
match (83 extremist posts do).[^rq4n] Every technique has a higher
false-positive rate on the 67 than on the other 213 (Table 7). The ratio is
1.91 to 4.24. Unadjusted Fisher exact p is below 0.05 for 04, 05 and 11 only.
With 67 such posts, only large ratios are detectable, so a non-significant
ratio is not evidence that the effect is absent.

[^rq4n]: `results_summary/rq/rq4_identity_fpr.csv`, 01 row (C464, C465,
    C466); 213 without identity terms from
    `results_summary/rq/rq2_rq4_relation_inputs.csv` (C474).

**Table 7.** False positives on non-extremist test posts with and without an
identity term, at each locked threshold, and the same after setting aside the
14 posts whose only matches are ambiguous words (colour words,
interjections, declared in `tools/identity_sensitivity.py`; the lexicon is
unchanged).[^t7]

| Technique | FP with (of 67) | FPR with | FP without (of 213) | FPR without | Ratio | Fisher p | Ratio, unambiguous (53) | Fisher p |
|---|---|---|---|---|---|---|---|---|
| 01 | 11 | 0.164 | 18 | 0.085 | 1.94 | 0.069 | 2.01 | 0.077 |
| 02 | 10 | 0.149 | 15 | 0.070 | 2.12 | 0.082 | 2.14 | 0.096 |
| 03 | 6 | 0.090 | 10 | 0.047 | 1.91 | 0.226 | 1.61 | 0.489 |
| 04 | 9 | 0.134 | 11 | 0.052 | 2.60 | 0.030 | 2.92 | 0.031 |
| 05 | 5 | 0.075 | 4 | 0.019 | 3.97 | 0.038 | 5.02 | 0.017 |
| 06 | 7 | 0.104 | 11 | 0.052 | 2.02 | 0.152 | 1.83 | 0.328 |
| 07 | 9 | 0.134 | 15 | 0.070 | 1.91 | 0.131 | 1.88 | 0.164 |
| 11 | 8 | 0.119 | 6 | 0.028 | 4.24 | 0.0064 | 4.69 | 0.0056 |

[^t7]: FP counts: `results_summary/rq/rq2_rq4_relation_inputs.csv`.
    Rates, ratio and Fisher p: `results_summary/rq/rq4_identity_fpr.csv`.
    Unambiguous columns: `results_summary/rq/rq4_identity_fpr_sensitivity.csv`,
    `variant=exclude_ambiguous_only`; 53, 14 and 24 posts from its 01 row
    (C467, C468, C469). Claims C400–C463.

Setting the ambiguous words aside keeps every ratio above 1 (1.61 to 5.02),
and the same three techniques stay below p = 0.05. 03 and 06 fall to 1.61
and 1.83, which at this n is not evidence that their effect is absent. The
false-positive counts are small (5 to 11 with identity terms), and they are
fewer than the 14-row gap, so Table 7 does not rank the techniques.

### 8.2 Intervention: removing the identity term

The correlation does not show that the identity term causes the false
positive. WP7 removes every identity-lexicon match from the 150 test posts
that have one (C623) and re-scores the edited posts with each technique's
restored predictor at its locked threshold. Before scoring, each restored predictor
reproduces the committed test probabilities to within 0.0016.[^cf0] This
re-scores frozen models and selects nothing, so it is a perturbation
analysis, not a second test evaluation. All p values are Holm-adjusted over
the 32 technique x population x operation rows.

[^cf0]: `results_summary/rq/rq4_identity_counterfactual.csv`, largest
    `verification_max_abs_diff` (C621) and row count (C622).

**Table 8.** Identity-term removal. Mean and median delta are p(extremist)
before minus after the edit, on the 67 non-extremist posts. Flag counts are
before -> after, with the posts that flip in brackets.[^t8]

| Technique | Mean delta | Median delta | Wilcoxon Holm p | Non-extremist flagged (cleared / new) | Extremist flagged (lost / gained) | McNemar Holm p, extremist |
|---|---|---|---|---|---|---|
| 01 | 0.035 | 0.024 | 2.5e-5 | 11 -> 8 (3 / 0) | 62 -> 56 (6 / 0) | 0.625 |
| 02 | 0.020 | 0.017 | 3.1e-4 | 10 -> 8 (3 / 1) | 60 -> 50 (10 / 0) | 0.051 |
| 03 | 0.001 | 0.000 | 0.92 | 6 -> 7 (0 / 1) | 51 -> 49 (4 / 2) | 1.000 |
| 04 | 0.022 | 0.012 | 1.4e-3 | 9 -> 4 (5 / 0) | 53 -> 45 (8 / 0) | 0.172 |
| 05 | 0.026 | 0.010 | 1.8e-4 | 5 -> 4 (1 / 0) | 53 -> 47 (7 / 1) | 1.000 |
| 06 | 0.040 | 0.034 | 1.6e-6 | 7 -> 7 (1 / 1) | 52 -> 43 (9 / 0) | 0.094 |
| 07 | 0.099 | 0.001 | 3.3e-5 | 9 -> 2 (7 / 0) | 71 -> 57 (15 / 1) | 0.016 |
| 11 | 0.095 | 0.003 | 9.8e-6 | 8 -> 2 (7 / 1) | 74 -> 52 (22 / 0) | 1.5e-5 |

[^t8]: `results_summary/rq/rq4_identity_counterfactual.csv`, rows
    `operation=remove`; non-extremist columns from `population=non_extremist`,
    extremist columns from `population=extremist`. Claims C500–C595.

On the non-extremist posts, removal lowers p(extremist) for 7 of 8
techniques. 03 shows no detectable shift (mean 0.001, Holm p = 0.92), which
is not evidence that the effect is absent. For 07 and 11 the mean drop
(0.099, 0.095) is far above the median (0.001, 0.003): the shift sits in a
minority of posts. Removal clears 7 of 9 original false positives for 07 and
7 of 8 for 11, against 0 to 5 for the classical techniques. These flip counts
are descriptive: no non-extremist flip test survives Holm, and the counts are
below the 14-row gap.

The same edit costs recall. On the 83 extremist posts, it unflags 15 of 71
flagged posts for 07 and 22 of 74 for 11 (Holm p = 0.016 and 1.5e-5); no
classical recall drop survives Holm. This matches the correlational side of
RQ4: recall for 07 and 11 is 0.855 and 0.892 on extremist posts with
identity terms, against 0.839 and 0.793 without.[^rec] Under removal the
transformers lose flagged posts on both sides of the label: they respond to
the edit as if the term were evidence of extremism whatever the label, and
removing it trades false positives for recall.

[^rec]: `results_summary/rq/rq4_identity_fpr.csv`, `recall_with_identity_terms`
    and `recall_without_identity_terms` (C470–C473).

Two sensitivity checks qualify this. First, on the 59 extremist posts that
keep an unambiguous identity match, removal unflags 15 of 51 flagged posts
for 11 (Holm p = 0.002, over that table's own rows) and 10 of 48 for 07
(Holm p = 0.33). 07's rate there (0.21) is close to its rate on all 83 posts
(15 of 71); with fewer discordant pairs the test has less power, so this is
about the same effect size, not significant at the reduced n. Removing only
the unambiguous matches unflags 11 of 51 for 11 (Holm p = 0.030) and 6 of 48
for 07. Only 11's recall loss is Holm-significant in every family
tried.[^cfs]

Second, replacing each match with a neutral control panel ("group",
"population", "humans", each checked for label balance and a small 01
coefficient) instead of deleting it removes the transformer-specific
pattern. The recall drop is then Holm-significant for 01, 04, 05 and 06
(11 posts lost each, Holm p = 0.029) and not for 07 (1.00) or 11 (0.094).
07's non-extremist shift shrinks to a mean of 0.012 (Holm p = 0.24). For the
transformers, the removal effect therefore cannot be attributed to the
identity term specifically: it may come from losing the group referent or
from the ungrammatical text deletion leaves.[^sub]

[^cfs]: `results_summary/rq/rq4_identity_counterfactual_sensitivity.csv`,
    `population=extremist`: C610–C620.

[^sub]: `results_summary/rq/rq4_identity_counterfactual.csv`,
    `operation=substitute`: C600–C608.

A flip shows that one perturbation moved one post across the locked
threshold. That is threshold-local necessity, not proof that the term causes
the error. Deletion also changes TF-IDF normalization and character n-grams
across the join, and the lexicon covers only the identity terms it lists.

### 8.3 Does RQ2 reliance predict RQ4 false positives?

Ranking the eight techniques by RQ2 identity-term share (Table 4) and by
FPR ratio (Table 7) gives Spearman rho = 0.83 at n = 8 (exact permutation
p = 0.017; |rho| >= 0.73 is needed for p <= 0.05 at this n; leave-one-out
0.74 to 0.95). Under the decision rules a rank correlation over eight
techniques is descriptive, and three further results argue against reading
it as a link between reliance and identity-term false positives.[^rel]

* The ranking follows the denominator. Identity share against FPR on
  identity-term posts is -0.22; against FPR on the other posts it is -0.43.
  The two highest ratios come from low FPR without identity terms (05: 4 of
  213, 11: 6 of 213), not from more identity-term false positives.
* The intervention does not follow it. The WP7 mean removal shift against
  the FPR ratio gives 0.06. Among the six classical techniques alone, the
  primary correlation is 0.60.
* It is unstable under resampling. Resampling explained posts for the shares
  and test rows for the ratios (2000 replicates) gives a replicate median of
  0.48. Rho is positive in 0.90 of replicates and below the point estimate
  in 0.94, and the replicates spread from -0.29 to 0.89. The instability
  comes mainly from the ratios: resampling test rows alone gives a median of
  0.43, posts alone 0.80. Each ratio rests on few false positives, and only
  04 (1.06 to 6.37) and 11 (1.40 to 16.1) keep their ratio above 1 across
  95% of replicates; 05, with the second-highest ratio, does not (lower
  bound 0.91).[^boot]

Resampling noisy ratios also pulls a rank correlation toward zero, so these
figures describe how fragile the ordering is; they are not an interval for
rho. RQ4 reading: every technique has a higher FPR on benign posts that
mention a community (Fisher p < 0.05, unadjusted, for three of eight);
deleting the term costs the transformers recall, but replacing it with a
neutral group noun does not, so the term-specific effect is not established
for them; and the cross-technique ordering of reliance and false positives
is not established either.

[^rel]: `results_summary/rq/rq2_rq4_relation.csv`: primary row (C630–C635),
    descriptive rows (C636, C637), secondary row (C638), classical-only
    sensitivity (C639, C640). FP counts: Table 7 (C434, C458).

[^boot]: `results_summary/rq/rq2_rq4_bootstrap_relation.csv`: C641–C654.

---

## 9. Why the task is hard (C3)

Each item cites the evidence above; "avenue" names what would address it.
None of these is a claim that the task is new: extremism and hate-speech
detection are established research areas. The aim is to say where the
remaining difficulty sits.

1. **Small test splits hide real differences.** With 450 test posts, 7 of 28
   pairs clear Holm-corrected McNemar significance, and a 25-row gap (06 vs
   07) does not, because 83 posts are discordant. More labelled data mainly
   buys statistical resolution (C057, C058, C085, C086, C087). Avenue: a
   larger test split, which also lowers the floor for RQ4.
2. **Some errors are shared by every model.** 12 test posts are wrong for all
   eight techniques, 11 of them extremist (C089, C090). They point at label
   ambiguity, or at signal the text does not contain, which fits the
   post-level, text-only limitation. Avenue: multi-annotator labels with
   reported agreement. **TODO(author):** a qualitative read of these 12 by
   the authors (outside the repository; no text may be committed).
3. **Identity terms are a shortcut that is hard to isolate.** Every
   technique has a higher FPR on benign posts with identity terms (Table 7),
   and deleting the term costs the transformers recall on extremist posts
   (Table 8). Replacing it with a neutral group noun does not, which suggests
   the transformers respond to losing the group referent (or to the
   ungrammatical edit) rather than to the specific term. Avenue:
   counterfactual data augmentation or identity-balanced training, evaluated
   with the same recall-cost reporting.
4. **Transformer explanations are faithful per post but unstable across
   samples.** Deletion faithfulness holds for every technique (Table 3), yet
   the transformers' word rankings agree least across post samples
   (Spearman, Table 6). Per-word claims about transformers are fragile.
   Avenue: report explanation stability and deletion faithfulness as
   standard metrics.
5. **Lexicons do not cover the vocabulary.** The topical category takes 0.81
   to 1.00 of positive mass (C158, C159), the framing lexicon matches almost
   none of the top words, the slur list is empty, and 27 likely identity
   references are missing from the identity list (C700). Category-level
   claims depend on curation. Avenue: hand-annotated top words rather than
   lexicons alone.
6. **Context matters.** Quotation, reporting and counter-speech use the same
   words as extremist posts, and a word-level attribution cannot separate
   them. Avenue: layer-wise probes (WP9) and LLM self-explanations (WP10).
   **TODO(author):** WP9 needs a Kaggle GPU run; WP10 needs an author
   decision on sending dataset text to a hosted model (an open-weight local
   model avoids it). Neither result exists; this item stays an avenue until
   one does.

---

## 10. The auditing framework in use (C1)

The framework is a protocol plus a toolkit for researchers, not a moderation
tool. A researcher with a new classifier wants to know three things: whether
it is measurably more accurate than existing models, what it relies on, and
whether that reliance shows up as false positives on benign posts that
mention a community. The framework answers all three from two exported files
without re-running the other models.

**The contract.** A classifier enters by exporting (1) validation and test
probabilities on the frozen split, exactly `row_id, split, y_true, y_prob`,
with a threshold locked on validation; and (2) word-level SHAP runs on test
and validation posts, `word, mean_abs_attribution, mean_attribution,
support` plus a sidecar naming the explainer, split and post count. A linear
model may add its coefficients, and a restorable `predict_proba_texts`
enables the perturbation tests (identity counterfactual, deletion
faithfulness). Without the two attribution runs a classifier cannot enter
RQ2 to RQ4, whatever its accuracy. The notebook template
(`tools/check_notebook.py --print-template`) and `tools/notebook_kit.py`
produce both files; `docs/FRAMEWORK.md` is the user guide.

**Three commands.**

```bash
python3 tools/audit.py check-contract --technique <TECHNIQUE>
python3 tools/audit.py run --technique <TECHNIQUE> [--with-perturbation]
python3 tools/audit.py report --all
```

`check-contract` prints OK, MISSING or INVALID per contract item. `run`
executes the tool chain in a fixed order (results-folder validation, the
coefficient check, categorization, reliance comparison, identity FPR and its
ambiguous-term sensitivity, McNemar pairs, word agreement, lexicon-gap
candidates, same-post tables, and with `--with-perturbation` the identity
counterfactuals and deletion faithfulness, then the RQ2-RQ4 relation and the
table renderer). It stops at the first failing stage and names it; a stage
whose inputs are absent prints SKIP with its reason. A coefficient-check
FAIL on the anchor (01) stops the chain, since it would mean the explainer is
wrong. `report` writes one text-free card per technique. Every number on a
card is recomputed from probabilities or attribution files; none is
transcribed.

**What the card tells the researcher.** For 01 (the anchor):[^card]

* RQ1: test accuracy 0.856. Holm runs over the comparison's 28 pairs. Only
  the pair with 11 clears it (11 right on 24 more posts, Holm p = 0.049); the
  other six read "not distinguishable at n = 450". The card never says
  "better".
* SHAP check: PASS, with the per-run figures from section 6.1.
* RQ2: the top words with category and support, word-level agreement with
  every other technique, and the deletion test on the 200 shared posts
  (k = 1 margin 0.065; Holm family of 32 rows).
* RQ3: test-versus-validation Spearman 0.711 and top-50 Jaccard 0.389.
* RQ4: FPR ratio 1.94 (Fisher p = 0.069), and 2.01 without the
  ambiguous-only posts; WP7 removal clears 3 of the 11 false positives
  (C819).

[^card]: `results_summary/rq/report_cards/01_LOG-REG_TF-IDF.json`:
    C800–C813.

**Worked example: auditing a classifier outside the comparison.** A
researcher sets `in_comparison: false` in the classifier's `CONFIG`. It is
copied into `best_config.json`. Every tool then gives it Holm families of its
own: its McNemar and word-agreement pairs go to separate `_outside_comparison`
tables, and its counterfactual and faithfulness rows are corrected only over
its own rows. It is left out of the rendered tables, so the eight-technique
results above do not change. The test suite exercises this path with a
synthetic classifier (`98_TOY` in `tools/tests/test_audit.py`): the toy's
McNemar pairs and faithfulness rows form families of their own, it is absent
from the comparison's word-agreement tables, and adding it leaves the
comparison techniques' Holm p values unchanged. Its card says it is outside
the comparison. Setting `in_comparison: true` instead changes every other
card: a ninth technique enlarges the Holm family, so pairs near alpha can
flip. The guide therefore requires it to be a recorded decision. On the committed state, `audit.py run --technique
01_LOG-REG_TF-IDF` rewrites every table, coefficient check and card byte for
byte, so the chain is deterministic.

**Properties to stress.** Derived, not transcribed: metrics come from
probabilities through one function. Text-free: every committed output passes
`tools/scan_text_leakage.py`, so results can be shared where the posts
cannot. Adversarial tests: each tool has a test proving that a specific
malformed input is refused. The same holds for this draft:
`tools/check_claims.py` refuses a claim whose value or artifact does not
match.

**TODO(author):** the report cards print the top words verbatim, including
a racial slur in 01's card. Any figure of a card in the paper must redact
it. **TODO(author):** external assets (weights, per-post files) are not yet
uploaded to a Kaggle Dataset or recorded with `run_manifest.py add-asset`.
The perturbation stages cannot be reproduced by a reader until they are.

---

## Limitations specific to these results

Labels are post-level and features text-only; one dataset and one frozen
split; 450 test posts; category shares depend on lexicons that are not yet
frozen; SHAP over subword models is aggregated to words with a word-level
masker; the bootstrap intervals cover post sampling only, not SHAP seeds or
other splits; transformer runs explain 200 posts per split against 450 for
the classical runs (the same-post tables control for this where stated). The
framework audits detectors of extremist content in single posts and is not a
deployment recommendation.

## TODO(author) summary

1. Notebook 12 (plain RoBERTa control): run it, rerun RQ1 (Holm family
   with 12's pairs), RQ3 and the relation, then revise sections 5 and 7.
2. WP3 lexicon freeze: populate `slurs.txt` from a citable source, review the
   identity list (27 gap candidates) and the framing list, freeze, rerun RQ2
   and RQ4, and remove "provisional" from sections 6 and 7.
3. Section 5 rule: two-run SD (preregistered) or bootstrap intervals for
   share differences.
4. Validation-split member runs for 11, if RQ3 prose needs member SDs.
5. WP9 (layer-wise probes) and WP10 (zero-shot LLM, data-sharing decision).
6. A qualitative read of the 12 shared errors, outside the repository.
7. Redact slurs in any report-card figure.
8. Upload and record external assets before submission.
9. Citations: place the related-work entries (`docs/RELATED_WORK.md`) in the
   Results prose; none are cited in this draft.
