# Results, challenges and framework: first draft

Status: first draft, 2026-11-04, written from the committed artifacts on
branch `research/integration`. It covers the paper's Results (RQ1 to RQ4),
the "Why this is hard" section (C3) and a short framework-usability section
(C1). The Introduction, Data, Models and Attribution-method sections are not
drafted here.

How numbers are bound. Every number taken from an artifact is written as
`value [Cnnn]`, where `Cnnn` is a row of `docs/paper/CLAIMS.md` naming the
artifact, the row selector and the column. `python3 tools/check_claims.py`
checks each value against its artifact and checks that the value printed
next to each id here is the claim's value, so a number changed in this text
without its claim fails. Footnotes give the artifact paths. Numbers that are
protocol constants rather than results (top-50, k in {1, 3, 5, 10}, alpha
0.05, the 14-row McNemar rule of thumb from the plan, section and technique
numbers) are not claims. The bracketed labels are claim ids, not
citations.

Items that need an author decision or a result that does not exist yet are
marked **TODO(author)**. All category-level results (RQ2, RQ3 and the
RQ2-RQ4 relation in section 8.3) are provisional until WP3 freezes the
lexicons: the slur list is empty today, so slurs are counted as topical.

Scope, stated once for every section below: labels are post-level and
features are text-only. The classifiers detect extremist content in a single
post. They do not detect accounts, intent or radicalization over time, and
nothing here supports a claim about those.

Technique names used throughout: 01 word TF-IDF + logistic regression (LR);
02 word TF-IDF + linear SVM; 03 word TF-IDF + single-layer perceptron (SLP);
04 character TF-IDF + linear SVM; 05 word+character TF-IDF + linear SVM;
06 FastText embeddings + LR; 07 fine-tuned Twitter-RoBERTa (one seed);
11 a logit-pooled ensemble of four fine-tuned transformer members, each of
which is itself the average of three seeds. 01 to 06 are "classical", 07 and
11 "transformer". The data are 2999 [C814] posts on one frozen split (2099 [C817]
training posts); the test split has 450 [C818] posts (280 [C815] non-extremist,
170 [C816] extremist) and is evaluated once per technique.[^data]

[^data]: `results_summary/foundation/dataset_manifest.json`,
    `results_summary/foundation/split_label_distribution.csv`.

---

## 5. RQ1: How accurate are the models, and which gaps are real?

Table 1 gives the test metrics at each technique's validation-locked
threshold. Accuracy runs from 0.831 [C016] (04) to 0.909 [C036] (11). We call one
technique more accurate than another only when the exact McNemar test on
discordant posts is significant after Holm correction over all 28 [C057]
pairs.[^rq1]

**Table 1.** Test metrics (one evaluation per technique).[^t1]

| Technique | Accuracy | Balanced acc. | Macro F1 | ROC-AUC | PR-AUC | FP | FN |
|---|---|---|---|---|---|---|---|
| 01 LR word TF-IDF | 0.856 [C001] | 0.842 [C002] | 0.845 [C003] | 0.914 [C004] | 0.891 [C005] | 29 [C041] | 36 [C042] |
| 02 SVM word TF-IDF | 0.853 [C006] | 0.835 [C007] | 0.841 [C008] | 0.907 [C009] | 0.889 [C010] | 25 [C043] | 41 [C044] |
| 03 SLP word TF-IDF | 0.840 [C011] | 0.807 [C012] | 0.820 [C013] | 0.904 [C014] | 0.879 [C015] | 16 [C045] | 56 [C046] |
| 04 SVM char TF-IDF | 0.831 [C016] | 0.800 [C017] | 0.811 [C018] | 0.915 [C019] | 0.887 [C020] | 20 [C047] | 56 [C048] |
| 05 SVM word+char TF-IDF | 0.856 [C021] | 0.819 [C022] | 0.836 [C023] | 0.920 [C024] | 0.902 [C025] | 9 [C049] | 56 [C050] |
| 06 LR FastText | 0.833 [C026] | 0.800 [C027] | 0.813 [C028] | 0.899 [C029] | 0.872 [C030] | 18 [C051] | 57 [C052] |
| 07 Twitter-RoBERTa | 0.889 [C031] | 0.881 [C032] | 0.882 [C033] | 0.951 [C034] | 0.932 [C035] | 24 [C053] | 26 [C054] |
| 11 Pooled ensemble | 0.909 [C036] | 0.896 [C037] | 0.902 [C038] | 0.968 [C039] | 0.955 [C040] | 14 [C055] | 27 [C056] |

[^t1]: Metrics: `results_summary/<TECHNIQUE>/metrics_test.json`, keys
    `accuracy`, `balanced_accuracy`, `f1_macro`, `roc_auc`, `pr_auc`.
    FP and FN: `results_summary/rq/rq1_error_overlap.csv`.

7 [C058] of the pairs clear Holm-corrected significance (Table 2): the
ensemble (11) against each of the six classical techniques, and 07 against
04. No pair of classical techniques is distinguishable on this split; the
largest classical row gaps are 11 [C088] and -11 [C094] rows (01 vs 04 at
11 [C096], 04 vs 05 at -11 [C095]). The ensemble and the single fine-tuned
transformer are not distinguishable either (07 vs 11: row gap -9 [C080],
27 [C081] discordant posts, Holm p = 1.00 [C082]). Against every classical
technique except 04, 07 has more correct posts but the gap does not survive
Holm (against 03, for example, row gap -22 [C084], Holm p = 0.21 [C083]). What
decides is the discordant count, not the row gap alone: 06 vs 07 has row gap
-25 [C085] with 83 [C086] discordant posts (Holm p = 0.17 [C087]), while 04 vs 07
has row gap -26 [C068] with 62 [C069] discordant posts (Holm p = 0.031 [C070]).[^rq1]

**Table 2.** Holm-significant McNemar pairs, and the transformer pair.
Row gap is the first technique's correct posts minus the second's.[^t2]

| Pair | Row gap | Discordant | Holm p |
|---|---|---|---|
| 01 vs 11 | -24 [C059] | 58 [C060] | 0.049 [C061] |
| 02 vs 11 | -25 [C062] | 51 [C063] | 0.016 [C064] |
| 03 vs 11 | -31 [C065] | 61 [C066] | 0.0023 [C067] |
| 04 vs 07 | -26 [C068] | 62 [C069] | 0.031 [C070] |
| 04 vs 11 | -35 [C071] | 57 [C072] | 9.3e-5 [C073] |
| 05 vs 11 | -24 [C074] | 54 [C075] | 0.034 [C076] |
| 06 vs 11 | -34 [C077] | 72 [C078] | 0.0020 [C079] |
| 07 vs 11 (not significant) | -9 [C080] | 27 [C081] | 1.00 [C082] |

[^t2]: `results_summary/rq/rq1_mcnemar_pairs.csv`, columns
    `accuracy_gap_rows`, `n_discordant`, `holm_p`.

[^rq1]: `results_summary/rq/rq1_mcnemar_pairs.csv`.

308 [C091] test posts are classified correctly by all eight techniques and
12 [C089] are misclassified by all eight, 11 [C090] of them extremist: the
errors every technique shares are almost all false negatives. 06 has the
most errors no other technique makes (17 [C092]; the maximum over techniques
is 17 [C093]).[^err]

[^err]: `results_summary/rq/rq1_shared_errors.csv`,
    `results_summary/rq/rq1_error_overlap.csv`.

RQ1 reading. The ensemble is more accurate than every classical technique on
this split; 07 is more accurate than 04 and than no other technique. Every
other ordering in Table 1 is descriptive. **TODO(author):** adding notebook
12 (plain RoBERTa control) adds 12's pairs to the Holm family. 01 vs 11
(Holm p 0.049 [C061]) then stays significant only if every new pair is itself
Holm-significant (RESEARCH_PLAN section 3). Rerun `mcnemar_pairs.py` and
update Table 2 and this paragraph after 12 runs.

---

## 6. RQ2: What does each model rely on?

### 6.1 The attribution pipeline recovers known coefficients

Before any cross-model claim, the SHAP pipeline (partition explainer over a
word-level text masker, one method for every technique) was checked against
the coefficients of the word-level logistic regression (01). The
preregistered bars are Spearman of at least 0.80 [C190] between mean
attribution and coefficient, and top-50 sign agreement of at least
0.90 [C191]. 01 reaches Spearman 0.873 [C100] on the test run and 0.865 [C101] on the
validation run, with sign agreement 1.00 [C102] and 1.00 [C103]. Restricted to
words seen in at least two explained posts the correlations are 0.932 [C104] and
0.939 [C105] (a diagnostic, not the pass rule). The FastText LR (06), whose
coefficients act on embedding dimensions rather than words, reaches
0.745 [C106] and 0.757 [C107] with sign agreement 1.00 [C109]; it does not pass
(passed: false [C108]) and is reported as the calibration point, as
planned.[^wp2]

[^wp2]: `results_summary/01_LOG-REG_TF-IDF/attributions/shap-partition_{test,validation}_seed30__coefficient_check.json`;
    `results_summary/06_FASTTEXT-EMB_LOG-REG/attributions/shap-partition_{test,validation}_seed30__coefficient_check.json`.

### 6.2 The attributions name words the models need

A deletion test checks that each technique's top-attributed words matter to
that technique. For each test post, the k words with the highest positive
attribution toward extremist are deleted and the drop in p(extremist) is
compared with deleting k random words of the same post (5 [C257] seeds). On
the posts all eight techniques explained, top-k deletion lowers
p(extremist) more than random deletion for every technique at every k
(largest Holm-adjusted Wilcoxon p 7.6e-21 [C248] over the 32 [C251] rows; restored
predictors match the committed probabilities within 0.0019 [C252]). Deletion
recovers 0.46 [C249] to 0.80 [C250] of the summed SHAP value of the deleted words.
Table 3 gives k = 1.[^faith]

**Table 3.** Deletion faithfulness at k = 1 on the shared test posts.
Margin is mean drop (top-1) minus mean drop (random). These are descriptive
per technique; no test compares techniques.[^faith]

| Technique | Posts | Margin | Median paired diff. | Margin, flagged posts (n) | Share top-1 > random |
|---|---|---|---|---|---|
| 01 | 199 [C200] | 0.065 [C201] | 0.045 [C202] | 0.091 [C203] (74 [C204]) | 0.98 [C205] |
| 02 | 200 [C206] | 0.055 [C207] | 0.049 [C208] | 0.067 [C209] (70 [C210]) | 0.99 [C211] |
| 03 | 198 [C212] | 0.018 [C213] | 0.016 [C214] | 0.016 [C215] (60 [C216]) | 0.98 [C217] |
| 04 | 199 [C218] | 0.050 [C219] | 0.041 [C220] | 0.064 [C221] (61 [C222]) | 0.98 [C223] |
| 05 | 199 [C224] | 0.052 [C225] | 0.043 [C226] | 0.064 [C227] (55 [C228]) | 0.99 [C229] |
| 06 | 200 [C230] | 0.057 [C231] | 0.034 [C232] | 0.081 [C233] (59 [C234]) | 0.99 [C235] |
| 07 | 199 [C236] | 0.109 [C237] | 0.001 [C238] | 0.231 [C239] (75 [C240]) | 0.80 [C241] |
| 11 | 196 [C242] | 0.094 [C243] | 0.005 [C244] | 0.204 [C245] (72 [C246]) | 0.87 [C247] |

[^faith]: `results_summary/rq/rq2_faithfulness.csv`, rows
    `population=shared` (Table 3: `k=1`).

For the transformers the effect is concentrated in the posts they flag.
Their median paired difference is 0.001 [C253] to 0.010 [C254] (07) and 0.005 [C255] to
0.024 [C256] (11) across k, while their margin on flagged posts at k = 1
(0.231 [C239] and 0.204 [C245]) is above every classical technique's (0.016 [C258] to
0.091 [C259]). The test shows that the attributions find words the models use,
not that the models are right to use them.

### 6.3 Category shares

Table 4 gives, for each technique's whole-model test run, the share of
positive attribution mass in the top 50 words that falls in each lexicon
category. Every share carries, in brackets, its across-run standard
deviation (test run versus validation run), as the decision rules require;
the last column adds a post-level bootstrap SD for the identity share.

**Table 4.** Share of positive attribution mass in the top 50 words, test
run, with across-run SD in brackets. Provisional until WP3: the slur lexicon
is empty, so the slur column is unavailable and slurs are counted as
topical.[^t4]

| Technique | Posts explained | Extremist framing (SD) | Identity term (SD) | Slur | Topical (SD) | Identity SD, bootstrap |
|---|---|---|---|---|---|---|
| 01 | 450 [C110] | 0.000 [C111] (0.000 [C260]) | 0.094 [C112] (0.006 [C114]) | n/a | 0.906 [C113] (0.006 [C261]) | 0.016 [C115] |
| 02 | 450 [C116] | 0.000 [C117] (0.000 [C262]) | 0.116 [C118] (0.015 [C120]) | n/a | 0.884 [C119] (0.015 [C263]) | 0.025 [C121] |
| 03 | 450 [C122] | 0.000 [C123] (0.000 [C264]) | 0.000 [C124] (0.023 [C126]) | n/a | 1.000 [C125] (0.023 [C265]) | 0.031 [C127] |
| 04 | 450 [C128] | 0.000 [C129] (0.000 [C266]) | 0.104 [C130] (0.019 [C132]) | n/a | 0.896 [C131] (0.019 [C267]) | 0.036 [C133] |
| 05 | 450 [C134] | 0.000 [C135] (0.000 [C268]) | 0.130 [C136] (0.003 [C138]) | n/a | 0.870 [C137] (0.003 [C269]) | 0.020 [C139] |
| 06 | 450 [C140] | 0.000 [C141] (0.001 [C270]) | 0.132 [C142] (0.014 [C144]) | n/a | 0.868 [C143] (0.013 [C271]) | 0.023 [C145] |
| 07 | 200 [C146] | 0.030 [C147] (0.015 [C272]) | 0.070 [C148] (0.003 [C150]) | n/a | 0.900 [C149] (0.018 [C273]) | 0.034 [C151] |
| 11 | 200 [C152] | 0.016 [C153] (0.007 [C274]) | 0.174 [C154] (0.002 [C156]) | n/a | 0.810 [C155] (0.009 [C275]) | 0.033 [C157] |

[^t4]: Shares: `results_summary/rq/rq3_reliance_comparison.csv`, rows
    `role=whole`, `split=test`. Posts explained:
    `results_summary/rq/rq2_top_word_support.csv`. Across-run SD:
    `results_summary/rq/rq3_stability.csv`; it is the population SD
    (ddof = 0) over the two runs, that is |test share - validation share| / 2.
    Bootstrap SD: `results_summary/rq/rq2_bootstrap_category_shares.csv`,
    rows `category=identity_term`, `split=test`, `post_set=own`.

The topical category takes 0.81 [C158] to 1.00 [C159] of the positive mass. The
extremist framing lexicon (seeded from single-word entries of an extremism
lexicon) matches almost none of the top words, and the identity-term share
is 0.000 [C124] to 0.174 [C154]. With the slur list empty, Table 4 cannot yet
answer the RQ2 hypothesis about slurs versus identity terms. The bootstrap
SDs (0.016 [C115] to 0.036 [C133]) are larger than the two-run SDs (0.002 [C156] to
0.023 [C126]), so the SD the decision rule names understates post-sampling
spread. Multi-word lexicon entries cannot match word-level artifacts; the
decision rules ask for their count, but `categorize_attributions.py` only
prints it and no committed artifact records it. **TODO(author):** commit
that count (for example in the category-share sidecar) and report it here.
**TODO(author):** decide whether the section 5 difference rule uses the
two-run SD (as preregistered) or the bootstrap intervals. **TODO(author):**
WP3 (populate `slurs.txt` from a citable source, review the identity and
framing lists, freeze), then rerun `categorize_attributions.py`,
`compare_reliance.py`, `identity_fpr.py` and this table.

The identity lexicon also misses identity references, though the evidence is
thin. Across the eight top-50 lists there are 27 [C700] candidate
(technique, word) pairs that look like identity references (plurals,
hashtags, group nouns) but are not lexicon entries; they cover 14 [C703]
distinct words, and 26 [C704] of the 27 [C701] pairs rest on two explained
posts or fewer. 6 [C702] of the pairs are 11's.[^gaps] They are a review list
for WP3, not additions.

[^gaps]: `results_summary/rq/rq2_identity_top_k_sensitivity.csv`,
    `results_summary/rq/rq2_identity_lexicon_gaps_sensitivity.csv`.

### 6.4 Word-level agreement, without lexicons

Comparing the word tables directly avoids the lexicons. 3 [C160] words are
in the top 50 by positive attribution of all eight techniques: a racial
slur, "kill" and "genocide". Of the 168 [C162] words that appear in any top-50
list, 73 [C161] appear in one technique's list only.[^agree]

Agreement follows feature family. 01 and 02, two of the three word TF-IDF
linear models, agree closely (Spearman over shared words 0.90 [C163], top-50
Jaccard 0.67 [C164]), as do char and word+char TF-IDF (Spearman 0.85 [C167]) and
the two transformers (0.76 [C165] and 0.54 [C166]). Across families agreement is
low: 07 against the six classical techniques gives Spearman 0.21 [C168] to
0.28 [C169], and 11 gives 0.26 [C170] to 0.32 [C171]. FastText agrees little even
with the word TF-IDF LR (0.17 [C172]).[^agree]

[^agree]: `results_summary/rq/rq2_shared_top_words.csv`,
    `results_summary/rq/rq2_word_agreement_pairs.csv`
    (`spearman_mean_attribution`, `jaccard_top_k_positive`).

The transformers' top words rest on fewer posts. In their own runs,
0.56 [C173] (07) and 0.52 [C174] (11) of the top-50 positive words occur in a
single explained post, against 0.22 [C175] (01) to 0.40 [C176] (04) for the
classical runs. The runs differ in size (200 versus 450 posts), so the same
measure was recomputed on the 200 test posts every technique explained.
There the classical shares rise: 04 reaches 0.52 [C178] and 06 0.48 [C179], while
the word TF-IDF models give 01 0.32 [C177], 02 0.34 [C181], 05 0.38 [C180] and 03
0.44 [C182].[^support]

Under a post-level bootstrap this share is a replicate spread, not an
interval: resampling shifts it upward (01's own-run point is 0.22 [C175], its
replicate median 0.32 [C184]), and the point for 01 falls outside its own
spread (point inside: False [C183]). On paired draws, 07's excess over the word
TF-IDF models 01, 02 and 05 has lower bounds of 0.02 [C185], 0.04 [C186] and
0.02 [C187], and its difference from 04 spans -0.10 [C188] to 0.14 [C189]. The
comparison is descriptive.[^support]

[^support]: `results_summary/rq/rq2_top_word_support.csv`;
    `results_summary/rq/rq2_same_post_support.csv` (`post_set=common`,
    `split=test`); `results_summary/rq/rq2_bootstrap_same_post.csv`;
    `results_summary/rq/rq2_rq3_bootstrap_paired_differences.csv`
    (`measure=share_single_post_top_k_positive`).

RQ2 reading. The deletion test supports reading the word tables as what each
model uses. The models share a small core of words (a racial slur, "kill",
"genocide") and otherwise rely on different words, and the split runs along
feature family. Category-level conclusions wait for WP3.

---

## 7. RQ3: Does pretraining or ensembling change that reliance?

### 7.1 Pretraining lineage and pooling

Table 5 compares the single Twitter-RoBERTa (07), the four members of the
ensemble's primary pool, their mean, and the pooled ensemble (11). The
members do not share 07's training recipe, so Table 5 is not a controlled
lineage comparison:

* 07 is one seed (30 [C711]) at learning rate 2e-5 [C710], starting from
  cardiffnlp/twitter-roberta-base-hate-latest [C740].
* Each member of 11 is the average of three seeds (17 [C713], 30 [C714],
  73 [C715] for member A; the same three for B, C and E: 17 [C719], 30 [C720],
  73 [C721]; 17 [C725], 30 [C726], 73 [C727]; 17 [C731], 30 [C732], 73 [C733]).
* Learning rates are set per member: 3e-5 [C712] (A), 3e-5 [C718] (B), 3e-5 [C724]
  (C) and 2e-5 [C730] (E).
* Classification heads differ: pretrained [C716] for A and pretrained [C722] for B, fresh [C728]
  for C and fresh [C734] for E.

**Table 5.** Share of positive attribution mass in the top 50 words, by
transformer, with across-run SD in brackets where two runs exist. Members
are explained on the test sample only (one run each), so they have no
across-run SD. Provisional until WP3.[^t5]

| Model | Checkpoint | Split | Extremist framing | Identity term | Topical |
|---|---|---|---|---|---|
| 07 | cardiffnlp/twitter-roberta-base-hate-latest [C740] | test | 0.030 [C300] (0.015 [C272]) | 0.070 [C301] (0.003 [C150]) | 0.900 [C302] (0.018 [C273]) |
| 07 | (same) | validation | 0.000 [C303] | 0.064 [C304] | 0.936 [C305] |
| 11 member A | cardiffnlp/twitter-roberta-base-hate-latest [C717] | test | 0.027 [C306] | 0.089 [C307] | 0.885 [C308] |
| 11 member B | facebook/roberta-hate-speech-dynabench-r4-target [C723] | test | 0.000 [C309] | 0.222 [C310] | 0.778 [C311] |
| 11 member C | GroNLP/hateBERT [C729] | test | 0.014 [C312] | 0.089 [C313] | 0.897 [C314] |
| 11 member E | cardiffnlp/twitter-roberta-large-hate-latest [C735] | test | 0.000 [C315] | 0.197 [C316] | 0.803 [C317] |
| 11 member mean | | test | 0.010 [C318] | 0.149 [C319] | 0.841 [C320] |
| 11 pooled | | test | 0.016 [C321] (0.007 [C274]) | 0.174 [C322] (0.002 [C156]) | 0.810 [C323] (0.009 [C275]) |
| 11 pooled | | validation | 0.030 [C324] | 0.178 [C325] | 0.792 [C326] |

[^t5]: Shares: `results_summary/rq/rq3_reliance_comparison.csv`, columns
    `share_of_positive_mass__*`. SDs: `results_summary/rq/rq3_stability.csv`
    (ddof = 0 over the two runs, as in Table 4). Recipes and checkpoints:
    `results_summary/07_TWITTER-ROBERTA_FINE-TUNE/best_config.json` and
    `results_summary/11_MULTI-CHECKPOINT_LOGIT-POOL/best_config.json`
    (`admission`, `component_resolution`,
    `hyperparameters.component_run_configs`).

Member A starts from 07's checkpoint, yet its identity share is 0.089 [C307]
against 07's 0.070 [C301]. That gap between two models from the same checkpoint,
trained with different seeds and learning rates, is a rough
within-checkpoint noise floor. Member C (0.089 [C313]) is within that distance
of A. B (0.222 [C310]) is further away and differs from A only in its checkpoint
(same learning rate, same head policy), which makes A versus B the closest
thing to a controlled contrast in Table 5; E (0.197 [C316]) also differs in
learning rate, head and model size. Both are single runs, so no lineage
reading is drawn.
Per-member differences cannot be claimed under the decision rule, because
each member has one run.

The pooled ensemble has a larger identity-term share than 07 (0.174 [C322]
against 0.070 [C301] on test; 0.178 [C325] against 0.064 [C304] on validation). The
difference exceeds the across-run SDs of the two (0.002 [C156] and 0.003 [C150]),
which is the preregistered criterion (provisional until WP3). On paired
bootstrap draws over the same 200 posts it is -0.104 [C327] (07 minus 11) with a
percentile interval of -0.140 [C328] to -0.025 [C329].[^rq3a] The difference cannot be
attributed to pooling: the ensemble differs from 07 in its checkpoints, its
seeds, its learning rates and heads, and in pooling itself. Its share sits
above the member mean (0.149 [C319]) and between the members.

[^rq3a]: `results_summary/rq/rq3_stability.csv`;
    `results_summary/rq/rq2_rq3_bootstrap_paired_differences.csv`
    (`measure=identity_share_of_positive_mass`).

Against the classical techniques the picture is weaker. On paired draws,
11's identity share is above 01, 02, 03 and 06 (lower bounds 0.026 [C332],
0.003 [C333], 0.064 [C334], 0.032 [C335]) but not above 04 or 05 (lower bounds -0.052 [C330]
and -0.017 [C331]). None of 07's six differences from a classical technique
excludes zero (the bounds nearest zero are -0.020 [C336] against 03 and 0.028 [C337]
against 04). These intervals cover post sampling only.[^rq3a]

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
| 01 | 0.704 [C340] | 0.644 [C341] to 0.752 [C342] | 0.333 [C343] |
| 02 | 0.711 [C344] | 0.650 [C345] to 0.758 [C346] | 0.333 [C347] |
| 03 | 0.695 [C348] | 0.656 [C349] to 0.760 [C350] | 0.316 [C351] |
| 04 | 0.705 [C352] | 0.632 [C353] to 0.747 [C354] | 0.333 [C355] |
| 05 | 0.743 [C356] | 0.706 [C357] to 0.799 [C358] | 0.471 [C359] |
| 06 | 0.616 [C360] | 0.566 [C361] to 0.681 [C362] | 0.190 [C363] |
| 07 | 0.427 [C364] | 0.344 [C365] to 0.480 [C366] | 0.176 [C367] |
| 11 | 0.488 [C368] | 0.405 [C369] to 0.538 [C370] | 0.220 [C371] |

[^t6]: `results_summary/rq/rq3_bootstrap_split_agreement.csv`, rows
    `post_set=common`; paired differences from
    `results_summary/rq/rq2_rq3_bootstrap_paired_differences.csv`
    (`measure=spearman_mean_abs_attribution`).

The transformers' attributions agree less across post samples than the
classical models'. On paired draws all twelve transformer-minus-classical
differences in Spearman have intervals that exclude zero: the largest upper
bound is -0.129 [C372] for 07's six and -0.078 [C373] for 11's six. 07 and 11 do not
separate from each other (-0.061 [C374], interval -0.126 [C375] to 0.005 [C376]). Top-50
Jaccard does not separate the transformers from FastText (06, 0.190 [C363]).
This is descriptive. It covers post sampling, not the SHAP seed or another
split, and lower agreement may reflect attributions that depend on context,
which a word-level mean over different posts reads as instability, rather
than less reliable explanations.[^t6]

RQ3 reading. The pooled ensemble has a larger identity-term share than the
single transformer on both splits, with recipe, seeds and pooling
confounded. Per-word claims about either transformer are fragile across
post samples. The pretraining-lineage comparison waits for notebook 12.

---

## 8. RQ4: Do identity terms drive false positives?

### 8.1 Correlation: false-positive rates with and without identity terms

Of the 280 [C465] non-extremist test posts, 67 [C464] contain at least one
identity-lexicon match (83 [C466] extremist posts do).[^rq4n] Every technique
has a higher false-positive rate on those 67 [C464] than on the other
213 [C474] (Table 7). The ratio is 1.91 [C420] to 4.24 [C460]. Unadjusted Fisher
exact p is below 0.05 for 04, 05 and 11 only. With so few such posts, only
large ratios are detectable, so a non-significant ratio is not evidence that
the effect is absent. The identity matches come from the lexicon, so these
rates are provisional until WP3.

[^rq4n]: `results_summary/rq/rq4_identity_fpr.csv`,
    `results_summary/rq/rq2_rq4_relation_inputs.csv`.

**Table 7.** False positives on non-extremist test posts with and without an
identity term, at each locked threshold, and the ratio after setting aside
the 14 [C468] non-extremist posts whose only matches are ambiguous words,
leaving 53 [C467] (the list is declared in `tools/identity_sensitivity.py`:
white, black, man, men, boy, boys, girl, girls, english, native, trans,
asylum, indian; the lexicon is unchanged; 24 [C469] extremist posts are set
aside the same way).[^t7]

| Technique | FP with | FPR with | FP without | FPR without | Ratio | Fisher p | Ratio, unambiguous | Fisher p |
|---|---|---|---|---|---|---|---|---|
| 01 | 11 [C400] | 0.164 [C401] | 18 [C402] | 0.085 [C403] | 1.94 [C404] | 0.069 [C405] | 2.01 [C406] | 0.077 [C407] |
| 02 | 10 [C408] | 0.149 [C409] | 15 [C410] | 0.070 [C411] | 2.12 [C412] | 0.082 [C413] | 2.14 [C414] | 0.096 [C415] |
| 03 | 6 [C416] | 0.090 [C417] | 10 [C418] | 0.047 [C419] | 1.91 [C420] | 0.226 [C421] | 1.61 [C422] | 0.489 [C423] |
| 04 | 9 [C424] | 0.134 [C425] | 11 [C426] | 0.052 [C427] | 2.60 [C428] | 0.030 [C429] | 2.92 [C430] | 0.031 [C431] |
| 05 | 5 [C432] | 0.075 [C433] | 4 [C434] | 0.019 [C435] | 3.97 [C436] | 0.038 [C437] | 5.02 [C438] | 0.017 [C439] |
| 06 | 7 [C440] | 0.104 [C441] | 11 [C442] | 0.052 [C443] | 2.02 [C444] | 0.152 [C445] | 1.83 [C446] | 0.328 [C447] |
| 07 | 9 [C448] | 0.134 [C449] | 15 [C450] | 0.070 [C451] | 1.91 [C452] | 0.131 [C453] | 1.88 [C454] | 0.164 [C455] |
| 11 | 8 [C456] | 0.119 [C457] | 6 [C458] | 0.028 [C459] | 4.24 [C460] | 0.0064 [C461] | 4.69 [C462] | 0.0056 [C463] |

[^t7]: FP counts: `results_summary/rq/rq2_rq4_relation_inputs.csv`. Rates,
    ratio and Fisher p: `results_summary/rq/rq4_identity_fpr.csv`.
    Unambiguous columns: `results_summary/rq/rq4_identity_fpr_sensitivity.csv`,
    `variant=exclude_ambiguous_only`.

Setting the ambiguous words aside keeps every ratio above 1 (1.61 [C422] to
5.02 [C438]), and the same three techniques stay below p = 0.05. 03 and 06 fall
to 1.61 [C422] and 1.83 [C446], which at this n is not evidence that their effect
is absent. The false-positive counts are small (5 [C432] to 11 [C400] with
identity terms) and below the 14-row rule of thumb, so Table 7 does not rank
the techniques.

### 8.2 Intervention: editing the identity term

The correlation does not show that the identity term causes the false
positive. WP7 edits every identity-lexicon match in the 150 [C623] test posts
that have one and re-scores the edited posts with each technique's restored
predictor at its locked threshold. Before scoring, each restored predictor
reproduces the committed test probabilities to within 0.0016 [C621]. This
re-scores frozen models and selects nothing, so it is a perturbation
analysis, not a second test evaluation. There are two edits. Removal deletes
the match (the preregistered primary operation). Substitution replaces it
with a neutral control panel ("group", "population", "humans", each checked
for label balance and a small 01 coefficient) and averages over the panel.
Substitution is the stricter control: it keeps a group noun in place and
keeps the sentence grammatical, so it isolates the identity term more cleanly than
deletion does. All p values are Holm-adjusted over the 32 [C622] technique x
population x operation rows.[^t8]

**Table 8.** Identity-term removal. Mean and median delta are p(extremist)
before minus after the edit, on the non-extremist posts. Flag counts are
before -> after, with the posts that flip in brackets.[^t8]

| Technique | Mean delta | Median delta | Wilcoxon Holm p | Non-extremist flagged (cleared / new) | Extremist flagged (lost / gained) | McNemar Holm p, extremist |
|---|---|---|---|---|---|---|
| 01 | 0.035 [C500] | 0.024 [C501] | 2.5e-5 [C502] | 11 [C503] -> 8 [C504] (3 [C505] / 0 [C506]) | 62 [C507] -> 56 [C508] (6 [C509] / 0 [C510]) | 0.625 [C511] |
| 02 | 0.020 [C512] | 0.017 [C513] | 3.1e-4 [C514] | 10 [C515] -> 8 [C516] (3 [C517] / 1 [C518]) | 60 [C519] -> 50 [C520] (10 [C521] / 0 [C522]) | 0.051 [C523] |
| 03 | 0.001 [C524] | 0.000 [C525] | 0.92 [C526] | 6 [C527] -> 7 [C528] (0 [C529] / 1 [C530]) | 51 [C531] -> 49 [C532] (4 [C533] / 2 [C534]) | 1.000 [C535] |
| 04 | 0.022 [C536] | 0.012 [C537] | 1.4e-3 [C538] | 9 [C539] -> 4 [C540] (5 [C541] / 0 [C542]) | 53 [C543] -> 45 [C544] (8 [C545] / 0 [C546]) | 0.172 [C547] |
| 05 | 0.026 [C548] | 0.010 [C549] | 1.8e-4 [C550] | 5 [C551] -> 4 [C552] (1 [C553] / 0 [C554]) | 53 [C555] -> 47 [C556] (7 [C557] / 1 [C558]) | 1.000 [C559] |
| 06 | 0.040 [C560] | 0.034 [C561] | 1.6e-6 [C562] | 7 [C563] -> 7 [C564] (1 [C565] / 1 [C566]) | 52 [C567] -> 43 [C568] (9 [C569] / 0 [C570]) | 0.094 [C571] |
| 07 | 0.099 [C572] | 0.001 [C573] | 3.3e-5 [C574] | 9 [C575] -> 2 [C576] (7 [C577] / 0 [C578]) | 71 [C579] -> 57 [C580] (15 [C581] / 1 [C582]) | 0.016 [C583] |
| 11 | 0.095 [C584] | 0.003 [C585] | 9.8e-6 [C586] | 8 [C587] -> 2 [C588] (7 [C589] / 1 [C590]) | 74 [C591] -> 52 [C592] (22 [C593] / 0 [C594]) | 1.5e-5 [C595] |

[^t8]: `results_summary/rq/rq4_identity_counterfactual.csv` (Table 8:
    `operation=remove`).

*Removal.* On the non-extremist posts, removal lowers p(extremist) for every
technique except 03 (the other seven have Holm p at most 1.4e-3 [C660]). 03 shows
no detectable shift (mean 0.001 [C524], Holm p = 0.92 [C526]), which is not evidence
that the effect is absent. For 07 and 11 the mean drop (0.099 [C572],
0.095 [C584]) is far above the median (0.001 [C573], 0.003 [C585]): the shift sits in a
minority of posts. Removal clears 7 [C577] of 9 [C575] original false
positives for 07 and 7 [C589] of 8 [C587] for 11, against 0 [C529] to
5 [C541] for the classical techniques. These flip counts are descriptive: no
non-extremist flip test survives Holm (smallest 0.33 [C661]), and the counts
are below the 14-row rule of thumb. On the extremist posts, removal unflags
15 [C581] of 71 [C579] flagged posts for 07 and 22 [C593] of 74 [C591] for 11
(Holm p = 0.016 [C583] and 1.5e-5 [C595]). 07 and 11 are the only techniques whose
recall loss survives Holm in this family; techniques are not compared with
each other. 02, for example, loses 10 [C521] of 60 [C519] (Holm p =
0.051 [C523]). Recall on extremist posts with and without identity terms differs
by little for 07 (0.855 [C470] against 0.839 [C471], one or two posts) and more for
11 (0.892 [C472] against 0.793 [C473]).[^rec]

[^rec]: `results_summary/rq/rq4_identity_fpr.csv`
    (`recall_with_identity_terms`, `recall_without_identity_terms`).

*Removal on the unambiguous posts.* On the 59 [C617] extremist posts that
keep an unambiguous identity match, removal unflags 15 [C611] of 51 [C610]
flagged posts for 11 (Holm p = 0.002 [C612], over that table's 32 [C670] rows)
and 10 [C614] of 48 [C613] for 07 (Holm p = 0.33 [C615]; flip rate 0.21 [C616],
close to 07's 15 [C581] of 71 [C579] on all posts, with less power at the
reduced n). Removing only the unambiguous matches unflags 11 [C618] of
51 [C610] for 11 (Holm p = 0.030 [C619]) and 6 [C620] of 48 [C613] for 07. Within the 8 [C626] extremist removal rows alone, 11 (4.9e-4 [C629]) and 02
(0.027 [C627]) are Holm-significant and 07 (0.070 [C628]) is not. So 11's recall
loss under removal is significant in every family tried, and 07's depends on
the family.[^cfs]

[^cfs]: `results_summary/rq/rq4_identity_counterfactual_sensitivity.csv`
    (`population=extremist`; the extremist-removal family is the column
    `mcnemar_p_holm_remove_extremist`).

*Substitution.* Under the stricter control the pattern changes. The four
classical techniques 01, 04, 05 and 06 each lose 11 [C604], 11 [C609],
11 [C624] and 11 [C625] flagged extremist posts (Holm p = 0.029 [C600],
0.029 [C601], 0.029 [C602], 0.029 [C603]), and the classical techniques other than 03
still lower p(extremist) on non-extremist posts by a mean of 0.028 [C666] to
0.045 [C667], about as much as under removal. For 07 and 11 the recall loss does
not survive Holm (1.00 [C605] and 0.094 [C606]). On non-extremist posts their shift
is small and not significant (07 mean 0.012 [C607], Holm p = 0.24 [C608]; 11 mean
0.038 [C669], Holm p = 0.92 [C664]). On extremist posts their p(extremist) still
falls (07 mean 0.053 [C665], Holm p = 1.4e-3 [C668]; 11 mean 0.111 [C663], Holm p =
1.7e-9 [C662]), but too few posts cross the threshold for the flip test.[^sub]

[^sub]: `results_summary/rq/rq4_identity_counterfactual.csv`
    (`operation=substitute`).

*One reading.* For the classical word-level techniques, both edits move
p(extremist) down on posts of both labels: replacing the identity term with
a neutral group noun has about the same effect as deleting it, which is
consistent with these models weighting the identity term itself as evidence
of extremism. For 07 and 11, the removal effects on non-extremist posts and
on recall are not reproduced by substitution, so they may partly reflect
deleting a noun (losing the group referent, or leaving ungrammatical text)
rather than the identity term specifically; the transformers' p(extremist)
on extremist posts still falls under substitution, so the term is not
irrelevant to them either. We therefore do not claim that the transformers
treat identity terms as evidence of extremism, and we do not rank techniques
by either edit.

A flip shows that one perturbation moved one post across the locked
threshold. That is threshold-local necessity, not proof that the term causes
the error. Deletion also changes TF-IDF normalization and character n-grams
across the join, the panel words are rare in training, and the lexicon
covers only the identity terms it lists.

### 8.3 Does RQ2 reliance predict RQ4 false positives?

This section is descriptive and, because both sides depend on the lexicons,
provisional until WP3. Ranking the eight techniques by RQ2 identity-term
share (Table 4) and by FPR ratio (Table 7) gives Spearman rho = 0.83 [C630] at
n = 8 [C631] (leave-one-out 0.74 [C634] to 0.95 [C635]; Holm p over the two
preregistered rows 0.034 [C655], from which no inference is drawn at this n).
The shares in Table 4 come from runs of different sizes (450 posts for the
classical runs, 200 for the transformers); the decision rules compare runs
on the same number of posts. On the 200 posts every technique explained the
correlation is 0.946 [C656], and with the validation-run shares 0.874 [C657]. Three
further results argue against reading any of these as a link between
reliance and identity-term false positives.[^rel]

* The ranking follows the denominator. Identity share against FPR on
  identity-term posts is -0.22 [C636]; against FPR on the other posts it is
  -0.43 [C637]. The two highest ratios come from low FPR without identity terms
  (05: 4 [C434] false positives, 11: 6 [C458]), not from more identity-term
  false positives.
* The intervention does not follow it. The WP7 mean removal shift against
  the FPR ratio gives 0.06 [C638]. On the subset of techniques whose
  attributions cover the whole test split (`full_split_posts`, the
  6 [C640] classical techniques) the primary correlation is 0.60 [C639].
* It is unstable under resampling. Resampling explained posts for the shares
  and test rows for the ratios (2000 [C648] replicates) gives a replicate median
  of 0.48 [C641]. Rho is positive in a share 0.90 [C642] of replicates and below
  the point estimate in 0.94 [C643], and the replicates spread from -0.29 [C644] to
  0.89 [C645]. The instability comes mainly from the ratios: resampling test
  rows alone gives a median of 0.43 [C646], posts alone 0.80 [C647]. The
  common-post version behaves the same way (replicate median 0.52 [C658],
  positive in 0.90 [C659]). Each ratio rests on few false positives; at the
  0.95 [C654] percentile level only 04 (1.06 [C649] to 6.37 [C650]) and 11 (1.40 [C651]
  to 16.1 [C652]) keep their ratio above 1, and 05, with the second-highest
  ratio, does not (lower bound 0.91 [C653]).[^boot]

Resampling noisy ratios also pulls a rank correlation toward zero, so these
figures describe how fragile the ordering is; they are not an interval for
rho. RQ4 reading: every technique has a higher FPR on benign posts that
mention a community (Fisher p < 0.05, unadjusted, for three of eight); for
the classical word-level techniques the identity term behaves as evidence of
extremism under both edits; for the transformers the removal effect is not
reproduced by the stricter substitution control; and the cross-technique
ordering of reliance and false positives is not established.

[^rel]: `results_summary/rq/rq2_rq4_relation.csv` (primary, secondary,
    sensitivity and descriptive rows); FP counts from
    `results_summary/rq/rq2_rq4_relation_inputs.csv`.

[^boot]: `results_summary/rq/rq2_rq4_bootstrap_relation.csv`.

---

## 9. Why the task is hard (C3)

Each item cites the evidence above; "avenue" names what would address it.
None of these is a claim that the task is new: extremism and hate-speech
detection are established research areas. The aim is to say where the
remaining difficulty sits.

1. **Small test splits hide real differences.** With 450 [C818] test posts,
   7 [C058] of 28 [C057] pairs clear Holm-corrected McNemar significance, and
   06 vs 07 (row gap -25 [C085]) does not, because 83 [C086] posts are
   discordant. More labelled data mainly buys statistical resolution.
   Avenue: a larger test split, which also helps RQ4.
2. **Some errors are shared by every model.** 12 [C089] test posts are wrong
   for all eight techniques, 11 [C090] of them extremist. They point at label
   ambiguity, or at signal the text does not contain, which fits the
   post-level, text-only limitation. Avenue: multi-annotator labels with
   reported agreement. **TODO(author):** a qualitative read of these posts
   by the authors (outside the repository; no text may be committed).
3. **Identity terms are hard to separate from what surrounds them.** Every
   technique has a higher FPR on benign posts with identity terms (Table 7).
   For the classical techniques, replacing the term with a neutral group
   noun lowers p(extremist) about as much as deleting it. For the
   transformers, deleting it costs recall but substituting it does not, so
   the edit, not only the term, may drive the effect (section 8.2). Avenue:
   counterfactual data augmentation or identity-balanced training,
   evaluated with substitution as well as deletion and with the
   recall-cost reporting used here.
4. **Transformer explanations are faithful on the posts they flag but
   unstable across samples.** The deletion test holds for every technique,
   and for the transformers its effect is concentrated in flagged posts
   (Table 3); yet their word rankings agree least across post samples
   (Spearman, Table 6). Per-word claims about transformers are fragile.
   Avenue: report explanation stability and deletion faithfulness as
   standard metrics.
5. **Lexicons do not cover the vocabulary.** The topical category takes
   0.81 [C158] to 1.00 [C159] of positive mass, the framing lexicon matches almost
   none of the top words, the slur list is empty, and 14 [C703] distinct
   words that look like identity references are missing from the identity
   list (most on thin support). Category-level claims depend on curation.
   Avenue: hand-annotated top words rather than lexicons alone.
6. **Context matters.** Quotation, reporting and counter-speech can use the
   same words as extremist posts, and a word-level attribution cannot
   separate them. No artifact here measures this. **TODO(author):** either
   support this item with evidence (for example a labelled sample of
   counter-speech among the false positives, kept outside the repository)
   or present it as a hypothesis. Avenue: layer-wise probes (WP9) and LLM
   self-explanations (WP10). **TODO(author):** WP9 needs a Kaggle GPU run;
   WP10 needs an author decision on sending dataset text to a hosted model
   (an open-weight local model avoids it). Neither result exists.

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

* RQ1: test accuracy 0.856 [C800]. Holm runs over the comparison's 28 [C801]
  pairs. Only the pair with 11 clears it (row gap -24 [C802], Holm p =
  0.049 [C803]); the other pairs read "not distinguishable at n = 450". The card
  never says "better".
* SHAP check: PASS [C804], with the per-run figures from section 6.1.
* RQ2: the top words with category and support, word-level agreement with
  every other technique, and the deletion test on the 200 [C813] shared posts
  (k = 1 margin 0.065 [C806]; Holm family of 32 [C805] rows).
* RQ3: on the technique's own runs (450 posts per split, so not the
  common-post figures of Table 6), test-versus-validation Spearman 0.711 [C807]
  and top-50 Jaccard 0.389 [C808].
* RQ4: FPR ratio 1.94 [C809] (Fisher p = 0.069 [C810]), and 2.01 [C811] without the
  ambiguous-only posts; WP7 removal clears 3 [C812] of the 11 [C819] false
  positives.

[^card]: `results_summary/rq/report_cards/01_LOG-REG_TF-IDF.json`.

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
flip. The guide therefore requires it to be a recorded decision. On the
committed state, `audit.py run --technique 01_LOG-REG_TF-IDF` rewrites every
table, coefficient check and card byte for byte, so the chain is
deterministic.

**Properties to stress.** Derived, not transcribed: metrics come from
probabilities through one function. Text-free: every committed output passes
`tools/scan_text_leakage.py`, so results can be shared where the posts
cannot. Adversarial tests: each tool has a test proving that a specific
malformed input is refused. The same holds for this draft:
`tools/check_claims.py` refuses a claim whose value or artifact does not
match, and a number in this text that differs from its claim.

**TODO(author):** the report cards print the top words verbatim, including
a racial slur in 01's card. Any figure of a card in the paper must redact
it. **TODO(author):** external assets (weights, per-post files) are not yet
uploaded to a Kaggle Dataset or recorded with `run_manifest.py add-asset`.
The perturbation stages cannot be reproduced by a reader until they are.

---

## Limitations specific to these results

Labels are post-level and features text-only; one dataset and one frozen
split; a small test split; category shares depend on lexicons that are not
yet frozen; SHAP over subword models is aggregated to words with a
word-level masker; the bootstrap intervals cover post sampling only, not
SHAP seeds or other splits; transformer runs explain fewer posts per split
than the classical runs (the same-post tables control for this where
stated); the transformers in RQ3 differ in more than pretraining. The
framework audits detectors of extremist content in single posts and is not a
deployment recommendation.

## TODO(author) summary

1. Notebook 12 (plain RoBERTa control): run it, rerun RQ1 (Holm family
   with 12's pairs), RQ3 and the relation, then revise sections 5 and 7.
2. WP3 lexicon freeze: populate `slurs.txt` from a citable source, review the
   identity list (the gap candidates) and the framing list, freeze, rerun RQ2
   and RQ4, and remove "provisional" from sections 6 to 8.
3. Commit the count of skipped multi-word lexicon entries, as the decision
   rules require it reported.
4. Section 5 rule: two-run SD (preregistered) or bootstrap intervals for
   share differences.
5. Validation-split member runs for 11, if RQ3 prose needs member SDs.
6. WP9 (layer-wise probes) and WP10 (zero-shot LLM, data-sharing decision).
7. A qualitative read of the shared errors, outside the repository; evidence
   or hypothesis status for C3 item 6.
8. Redact slurs in any report-card figure.
9. Upload and record external assets before submission.
10. Citations: place the related-work entries (`docs/RELATED_WORK.md`) in the
    Results prose; none are cited in this draft.
