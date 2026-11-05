# Related work

Written 2026-11-04. This is the working bibliography for the paper. It is
grouped into six sections (a) to (f). Each entry gives the citation and then
one line on how it bears on this repository's research questions: RQ1
accuracy, RQ2 reliance, RQ3 pretraining and ensembling, RQ4 identity-term
false positives. It also covers the WP7 identity counterfactual and the
planned WP9 (probes) and WP10 (zero-shot LLM) packages. `docs/POSITIONING.md`
decides how the paper is framed. This file only collects and places the prior
work.

Framing rules that apply to every line below (CLAUDE.md and the plan,
section 1):

* Extremism and hate-speech detection are established research areas with
  datasets, shared tasks and surveys (section a). This paper does not claim
  the task is new. Its contribution is the attribution comparison and the
  auditing framework.
* Labels are post-level and features are text-only. Work on accounts,
  networks or radicalisation over time is cited as context, not as something
  this paper does.
* The attribution categories are the four in `data/lexicons/`: extremist
  framing, identity term, slur, topical. Other papers' taxonomies (for
  example HateXplain's target communities) are described in their own terms
  and are not mapped onto a new category.

## How the entries were verified

Every entry was checked on 2026-11-04 against at least one primary record:

* the arXiv API (`export.arxiv.org`) for title, author list, first-submission
  date and arXiv id;
* Crossref (`api.crossref.org`) for the DOI, the container title and the
  publication year;
* the venue's own page where Crossref has no record: the ACL Anthology, AAAI
  OJS, PMLR, NeurIPS proceedings, OpenReview/TMLR or JSTOR (found by web
  search);
* Semantic Scholar's batch API, as a cross-check of venue and ACL Anthology
  id.

Conventions:

* "et al." stands for more than six authors. Author lists of six or fewer
  are complete.
* Year is the year of the venue. For an arXiv-only paper it is the year of
  first submission.
* A DOI of the form `10.48550/arXiv.*` is not used. The arXiv id is given
  instead.
* **UNVERIFIED** marks an item whose primary record could not be confirmed.
  Do not cite it in the paper until it is checked.
* Several papers have a published venue that differs from the one printed
  on the arXiv abstract. Where they differ, the venue's own record was used.

---

## (a) Extremism and hate-speech detection: datasets, models, evaluation

### Surveys of extremism detection

* **Gaikwad, M., Ahirrao, S., Phansalkar, S., Kotecha, K. (2021).** Online
  Extremism Detection: A Systematic Literature Review With Emphasis on
  Datasets, Classification Techniques, Validation Methods, and Tools.
  *IEEE Access* 9. DOI 10.1109/ACCESS.2021.3068313.
  *Relation:* shows that text-based extremism detection is an established
  field with many datasets and classifiers. The introduction cites it to place
  RQ1 as a brief replication, not a new result.
* **Aldera, S., Emam, A., Al-Qurishi, M., Alrubaian, M., Alothaim, A. (2021).**
  Online Extremism Detection in Textual Content: A Systematic Literature
  Review. *IEEE Access* 9. DOI 10.1109/ACCESS.2021.3064178.
  *Relation:* the second systematic review. Its emphasis on textual content
  matches our text-only scope, and it is the reference for "the task is
  well studied".
* **Torregrosa, J., Bello-Orgaz, G., Martínez-Cámara, E., Del Ser, J.,
  Camacho, D. (2022).** A survey on extremism analysis using natural language
  processing: definitions, literature review, trends and challenges.
  *Journal of Ambient Intelligence and Humanized Computing* 14(8),
  9869-9905.
  DOI 10.1007/s12652-021-03658-z.
  *Relation:* covers definitions of extremism and the open problems. It
  supports the C3 "why the task is hard" section (context, label ambiguity).

### Extremism-specific detection studies

* **Ferrara, E., Wang, W.-Q., Varol, O., Flammini, A., Galstyan, A. (2016).**
  Predicting Online Extremism, Content Adopters, and Interaction Reciprocity.
  *SocInfo 2016*, LNCS. DOI 10.1007/978-3-319-47874-6_3.
  *Relation:* uses user and network signals as well as content. It is the
  contrast for our stated limitation that we classify posts from text only
  and do not model accounts or interaction.
* **Saif, H., Dickinson, T., Kastler, L., Fernandez, M., Alani, H. (2017).**
  A Semantic Graph-Based Approach for Radicalisation Detection on Social
  Media. *ESWC 2017*, LNCS. DOI 10.1007/978-3-319-58068-5_35.
  *Relation:* goes beyond surface word features with semantic graph features.
  This is the kind of signal C3 item 6 (context) says word-level attribution
  cannot see.
* **Fernandez, M., Asif, M., Alani, H. (2018).** Understanding the Roots of
  Radicalisation on Twitter. *Proceedings of the 10th ACM Conference on Web
  Science*. DOI 10.1145/3201064.3201082.
  *Relation:* models radicalisation as a process over a user's history. We
  cite it to state that our post-level labels do not support claims about
  radicalisation over time.
* **Nouh, M., Nurse, J. R. C., Goldsmith, M. (2019).** Understanding the
  Radical Mind: Identifying Signals to Detect Extremist Content on Twitter.
  *IEEE ISI 2019*. DOI 10.1109/ISI.2019.8823548.
  *Relation:* combines textual, psychological and behavioural signals for
  extremist content. Its textual signals are close to our "extremist
  framing" lexicon category (RQ2).

### Hate-speech and abusive-language datasets and surveys

* **Waseem, Z., Hovy, D. (2016).** Hateful Symbols or Hateful People?
  Predictive Features for Hate Speech Detection on Twitter. *NAACL Student
  Research Workshop*. DOI 10.18653/v1/N16-2013.
  *Relation:* an early feature study with character n-grams on tweets. It
  is prior art for our character-level TF-IDF SVM (technique 04) in RQ1.
* **Davidson, T., Warmsley, D., Macy, M., Weber, I. (2017).** Automated Hate
  Speech Detection and the Problem of Offensive Language. *ICWSM 2017*.
  arXiv:1703.04009.
  *Relation:* separates hate from offensive language. It is the classic
  example of the label ambiguity behind the twelve posts every technique gets
  wrong (C3 item 2).
* **Schmidt, A., Wiegand, M. (2017).** A Survey on Hate Speech Detection
  using Natural Language Processing. *SocialNLP Workshop (EACL 2017)*.
  DOI 10.18653/v1/W17-1101.
  *Relation:* surveys the feature families (word and character n-grams,
  lexicons, embeddings) that techniques 01 to 06 use.
* **Fortuna, P., Nunes, S. (2018).** A Survey on Automatic Detection of Hate
  Speech in Text. *ACM Computing Surveys* 51(4). DOI 10.1145/3232676.
  *Relation:* the second standard survey. Its discussion of definitions
  supports the scoping in the introduction.
* **Founta, A.-M., Djouvas, C., Chatzakou, D., Leontiadis, I., Blackburn, J.,
  Stringhini, G., et al. (2018).** Large Scale Crowdsourcing and
  Characterization of Twitter Abusive Behavior. *ICWSM 2018*.
  arXiv:1802.00393.
  *Relation:* a large crowdsourced annotation study. It is the reference for
  the multi-annotator labels C3 proposes as future work.
* **Zampieri, M., Malmasi, S., Nakov, P., Rosenthal, S., Farra, N.,
  Kumar, R. (2019).** Predicting the Type and Target of Offensive Posts in
  Social Media (OLID). *NAACL 2019*. DOI 10.18653/v1/N19-1144.
  *Relation:* labels the target of offence (individual, group, other).
  Group-targeted posts are where identity terms occur, which is RQ4's
  population.
* **Basile, V., Bosco, C., Fersini, E., Nozza, D., Patti, V., Rangel Pardo,
  F. M., et al. (2019).** SemEval-2019 Task 5: Multilingual Detection of Hate
  Speech Against Immigrants and Women in Twitter (HatEval). *SemEval 2019*.
  DOI 10.18653/v1/S19-2007.
  *Relation:* a shared task whose targets are named groups. It shows that
  identity-term confounding (RQ4) is a known problem in benchmark design.
* **Vidgen, B., Derczynski, L. (2020).** Directions in abusive language
  training data, a systematic review: Garbage in, garbage out. *PLOS ONE*
  15(12). DOI 10.1371/journal.pone.0243300.
  *Relation:* audits how abuse datasets are built and documented. It
  supports our dataset limitations section and the text-free release policy.
* **Sap, M., Gabriel, S., Qin, L., Jurafsky, D., Smith, N. A., Choi, Y.
  (2020).** Social Bias Frames: Reasoning about Social and Power
  Implications of Language. *ACL 2020*. DOI 10.18653/v1/2020.acl-main.486.
  *Relation:* annotates implied statements about groups. It shows what
  word-level attribution cannot capture (C3 item 6).
* **Mathew, B., Saha, P., Yimam, S. M., Biemann, C., Goyal, P.,
  Mukherjee, A. (2021).** HateXplain: A Benchmark Dataset for Explainable
  Hate Speech Detection. *AAAI 2021*, 35(17), 14867-14875.
  DOI 10.1609/aaai.v35i17.17745. arXiv:2012.10289.
  *Relation:* the closest prior work on explainability. It provides human
  rationales and target-community bias metrics, and it reports that accurate
  models can score poorly on explanation metrics. We have no human
  rationales. Our RQ2 compares model attributions with each other and with
  lexicons, so HateXplain is the model for the hand-annotated top words C3
  proposes.
* **Vidgen, B., Thrush, T., Waseem, Z., Kiela, D. (2021).** Learning from
  the Worst: Dynamically Generated Datasets to Improve Online Hate Detection.
  *ACL 2021*. DOI 10.18653/v1/2021.acl-long.132.
  *Relation:* the source of `facebook/roberta-hate-speech-dynabench-r4-target`,
  the `B_dynabench` member of ensemble 11 (RQ3 lineage). Cited as on the
  Hugging Face model card.
* **Hartvigsen, T., Gabriel, S., Palangi, H., Sap, M., Ray, D., Kamar, E.
  (2022).** ToxiGen: A Large-Scale Machine-Generated Dataset for Adversarial
  and Implicit Hate Speech Detection. *ACL 2022*.
  DOI 10.18653/v1/2022.acl-long.234. arXiv:2203.09509.
  *Relation:* its benign statements that mention minority groups are the
  same population as RQ4's identity-term non-extremist posts. It is a
  candidate external test set for future work.

### Pretrained encoders used in RQ1 and RQ3

* **Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., et al.
  (2019).** RoBERTa: A Robustly Optimized BERT Pretraining Approach.
  arXiv preprint arXiv:1907.11692.
  *Relation:* the base architecture of every transformer member. It is the
  plain-RoBERTa, no-domain-pretraining control in RQ3.
* **Barbieri, F., Camacho-Collados, J., Espinosa-Anke, L., Neves, L.
  (2020).** TweetEval: Unified Benchmark and Comparative Evaluation for Tweet
  Classification. *Findings of EMNLP 2020*.
  DOI 10.18653/v1/2020.findings-emnlp.148.
  *Relation:* introduced the Twitter-RoBERTa lineage that technique 07
  fine-tunes (RQ1, RQ3).
* **Caselli, T., Basile, V., Mitrović, J., Granitzer, M. (2021).** HateBERT:
  Retraining BERT for Abusive Language Detection in English. *WOAH 2021*.
  ACL Anthology 2021.woah-1.3. arXiv:2010.12472.
  *Relation:* `GroNLP/hateBERT`, the `C_hatebert` member of 11. It is RQ3's
  test of whether abuse-domain pretraining changes word reliance.
* **Antypas, D., Camacho-Collados, J. (2023).** Robust Hate Speech Detection
  in Social Media: A Cross-Dataset Empirical Evaluation. *WOAH 2023*,
  pp. 231-242. DOI 10.18653/v1/2023.woah-1.25.
  *Relation:* the paper the `cardiffnlp/twitter-roberta-{base,large}-hate-latest`
  model cards ask users to cite (members of 11). Its cross-dataset framing
  supports reading RQ1 accuracies as specific to this split.

### Statistical comparison of classifiers (RQ1)

* **McNemar, Q. (1947).** Note on the Sampling Error of the Difference
  Between Correlated Proportions or Percentages. *Psychometrika* 12(2).
  DOI 10.1007/BF02295996.
  *Relation:* the paired test behind `mcnemar_pairs.py` and the "about 14
  rows" floor at n = 450.
* **Holm, S. (1979).** A Simple Sequentially Rejective Multiple Test
  Procedure. *Scandinavian Journal of Statistics* 6(2), 65-70. JSTOR 4615733.
  *Relation:* the family-wise correction for RQ1 (7 of 28 pairs) and for the
  32 WP7 rows.
* **Dietterich, T. G. (1998).** Approximate Statistical Tests for Comparing
  Supervised Classification Learning Algorithms. *Neural Computation* 10(7).
  DOI 10.1162/089976698300017197.
  *Relation:* recommends McNemar's test for a single held-out split. This
  is why RQ1 uses it, and why "better" requires a significant gap.

---

## (b) SHAP and attribution for text classifiers, and their failure modes

### Foundations

* **Shapley, L. S. (1953).** A Value for n-Person Games. In *Contributions
  to the Theory of Games II* (Annals of Mathematics Studies 28).
  DOI 10.1515/9781400881970-018.
  *Relation:* the axiomatic basis of every attribution in RQ2 to RQ4.
* **Owen, G. (1977).** Values of Games with a Priori Unions. *Lecture Notes in Economics and
  Mathematical Systems*, pp. 76-88. DOI 10.1007/978-3-642-45494-3_7.
  *Relation:* SHAP's Partition explainer, which `nk.explain` uses over the
  word-level text masker, computes Owen values over the masker's word
  hierarchy. So our attributions are Owen values, not exact Shapley values.
* **Ribeiro, M. T., Singh, S., Guestrin, C. (2016).** "Why Should I Trust
  You?": Explaining the Predictions of Any Classifier (LIME). *KDD 2016*.
  DOI 10.1145/2939672.2939778. arXiv:1602.04938.
  *Relation:* the main model-agnostic alternative. We keep one attribution
  method on purpose (CLAUDE.md), and LIME is named as the method not used.
* **Lundberg, S. M., Lee, S.-I. (2017).** A Unified Approach to Interpreting
  Model Predictions (SHAP). *NeurIPS 2017* (Advances in Neural Information
  Processing Systems 30). arXiv:1705.07874.
  *Relation:* the method behind every attribution artifact. Its
  linear-model case (SHAP equals coefficient times centred feature) is what
  `validate_shap.py` checks before any cross-model claim.
* **SHAP library documentation, `shap.PartitionExplainer` and
  `shap.maskers.Text`.** https://shap.readthedocs.io (checked 2026-11-04).
  *Relation:* software reference for the Partition explainer and for its
  documented property that Owen values equal Shapley values for linear
  models. Cite the exact `shap` version from `requirements-lock.txt`.
* **Sundararajan, M., Taly, A., Yan, Q. (2017).** Axiomatic Attribution for
  Deep Networks (Integrated Gradients). *ICML 2017*, PMLR 70.
  arXiv:1703.01365.
  *Relation:* a gradient-based alternative available only for the
  transformers. Using it would break comparability across model families,
  which is why the paper does not.
* **Kokalj, E., Škrlj, B., Lavrač, N., Pollak, S., Robnik-Šikonja, M.
  (2021).** BERT meets Shapley: Extending SHAP Explanations to
  Transformer-based Classifiers. *EACL Hackashop on News Media Content
  Analysis and Automated Report Generation*, pp. 16-21. ACL Anthology
  2021.hackashop-1.3.
  *Relation:* adapts SHAP to sub-word BERT inputs. It justifies our
  word-level masker, which puts classical and transformer attributions on
  the same vocabulary (RQ2, RQ3).
* **Mosca, E., Szigeti, F., Tragianni, S., Gallagher, D., Groh, G. (2022).**
  SHAP-Based Explanation Methods: A Review for NLP Interpretability.
  *COLING 2022*, pp. 4593-4603. ACL Anthology 2022.coling-1.406.
  *Relation:* reviews 41 SHAP variants for NLP. It is the reference for our
  choice of masker and explainer.
* **Madsen, A., Reddy, S., Chandar, S. (2022).** Post-hoc Interpretability
  for Neural NLP: A Survey. *ACM Computing Surveys* 55(8), Article 155.
  DOI 10.1145/3546577. arXiv:2108.04840.
  *Relation:* the general survey that places word-level SHAP among post-hoc
  methods for the related-work section.

### Known failure modes

* **Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., Kim, B.
  (2018).** Sanity Checks for Saliency Maps. *NeurIPS 2018*.
  arXiv:1810.03292.
  *Relation:* shows that attributions can look plausible yet not depend on
  the model. `validate_shap.py`, which recovers known LR coefficients, is our
  sanity check of this kind.
* **Kindermans, P.-J., Hooker, S., Adebayo, J., Alber, M., Schütt, K. T.,
  Dähne, S., et al. (2019).** The (Un)reliability of Saliency Methods. In
  *Explainable AI: Interpreting, Explaining and Visualizing Deep Learning*,
  LNCS, pp. 267-280. DOI 10.1007/978-3-030-28954-6_14.
  *Relation:* attributions can change under input shifts that do not change
  the model's function. This is a caution on the transformer instability in
  RQ3 (test/validation top-50 overlap of 0.18 to 0.22).
* **Jain, S., Wallace, B. C. (2019).** Attention is not Explanation.
  *NAACL 2019*. DOI 10.18653/v1/N19-1357.
  *Relation:* the reason we do not use attention weights as a second
  attribution for the transformers.
* **Wiegreffe, S., Pinter, Y. (2019).** Attention is not not Explanation.
  *EMNLP-IJCNLP 2019*. DOI 10.18653/v1/D19-1002.
  *Relation:* the reply to Jain and Wallace. Cited together with it so the
  attention debate is represented on both sides.
* **Slack, D., Hilgard, S., Jia, E., Singh, S., Lakkaraju, H. (2020).**
  Fooling LIME and SHAP: Adversarial Attacks on Post hoc Explanation Methods.
  *AIES 2020*. DOI 10.1145/3375627.3375830.
  *Relation:* perturbation explainers query the model off the data
  distribution, and a model can be built to hide its bias there. Masked texts
  in our SHAP runs are off-distribution in the same way. This is a threat to
  validity for RQ4 attribution claims, and WP7's intervention addresses it.
* **Kumar, I. E., Venkatasubramanian, S., Scheidegger, C., Friedler, S.
  (2020).** Problems with Shapley-value-based explanations as feature
  importance measures. *ICML 2020*, PMLR 119. arXiv:2002.11097.
  *Relation:* argues that Shapley values do not answer "what should change
  to change the outcome". This is why RQ4 adds a counterfactual (WP7)
  instead of reading SHAP alone as causal.
* **Sundararajan, M., Najmi, A. (2020).** The Many Shapley Values for Model
  Explanation. *ICML 2020*, PMLR 119. arXiv:1908.08474.
  *Relation:* different baselines and value functions give different
  "Shapley values". We fix one masker and one explainer for every technique
  so that RQ2 compares models, not choices of value function.
* **Janzing, D., Minorics, L., Blöbaum, P. (2020).** Feature relevance
  quantification in explainable AI: A causal problem. *AISTATS 2020*, PMLR
  108, 2907-2916. arXiv:1910.13413.
  *Relation:* argues for interventional rather than conditional removal of
  features. Our masker deletes words (an interventional choice), which the
  methods section should state.
* **Krishna, S., Han, T., Gu, A., Wu, S., Jabbari, S., Lakkaraju, H.
  (2024).** The Disagreement Problem in Explainable Machine Learning: A
  Practitioner's Perspective. *Transactions on Machine Learning Research*
  (2024). OpenReview jESY2WTZCe. arXiv:2202.01602.
  *Relation:* documents disagreement between explanation methods on one
  model. RQ2 studies disagreement between models under one method, and the
  between-family agreement of 0.2 to 0.3 should be read next to this
  baseline.

---

## (c) Faithfulness evaluation

* **Hooker, S., Erhan, D., Kindermans, P.-J., Kim, B. (2019).** A Benchmark
  for Interpretability Methods in Deep Neural Networks (ROAR). *NeurIPS 2019*
  (Advances in Neural Information Processing Systems 32), 9737-9748.
  arXiv:1806.10758.
  *Relation:* remove-and-retrain is the strict form of deletion-based
  faithfulness. WP7 removes identity terms without retraining, which is the
  cheaper, non-retrained variant, and the limitations section should say so.
* **DeYoung, J., Jain, S., Rajani, N. F., Lehman, E., Xiong, C., Socher, R.,
  et al. (2020).** ERASER: A Benchmark to Evaluate Rationalized NLP Models.
  *ACL 2020*. ACL Anthology 2020.acl-main.408. arXiv:1911.03429.
  *Relation:* defines comprehensiveness and sufficiency (the probability
  change when the top-k tokens are removed or kept alone). These can be
  computed from our restored predictors and attribution runs, and are the
  natural faithfulness metric for RQ2's top words. WP7 is already a
  comprehensiveness test restricted to identity terms.
* **Jacovi, A., Goldberg, Y. (2020).** Towards Faithfully Interpretable NLP
  Systems: How Should We Define and Evaluate Faithfulness? *ACL 2020*.
  ACL Anthology 2020.acl-main.386. arXiv:2004.03685.
  *Relation:* separates faithfulness from plausibility. Our lexicon
  categories measure plausibility-style alignment, and WP7 measures
  faithfulness. The paper should not let the first stand in for the second.
* **Atanasova, P., Simonsen, J. G., Lioma, C., Augenstein, I. (2020).** A
  Diagnostic Study of Explainability Techniques for Text Classification.
  *EMNLP 2020*. ACL Anthology 2020.emnlp-main.263. arXiv:2009.13295.
  *Relation:* compares explainers across architectures on agreement with
  humans, faithfulness and consistency. It is the closest methodological
  precedent for a cross-architecture attribution comparison (RQ2, RQ3). It
  holds the task fixed and varies the explainer, while we hold the explainer
  fixed and vary the model.
* **Ding, S., Koehn, P. (2021).** Evaluating Saliency Methods for Neural
  Language Models. *NAACL-HLT 2021*. DOI 10.18653/v1/2021.naacl-main.399.
  *Relation:* evaluates saliency on plausibility and faithfulness, including
  stability. It supports reporting explanation stability as a metric (RQ3,
  C3 item 4).
* **Kaushik, D., Hovy, E., Lipton, Z. C. (2020).** Learning the Difference
  that Makes a Difference with Counterfactually-Augmented Data. *ICLR 2020*.
  arXiv:1909.12434.
  *Relation:* minimal human edits that flip the label. It is the
  data-augmentation route C3 proposes for the identity shortcut, and a
  reference point for WP7's minimal edits.
* **Ribeiro, M. T., Wu, T., Guestrin, C., Singh, S. (2020).** Beyond
  Accuracy: Behavioral Testing of NLP Models with CheckList. *ACL 2020*.
  DOI 10.18653/v1/2020.acl-main.442.
  *Relation:* invariance tests under perturbation. WP7's substitution panel
  is an invariance test in this sense, applied to identity terms.

---

## (d) Identity-term bias and its mitigations

### Measurement

* **Dixon, L., Li, J., Sorensen, J., Thain, N., Vasserman, L. (2018).**
  Measuring and Mitigating Unintended Bias in Text Classification. *AIES
  2018*. DOI 10.1145/3278721.3278729.
  *Relation:* the founding paper for RQ4. It shows that toxicity classifiers
  over-flag benign text that names identity groups, and it introduces
  templated identity-term tests and data rebalancing. RQ4's per-technique
  FPR ratio (1.9 to 4.2) is the same measurement on extremism labels.
* **Park, J. H., Shin, J., Fung, P. (2018).** Reducing Gender Bias in
  Abusive Language Detection. *EMNLP 2018*. DOI 10.18653/v1/D18-1302.
  *Relation:* gender-swap augmentation and debiased embeddings for abuse
  classifiers. It is an early identity-term mitigation and a precedent for
  the substitution arm of WP7.
* **Borkan, D., Dixon, L., Sorensen, J., Thain, N., Vasserman, L. (2019).**
  Nuanced Metrics for Measuring Unintended Bias with Real Data for Text
  Classification. *Companion Proceedings of WWW 2019*.
  DOI 10.1145/3308560.3317593. arXiv:1903.04561.
  *Relation:* subgroup AUC and BPSN/BNSP AUCs. They are threshold-free
  complements to RQ4's FPR at the locked threshold and can be computed from
  the committed probability artifacts.
* **Garg, S., Perot, V., Limtiaco, N., Taly, A., Chi, E. H., Beutel, A.
  (2019).** Counterfactual Fairness in Text Classification through
  Robustness. *AIES 2019*. DOI 10.1145/3306618.3317950. arXiv:1809.10610.
  *Relation:* defines counterfactual token fairness (swap the identity
  token, measure the change). WP7 is this test, with removal as the primary
  edit and a balanced substitution panel as the sensitivity check.
* **Prabhakaran, V., Hutchinson, B., Mitchell, M. (2019).** Perturbation
  Sensitivity Analysis to Detect Unintended Model Biases. *EMNLP-IJCNLP
  2019*. DOI 10.18653/v1/D19-1578.
  *Relation:* scores models by how much predictions move when a named
  entity is perturbed. This is the same design as WP7's probability-shift
  statistic, applied to identity terms.
* **Sap, M., Card, D., Gabriel, S., Choi, Y., Smith, N. A. (2019).** The
  Risk of Racial Bias in Hate Speech Detection. *ACL 2019*.
  DOI 10.18653/v1/P19-1163.
  *Relation:* dialect-correlated false positives. They are a confounder our
  identity lexicon does not capture, and the RQ4 limitation should say so.
* **Davidson, T., Bhattacharya, D., Weber, I. (2019).** Racial Bias in Hate
  Speech and Abusive Language Detection Datasets. *Third Workshop on Abusive
  Language Online*. DOI 10.18653/v1/W19-3504.
  *Relation:* shows that the bias comes from the datasets, not one model. It
  fits RQ4's finding that every technique, classical or transformer, has an
  elevated identity-term FPR.
* **Kennedy, B., Jin, X., Mostafazadeh Davani, A., Dehghani, M., Ren, X.
  (2020).** Contextualizing Hate Speech Classifiers with Post-hoc
  Explanation. *ACL 2020*. ACL Anthology 2020.acl-main.483.
  arXiv:2005.02439.
  *Relation:* the closest prior use of post-hoc attribution to diagnose
  identity-term over-reliance in hate classifiers, followed by explanation
  regularisation. It is the main precedent for linking RQ2 identity share to
  RQ4 FPR. Our difference is the cross-family comparison and the
  intervention (WP7).
* **Röttger, P., Vidgen, B., Nguyen, D., Waseem, Z., Margetts, H.,
  Pierrehumbert, J. (2021).** HateCheck: Functional Tests for Hate Speech
  Detection Models. *ACL-IJCNLP 2021*. DOI 10.18653/v1/2021.acl-long.4.
  *Relation:* functional tests that include non-hateful identity mentions,
  counter-speech and quotation. They target C3 item 6 (context) directly,
  and a HateCheck-style curated set fits `curated_examples.py`.

### Mitigation by training objective

* **Zhang, B. H., Lemoine, B., Mitchell, M. (2018).** Mitigating Unwanted
  Biases with Adversarial Learning. *AIES 2018*.
  DOI 10.1145/3278721.3278779. arXiv:1801.07593.
  *Relation:* adversarial removal of a protected attribute from predictions.
  It is a mitigation for the identity shortcut in C3 item 3. Out of scope
  here, because this paper audits and does not retrain.
* **Sagawa, S., Koh, P. W., Hashimoto, T. B., Liang, P. (2020).**
  Distributionally Robust Neural Networks for Group Shifts: On the
  Importance of Regularization for Worst-Case Generalization (Group DRO).
  *ICLR 2020*. arXiv:1911.08731.
  *Relation:* worst-group training with known groups. With "label x
  identity term present" as groups, it is the direct mitigation of RQ4's FPR
  gap (future work).
* **Liu, E. Z., Haghgoo, B., Chen, A. S., Raghunathan, A., Koh, P. W.,
  Sagawa, S., et al. (2021).** Just Train Twice: Improving Group Robustness
  without Training Group Information (JTT). *ICML 2021*, PMLR 139.
  arXiv:2107.09044.
  *Relation:* upweights the errors of a first model, without group labels.
  This matters here because our identity lexicon is incomplete (C3 item 5).
* **Attanasio, G., Nozza, D., Hovy, D., Baralis, E. (2022).** Entropy-based
  Attention Regularization Frees Unintended Bias Mitigation from Lists.
  *Findings of ACL 2022*. DOI 10.18653/v1/2022.findings-acl.88.
  *Relation:* mitigation that does not need an identity-term list. It
  answers the same lexicon-coverage limitation (C3 item 5) from the
  mitigation side.

### Mitigation by known-bias ensembling (product of experts)

* **Clark, C., Yatskar, M., Zettlemoyer, L. (2019).** Don't Take the Easy
  Way Out: Ensemble Based Methods for Avoiding Known Dataset Biases.
  *EMNLP-IJCNLP 2019*. DOI 10.18653/v1/D19-1418.
  *Relation:* trains the main model with a bias-only model, as a product of
  experts. An identity-term-only classifier is the obvious bias model for
  this task. Its relevance to RQ3 is that ensembling can be used to remove a
  shortcut, not only to add accuracy as ensemble 11 does.
* **He, H., Zha, S., Wang, H. (2019).** Unlearn Dataset Bias in Natural
  Language Inference by Fitting the Residual (DRiFt). *DeepLo Workshop
  (EMNLP-IJCNLP 2019)*. DOI 10.18653/v1/D19-6115.
  *Relation:* the same product-of-experts idea, fitting what the bias model
  cannot explain. Cited together with Clark et al. as the second
  known-bias mitigation.

### Mitigation by representation erasure

* **Ravfogel, S., Elazar, Y., Gonen, H., Twiton, M., Goldberg, Y. (2020).**
  Null It Out: Guarding Protected Attributes by Iterative Nullspace
  Projection (INLP). *ACL 2020*. ACL Anthology 2020.acl-main.647.
  arXiv:2004.07667.
  *Relation:* removes a linearly decodable attribute from representations.
  It is the identity-direction ablation WP9 proposes for the transformer
  members.
* **Belrose, N., Schneider-Joseph, D., Ravfogel, S., Cotterell, R.,
  Raff, E., Biderman, S. (2023).** LEACE: Perfect linear concept erasure in
  closed form. *NeurIPS 2023*. arXiv:2306.03819.
  *Relation:* closed-form linear erasure with a guarantee that no linear
  probe can recover the concept. It is the preferred erasure operator for
  WP9 over iterative INLP.

---

## (e) Probing and representation-level interpretability (WP9)

* **Alain, G., Bengio, Y. (2017).** Understanding intermediate layers using
  linear classifier probes. *ICLR 2017* (listed by MLAnthology. Whether it
  was the workshop or main track was not confirmed). arXiv:1610.01644.
  *Relation:* the original layer-wise linear probe. It is the template for
  WP9's per-layer probes for "extremist" and "identity term present".
* **Conneau, A., Kruszewski, G., Lample, G., Barrault, L., Baroni, M.
  (2018).** What you can cram into a single $&!#* vector: Probing sentence
  embeddings for linguistic properties. *ACL 2018*.
  DOI 10.18653/v1/P18-1198.
  *Relation:* standard probing-task methodology for sentence
  representations, which applies to the pooled representations of 07 and 11.
* **Kim, B., Wattenberg, M., Gilmer, J., Cai, C., Wexler, J., Viegas, F.,
  et al. (2018).** Interpretability Beyond Feature Attribution: Quantitative
  Testing with Concept Activation Vectors (TCAV). *ICML 2018*, PMLR 80.
  arXiv:1711.11279.
  *Relation:* tests a concept direction's influence on the output. An
  "identity term" concept vector would ask the RQ4 question at the
  representation level.
* **Tenney, I., Das, D., Pavlick, E. (2019).** BERT Rediscovers the
  Classical NLP Pipeline. *ACL 2019*. DOI 10.18653/v1/P19-1452.
  *Relation:* locates information by layer. This is WP9's sharper RQ3
  question: at which layer identity information becomes predictive of the
  label.
* **Hewitt, J., Liang, P. (2019).** Designing and Interpreting Probes with
  Control Tasks. *EMNLP-IJCNLP 2019*. DOI 10.18653/v1/D19-1275.
  *Relation:* probe accuracy must be compared against a control task, or
  the probe's own capacity is measured. WP9 needs selectivity, not raw probe
  accuracy.
* **Voita, E., Titov, I. (2020).** Information-Theoretic Probing with Minimum
  Description Length. *EMNLP 2020*. ACL Anthology 2020.emnlp-main.14.
  arXiv:2003.12298.
  *Relation:* an MDL alternative to control tasks. Useful at our small
  sample sizes, where probe accuracy is noisy.
* **Rogers, A., Kovaleva, O., Rumshisky, A. (2020).** A Primer in BERTology:
  What We Know About How BERT Works. *TACL* 8. DOI 10.1162/tacl_a_00349.
  *Relation:* background survey for the WP9 section.
* **Elazar, Y., Ravfogel, S., Jacovi, A., Goldberg, Y. (2021).** Amnesic
  Probing: Behavioral Explanation with Amnesic Counterfactuals. *TACL* 9.
  DOI 10.1162/tacl_a_00359. arXiv:2006.00995.
  *Relation:* erases a property (with INLP) and measures the change in
  behaviour, not decodability. It is the representation-level counterpart
  of WP7's input-level removal.
* **Belinkov, Y. (2022).** Probing Classifiers: Promises, Shortcomings, and
  Advances. *Computational Linguistics* 48(1). DOI 10.1162/coli_a_00422.
  arXiv:2102.12452.
  *Relation:* the reference for what probes can and cannot show. Decodable
  does not mean used, which is why WP9 pairs probes with erasure.

---

## (f) LLM zero-shot classification and self-explanation faithfulness (WP10)

### Zero-shot and few-shot classification

* **Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P.,
  et al. (2020).** Language Models are Few-Shot Learners. *NeurIPS 2020*.
  arXiv:2005.14165.
  *Relation:* the basis for in-context classification. WP10's zero-shot
  baseline follows it.
* **Chiu, K.-L., Collins, A., Alexander, R. (2021).** Detecting Hate Speech
  with GPT-3. arXiv preprint arXiv:2103.12407. (No peer-reviewed venue
  found.)
  *Relation:* an early zero-shot and few-shot hate classification
  experiment. It is the precedent for WP10's accuracy-against-07 McNemar
  comparison.
* **Plaza-del-Arco, F. M., Nozza, D., Hovy, D. (2023).** Respectful or Toxic?
  Using Zero-Shot Learning with Language Models to Detect Hate Speech.
  *WOAH 2023*. DOI 10.18653/v1/2023.woah-1.6.
  *Relation:* prompt and verbaliser sensitivity in zero-shot hate detection.
  WP10 must fix its prompt and label tokens before touching test, in the
  same way thresholds are locked on validation.
* **Roy, S., Harshvardhan, A., Mukherjee, A., Saha, P. (2023).** Probing
  LLMs for hate speech detection: strengths and vulnerabilities. *Findings
  of EMNLP 2023*. DOI 10.18653/v1/2023.findings-emnlp.407.
  *Relation:* reports LLM sensitivity to target-group information in
  prompts, which is RQ4's identity question for WP10.
* **Ziems, C., Held, W., Shaikh, O., Chen, J., Zhang, Z., Yang, D. (2024).**
  Can Large Language Models Transform Computational Social Science?
  *Computational Linguistics* 50(1). DOI 10.1162/coli_a_00502.
  arXiv:2305.03514.
  *Relation:* a broad zero-shot benchmark on social-science labelling tasks,
  including hate and ideology. It gives the expected range for WP10's
  accuracy.

### Self-explanation faithfulness

* **Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., et al.
  (2022).** Chain-of-Thought Prompting Elicits Reasoning in Large Language
  Models. *NeurIPS 2022*. arXiv:2201.11903.
  *Relation:* the elicitation method whose faithfulness the next entries
  question. Cited to define what WP10 means by a self-explanation.
* **Turpin, M., Michael, J., Perez, E., Bowman, S. R. (2023).** Language
  Models Don't Always Say What They Think: Unfaithful Explanations in
  Chain-of-Thought Prompting. *NeurIPS 2023*. arXiv:2305.04388.
  *Relation:* stated reasons can omit the feature that actually drove the
  answer. WP10 therefore checks the LLM's named words against its own SHAP
  attribution, not against plausibility.
* **Lanham, T., Chen, A., Radhakrishnan, A., Steiner, B., Denison, C.,
  Hernandez, D., et al. (2023).** Measuring Faithfulness in Chain-of-Thought
  Reasoning. arXiv preprint arXiv:2307.13702.
  *Relation:* intervention-based faithfulness tests on the explanation
  (truncate, corrupt). This is the WP10 analogue of WP7's input
  intervention.
* **Huang, S., Mamidanna, S., Jangam, S., Zhou, Y., Gilpin, L. H. (2023).**
  Can Large Language Models Explain Themselves? A Study of LLM-Generated
  Self-Explanations. arXiv preprint arXiv:2310.11207.
  *Relation:* compares LLM self-explanations with occlusion and LIME
  attributions on sentiment. This is the WP10 design (self-explanation
  against an attribution method) on another task.
* **Madsen, A., Chandar, S., Reddy, S. (2024).** Are self-explanations from
  Large Language Models faithful? *Findings of ACL 2024*, pp. 295-337.
  DOI 10.18653/v1/2024.findings-acl.19. arXiv:2401.07927.
  *Relation:* self-consistency checks. If the LLM names words as important,
  removing them should change its answer. WP10 can run this with the WP7
  removal machinery and no model internals.
* **Parcalabescu, L., Frank, A. (2024).** On Measuring Faithfulness or
  Self-consistency of Natural Language Explanations. *ACL 2024*.
  DOI 10.18653/v1/2024.acl-long.329. arXiv:2311.07466.
  *Relation:* argues that many "faithfulness" tests measure only
  self-consistency. WP10's prose should use the narrower term unless
  internals are inspected.
* **Agarwal, C., Tanneru, S. H., Lakkaraju, H. (2024).** Faithfulness vs.
  Plausibility: On the (Un)Reliability of Explanations from Large Language
  Models. arXiv preprint arXiv:2402.04614. (No peer-reviewed venue found.)
  *Relation:* a position paper on the plausibility/faithfulness gap in LLM
  explanations. It is the framing for WP10's agreement statistic.

---

## Items not verified

* **The project dataset and competition.** Kaggle dataset
  `adityasureshgithub/digital-extremism-detection-curated-dataset` and the
  Kaggle competition "Social Media Extremism Detection Challenge" (URLs in
  `README.md`). **UNVERIFIED** as scholarly sources. No peer-reviewed paper
  describing how the posts were collected or labelled was found or checked.
  Before submission the authors must confirm the original collection and
  annotation source, and cite it if one exists.

Every other entry above matched a primary record (arXiv, Crossref, ACL
Anthology, AAAI OJS, PMLR, NeurIPS proceedings, JSTOR or OpenReview).

