# WP9 preregistration: layer-wise linear probes on the transformers

Committed before notebook 13 was written and before any probe was fitted.
Nothing below may change after the first Kaggle run. If something has to,
the change is a new version of this file, both runs are kept, and the change
is stated with the results.

## Question

SHAP sees only inputs and outputs. WP9 looks inside the fine-tuned
transformers of the comparison and asks three things, in support of RQ3 and
RQ4:

1. At which layer does the extremist label become linearly decodable from a
   post's representation, and how much of that is already there before
   fine-tuning on this dataset (pretraining) versus added by it (RQ3)?
2. Is "an identity term is present" linearly decodable at every layer, and
   does fine-tuning change that (RQ3, RQ4)?
3. Does the linear direction that encodes identity-term presence in the final
   layer carry label-relevant signal: does removing it change p(extremist)
   on test posts that mention a community (RQ4)?

This is an analysis of frozen models. It trains no classifier of posts that
enters RQ1, selects no threshold, model or configuration, and evaluates each
probe on the test split once. It does not count against the test budget of
any technique: the fine-tuned models re-score test posts only to verify that
the restored weights reproduce the committed probabilities and for the
direction ablation, as WP7 does.

## Models

* **Fine-tuned (primary).** `07_TWITTER-ROBERTA_FINE-TUNE`, and every run of
  the four components admitted to `11_MULTI-CHECKPOINT_LOGIT-POOL`'s primary
  pool: `A_anchor`, `B_dynabench`, `C_hatebert`, `E_twitter_large`, each
  with seeds 17, 30 and 73 (12 runs). `D_deberta` (not admitted) and
  `F_anchor_fgm` (not in the primary pool) are excluded. Weights are the
  ones the notebooks saved under `external/`, attached as Kaggle datasets.
  Each is served as its notebook served it: its text normalisation (07
  `identity`, 11 `minimal_twitter`), truncation at the `max_length` in its
  committed `best_config.json` (192), float16 autocast on the GPU, and, for
  11's runs, the inference precision recorded in `run_summary.csv`.
* **Pretrained bases (control).** The Hugging Face checkpoints each
  fine-tuned model started from, loaded as encoders without a
  classification head and never trained here:
  `cardiffnlp/twitter-roberta-base-hate-latest` (twice: with 07's
  normalisation, as 07's control, and with 11's, as `A_anchor`'s),
  `facebook/roberta-hate-speech-dynabench-r4-target`, `GroNLP/hateBERT`, and
  `cardiffnlp/twitter-roberta-large-hate-latest`. "Pretrained" means "before
  fine-tuning on this dataset": three of these were themselves trained on
  hate-speech data, which is part of their lineage. The resolved Hub commit
  of each is recorded in the run sidecar.

The notebook refuses to continue unless each fine-tuned model reproduces its
committed probability artifact: 07 directly, 11 by pooling its 12 runs
exactly as notebook 11 does. Maximum absolute difference at most 0.01 (the
tolerance of `tools/curated_examples.py`), checked on validation before any
probe is fitted and again on test in the evaluation section.

## Representation

For every hidden-state output of every model (the embedding output as layer
0, then each transformer layer: 13 for base models, 25 for the large one),
the **mean over the post's attended tokens**, special tokens included, cast
to float32. This is the only pooling; the first-token (CLS) state is not
probed, because at the final layer it is the classification head's own
input. The same inputs the model is served with are used.

## Probe tasks

Each post gets three binary targets.

* `extremist_label`: the dataset label.
* `identity_term_present`: at least one match of
  `data/lexicons/identity_terms.txt` in the post text, with RQ4's matching
  rule (`tools/lexicons.py::phrase_pattern`: case-insensitive, non-word
  boundaries, multi-word entries kept, every entry including the ambiguous
  ones, as in RQ4's primary analysis). The kit carries a copy of that rule,
  pinned to the tool by a test on every dataset post.
* `control_word_type`: a control task after Hewitt and Liang (2019), adapted
  from tokens to posts. Each lower-cased word type (the kit's word split on
  `\W+`) gets a pseudo-random bit from SHA-256 of `"30|<word>"`. A post's
  score is the mean bit over its words (0 for a post with no word). The
  control label is `score > c`, where `c` is the smallest observed train
  score whose train positive share does not exceed the train extremist share
  (790 / 2099). The control is therefore a function of word identity only,
  learnable by memorising word types, with the label's base rate on train.
  The achieved base rate per split is recorded.

## Probes

One probe per model run x layer x task: standardisation fitted on the train
split, then L2 logistic regression (lbfgs, up to 5000 iterations, no class
weights). The regularisation `C` is chosen from {0.001, 0.01, 0.1, 1} by
5-fold stratified cross-validated ROC-AUC **within the train split**
(shuffled with seed 30), ties to the smaller `C`, then refitted on the whole
train split. The scaler is fitted once on the whole train split before the
folds; the folds never see validation or test. Validation is scored and
reported as a held-out check and is used for nothing.

## Metrics

Per model run x layer x task x split (validation, test):

* ROC-AUC (primary) and accuracy at probability 0.5 (secondary), with the
  majority-class accuracy of that split next to it.
* 95% percentile bootstrap intervals, 1000 resamples of the split's posts,
  seed 30. The same resample indices are used for every model, layer and
  task on a split, so intervals of paired differences are paired.
* **Selectivity** (Hewitt and Liang): AUC of the task minus AUC of
  `control_word_type` on the same run and layer, with a paired bootstrap
  interval. Reported for `extremist_label` and `identity_term_present`.
* **Fine-tuning contrast** (RQ3): fine-tuned run minus its base at the same
  layer, task and split, with a paired bootstrap interval of the AUC
  difference. Pairs: 07 - twitter-roberta-base (07's normalisation);
  `A_anchor` runs - twitter-roberta-base (11's normalisation); `B_dynabench`
  runs - dynabench; `C_hatebert` runs - hateBERT; `E_twitter_large` runs -
  twitter-roberta-large.

No hypothesis test is run on probe metrics and no family correction is
applied to them: they are descriptive, with intervals.

## Identity-direction ablation (included)

For each fine-tuned run, the identity probe at the final layer gives a
direction in the raw hidden space: its weight vector divided by the scaler's
standard deviations, normalised to unit length. During a forward pass, a
hook on the last encoder layer removes that component from every token's
state (h - (h . u) u); the classification head is unchanged. Population: the
test posts with `identity_term_present`, split by label
(`non_extremist`, `extremist`).

* Rows: 07 at its locked threshold; 11 pooled (the direction of each run
  removed in that run, then pooled as notebook 11 pools) at 11's locked
  threshold; each of the 12 runs alone at 0.5, descriptive only.
* Directions: `identity_probe`, and `random`, the mean probability over 10
  random unit directions (seed 30) removed the same way. The 95th percentile
  of the 10 random directions' mean deltas is reported with every row.
* Per row: n, mean p before and after, mean and median delta
  (before - after), flagged before and after, flips both ways, Wilcoxon
  signed-rank p (before vs after) and exact McNemar p on the flips. Holm over
  the four `identity_probe` rows of 07 and 11 pooled (2 models x 2
  populations); every other row is descriptive.

## Reading rules (fixed before the run)

* **Where the label is decodable.** Descriptive: per run, the test AUC curve
  over layers and the layer of its maximum. No claim that one model "encodes
  more" than another unless the paired interval of the difference excludes
  zero, and never "better".
* **Fine-tuning versus pretraining (RQ3).** For a pair at a layer,
  fine-tuning *added decodable label information* if the 95% paired interval
  of the test AUC difference lies above zero; *removed* if below; otherwise
  no detectable change. For 11's components the statement is made for seed
  30 and accompanied by how many of the three seeds agree. Per-layer
  statements are descriptive (no family correction); the paper reports the
  layer ranges, not single layers.
* **Identity.** `identity_term_present` is lexical, so high AUC is expected
  everywhere. It is *decodable beyond the control* at a layer if the
  selectivity interval lies above zero. Whether fine-tuning changes it is
  read from the contrast at the final layer, by the same rule as above.
* **Ablation.** For 07 and for 11 pooled, on each population: *the identity
  direction carries label-relevant signal* if the Holm-adjusted Wilcoxon
  p < 0.05 and the mean delta exceeds the 95th percentile of the random
  directions' mean deltas. *No evidence* if Holm p >= 0.05. Anything else is
  reported as partial and described. A false-positive change is never
  reported without the extremist-post change on the same edit, flips are
  descriptive unless they survive Holm, and none of this is read as "the
  model treats identity terms as evidence of extremism": WP7 and WP7b keep
  the cautious reading (decision 5), and a linear direction is not the only
  way a network can carry information.

## Outputs

Text-free tables written by the notebook through the kit
(`nk.export_probe_tables`), and checked locally by
`tools/layer_probes_report.py --check`:

* `results_summary/rq/rq3_layer_probes.csv`: model run x layer x task x
  split.
* `results_summary/rq/rq3_layer_probe_contrasts.csv`: fine-tuned minus base,
  and selectivity.
* `results_summary/rq/rq4_identity_direction_ablation.csv`: the ablation
  rows.
* `results_summary/rq/rq3_layer_probes_run.json`: configuration, models and
  their sources, Hub commits, verification results, achieved control base
  rates, run id and kit version.

Nothing per post is committed. Hidden-state caches are off by default; when
switched on they are written under `external/13_TRANSFORMER-LAYER-PROBES/`,
uploaded to Kaggle, and recorded with `tools/run_manifest.py add-asset`
against the technique whose weights produced them, with `--contains-text`.

## Limitations stated in advance

A linear probe measures what is decodable, not what the model uses; the
ablation is the only part that intervenes, and only on one linear direction
at one layer. Mean pooling is one choice of post representation. The base
checkpoints are fetched from the Hub at run time and may differ from the
revision notebook 11 started from; the recorded commits make that visible.
float16 serving differs slightly from float32. The word-type control is an
adaptation of a token-level method to posts. With 450 test posts the
intervals are wide. Labels are post-level and features text-only: this
describes representations of posts, not accounts or radicalisation over
time.
