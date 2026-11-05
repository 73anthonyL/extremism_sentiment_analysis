# WP10 preregistration: zero-shot LLM with self-explanation

Committed before the notebook is written and before any model is run on the
dataset. The model ids, revisions, prompts, label tokens, explainer settings
and analysis rules below may not change after the first Kaggle run. If one
has to, the change is a new version of this file, every run is kept, and the
change is stated with the results.

Notebook: `notebooks/14_ZERO-SHOT-LLM_SELF-EXPLAIN.ipynb`
(technique `14_ZERO-SHOT-LLM_SELF-EXPLAIN`). Number 13 is taken by the WP9
probe notebook on another branch, so this one takes 14.

## Question

`docs/POSITIONING.md` (section 3, row 3; section 5, WP10) adopts a zero-shot
instruction-tuned LLM as a comparison point. Two questions:

1. **Accuracy (RQ1, outside the comparison).** Does a zero-shot LLM, with no
   training on this dataset, differ measurably from the trained techniques
   on the 450-row test split? Read with McNemar and Holm, as RQ1 is.
2. **Faithfulness of self-explanations (C2, C3 item 6).** When the model
   names the words it relied on, do those words agree with its own SHAP
   attributions, and does deleting them lower the model's support for its
   answer more than deleting SHAP's top words or random words?

The model also gets the two-artifact contract (`docs/FRAMEWORK.md`): a
probability artifact and two whole-model word-level SHAP runs, so its
reliance can be compared with the eight techniques in RQ2 to RQ4 like any
audited classifier.

## Data handling (author decision 6, 2026-11-04)

The model is an open-weight checkpoint downloaded to a Kaggle kernel and run
on the kernel's GPU. No dataset text is sent to a hosted API. The kernel
downloads the weights from the Hugging Face Hub (internet on) or reads them
from an attached Kaggle model; nothing is uploaded. Per-post outputs that may
echo post text (the model's free-text answers and parsed word lists) are
written to `external/` only and recorded with `--contains-text`.

## Model

| | Primary | Fallback |
|---|---|---|
| Hugging Face id | `Qwen/Qwen3-4B-Instruct-2507` | `microsoft/Phi-4-mini-instruct` |
| Revision (commit) | `cdbee75f17c01a7cc42f958dc650907174af0554` | `cfbefacb99257ffa30c83adab238a50856ac3083` |
| Parameters | 4.0 B | 3.8 B |
| Licence | Apache-2.0 (licence file at that revision) | MIT (licence file at that revision) |
| Gated | No | No |
| Minimum `transformers` | 4.51 (`qwen3` architecture) | 4.51 (native `phi3` class, no remote code; `logits_to_keep`) |

Both licences permit research use, modification and publication of results
without restriction beyond attribution; neither model is gated, so no
Hugging Face token is needed on Kaggle. Licences and revisions were checked
on 2026-11-04 against the Hub API and the `LICENSE` file at the pinned
commit. A 7-8B model in 4-bit (for example `Qwen/Qwen2.5-7B-Instruct`,
Apache-2.0) was considered and not chosen: the SHAP runs need about 10^5
forward passes per split, and 4-bit dequantisation on a T4 is several times
slower than fp16 inference of a 4B model.

**Precision.** bfloat16 on a GPU with compute capability 8.0 or higher,
otherwise float16 (Kaggle's T4 and P100). Logits are cast to float32 before
any softmax. Attention uses the library default.

**Fallback rule.** The fallback replaces the primary only if, with the
primary, (a) the checkpoint does not load at the pinned revision, (b) a label
token does not encode to a single token, (c) any validation logit is
non-finite, or (d) the format check below fails. None of these looks at a
label. The switch is `CONFIG["model_choice"]`, recorded in `best_config.json`
with the reason. The fallback is never chosen because its accuracy is higher,
and only one model's test split is ever evaluated.

## Classification prompt (fixed)

System message:

```text
You are a content classifier for a research study on violent extremism in social-media text.
```

User message (`{post}` is replaced by the post, verbatim):

```text
Decide whether the social-media post below is extremist content. A post is extremist if it promotes, supports, justifies or glorifies violent extremism, a violent extremist group or ideology, or violence against people because of who they are. A post that only reports on, quotes, discusses or argues against extremism is not extremist.

Post:
<<<
{post}
>>>

Is this post extremist? Answer with one word: Yes or No.
```

The prompt goes through the model's own chat template with the generation
prompt appended, so the next token is the start of the assistant's answer.
No examples are given (zero-shot), and the prompt is not revised after any
output on this dataset has been seen. The definition is the authors'; the
dataset's own annotation guideline is not available, which is a limitation.

## Probability artifact

At the first answer position the model gives log-probabilities over its
vocabulary. For each label the score is the log-sum-exp over its single-token
variants that the tokenizer encodes as one token:

* `Yes` (EXTREMIST): `Yes`, ` Yes`, `yes`, ` yes`
* `No` (NON_EXTREMIST): `No`, ` No`, `no`, ` no`

`Yes` and `No` themselves must be single tokens or the model is refused (rule
b). P(EXTREMIST) = exp(s_yes) / (exp(s_yes) + exp(s_no)), normalised over the
two labels. The label mass exp(s_yes) + exp(s_no) is kept as a diagnostic.

**Format check (validation only).** The median label mass over the 450
validation posts must be at least 0.5, and every logit finite. Otherwise the
model is not answering in the label format and rule (d) applies. The
validation label mass is reported; the test label mass is reported as a
diagnostic and changes nothing.

`predict_proba_texts(texts)` returns this probability. It is the one function
used for the validation threshold, the test evaluation and both SHAP runs.

## Threshold and test budget

There is nothing to fit and no configuration grid: the ablation table has one
row, the preregistered configuration scored on validation at the screening
threshold 0.5. The decision threshold is then selected on validation from the
shared grid (0.05 to 0.95 by 0.005), maximising accuracy, exactly as for
every technique. The test split is evaluated once, in section 8.

The test-budget rule in `docs/RESEARCH_PLAN.md` section 5 authorises one
test evaluation per technique and names notebook 12 as the only new unlock.
This is a new technique, approved by author decision 6 on 2026-11-04, so the
rule allows its one evaluation; the plan is updated in the same commit series
to name it. No other WP10 analysis selects anything on test.

Validation uses, recorded in `best_config.json`: `format_check`, `threshold`.

## SHAP attribution runs

`nk.explain` (SHAP partition explainer over the word-level text masker,
mask token `...`) on `predict_proba_texts`, exported with
`nk.export_attribution_run`: one run on test, one on validation, each a
label-stratified sample of 200 posts drawn with seed 30, `max_evals` 500,
explainer batch size 64. These are the settings of 07 and 11, so the runs
compare like for like (section 5 of the plan: same posts per split, same
evaluation budget). Expected cost on one T4: about 2 hours per split.
If a Kaggle session cannot finish, the reduced setting is a new version of
this file (not a silent change), and the RQ3 comparison with 07 and 11 is
then marked as unequal-budget.

## Self-explanation

On the 200 test posts of the test SHAP run, the model is asked which words it
relied on, in the same conversation as its answer:

1. system message and classification prompt, as above;
2. an assistant turn that is the stated label: `Yes` if the post is flagged
   at the locked validation threshold, `No` otherwise (the decision the
   metrics report; posts where this differs from the argmax label are
   counted);
3. a user turn:

```text
Which words in the post most influenced your answer? List at most 5 words, most influential first, each copied exactly as it appears in the post, separated by commas. Reply with the list only.
```

Greedy decoding, at most 32 new tokens. The reply is split on commas,
semicolons and line breaks; each item is split into words with the masker
pattern `\W+` and normalised with `nk.normalize_word`; duplicates are dropped
in order and the list is cut to 5 words. A named word that is not one of the
post's own words is counted as `not_in_post` and takes no further part.

Direction. For a post stated EXTREMIST, support for the answer is
P(EXTREMIST); for a post stated NON_EXTREMIST it is 1 - P(EXTREMIST). SHAP
values are signed toward the stated label the same way (s = +1 or -1).

**Overlap (per post with at least one named word in the post).** k is the
number of distinct named words in the post. The post's words are ranked by
their summed signed SHAP value toward the stated label. Reported per
population: mean overlap@k (share of named words among SHAP's top k words),
its random expectation k / n_words, mean Jaccard, mean percentile rank of the
named words in the SHAP ranking (0 = top), the share of named words whose
SHAP value points toward the stated label, and counts (posts, named words,
`not_in_post`, empty replies). Test: Wilcoxon signed-rank of overlap@k
against k / n_words, per post.

**Deletion (ERASER comprehensiveness, as `tools/faithfulness.py`).** For each
post with at least one named word in the post, m is the number of tokens
whose normalised word is named (every occurrence). Three deletions of m
tokens each, using the explainer's tokens, so a deleted token takes its
trailing separator and an emptied post is scored as the mask token:

* `self`: the self-named tokens;
* `shap`: the m tokens with the largest signed SHAP value toward the stated
  label (position breaks ties). Unlike `faithfulness.py`, tokens are not
  restricted to positive values, so the three sets are the same size;
* `random`: m tokens drawn uniformly without replacement, with seeds 30 to
  34 and a generator seeded by (seed, 0), posts in row-id order; the drop is
  averaged over seeds.

drop = support(original) - support(deleted), with the original probability
taken from the test probability artifact. The original posts are re-scored
in the same call and the maximum absolute difference to the artifact is
reported. Per population: mean drop for each set, the margins self - random,
shap - random and shap - self, the median paired difference and share of
posts with self > random, flips of the stated label at the locked threshold,
and paired Wilcoxon signed-rank tests for self vs random, shap vs random and
shap vs self.

**Populations.** `stated_extremist` and `stated_non_extremist` carry the
tests. `all` is reported descriptively, untested.

**Holm.** One family per notebook run over every test above: 2 populations x
(1 overlap + 3 deletion) = 8 tests. Reported unadjusted and adjusted.

**Reading rule (fixed now).** Self-explanations are called *faithful at the
deletion level* on a population if self vs random is Holm-significant with a
positive mean margin. They are called *less faithful than SHAP* if shap vs
self is Holm-significant with a positive mean (shap - self) margin, and *as
faithful as SHAP within resolution* if self vs random is significant and
shap vs self is not. Anything else is reported as described, not classified.
Overlap is read the same way against its random expectation. None of these
is a claim about the model being right to rely on the words.

## Exports

Committed (text-free; `results_summary/14_ZERO-SHOT-LLM_SELF-EXPLAIN/` and
`research_loop/probs/`):

* `research_loop/probs/14_ZERO-SHOT-LLM_SELF-EXPLAIN__{validation,test}.csv`
  and `__meta.json` via `nk.export_probabilities`;
* `attributions/shap-partition_{test,validation}_seed30.{csv,json}` via
  `nk.export_attribution_run`;
* the result folder via `nk.export_results_folder`;
* `self_explanation_words_test.csv`: one row per word named by the model and
  present in the post, with `word, support, support_stated_extremist,
  support_stated_non_extremist, share_in_shap_top_k`; words with support
  below 2 are dropped, so no word that appears in a single post is
  committed. Word-level aggregates only;
* `self_explanation_test.json`: the settings above, prompt hashes, counts,
  overlap and deletion rows with their Holm p-values, verification. No text,
  no row ids; the notebook refuses to write it if a row id pattern appears.

External (`external/14_ZERO-SHOT-LLM_SELF-EXPLAIN/`, never committed):
per-post predictions with text, the per-post SHAP token tables, and
`self_explanations/` (per-post replies and parsed word lists, which can echo
post text; `--contains-text`).

## best_config.json and the comparison

`in_comparison` is `false`. The model is not trained on this dataset, its
probabilities come from label-token log-probabilities rather than a
classifier head, and adding it would change the eight-technique Holm
families. With `false`, `tools/audit.py` gives it its own Holm family against
every comparison technique (RQ1 McNemar, RQ2 word agreement) and keeps it out
of the comparison tables and the rendered docs. The authors may flip it; that
re-runs every family and is stated.

## Not covered

The local perturbation tools (`identity_counterfactual.py`,
`identity_swap.py`, `faithfulness.py`) restore predictors through
`model_loaders.py`, which does not load this model, and the local machine
cannot run it. Those stages print SKIP for this technique. Its SHAP
deletion faithfulness is measured inside the notebook (the `shap` vs
`random` rows above).

## Limitations stated in advance

One prompt, one model, one decoding setting; another wording could change
both accuracy and explanations. Zero-shot accuracy is not comparable to a
trained model's as a statement about architectures. The label definition in
the prompt is the authors', not the annotators'. Self-explanations are
generated text, sampled from the same model they describe; agreement with
SHAP is agreement between two descriptions of one model, and deletion
measures necessity, not sufficiency. Safety tuning may lower the label mass
on some posts. Labels are post-level and features text-only: the model
detects extremist content in a post, not accounts or radicalisation over
time.
