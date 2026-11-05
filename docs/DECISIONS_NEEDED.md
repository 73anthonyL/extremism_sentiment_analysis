# Decisions needed from the authors

Every item below blocks work that cannot proceed without a human choice.
Each has a recommendation; replying "accept all recommendations" is enough
to unblock the local items. Status as of 2026-11-04 (`research/integration`).

| # | Decision | Blocks | Options | Recommendation |
|---|---|---|---|---|
| 1 | Source for `data/lexicons/slurs.txt` (WP3) | Freezing lexicons; removing "provisional" from RQ2, RQ3, RQ4 and the RQ2-RQ4 relation | (a) a published hate-speech lexicon with a licence that allows redistribution of the word list; (b) leave empty and report the slur column as unavailable | (a), choose the source and licence yourselves; the list is committed, so the licence matters |
| 2 | Identity-list review (WP3) | Same as 1 | Add some of the 14 distinct gap words in `rq2_identity_lexicon_gaps_sensitivity.csv` (26 of 27 candidates have support <= 2); keep the 13 ambiguous entries or drop them | Add only terms with support >= 3; keep ambiguous entries and keep reporting the `exclude_ambiguous_only` sensitivity |
| 3 | Section 5 share-difference rule | Any "technique A relies more on X than B" sentence | (a) two-run SD as preregistered; (b) post-level bootstrap intervals (`rq2_bootstrap_category_shares.csv`), whose SDs are 2-10x larger | Keep (a) as the preregistered rule and report (b) alongside as a stricter check; say so in the paper |
| 4 | Run notebook 12 (plain RoBERTa) on Kaggle | RQ3 lineage comparison; RQ1 Holm family grows 28 -> 36 pairs and 01 vs 11 very likely loses significance | Run / do not run | Run it; accept the RQ1 change and report it |
| 5 | Reading of WP7 and WP7b | RQ4 prose and C3 item 3 | See `docs/RESEARCH_PLAN.md` WP7b; current draft treats substitution as the stricter control and calls 07/11's extremist-post response "group mention, not which group" | Adopt the draft's reading; do not claim the transformers treat identity terms as evidence of extremism |
| 6 | WP10 zero-shot LLM baseline | C2/C3 self-explanation comparison | Send test-post text to an external API, use a local open-weight model, or drop | Local open-weight model if it fits on Kaggle; otherwise drop to future work |
| 7 | WP9 probes | Professor's "internal neural nets" suggestion | Kaggle GPU run / future work | Future work unless time allows; it is not needed for C1-C3 |
| 8 | Report cards print top words verbatim, including a racial slur | Any report-card figure in the paper | Redact in figures only / redact in committed cards | Redact in figures only; committed cards are an audit record |
| 9 | Make support >= 2 the SHAP-validation pass rule | WP2 rule | Keep the preregistered rule / switch | Keep; report support >= 2 as a diagnostic (already done) |
| 10 | Drop `--primed` from the CLAUDE.md pre-commit list | Contributor docs | Drop / keep | Drop; notebooks now carry outputs and the flag fails |
| 11 | Upload external assets (weights, per-post files) to Kaggle Datasets and record them | Reproducing the perturbation stages (faithfulness, WP7, WP7b) | Upload / leave local | Upload before submission; files with text marked `--contains-text` |
| 12 | Push and open PRs | Everything on `research/integration` (stacked on PR #22) | One stacked PR / several | One PR from `research/integration` after PR #22 merges |

Items 1, 2, 3, 5, 9 and 10 are local once decided. Items 4, 6, 7 and 11
need Kaggle or an upload. Item 12 needs a push.
