# Lexicons

Word lists used by `tools/categorize_attributions.py` (RQ2) and
`tools/identity_fpr.py` (RQ4). One entry per line, `#` starts a comment,
matching is case-insensitive at word boundaries.

| File | Category | Notes |
|---|---|---|
| `extremist_framing.txt` | `extremist_framing` | Movement names, slogans, codes, and violent-framing vocabulary. Seeded from the single-word entries of `data/extremism_lexicon.txt`; curate freely. |
| `identity_terms.txt` | `identity_term` | Neutral references to religions, ethnicities, nationalities, gender, sexuality, and migration status. These are the terms RQ4 measures false positives against. |
| `slurs.txt` | `slur` | Slurs against ethnic, racial and national groups and against gay people, selected from HurtLex (below). Licensed CC BY-NC-SA 4.0, unlike the rest of the repository. |

A word matched by more than one list takes the first category in the order
`slur`, `extremist_framing`, `identity_term`; everything unmatched is
`topical`. The categorizer records every matched category alongside the chosen
one, so overlaps are visible rather than hidden.

## Slur list: source, licence and selection

**Source.** HurtLex (Bassignana, Basile and Patti 2018), English lexicon
version 1.2, `lexica/EN/1.2/hurtlex_EN.tsv` from
<https://github.com/valeriobasile/hurtlex>, read at repository commit
`d4d5cf1199c09868486f978fcea58af0e8936a1e` (2025-10-10; the EN 1.2 file
itself was last changed in commit `10ae72a4`, 2020-01-27). The source file's
sha256 is `a734820a63c87994781d182692e6dc7ec262c402016971a7fa31946ced0d470c`.

> Elisa Bassignana, Valerio Basile, Viviana Patti. 2018. Hurtlex: A
> Multilingual Lexicon of Words to Hurt. In *Proceedings of the Fifth Italian
> Conference on Computational Linguistics (CLiC-it 2018)*, CEUR Workshop
> Proceedings Vol-2253, paper 49. <http://ceur-ws.org/Vol-2253/paper49.pdf>

**Licence.** HurtLex is released under CC BY-NC-SA 4.0
(<https://creativecommons.org/licenses/by-nc-sa/4.0/>), which permits copying
and redistributing the material and adaptations of it, with attribution, for
non-commercial purposes, provided adaptations carry the same licence.
`slurs.txt` is such an adaptation. It is therefore distributed under CC
BY-NC-SA 4.0, separately from the rest of this repository, and its header
repeats the source, licence and changes. The other lexicons are not derived
from HurtLex.

**Selection.** HurtLex has 17 categories and two levels. We use:

* **Level `conservative`.** HurtLex builds it by translating only the
  offensive senses of the words in its source lexicon; `inclusive` adds every
  potentially relevant sense, which brings in many neutral words.
* **Categories `ps` and `om`.** `ps` ("negative stereotypes, ethnic slurs")
  and `om` ("words related to homosexuality") are the two categories whose
  entries target a group defined by ethnicity, race, nationality or sexual
  orientation, the dimensions the identity lexicon covers. The others are
  excluded: `rci` (locations and demonyms) holds neutral group references,
  which belong to the identity list if anywhere; `pa` (professions), `an`
  (animals), `or` (plants), `asm`/`asf` (genitalia), `pr` (prostitution),
  `re` (crime), `svp` (the seven deadly sins), `dmc` (moral defects), `cds`
  (derogatory words), `qas` (negative connotations) and `is` (social and
  economic disadvantage) are insults or offensive vocabulary not aimed at an
  identity group; `ddf` and `ddp` (physical and cognitive disability) are
  mostly generic insults about appearance and intelligence, and disability
  is not a dimension of the identity lexicon, so a disability-slur category
  would have no RQ4 counterpart.

`ps` and `om` at the conservative level give 191 distinct lemmas. HurtLex
assigns categories by translation and sense, so they also contain words that
are not slurs against a group. 133 are removed, in five classes, and the
exact words of each class are listed in the header of `slurs.txt` so the
derivation can be repeated:

| Removed because | Count |
|---|---|
| already a neutral entry of `identity_terms.txt` | 2 |
| a neutral or descriptive group reference, not a slur (for example `homosexual`, `mestizo`); these forms were not added to `identity_terms.txt` either: they occur in no non-extremist test post, so they could not change RQ4 | 20 |
| a generic insult about intelligence, character, class or conduct, not aimed at a group (for example `idiot`, `barbarian`, `peasant`), including cognitive-disability insults | 84 |
| a word whose everyday sense dominates (a name, object, animal or other meaning), so a word-level match would mostly not be a slur | 12 |
| not a usable lemma (an organisation name, a phrase about slurs, a crime term, an unidentified string) | 15 |

58 entries remain: 42 from `ps` (slurs for Black people, for people of mixed
descent, for Germans, for Travellers, for East Asian people, and one Russian
ethnic slur in transliteration and Cyrillic) and 16 from `om` (slurs for gay
men). One `ps` family is also used as a disability slur; it stays, since
HurtLex lists it as an ethnic slur. The list is not a complete slur
inventory: the conservative English HurtLex has no slurs for several groups
in the identity lexicon (for example religious groups), and we did not
supplement it from other sources, so that every entry has one documented
origin. A post that uses a slur outside this list is counted under its other
categories.

**Neutral references stay identity terms.** Where a HurtLex slur category
contains a word that is also a neutral identity reference (`gay`, `queer`),
the word is removed from the slur list and stays in `identity_terms.txt`.
WP3 as first written did the opposite (remove the overlap from the identity
list, since `slur` takes precedence); that would have dropped neutral
mentions of gay and queer people from RQ4, which measures false positives on
exactly such mentions. After this step no word is in both lists.

**Overlaps with the other lists.** No slur entry is an entry of
`extremist_framing.txt` or `identity_terms.txt`, and no entry of either
contains a slur entry at word boundaries (or the reverse), so the precedence
order above never has to choose between `slur` and another category.

## Matching

Multi-word entries can match post text (RQ4) but not a word-level attribution
artifact (RQ2). The categorizer prints how many it skipped.

Changing a lexicon changes RQ2 and RQ4 results. Re-run both tools and the table
renderer after any edit, and describe the edit in the commit.

`identity_swap_pairs.csv` is not a lexicon and neither RQ2 nor RQ4 reads it.
It is the preregistered swap map of WP7b (`tools/identity_swap.py`,
`docs/preregistrations/WP7b_identity_swap.md`): one row per identity-term
entry, either three same-category, same-form targets or `excluded` with a
reason. Every word in it is an entry of `identity_terms.txt`.

## Version note: frozen at WP3 (2026-11-04; identity list revised 2026-11-05)

The three lexicons are frozen as of this note (`docs/RESEARCH_PLAN.md`, WP3
and section 5 "Lexicons"). Any later change re-runs RQ2 and RQ4 in full and
keeps both versions of the affected tables.

| File | Entries | sha256 |
|---|---|---|
| `slurs.txt` | 58 | `8164361489ce79690ff6a7910d716ff4f00f8aa8a170f228b37af839f981618d` |
| `extremist_framing.txt` | 382 | `377179826ce2220dde8874b8707724028eacd44ad33eb9696c4b5a132318c749` |
| `identity_terms.txt` | 96 | `4c6eb3cf088d3e47aec7141605b61c25d6ea31bbf559bbfd1db98ffd47a4b77c` |

Entries are counted after `tools/lexicons.py` normalisation (comments and
blank lines dropped, lower-cased, de-duplicated). Changes at the freeze:

* `slurs.txt`: populated from HurtLex (above); it was empty before.
* `identity_terms.txt`: unchanged from before the freeze (96 entries). At
  the freeze on 2026-11-04 `aliens` was added (97 entries, sha256
  `df98c56d…`), as the only gap word in
  `rq2_identity_lexicon_gaps_sensitivity.csv` with test support of at least
  3. It was dropped again on 2026-11-05 (author decision 2,
  `docs/DECISIONS_NEEDED.md`): it came from 06's own test top-50 gap list,
  so the addition was derived from test-split attributions; it always
  appears in the loaded phrase "illegal aliens", not as a neutral
  reference; and it alone moved 06's RQ4 Fisher p below 0.05. Both runs are
  in git history. The 13 ambiguous entries stay, and the
  `exclude_ambiguous_only` sensitivity keeps being reported.
* `extremist_framing.txt`: reviewed, unchanged. It shares no entry with the
  other two lists; four of its entries contain an identity entry inside a
  hyphenated framing word (for example `anti-muslim`), and these stay
  framing.

`identity_swap_pairs.csv` (not a lexicon) is byte-identical to the WP7b
preregistration (sha256
`1e94dd70e48e511d0530592a95044f4e15bb74235853ec04c2356cb32dab406f`); the
excluded `aliens` row of Amendment 1 was removed by Amendment 2 of
`docs/preregistrations/WP7b_identity_swap.md`.

## Rule for adding terms

A term may be added to any list only on grounds that do not come from the
test split. In particular, no term is added because it appears in a
test-split attribution artifact, a test-split gap list or test-split
false positives: that would fit the lexicon to the results it is used to
measure. Candidates may come from the training split, from a published
source, or from linguistic review; the source is recorded with the
addition, and the change follows the freeze rule above.
