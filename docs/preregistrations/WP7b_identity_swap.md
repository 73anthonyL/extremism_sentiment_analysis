# WP7b preregistration: identity-term swap (RQ4 sensitivity analysis)

Committed before any model was run on swapped text. The swap map is
`data/lexicons/identity_swap_pairs.csv`, committed in the same commit as this
file. Neither may change after the first run. If one has to, the change is a
new version of this file, both runs are kept, and the change is stated.

## Question

WP7 found that deleting identity terms lowers p(extremist) a lot for 07 and
11, while replacing them with a neutral group noun ("group", "population",
"humans") mostly does not. The classical techniques move under both edits.
So the transformers' removal effect may partly come from deleting a noun.

WP7b asks whether the transformers respond to **which** group is named, or
only to **a** group being named. It replaces each identity term with other
identity terms from the same lexicon category and form, and asks whether
p(extremist) changes.

This is a sensitivity analysis of RQ4, not a new research question. It
re-scores perturbed test texts with frozen models at their locked thresholds,
as WP7 does. It selects no threshold, model or configuration, and it does not
count against the test budget.

## Swap map

Every entry of `data/lexicons/identity_terms.txt` has exactly one row in the
map. A row is either swapped (three targets) or `excluded` (with a reason).

* **Categories.** A swap stays inside the lexicon section of its source:
  religion, ethnicity/race/nationality, migration status, gender and
  sexuality. The tool reads the section of every entry from the lexicon's own
  section headers and refuses a row whose source or target lies in another
  section.
* **Forms.** Inside a category, a swap keeps the grammatical form:
  `person` (singular noun or adjective), `person_plural`, `faith` (the name of
  a religion), `identity` (sexuality and gender-identity adjectives). Every
  target must itself be a swapped row with the same category and form.
* **Panel rule.** Each (category, form) has a fixed ordered panel of four
  entries. A source maps to the first three panel entries that are not of
  its own `group`. A source whose group is outside the panel therefore gets
  the first three; a panel member gets the other three. `group` merges
  entries that name the same population (muslim / islamic / muslims / islam;
  catholic and protestant with christian; somali with african; pakistani and
  chinese with asian; mexican, latino, latina and hispanic; the European
  nationalities with european). Panels:

  | Category | Form | Panel (in order) |
  |---|---|---|
  | religion | person | muslim, christian, hindu, buddhist |
  | religion | person_plural | muslims, christians, hindus, buddhists |
  | religion | faith | islam, christianity, judaism, hinduism |
  | ethnicity_nationality | person | african, asian, mexican, european |
  | ethnicity_nationality | person_plural | africans, asians, mexicans, europeans |
  | migration | person | immigrant, refugee, foreigner, migrant |
  | migration | person_plural | immigrants, refugees, foreigners, migrants |
  | gender_sexuality | identity | gay, lesbian, bisexual, transgender |

* **Excluded entries (21).** The 13 ambiguous entries of
  `AMBIGUOUS_IDENTITY_TERMS` in `tools/identity_sensitivity.py` are neither
  sources nor targets. Also excluded: `indians` (the same ambiguity as
  `indian`), `woman` and `women` (every same-form target is ambiguous),
  `feminist` and `feminists` (a stance, not a group of the same kind as any
  other entry), and `queer`, `lgbt`, `lgbtq` (umbrella terms whose same-form
  targets are their own members). Matches of excluded entries stay in the
  text unchanged under every operation.
* **Refusals.** The tool refuses the map if any source or target is not an
  identity-lexicon entry, crosses a category, changes form, repeats its own
  group, is ambiguous or excluded, or is matched by the slur or
  extremist-framing lexicon; if a row does not have exactly three distinct
  targets; or if a lexicon entry has no row.

The panel was fixed on linguistic grounds before any model was run on swapped
text. No per-term counts or model outputs were looked at to choose it.

## Population and operations

The population is the test posts with at least one match of a swapped source
(a subset of the 150 WP7 posts), split by label. Every operation edits the
same spans: the matches of swapped sources. Case follows the matched text
(all-caps stays all-caps, a capital first letter stays capital, otherwise
lower case).

* `swap` (primary). Variant k (k = 1, 2, 3) replaces every swapped match by
  target_k of its own source. p_swap is the mean probability over the three
  variants.
* `remove`. The same spans are deleted and whitespace collapsed.
* `substitute`. The same spans are replaced by each word of the WP7 neutral
  panel ("group", "population", "humans", vetted by
  `identity_counterfactual.py`), case-matched as above, and the
  probabilities averaged.

These differ from WP7's rows in two ways, both stated with the results: only
the swapped spans are edited (WP7 edits every identity match), and the
substitute word is case-matched (WP7 inserts it in lower case).

## Metrics and tests

Per technique x population (non_extremist, extremist) x operation, with
delta = p_original - p_counterfactual and p_original the committed artifact
probability:

* **Primary:** mean delta, median delta, flagged before and after at the
  locked threshold, flips both ways; Wilcoxon signed-rank on p_original
  versus p_counterfactual, and exact McNemar on the flips.
* **Holm.** Over every technique x population x operation row of
  `rq4_identity_swap.csv` for the techniques in the comparison (8 x 2 x 3 =
  48 rows), one family per test, as WP7. An outside-comparison technique is
  its own family. A secondary Holm column covers the 16 swap rows only.
* **Descriptive, untested:** mean |delta| over posts and variants (an
  averaged p can cancel a target that raises the score against one that
  lowers it), and the mean within-post spread (max minus min) across the
  three variants, for swap and substitute.
* **Directions, descriptive:** for each post and each distinct swapped
  source in it, three single-source variants that replace only that source's
  matches by each of its targets. Mean delta per technique x population x
  category x source x target, with its post count. These are not tested.
* **Target profile:** each map term's training-split post count, extremist
  share and 01_LOG-REG_TF-IDF coefficient and rank, reported, not used to
  refuse: an identity target is expected to be label-skewed, and the profile
  lets a reader see when it is.

## Reading rule (fixed before the run)

For 07 and 11, on each population:

* **Group-specific sensitivity** if the swap row has Holm-adjusted Wilcoxon
  p < 0.05 (48-row family) and mean swap delta is at least half the mean
  remove delta on the same population.
* **Group mention (or noun deletion), not group identity** if the remove row
  is Holm-significant, the swap row is not, and mean swap delta is below half
  the mean remove delta.
* Anything else is reported as partial and described, not classified.

The classical techniques are read the same way. A large mean |delta| with a
small mean delta is reported as direction-dependent sensitivity that the
averaged test cannot see, not as group-specific sensitivity in either
direction. Flip counts follow WP7's rule: descriptive unless they survive
Holm, never ranked across techniques below the 14-row gap, and a
false-positive change is never reported without the extremist-post change.

## Limitations stated in advance

The panel fixes one set of comparison groups; another panel could give a
different answer. Swapping two co-occurring groups can leave the post's set
of groups unchanged. immigrant and migrant are near-synonyms. Word-level
models score a target unseen in training like a deletion. Labels are
post-level and features text-only.

## Amendment 1 (2026-11-04): WP3 lexicon freeze

The WP3 freeze added one entry, `aliens`, to the migration section of
`data/lexicons/identity_terms.txt` (the only identity-lexicon gap word with
test support of at least 3; author decision 2 in `docs/DECISIONS_NEEDED.md`)
and populated `data/lexicons/slurs.txt` from HurtLex. Because the tool
refuses a map without a row for every identity entry, the map gains one row:

    migration,excluded,,aliens,,,,"added at WP3 freeze, after the WP7b preregistration"

`aliens` is excluded rather than swapped so that the swap population and the
panels stay as preregistered: it is neither a source nor a target, and its
matches stay in the text unchanged under every operation, like the other
excluded entries. Excluded entries are now 22. No other row changed. The map's
sha256 moves from `1e94dd70e48e511d0530592a95044f4e15bb74235853ec04c2356cb32dab406f`
to `89c99e5b3b69779c70003d2317da0cbf39957485d399118740ffbbd4a8ac47f4`.

The slur lexicon could change the population, since a source match that
overlaps a slur match is not swapped. No slur entry contains an identity
entry, and `identity_swap.py --all --dry-run-population` with the frozen
lexicons gives the preregistered population: 68 test posts (23
non-extremist, 45 extremist) for every technique. Both runs are kept: the
first, on the preregistered map (commit `f08193b1`), is the
`rq4_identity_swap.csv` and `identity_swap_test.json` files of commit
`0ca755d4`; the re-run on the amended map supersedes them in the tree.
