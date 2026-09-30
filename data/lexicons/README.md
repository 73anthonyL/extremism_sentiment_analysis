# Lexicons

Word lists used by `tools/categorize_attributions.py` (RQ2) and
`tools/identity_fpr.py` (RQ4). One entry per line, `#` starts a comment,
matching is case-insensitive at word boundaries.

| File | Category | Notes |
|---|---|---|
| `extremist_framing.txt` | `extremist_framing` | Movement names, slogans, codes, and violent-framing vocabulary. Seeded from the single-word entries of `data/extremism_lexicon.txt`; curate freely. |
| `identity_terms.txt` | `identity_term` | Neutral references to religions, ethnicities, nationalities, gender, sexuality, and migration status. These are the terms RQ4 measures false positives against. |
| `slurs.txt` | `slur` | Left for the authors to populate from a published slur lexicon. Ships empty on purpose. |

A word matched by more than one list takes the first category in the order
`slur`, `extremist_framing`, `identity_term`; everything unmatched is
`topical`. The categorizer records every matched category alongside the chosen
one, so overlaps are visible rather than hidden.

Multi-word entries can match post text (RQ4) but not a word-level attribution
artifact (RQ2). The categorizer prints how many it skipped.

Changing a lexicon changes RQ2 and RQ4 results. Re-run both tools and the table
renderer after any edit, and describe the edit in the commit.
