# Data card: Social Media Extremism Detection Dataset

## Dataset overview

The dataset supports binary classification of social-media text into two labels:

- `EXTREMIST`
- `NON_EXTREMIST`

The research goal is to study violent-extremism detection as a distinct NLP task rather than treating it as identical to general toxicity, hate speech, or sentiment classification.

## Files

| File | Description |
|---|---|
| `data/dataset.csv` | Source dataset used by the repository notebooks. |
| `data/extremism_lexicon.txt` | Supporting lexicon file used for inspection or exploratory analysis, not a replacement for model evaluation. |
| `splits/split_assignments.csv` | Fixed row-level train/validation/test split assignments (a mirror of the frozen assignment — see below). |
| `splits/split_assignments.PRE-REPAIR.csv` | The stale pre-2026-07-29 mirror, kept for provenance. Not for use. |
| `results_summary/foundation/dataset_manifest.json` | Dataset manifest produced by the foundation notebook. |
| `results_summary/foundation/label_distribution.csv` | Label counts and proportions. |
| `results_summary/foundation/split_label_distribution.csv` | Label counts by split. |
| `results_summary/foundation/text_length_summary.csv` | Text-length summary statistics. |
| `results_summary/foundation/duplicate_text_report.csv` | Duplicate text-hash report. |
| `results_summary/foundation/rows_removed_summary.json` | Summary of removed rows during preprocessing. |

## Current dataset summary

| Item | Value |
|---|---:|
| Raw rows | 3000 |
| Processed rows | 2999 |
| Rows removed | 1 |
| Duplicate text hashes | 0 |
| Non-extremist rows | 1870 |
| Extremist rows | 1129 |
| Non-extremist proportion | 62.35% |
| Extremist proportion | 37.65% |
| Random seed | 30 |
| Split version | `split_v1_stratified_70_15_15_seed30` |

## Fixed split summary

| Split | Rows | Non-extremist | Extremist |
|---|---:|---:|---:|
| Train | 2099 | 1309 | 790 |
| Validation | 450 | 281 | 169 |
| Test | 450 | 280 | 170 |

## Split-mirror integrity

`splits/split_assignments.csv` is a *mirror* of the frozen split assignment, and
it has been wrong. Until 2026-07-29 the committed file was a stale
pre-correction artifact that disagreed with the frozen assignment on **728
labels and 1420 split assignments**. Results produced against it could not match
the reported ones, and nothing in the file itself announced the problem.

The mirror was repaired on 2026-07-29 with `tools/repair_split_mirror.py`, which
refuses to write unless its reconstruction reproduces
`results_summary/foundation/` exactly, and it currently verifies clean. Confirm
that before relying on it:

```bash
python3 tools/protocol_check.py --all   # first invariant: split mirror
```

Repairing the mirror is not the same as regenerating the split. The assignment
itself is frozen at `split_v1_stratified_70_15_15_seed30` and must never be
regenerated; every committed result depends on it.

## A note on committed dataset text

Notebooks `00`–`08` and `10` carry saved cell outputs containing dataset text
and row ids. `tools/scan_text_leakage.py` reports this, and it is a known
cleanup backlog rather than an intended release of the text in that form.
Anyone redistributing this repository should be aware the text is present in
those notebooks, not only in `data/`.

## Label interpretation

The labels are intended to capture whether a social-media text expresses or supports extremist content under the project definition. This task is inherently sensitive and context-dependent.

Important interpretation notes:

- The label is not a statement about the identity, belief system, or intent of an author.
- The label applies to the text instance under the project guidelines.
- Borderline examples may require context that is unavailable in a standalone text record.
- Models may learn lexical shortcuts rather than robust understanding of ideology, violence, or intent.

## Intended research uses

Appropriate uses include:

- Reproducing the baseline experiments in this repository.
- Comparing text-classification approaches under the fixed split protocol.
- Studying model errors, false positives, and false negatives.
- Investigating explainability methods for high-stakes NLP classification.
- Discussing dataset limitations and the difficulty of violent-extremism detection.

## Uses that are not supported

This dataset should not be used as-is for:

- Automated moderation decisions.
- Account bans, user-level risk scoring, law-enforcement triage, or disciplinary processes.
- Claims about an individual's beliefs, affiliations, or future behavior.
- Production deployment without domain-expert review, fairness analysis, privacy review, and human oversight.

## Known limitations

- The dataset is small for a high-stakes NLP task.
- Human labeling is subjective and may vary by reviewer background, context, and interpretation.
- The binary label collapses a complex phenomenon into two categories.
- Social-media language, coded references, and extremist discourse evolve over time.
- Some examples may lack enough context for a fully reliable judgment.
- Reported metrics on fixed held-out splits should not be interpreted as deployment readiness.

## Versioning guidance

Dataset changes should create a new dataset version and, if necessary, a new split version.

Examples:

- Minor metadata/documentation correction: keep the dataset version unchanged.
- Added or removed rows: create a new dataset version.
- Relabeled examples: create a new dataset version.
- New train/validation/test assignments: create a new split version.
- New preprocessing that changes model input text: document as a new experiment condition or dataset version, depending on scope.
