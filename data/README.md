# Data folder

This folder contains the source data files used by the repository notebooks.

## Expected files

| File | Purpose |
|---|---|
| `dataset.csv` | Main dataset used for binary extremist/non-extremist text classification. |
| `extremism_lexicon.txt` | Supporting lexicon file for inspection or exploratory analysis. |

## Handling rules

- Do not overwrite `dataset.csv` casually.
- Any dataset change that adds rows, removes rows, changes labels, or changes source text should be treated as a new dataset version.
- If the dataset changes, re-run `notebooks/00_create_dataset_and_splits.ipynb` and update the foundation artifacts.
- Keep the fixed split assignments in `splits/split_assignments.csv` aligned with the dataset version.

## Notes

The dataset is used for research replication and controlled model comparison. It should not be used as a production moderation dataset without substantial external validation, domain review, bias analysis, and human oversight.
