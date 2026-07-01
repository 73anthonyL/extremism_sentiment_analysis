# Kaggle competition context

The project also includes a related Kaggle competition:

<https://www.kaggle.com/competitions/social-media-extremism-detection-challenge>

The competition was created to encourage external experimentation on the same broad binary classification task: identifying social-media text as extremist or non-extremist.

## Role of the competition

The competition is useful as:

- A public benchmark for external participants.
- A way to observe alternative model strategies.
- A signal that the task can be explored beyond the internal baseline models.
- A source of future ideas for model comparison and error analysis.

## Relationship to this repository

This repository is the controlled research and replication artifact.

The competition and this repository should not be treated as identical evaluation settings unless the exact same dataset version, split protocol, preprocessing, metric definitions, and threshold policy are used.

| Source | Purpose | Treatment in this repository |
|---|---|---|
| Kaggle dataset | Public dataset release | Source dataset and reference point |
| Kaggle competition | External benchmark and model-exploration setting | Separate context for comparison ideas |
| GitHub repository | Controlled research pipeline | Main source for reproducible notebooks, metrics, and analysis |

## Reporting guidance

When discussing competition results in a paper, README, or presentation, keep them separate from controlled baseline results unless they are rerun under the repository protocol.

Recommended phrasing:

> We hosted a Kaggle competition to encourage external experimentation on the task. Competition submissions are treated as a separate benchmark context, while the repository reports controlled baseline results using fixed train/validation/test splits.

## Future work

A future competition-analysis document may summarize:

- Common modeling strategies used by participants.
- High-level lessons from the leaderboard.
- Whether competition-inspired approaches improve under the fixed repository split.
- Which approaches are worth incorporating into the controlled experiment suite.
