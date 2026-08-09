# Documentation index

This folder contains supporting documentation for the Social Media Extremism Detection research repository.

The repository is intended for review, replication, and research transparency. It is not structured as a general community-maintained software project.

## Documents

| File | Purpose |
|---|---|
| `DATA_CARD.md` | Documents the dataset, label definitions, fixed splits, intended use, and known limitations. |
| `EXPERIMENTS.md` | Defines the controlled experiment protocol used to compare models. |
| `MODEL_CARD.md` | Summarizes the model task, current baselines, evaluation metrics, risks, and non-deployment status. |
| `RESEARCH_LOOP.md` | Describes how a candidate moves from hypothesis to registered result, and how test unlocks are rationed and recorded. |
| `RESPONSIBLE_USE.md` | States how the dataset and models should and should not be used. |
| `RESULTS_SCHEMA.md` | Defines expected output files and metric fields for each experiment folder. |
| `REPLICATION_GUIDE.md` | Gives a step-by-step workflow for reproducing the dataset foundation and baseline experiments. |
| `COMPETITION.md` | Explains how the Kaggle competition relates to this controlled research repository. |
| `RELEASE_CHECKLIST.md` | Provides a pre-release checklist before public result updates or manuscript-aligned releases. |

Two files outside this folder are part of the same documentation set:

| File | Purpose |
|---|---|
| `research_loop/STATE.md` | The live status of the research loop: current cycle, stage, and blockers. Read it before starting work. |
| `CHANGELOG.md` | Human-readable project history of research, data, modeling, and reproducibility milestones. |

## Documentation philosophy

Good research repositories should make it easy for another reader to answer five questions:

1. What data was used?
2. What exact split and protocol were used?
3. What model produced each result?
4. What limitations or risks affect interpretation?
5. Which claims are actually supported by the evidence, and which are not?

These files exist to answer those questions without making the repository look like a production library or an open contribution hub.

The fifth question is the one this project takes most seriously, because a
450-row test split cannot support most of the claims a metrics table invites.
Documentation here is expected to say plainly when a result is unregistered,
when a comparison is inconclusive, and when a number in prose has no artifact
behind it.
