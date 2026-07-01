# Changelog

This changelog summarizes the major research, data, modeling, reproducibility, and documentation milestones for the **Social Media Extremism Detection** repository.

The repository is maintained as a research and replication artifact. This changelog is intentionally written as a human-readable project history rather than a raw list of commits.

## [0.1.0] - 2026-06-30

### Added

- Established the repository as a public-facing research and replication artifact for binary classification of social-media text as `EXTREMIST` or `NON_EXTREMIST`.
- Added a professional project `README.md` describing the research motivation, repository structure, dataset/split protocol, baseline results, reproducibility workflow, limitations, and responsible-use expectations.
- Added citation metadata through `CITATION.cff`.
- Added a repository `LICENSE` file describing current visibility and usage boundaries.
- Added a clean `requirements.txt` for direct research dependencies.
- Added `requirements-lock.txt` to preserve the frozen working environment for stricter replication.
- Added documentation pages under `docs/`, including:
  - `DATA_CARD.md`
  - `EXPERIMENTS.md`
  - `MODEL_CARD.md`
  - `RESPONSIBLE_USE.md`
  - `RESULTS_SCHEMA.md`
  - `REPLICATION_GUIDE.md`
  - `COMPETITION.md`
  - `RELEASE_CHECKLIST.md`
- Added folder-level documentation for:
  - `data/`
  - `splits/`
  - `notebooks/`
  - `results_summary/`
- Added repository cleanup and formatting support through an updated `.gitignore`.

### Changed

- Reframed the repository from an exploratory project into a structured research repository focused on review, replication, and evaluation.
- Clarified the distinction between:
  - the Kaggle dataset,
  - the Kaggle competition,
  - and the controlled experimental workflow maintained in this repository.
- Documented the intended use of the repository as a research artifact, not a production moderation system or community-maintained software package.
- Clarified the role of `requirements.txt` versus `requirements-lock.txt`.

### Fixed

- Removed `.DS_Store` from version control.
- Corrected README formatting and repository-tree documentation.
- Updated documentation to better align with the current dataset, fixed splits, baseline model outputs, and responsible-use posture.

## [Research baseline consolidation] - 2026-06-27 to 2026-06-29

### Added

- Replaced the earlier dataset artifact with a corrected dataset version.
- Created a foundation data report under `results_summary/foundation/`.
- Added dataset-level artifacts, including:
  - dataset manifest,
  - label distribution summary,
  - split label distribution summary,
  - duplicate-text report,
  - text-length summary,
  - removed-row summary.
- Established the canonical split version:

  ```text
  split_v1_stratified_70_15_15_seed30
  ```

- Added corrected and standardized model notebooks for:
  - `LOG-REG_TF-IDF`
  - `LIN-SVM_TF-IDF`
  - `SLP_TF-IDF`
- Added standardized result artifacts for the completed baseline model families:
  - `ablation_results.csv`
  - `best_config.json`
  - `classification_report_test.json`
  - `confusion_matrix_test.csv`
  - `metrics_validation.json`
  - `metrics_test.json`

### Dataset snapshot

The corrected dataset foundation contains:

| Item | Value |
|---|---:|
| Raw rows | 3000 |
| Processed rows | 2999 |
| Rows removed | 1 |
| Duplicate text hashes | 0 |
| Non-extremist rows | 1870 |
| Extremist rows | 1129 |
| Random seed | 30 |

The fixed split assignments contain:

| Split | Rows | Non-extremist | Extremist |
|---|---:|---:|---:|
| Train | 2099 | 1309 | 790 |
| Validation | 450 | 281 | 169 |
| Test | 450 | 280 | 170 |

### Current baseline results

The following held-out test results were added or corrected during this consolidation phase. The positive class is `EXTREMIST`.

| Technique | Accuracy | Positive F1 | Positive precision | Positive recall | ROC-AUC | PR-AUC | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| `LOG-REG_TF-IDF` | 0.8533 | 0.8024 | 0.8171 | 0.7882 | 0.9111 | 0.8881 | 0.45 |
| `LIN-SVM_TF-IDF` | 0.8556 | 0.7962 | 0.8523 | 0.7471 | 0.9037 | 0.8825 | 0.47 |
| `SLP_TF-IDF` | 0.8378 | 0.7768 | 0.8089 | 0.7471 | 0.9022 | 0.8777 | 0.50 |

### Changed

- Updated and corrected result summaries for `LIN-SVM_TF-IDF` and `SLP_TF-IDF`.
- Removed older summary files that no longer matched the corrected dataset and split protocol.
- Renamed and normalized notebook filenames for consistency.
- Reorganized results so model outputs could be reviewed without rerunning every notebook.

### Fixed

- Corrected file naming and file-path errors in the model notebook workflow.
- Corrected summary artifacts after the dataset update.
- Ensured final reported model results use the corrected dataset and controlled split workflow.

## [Reproducible pipeline foundation] - 2026-06-20

### Added

- Added the first version of the reproducible dataset creation and split-assignment notebook:
  - `00_create_dataset_and_splits.ipynb`
- Added the first controlled baseline notebooks:
  - `01_LOG-REG_TF-IDF.ipynb`
  - `02_LIN-SVM_TF-IDF.ipynb`
  - `03_SLP_TF-IDF.ipynb`
- Added split assignments for train, validation, and test evaluation.
- Added initial summarized result files for:
  - Logistic Regression with TF-IDF features,
  - Linear SVM with TF-IDF features,
  - Single-Layer Perceptron with TF-IDF features.

### Changed

- Shifted the repository away from earlier exploratory scripts toward a numbered notebook workflow.
- Organized experiments around a consistent dataset/split foundation.
- Began separating model-family outputs into dedicated `results_summary/<TECHNIQUE>/` folders.

### Removed

- Removed older tools, utilities, and study folders that were no longer needed after the repository reorganization.
- Removed unnecessary legacy files to reduce confusion and improve reproducibility.

## [Experiment planning and SLP research notebook] - 2026-05 to 2026-06

### Added

- Added a more formal SLP-focused research notebook for extremism detection.
- Added an initial notebook version focused on:
  - single-layer perceptron modeling,
  - ablation setup,
  - experimental organization,
  - and preparation for comparison across techniques.
- Added model experiment folders for planned technique families, including:
  - `HATEBERT-FEATS_LOG-REG`
  - `LIN-SVM_TF-IDF`
  - `LOG-REG_TF-IDF`
  - `LOG-REG_WORD-CHAR-TF-IDF`
  - `RAND-FOREST_TF-IDF`
  - `SENT-EMB_LOG-REG`
  - `SENT-EMB_XGBOOST`
  - `SLP_TF-IDF`
  - `XGBOOST_TF-IDF`
  - `XGBOOST_WORD-CHAR-TF-IDF`

### Changed

- Reorganized notebooks and archived older notebook structures.
- Updated the final extremism dataset file used by the project.
- Began aligning the project around a broader comparison study rather than a single-model experiment.

### Removed

- Removed outdated files from earlier exploratory stages.

## [Model preservation and scripting] - 2026-02

### Added

- Added scripts and model-related files used to preserve intermediate model information.
- Saved model artifacts and supporting outputs from earlier modeling work.

### Notes

- This period appears to represent an intermediate preservation stage between the late-2025 exploratory modeling work and the later 2026 repository overhaul.

## [Notebook professionalization, SHAP, and error analysis] - 2025-12

### Added

- Implemented SHAP-oriented analysis in the research workflow.
- Added more evaluation metrics to the notebook workflow.
- Added support for retrieving potential mislabels and model-error examples.
- Generated a `potential_mislabels_by_model.csv` artifact for reviewing cases where model predictions and labels may require further inspection.
- Added Kaggle-related notebook and requirement support.
- Added top-contestant submission material from the Kaggle competition for reference.

### Changed

- Performed a major notebook overhaul to make the analysis more professional and reviewable.
- Updated notebook HTML exports for easier viewing.
- Updated `requirements.txt` to support the expanded analysis stack.
- Updated the dataset by one entry.

### Fixed

- Removed an extra notebook cell.
- Improved file naming for consistency.

## [Initial SVM/SLP modeling and data cleanup] - 2025-11

### Added

- Completed early SVM and SLP model implementations.
- Added utilities for data cleaning and duplicate handling.
- Added an archive of an older dataset version that contained duplicates.
- Generated model-output files and mislabel-review outputs.
- Added a reported early model result of approximately 84.1% accuracy.

### Changed

- Reformed the dataset for cleaner modeling.
- Updated file names to stay consistent with the notebook workflow.
- Removed duplicate-dropping steps from training notebooks after determining they were no longer necessary.
- Removed unnecessary files from earlier exploratory work.

## [Initial feature engineering and modeling exploration] - 2025-07

### Added

- Created the initial repository structure for the analysis system.
- Added the first social-media violence/extremism dataset materials.
- Added tools for counting entries related to violence in the dataset.
- Added early XGBoost and SVM modeling support.
- Added early model-validation and classification-report functionality.
- Added initial feature modules and exploratory signal extractors, including:
  - hate sentiment,
  - violent sentiment,
  - overall sentiment,
  - extremist-reference features,
  - action-indication features,
  - VADER-based sentiment features,
  - HateBERT-related exploratory components.
- Added `extremism_lexicon.txt` / noise-cleaned lexicon support.
- Added initial project dependencies, including support for:
  - Label Studio,
  - XGBoost,
  - VADER sentiment,
  - and a frozen Python environment.

### Changed

- Reorganized feature code into a dedicated `features/` folder.
- Moved utility scripts into a `tools/` directory.
- Made dataset title and formatting tweaks to improve parsing and downstream analysis.
- Improved several exploratory feature modules after initial implementation.

### Fixed

- Fixed early spelling and filename errors.
- Fixed early implementation errors during exploratory feature development.

## [Initial repository setup] - 2025-07-16

### Added

- Created the initial repository.
- Added the first organization structure for the analysis system.
- Began the first version-controlled workflow for the extremism/sentiment-analysis research project.

---

## Development notes

- This changelog summarizes the major visible repository milestones from July 2025 through June 2026.
- It does not include every commit, notebook run, file rename, or temporary artifact.
- Model metrics should be interpreted as research baselines, not deployment claims.
- Kaggle competition materials are treated as external research context unless rerun under the same controlled split and metric protocol used in this repository.
- Future entries should continue to separate dataset changes, split changes, model changes, result changes, documentation changes, and responsible-use updates.
