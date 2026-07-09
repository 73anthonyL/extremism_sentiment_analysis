# Changelog

This changelog summarizes the major research, data, modeling, reproducibility, and documentation milestones for the Social Media Extremism Detection repository.

The repository is maintained as a research and replication artifact. This changelog is intentionally written as a human-readable project history rather than a raw list of commits.

## [0.2.0] - 2026-07-08

### Added

* Added four additional controlled model families beyond the original TF-IDF baselines:
  * `04_CHAR-TF-IDF_LIN-SVM`
  * `05_WORD-CHAR-TF-IDF_LIN-SVM`
  * `06_FASTTEXT-EMB_LOG-REG`
  * `07_TWITTER-ROBERTA_FINE-TUNE`
* Added compact result folders for the four new experiments under `results_summary/`.
* Added transformer-specific result artifacts for the RoBERTa experiment, including validation threshold sweep and confusion-matrix plot.

### Changed

* Updated the repository documentation to describe the full seven-technique controlled comparison.
* Updated model-result tables to include the new character TF-IDF, hybrid TF-IDF, FastText embedding, and Twitter-RoBERTa results.
* Updated the experiment protocol to distinguish classical baselines, embedding baselines, and contextual transformer fine-tuning.
* Updated the results schema to allow transformer-specific compact summaries such as `confusion_matrix_test.png` and `threshold_sweep_validation.csv`.
* Updated dependency requirements for FastText/Gensim and Hugging Face transformer fine-tuning support.

### Result snapshot

The strongest current held-out test result is `07_TWITTER-ROBERTA_FINE-TUNE`, with accuracy 0.8889, positive F1 0.8555, ROC-AUC 0.9496, and PR-AUC 0.9233.

### Notes

The new transformer result is a research result under the fixed split protocol. It is not deployment evidence and should be interpreted alongside responsible-use limitations, error analysis, and XAI review.

## Earlier history

Earlier repository history established the public research repository, dataset foundation, fixed stratified split protocol, and the first three controlled baseline notebooks:

* `01_LOG-REG_TF-IDF`
* `02_LIN-SVM_TF-IDF`
* `03_SLP_TF-IDF`

For detailed commit-level history, use GitHub's commit log.
