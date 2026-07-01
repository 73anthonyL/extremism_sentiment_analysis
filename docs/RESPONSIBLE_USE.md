# Responsible use statement

This repository studies violent-extremism detection as a research problem in NLP. The dataset, models, and analysis artifacts are intended for replication, benchmarking, and interpretability research.

They are not intended for direct deployment.

## High-stakes nature of the task

Extremism detection can affect speech, safety, reputation, and potentially access to digital platforms or institutional processes. False positives and false negatives both carry serious risks.

A model prediction from this repository should not be treated as an authoritative judgment about a person, community, or piece of content.

## Do not use this repository for

- Automated content removal.
- Account banning, suspension, or ranking.
- User-level risk scoring.
- Law-enforcement, school-discipline, employment, immigration, or housing decisions.
- Surveillance, profiling, or demographic inference.
- Production moderation without independent validation and human review.

## Acceptable research uses

- Reproducing the reported model baselines.
- Studying model behavior on a fixed split.
- Comparing NLP techniques under a documented protocol.
- Investigating model explanations and error modes.
- Discussing dataset limitations and annotation challenges.

## Human oversight

Any applied system in this domain would require, at minimum:

- Domain-expert review.
- Clear escalation criteria.
- Bias and fairness evaluation.
- Privacy and legal review.
- Human-in-the-loop decision-making.
- Transparent appeal or correction processes.
- Regular re-evaluation against distribution shift.

This repository does not provide those safeguards.

## Interpretation cautions

Model outputs are sensitive to:

- Missing context.
- Quotation, sarcasm, counterspeech, or news discussion.
- Coded references or changing slang.
- Class imbalance.
- Annotation subjectivity.
- Dataset source and preprocessing choices.

Strong benchmark metrics do not imply deployment safety.
