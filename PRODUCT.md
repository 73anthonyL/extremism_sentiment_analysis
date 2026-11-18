# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary, confirmed by an author on 2026-11-18: a machine-learning or NLP
researcher who has trained their own text classifier of extremist content and
wants to audit it. They work on their own machine with this repository cloned,
usually after a training run on Kaggle, and they have not read the project's
documentation. Their job is to learn three things about their model before
trusting it: whether it is measurably different from existing models, what it
relies on, and whether that reliance shows up as false positives on posts that
mention a community.

Not confirmed as an audience: reviewers and readers of the paper who only want
to inspect the published results. The interface is not designed around them.

## Product Purpose

An auditing framework for text classifiers of extremist content, the first
contribution of a research paper. A classifier enters by exporting two kinds of
file (probabilities on a fixed split, and word-level attribution tables). The
framework checks that contract, runs a fixed chain of analyses and writes one
report card per model.

Success, confirmed by an author on 2026-11-18: a researcher who has never read
the documentation can go from "I have a model" to a finished report card
through the interface alone.

## Positioning

The audit takes exported files, not a model, so a classifier can be audited
without sharing weights and without re-running anyone else's models. Every
output is free of post text, so results can be published where the posts
cannot.

## Operating Context

* Run locally from a clone of the repository: `python3 tools/audit_gui.py`
  serves one page on the loopback address. It must work offline.
* The same actions exist as command-line tools (`tools/audit.py`); the page
  starts only the command a user would type.
* Models are trained in notebooks, usually on Kaggle, from a template. The
  notebook's last cell writes a zip that is unpacked into the repository.
* The reference study in the repository is eight techniques on one labelled
  dataset with a frozen 70/15/15 split and a 450-post test split.

## Capabilities and Constraints

* Shows, per model: the contract check item by item, a live run of the audit
  chain, the files a run changed, and the committed report card.
* Shows, across models: the committed tables by research question.
* Explains how to add a classifier, with the notebook template's sections.
* Does not classify text. There is no free-text input. The framework audits
  classifiers; it is not moderation tooling.
* Shows no post text, row id or text hash. A payload carrying one is refused.
* Words a report card lists as slurs are masked unless the reader asks for
  them.
* Does not rank models. A gap that a significance test does not support is
  described as not distinguishable, never as one model being ahead.
* Standard library only on the server; no build step; no network requests.

Undecided on 2026-11-18:

* The framework's name. Candidates are to be proposed to the authors.
* Whether the lexicon-category analysis stays a headline analysis or becomes
  one optional lens among several. Under review against the literature.
* The licence. The repository is "all rights reserved" today.

## Brand Commitments

No name, logo or visual identity exists yet. Voice already in use across the
project: plain declarative sentences, numbers with their source, limits stated
next to results.

## Evidence on Hand

* Eight audited techniques with committed report cards under
  `results_summary/rq/report_cards/`, and the cross-technique tables under
  `results_summary/rq/`.
* The user guide, `docs/FRAMEWORK.md`.
* No user study, testimonial, adoption figure or external user exists. Future
  work must not invent one.

## Product Principles

1. The contract comes first: a model is in or out, item by item, with the
   validator's own words.
2. Every number names the file it was read from.
3. Limits sit next to results, in the same weight.
4. Nothing on the page is a claim the committed files do not support.
5. Harmful words are handled as data, with care, never as decoration.

## Accessibility & Inclusion

Fully operable by keyboard. Colour never carries meaning alone: every status
has a symbol and a word. Harmful words are masked by default.
