"""One entry point for auditing a classifier with the framework (WP8).

A technique that meets the framework contract (docs/FRAMEWORK.md) is run
through the existing tool chain and summarised in one text-free report card.
This module adds no analysis of its own: every number comes from a tool that
already exists, either by calling its CLI with this interpreter or, where the
result has to be inspected, by calling its function.

Subcommands
-----------
check-contract --technique T
    One line per contract item: OK, MISSING or INVALID (with the validator's
    message). Exits 1 when a required item is missing or invalid.

run --technique T [--with-perturbation] [--device D] [--sequential-checkpoints]
    check-contract, then the chain in the documented order (CLAUDE.md, "Order
    for a new technique"; RESEARCH_PLAN.md WP5), a manifest refresh for files
    the chain wrote into the technique folder, then every report card. Stops
    at the first failing stage, whatever the error, and names it. A stage
    whose inputs are absent (not requested, no per-post files under
    external/, outside the comparison where the tool has no separate family)
    prints SKIP with the reason and is listed at the end; a skip is never
    reported as OK. --with-perturbation runs the stages that restore the
    predictor from external/ and re-score edited texts (identity
    counterfactual, its ambiguous-term sensitivity, deletion faithfulness);
    --with-counterfactual is kept as an alias. With --with-perturbation, a
    perturbation stage that skips for missing inputs makes the run exit 2.

report --technique T | --all
    Reads committed outputs only and writes
    results_summary/rq/report_cards/<T>.md and <T>.json. Refuses when a
    required RQ table has no row for T (a stale table), rather than leaving a
    section out.

USAGE
-----
    python3 tools/audit.py check-contract --technique 01_LOG-REG_TF-IDF
    python3 tools/audit.py run --technique 01_LOG-REG_TF-IDF
    python3 tools/audit.py report --all
"""

import argparse
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import probs_artifact  # noqa: E402
from attributions import (  # noqa: E402
    AttributionArtifactError,
    attributions_dir,
    list_runs,
    load_run,
)
from repo_paths import (  # noqa: E402
    COEFFICIENTS_FILENAME,
    EXPECTED_SPLIT_COUNTS,
    EXTERNAL_DIR,
    FAITHFULNESS_FILENAME,
    REPO_ROOT,
    REPORT_CARD_JSON_FILENAME,
    REPORT_CARD_MD_FILENAME,
    REPORT_CARDS_DIR,
    RESULTS_DIR,
    RQ1_MCNEMAR_PAIRS_CSV,
    RQ1_MCNEMAR_PAIRS_OUTSIDE_CSV,
    RQ2_FAITHFULNESS_CSV,
    RQ2_RQ4_RELATION_CSV,
    RQ2_SAME_POST_CATEGORY_SHARES_CSV,
    RQ2_WORD_AGREEMENT_CSV,
    RQ2_WORD_AGREEMENT_OUTSIDE_CSV,
    RQ3_STABILITY_CSV,
    RQ4_IDENTITY_COUNTERFACTUAL_CSV,
    RQ4_IDENTITY_COUNTERFACTUAL_SENSITIVITY_CSV,
    RQ4_IDENTITY_FPR_CSV,
    RQ4_IDENTITY_FPR_SENSITIVITY_CSV,
    RQ_DIR,
    TOOLS_DIR,
    comparison_technique_dirs,
    in_comparison,
    technique_dirs,
)

EXIT_OK = 0
EXIT_FAILED = 1
# run --with-perturbation completed, but a perturbation stage skipped for missing inputs.
EXIT_PERTURBATION_SKIPPED = 2

TEST_ROWS = EXPECTED_SPLIT_COUNTS["test"][0]
REPORT_TOP_WORDS = 10
WHOLE = "whole"

# The SHAP pipeline is validated on this technique (CLAUDE.md, RESEARCH_PLAN
# WP2). Its coefficient check must pass before any cross-model claim.
ANCHOR_TECHNIQUE = "01_LOG-REG_TF-IDF"

# Techniques whose coefficient check FAIL is expected, because their
# coefficients act on something other than words (RESEARCH_PLAN WP2: 06's act
# on embedding dimensions). This only changes how a FAIL is labelled; a FAIL
# for any technique other than the anchor is recorded on its report card
# rather than stopping the chain (docs/FRAMEWORK.md, "If the SHAP check fails").
CALIBRATION_POINT_TECHNIQUES = {
    "06_FASTTEXT-EMB_LOG-REG": "coefficients act on embedding dimensions (RESEARCH_PLAN WP2)",
}

# Keys that would mean per-post data in a report card; refused at write time.
FORBIDDEN_REPORT_KEYS = ("row_id", "row_ids", "text", "text_hash", "post", "message")
ROW_ID_RE = re.compile(r"\bex_\d{6}\b")

OK, MISSING, INVALID = "OK", "MISSING", "INVALID"


class AuditError(RuntimeError):
    """A contract item, a stage, or a report input is not usable."""


def _rel(path):
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# check-contract
# ---------------------------------------------------------------------------
@dataclass
class ContractItem:
    name: str
    status: str
    detail: str
    required: bool = True

    @property
    def failed(self):
        return self.required and self.status != OK

    def line(self):
        flag = "" if self.required else " (optional)"
        return f"{self.status:<8} {self.name}{flag}: {self.detail}"


def _check_probs(technique, split):
    path = probs_artifact.probs_path(technique, split)
    name = f"probability artifact ({split})"
    if not path.exists():
        return ContractItem(name, MISSING, f"{_rel(path)} not found")
    try:
        frame = probs_artifact.load_probs(path, expected_split=split)
    except (probs_artifact.ProbsArtifactError, ValueError, KeyError) as error:
        return ContractItem(name, INVALID, str(error))
    return ContractItem(name, OK, f"{_rel(path)} ({len(frame)} rows)")


def _check_best_config(technique, results_dir):
    path = Path(results_dir) / technique / "best_config.json"
    name = "locked validation threshold"
    if not path.exists():
        return ContractItem(name, MISSING, f"{_rel(path)} not found")
    try:
        with open(path) as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        return ContractItem(name, INVALID, f"{path.name}: {error}")
    if not isinstance(config, dict) or "selected_threshold" not in config:
        return ContractItem(name, INVALID, f"{path.name}: no selected_threshold")
    threshold = config["selected_threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0.0 <= threshold <= 1.0:
        return ContractItem(name, INVALID, f"{path.name}: selected_threshold {threshold!r} is not in [0, 1]")
    return ContractItem(name, OK, f"{_rel(path)} (selected_threshold {threshold})")


def _whole_runs(technique, results_dir):
    """Return ({split: [run_id]}, [problems]) over the technique's whole-model runs."""
    by_split, problems = {}, []
    folder = attributions_dir(technique, results_dir)
    if folder.exists():
        for csv_path in sorted(folder.glob("*.csv")):
            if "__" in csv_path.stem or csv_path.name == COEFFICIENTS_FILENAME:
                continue
            if not csv_path.with_suffix(".json").exists():
                problems.append(f"{csv_path.name} has no sidecar {csv_path.stem}.json")
    for run_id in list_runs(technique, results_dir):
        try:
            _, meta = load_run(technique, run_id, results_dir)
        except AttributionArtifactError as error:
            problems.append(f"{run_id}: {error}")
            continue
        if meta.get("member") in (None, "", WHOLE):
            by_split.setdefault(meta["split"], []).append(run_id)
    return by_split, problems


def _check_attributions(technique, results_dir, by_split, problems):
    items = []
    for split in ("test", "validation"):
        name = f"whole-model attribution run ({split})"
        runs = by_split.get(split, [])
        if problems:
            items.append(ContractItem(name, INVALID, "; ".join(problems)))
        elif not runs:
            items.append(ContractItem(
                name, MISSING,
                f"no whole-model run for the {split} split under "
                f"{_rel(attributions_dir(technique, results_dir))}; without word-level "
                "attribution runs the technique cannot take part in RQ2-RQ4",
            ))
        else:
            items.append(ContractItem(name, OK, ", ".join(runs)))
    return items


def _check_coefficients(technique, results_dir):
    from validate_shap import load_coefficients

    path = attributions_dir(technique, results_dir) / COEFFICIENTS_FILENAME
    name = "coefficients.csv (SHAP check, linear models)"
    if not path.exists():
        return ContractItem(name, MISSING, "absent; the SHAP coefficient check is skipped", required=False)
    try:
        frame = load_coefficients(path)
    except AttributionArtifactError as error:
        # Present but broken is a contract violation even though absence is not.
        return ContractItem(name, INVALID, str(error))
    return ContractItem(name, OK, f"{_rel(path)} ({len(frame)} words)", required=False)


def _check_external(technique, external_dir):
    from model_loaders import ModelLoadError, detect_family

    folder = Path(external_dir) / technique
    name = "restorable predictor under external/"
    needed = "needed only for --with-perturbation (counterfactuals, faithfulness) and curated examples"
    if not folder.is_dir():
        return ContractItem(name, MISSING, f"{_rel(folder)} not found; {needed}", required=False)
    try:
        family = detect_family(folder)
    except ModelLoadError as error:
        return ContractItem(name, INVALID, f"{error}; {needed}", required=False)
    return ContractItem(name, OK, f"{_rel(folder)} ({family})", required=False)


def check_contract(technique, results_dir=None, external_dir=None):
    """Return the contract items for a technique, in the order of the contract table."""
    results_dir = RESULTS_DIR if results_dir is None else Path(results_dir)
    external_dir = EXTERNAL_DIR if external_dir is None else Path(external_dir)
    by_split, problems = _whole_runs(technique, results_dir)
    items = [
        _check_probs(technique, "test"),
        _check_probs(technique, "validation"),
        _check_best_config(technique, results_dir),
        *_check_attributions(technique, results_dir, by_split, problems),
        _check_coefficients(technique, results_dir),
        _check_external(technique, external_dir),
    ]
    return items


def print_contract(technique, items):
    print(f"contract for {technique}")
    for item in items:
        print(f"  {item.line()}")
    failed = [item for item in items if item.failed]
    if failed:
        print(f"contract NOT met: {len(failed)} required item(s) missing or invalid")
        if any(item.name.startswith("whole-model attribution") for item in failed):
            print(f"  {technique} cannot take part in RQ2-RQ4 until it exports word-level "
                  "attribution runs for test and validation (nk.explain + nk.export_attribution_run)")
    else:
        print("contract met")
    return not failed


# ---------------------------------------------------------------------------
# run: the chain
# ---------------------------------------------------------------------------
@dataclass
class Context:
    technique: str
    # Opt in to the stages that restore the predictor from external/ and
    # re-score edited texts. One flag for all of them: they share the
    # prerequisite (a verified predictor), the memory cost and --device /
    # --sequential-checkpoints.
    with_perturbation: bool = False
    device: str = "auto"
    sequential_checkpoints: bool = False
    results_dir: Path = RESULTS_DIR
    external_dir: Path = EXTERNAL_DIR
    # sha256 of the technique folder once validate_results_folder has passed,
    # so the manifest is refreshed only for files the chain itself changed.
    initial_hashes: dict = field(default_factory=dict)
    # (stage, reason) for every stage that returned Skip, in order.
    skipped: list = field(default_factory=list)


@dataclass
class Skip:
    """Returned by a stage that did not run; the reason is printed and listed, never shown as OK."""
    reason: str


NOT_REQUESTED = "not requested (--with-perturbation)"


class StageFailed(AuditError):
    pass


def _tool(script, *args):
    """Run a tool CLI with this interpreter from the repo root; raise on a nonzero exit."""
    command = [sys.executable, str(TOOLS_DIR / script), *args]
    completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    output = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0:
        tail = "\n".join(output.splitlines()[-15:])
        raise StageFailed(f"{script} {' '.join(args)} exited {completed.returncode}\n{tail}")
    # The tool's own last line, not a library warning on stderr.
    lines = completed.stdout.strip().splitlines() or output.splitlines()
    return lines[-1] if lines else "ok"


def stage_validate_results_folder(ctx):
    from run_manifest import hash_committed

    message = _tool("validate_results_folder.py", "--technique", ctx.technique)
    ctx.initial_hashes = hash_committed(ctx.technique, ctx.results_dir)
    return message


def stage_refresh_manifest(ctx):
    """Record the files the chain wrote into the technique folder in its run_manifest.json.

    The folder matched its manifest when validate_results_folder passed, so a
    refresh here records tool outputs only (coefficient checks, category
    shares, identity-FPR, counterfactual and faithfulness JSONs). Nothing changed, nothing
    is rewritten, which keeps a rerun byte-identical.
    """
    from run_manifest import hash_committed, manifest_path, refresh

    if not manifest_path(ctx.technique, ctx.results_dir).exists():
        return "no run_manifest.json; skipped (create one with run_manifest.py init)"
    after = hash_committed(ctx.technique, ctx.results_dir)
    changed = sorted(p for p in after if ctx.initial_hashes.get(p) != after[p])
    if not changed:
        return "no file in the technique folder changed; manifest left as is"
    refresh(ctx.technique, ctx.results_dir)
    return f"refreshed for {len(changed)} file(s) the chain wrote: {', '.join(changed)}"


def _anchor_passes(results_dir):
    folder = attributions_dir(ANCHOR_TECHNIQUE, results_dir)
    checks = sorted(folder.glob("*__coefficient_check.json")) if folder.exists() else []
    if not checks:
        return False, f"no committed coefficient check for the anchor {ANCHOR_TECHNIQUE}"
    for path in checks:
        with open(path) as handle:
            if not json.load(handle).get("passed"):
                return False, f"anchor check {_rel(path)} did not pass"
    return True, f"{len(checks)} run(s) checked"


def stage_validate_shap(ctx):
    """The coefficient check on every whole-model run, when coefficients exist.

    The anchor's check is the pipeline gate (CLAUDE.md, RESEARCH_PLAN WP2):
    a FAIL on the anchor stops the chain, and any other technique needs the
    anchor's committed check to pass before it is compared. For any other
    technique a FAIL is a finding about that model, not about the pipeline:
    it is written to <run_id>__coefficient_check.json, reported, and carried
    onto the report card, whose RQ2 section then says the word list must not
    support a cross-model claim. The chain goes on.
    """
    from validate_shap import validate

    notes = []
    if ctx.technique != ANCHOR_TECHNIQUE:
        ok, message = _anchor_passes(ctx.results_dir)
        if not ok:
            raise StageFailed(f"{message}; no cross-model attribution claim is licensed")
        notes.append(message)

    if not (attributions_dir(ctx.technique, ctx.results_dir) / COEFFICIENTS_FILENAME).exists():
        notes.append("no coefficients.csv; skipped")
        return "; ".join(notes)

    by_split, _ = _whole_runs(ctx.technique, ctx.results_dir)
    for split in sorted(by_split):
        for run_id in by_split[split]:
            try:
                stats, _ = validate(ctx.technique, run_id, results_dir=ctx.results_dir)
            except AttributionArtifactError as error:
                raise StageFailed(str(error)) from error
            summary = (f"{run_id}: Spearman {stats['spearman_mean_attribution_vs_coefficient']:.3f}, "
                       f"top-{stats['top_k']} sign agreement {stats['top_k_sign_agreement']:.2f}")
            if stats["passed"]:
                notes.append(f"{summary} PASS")
            elif ctx.technique in CALIBRATION_POINT_TECHNIQUES:
                notes.append(f"{summary} FAIL, recorded as a calibration point: "
                             f"{CALIBRATION_POINT_TECHNIQUES[ctx.technique]}")
            elif ctx.technique == ANCHOR_TECHNIQUE:
                raise StageFailed(
                    f"{summary} FAIL on the anchor {ANCHOR_TECHNIQUE}; the SHAP pipeline is not "
                    "validated and no cross-model attribution claim is licensed"
                )
            else:
                secondary = stats.get("spearman_support_ge_min")
                extra = (f"; Spearman over words with support >= {stats['secondary_min_support']} "
                         f"{secondary:.3f}" if secondary is not None else "")
                notes.append(f"{summary} FAIL{extra}. Recorded on the report card; its RQ2 words "
                             "must not support a cross-model claim (docs/FRAMEWORK.md)")
    return "; ".join(notes)


def stage_categorize_attributions(ctx):
    return _tool("categorize_attributions.py", "--technique", ctx.technique)


def stage_compare_reliance(ctx):
    return _tool("compare_reliance.py")


def stage_identity_fpr(ctx):
    return _tool("identity_fpr.py", "--technique", ctx.technique)


def stage_mcnemar_pairs(ctx):
    return _tool("mcnemar_pairs.py")


def stage_word_agreement(ctx):
    return _tool("word_agreement.py")


def stage_identity_sensitivity_fpr(ctx):
    """RQ4 FPR without the ambiguous identity terms; per-technique rows, no model loaded."""
    return _tool("identity_sensitivity.py", "fpr", "--technique", ctx.technique)


def stage_identity_lexicon_gaps(ctx):
    """WP3 candidates among this technique's top words; replaces only its rows."""
    return _tool("identity_sensitivity.py", "lexicon-gaps", "--technique", ctx.technique)


def _missing(paths):
    return [_rel(p) for p in paths if not Path(p).is_file()]


def _missing_phrase(missing):
    shown = ", ".join(missing[:3]) + (f" and {len(missing) - 3} more" if len(missing) > 3 else "")
    return f"{len(missing)} per-post attribution file(s) absent ({shown})"


def same_post_inputs(results_dir, external_dir):
    """The per-post file of every whole-model run of every comparison technique."""
    from same_post_stability import local_path

    paths = []
    for folder in comparison_technique_dirs(results_dir):
        by_split, _ = _whole_runs(folder.name, results_dir)
        for split in ("test", "validation"):
            paths += [local_path(folder.name, run_id, external_dir) for run_id in by_split.get(split, [])]
    return paths


def stage_same_post_stability(ctx):
    """Same-post support, split agreement and category shares over the comparison's common posts.

    The tool defines the common post set over the comparison techniques and
    rewrites its three tables whole, with no separate family for a technique
    outside the comparison, so it is kept out of such a run.
    """
    if not in_comparison(ctx.technique, ctx.results_dir):
        return Skip(f"{ctx.technique} is outside the comparison; same_post_stability.py defines the common "
                    "post set over the comparison and has no outside family, so its tables are left as is")
    missing = _missing(same_post_inputs(ctx.results_dir, ctx.external_dir))
    if missing:
        return Skip(f"{_missing_phrase(missing)}; download them to external/ to rebuild the same-post tables, "
                    "which are left as is")
    return _tool("same_post_stability.py")


def _perturbation_args(ctx):
    args = ["--technique", ctx.technique, "--device", ctx.device]
    if ctx.sequential_checkpoints:
        args.append("--sequential-checkpoints")
    return args


def stage_identity_counterfactual(ctx):
    if not ctx.with_perturbation:
        return Skip(NOT_REQUESTED)
    return _tool("identity_counterfactual.py", *_perturbation_args(ctx))


def stage_identity_sensitivity_counterfactual(ctx):
    """WP7 remove on the posts with an unambiguous identity match; loads the predictor."""
    if not ctx.with_perturbation:
        return Skip(NOT_REQUESTED)
    return _tool("identity_sensitivity.py", "counterfactual", *_perturbation_args(ctx))


def faithfulness_inputs(technique, results_dir, external_dir):
    """(committed word table, per-post files) faithfulness.py needs for this technique.

    The shared population is the posts every comparison technique explained,
    so every comparison technique's per-post test file is needed too.
    """
    from split_beeswarms import LOCAL_SUBDIR
    from split_beeswarms import run_id as local_run_id

    name = f"{local_run_id('test')}.csv"
    techniques = [d.name for d in comparison_technique_dirs(results_dir)]
    if technique not in techniques:
        techniques.append(technique)
    word_table = attributions_dir(technique, results_dir) / name
    return word_table, [Path(external_dir) / t / LOCAL_SUBDIR / name for t in techniques]


def stage_faithfulness(ctx):
    """Deletion faithfulness of the whole-model test run; loads the predictor.

    faithfulness.py gives a technique outside the comparison its own Holm
    family, so the comparison rows are unchanged.
    """
    if not ctx.with_perturbation:
        return Skip(NOT_REQUESTED)
    word_table, per_post = faithfulness_inputs(ctx.technique, ctx.results_dir, ctx.external_dir)
    if not word_table.is_file():
        return Skip(f"no committed whole-model test run named {word_table.stem}; faithfulness.py reads only "
                    f"that run, so {RQ2_FAITHFULNESS_CSV.name} is left as is")
    missing = _missing(per_post)
    if missing:
        return Skip(f"{_missing_phrase(missing)}; the shared population needs every comparison technique's "
                    f"test file, so {RQ2_FAITHFULNESS_CSV.name} is left as is")
    return _tool("faithfulness.py", *_perturbation_args(ctx))


def relation_gaps(results_dir, rq_dir):
    """Comparison techniques missing from an input rq2_rq4_relation.py needs beyond the always-run tables."""
    techniques = {d.name for d in comparison_technique_dirs(results_dir)}
    gaps = []
    for path, keep in ((RQ4_IDENTITY_COUNTERFACTUAL_CSV,
                        lambda t: (t["population"] == "non_extremist") & (t["operation"] == "remove")),
                       (RQ2_SAME_POST_CATEGORY_SHARES_CSV, lambda t: t["post_set"] == "common")):
        path = Path(rq_dir) / path.name
        present = set()
        if path.exists():
            table = pd.read_csv(path)
            present = set(table.loc[keep(table), "technique"])
        missing = sorted(techniques - present)
        if missing:
            gaps.append(f"{path.name} has no row for {missing}")
    return gaps


def counterfactual_threshold_gaps(results_dir, rq_dir):
    """Comparison techniques whose WP7 rows were scored at a threshold other than their locked one."""
    path = Path(rq_dir) / RQ4_IDENTITY_COUNTERFACTUAL_CSV.name
    if not path.exists():
        return []
    table = pd.read_csv(path)
    off = []
    for folder in comparison_technique_dirs(results_dir):
        thresholds = table.loc[table["technique"] == folder.name, "threshold"]
        locked = _locked_threshold(folder.name, results_dir)
        if any(not _same_number(t, locked) for t in thresholds):
            off.append(folder.name)
    return off


def stage_rq2_rq4_relation(ctx):
    """The RQ2-RQ4 rank relation over the comparison techniques; committed tables only."""
    if not in_comparison(ctx.technique, ctx.results_dir):
        return Skip(f"{ctx.technique} is outside the comparison; the relation is computed over the comparison "
                    "techniques only")
    if "same_post_stability" in {name for name, _ in ctx.skipped}:
        return Skip("same_post_stability was skipped in this run, so the common-post shares the relation reads "
                    f"may be stale; {RQ2_RQ4_RELATION_CSV.name} is left as is")
    off = counterfactual_threshold_gaps(ctx.results_dir, RQ_DIR)
    if off:
        return Skip(f"{RQ4_IDENTITY_COUNTERFACTUAL_CSV.name} was scored at a threshold other than the locked one "
                    f"for {off}; {RQ2_RQ4_RELATION_CSV.name} is left as is and is stale until "
                    "identity_counterfactual.py reruns for them")
    gaps = relation_gaps(ctx.results_dir, RQ_DIR)
    if gaps:
        return Skip("; ".join(gaps) + f". {RQ2_RQ4_RELATION_CSV.name} is left as is and is stale until those "
                    "stages run (--with-perturbation, per-post files under external/)")
    return _tool("rq2_rq4_relation.py")


def stage_render_tables(ctx):
    return _tool("render_tables.py", "--write")


def stage_report(ctx):
    """Every card, not only this technique's: a new technique changes the others' RQ1 and RQ2 rows."""
    written = write_reports([d.name for d in technique_dirs(ctx.results_dir)], results_dir=ctx.results_dir)
    return f"wrote {len(written)} report card(s) under {_rel(REPORT_CARDS_DIR)}"


# Resolved by name at run time so a test can replace one stage.
STAGE_ORDER = (
    "validate_results_folder",
    "validate_shap",
    "categorize_attributions",
    "compare_reliance",
    "identity_fpr",
    "identity_sensitivity_fpr",
    "mcnemar_pairs",
    "word_agreement",
    "identity_lexicon_gaps",
    "same_post_stability",
    # The three stages that load the predictor, one process at a time.
    "identity_counterfactual",
    "identity_sensitivity_counterfactual",
    "faithfulness",
    "rq2_rq4_relation",
    "refresh_manifest",
    "render_tables",
    "report",
)

# Stages that restore the predictor from external/ (--with-perturbation).
PERTURBATION_STAGES = ("identity_counterfactual", "identity_sensitivity_counterfactual", "faithfulness")


def run_chain(ctx, external_dir=None):
    """Return (ok, failed_stage). Prints one line per stage; skipped stages land in ctx.skipped."""
    if external_dir is not None:
        ctx.external_dir = Path(external_dir)
    items = check_contract(ctx.technique, results_dir=ctx.results_dir, external_dir=ctx.external_dir)
    if not print_contract(ctx.technique, items):
        print("STOPPED at stage check-contract")
        return False, "check-contract"
    if ctx.with_perturbation:
        external = next(item for item in items if item.name.startswith("restorable predictor"))
        if external.status != OK:
            print(f"STOPPED at stage check-contract: --with-perturbation needs {external.detail}")
            return False, "check-contract"

    module = sys.modules[__name__]
    for name in STAGE_ORDER:
        stage = getattr(module, f"stage_{name}")
        try:
            message = stage(ctx)
        except (AuditError, AttributionArtifactError, probs_artifact.ProbsArtifactError) as error:
            print(f"FAIL  {name}")
            print(f"STOPPED at stage {name}: {error}")
            return False, name
        except Exception as error:  # noqa: BLE001 - any stage error must still name the stage
            print(f"FAIL  {name}")
            print(f"STOPPED at stage {name}: unexpected {type(error).__name__}: {error}")
            return False, name
        if isinstance(message, Skip):
            ctx.skipped.append((name, message.reason))
            print(f"SKIP  {name}: {message.reason}")
            continue
        print(f"OK    {name}: {message}")
    if ctx.skipped:
        print(f"audit of {ctx.technique} complete; {len(ctx.skipped)} stage(s) skipped: "
              f"{', '.join(name for name, _ in ctx.skipped)}")
        skipped_perturbation = [name for name, _ in ctx.skipped if name in PERTURBATION_STAGES]
        if skipped_perturbation:
            print(f"perturbation stages skipped this run ({', '.join(skipped_perturbation)}): their tables were "
                  "not rewritten, so the cards show the outputs committed earlier, or \"not run\" where none exist")
    else:
        print(f"audit of {ctx.technique} complete")
    return True, None


def perturbation_shortfall(ctx):
    """Perturbation stages that were requested but skipped for a reason other than not being requested."""
    if not ctx.with_perturbation:
        return []
    return [(name, reason) for name, reason in ctx.skipped
            if name in PERTURBATION_STAGES and reason != NOT_REQUESTED]


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _read_table(path):
    path = Path(path)
    if not path.exists():
        raise AuditError(f"{_rel(path)} not found; run `python3 tools/audit.py run` first")
    return pd.read_csv(path)


def _stale(path, technique, tool):
    return AuditError(
        f"{_rel(path)} has no row for {technique}; the table is stale. Rerun {tool} "
        f"(or `python3 tools/audit.py run --technique {technique}`) before reporting"
    )


def _require_every_other(technique, others, results_dir, path, tool):
    """A pair table must pair the technique with every other technique in the comparison."""
    expected = {d.name for d in comparison_technique_dirs(results_dir)} - {technique}
    missing = sorted(expected - set(others))
    if missing:
        raise AuditError(
            f"{_rel(path)} has no {technique} row paired with {missing}; the table is stale. "
            f"Rerun {tool} before reporting"
        )


def _num(value, digits=6):
    """A JSON-safe rounded float, or None for NaN."""
    if value is None:
        return None
    value = float(value)
    if math.isnan(value):
        return None
    return float(f"{value:.{digits}g}")


def _int(value):
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else int(value)


def _is_whole(value):
    return value is None or (isinstance(value, float) and np.isnan(value)) or value in ("", WHOLE)


def _rq1(technique, results_dir, rq_dir):
    metrics_path = Path(results_dir) / technique / "metrics_test.json"
    if not metrics_path.exists():
        raise AuditError(f"{_rel(metrics_path)} not found")
    with open(metrics_path) as handle:
        accuracy = float(json.load(handle)["accuracy"])

    inside = in_comparison(technique, results_dir)
    path = Path(rq_dir) / (RQ1_MCNEMAR_PAIRS_CSV if inside else RQ1_MCNEMAR_PAIRS_OUTSIDE_CSV).name
    table = _read_table(path)
    pairs = table[(table["split"] == "test")
                  & ((table["technique_a"] == technique) | (table["technique_b"] == technique))]
    if pairs.empty:
        raise _stale(path, technique, "tools/mcnemar_pairs.py")

    rows = []
    for _, row in pairs.iterrows():
        as_a = row["technique_a"] == technique
        own_accuracy = row["accuracy_a"] if as_a else row["accuracy_b"]
        if abs(float(own_accuracy) - accuracy) > 1e-5:
            raise AuditError(
                f"{_rel(path)} gives {technique} accuracy {own_accuracy} but {_rel(metrics_path)} "
                f"gives {accuracy}; the table is stale. Rerun tools/mcnemar_pairs.py"
            )
        gap = int(row["accuracy_gap_rows"]) * (1 if as_a else -1)
        significant = bool(row["significant_holm"])
        rows.append({
            "other_technique": row["technique_b"] if as_a else row["technique_a"],
            "other_accuracy": _num(row["accuracy_b"] if as_a else row["accuracy_a"]),
            "row_gap": gap,
            "n_discordant": int(row["n_discordant"]),
            "mcnemar_p": _num(row["mcnemar_p"]),
            "holm_p": _num(row["holm_p"]),
            "significant_holm": significant,
        })
    rows.sort(key=lambda r: r["other_technique"])
    _require_every_other(technique, [r["other_technique"] for r in rows], results_dir, path,
                         "tools/mcnemar_pairs.py")
    return {
        "sources": [_rel(metrics_path), _rel(path)],
        "test_accuracy": _num(accuracy),
        "n_test_rows": TEST_ROWS,
        "in_comparison": inside,
        # The comparison's family is every test pair in the main table; an
        # outside technique's family is its own pairs only.
        "holm_family_size": int((table["split"] == "test").sum()) if inside else len(rows),
        "pairs": rows,
    }


def _rq2(technique, results_dir, rq_dir):
    from categorize_attributions import build_word_index, categorize_word
    from lexicons import load_lexicons

    by_split, problems = _whole_runs(technique, results_dir)
    if problems:
        raise AuditError(f"{technique} attribution runs are invalid: {'; '.join(problems)}")
    if not by_split.get("test"):
        raise AuditError(f"{technique} has no whole-model test attribution run; "
                         "it cannot take part in RQ2-RQ4")
    run_id = sorted(by_split["test"])[0]  # the run word_agreement.py compares
    frame, meta = load_run(technique, run_id, results_dir)
    lexicons = load_lexicons()
    index, _ = build_word_index(lexicons)
    empty_categories = sorted(c for c, entries in lexicons.items() if not entries)
    top = frame[frame["mean_attribution"] > 0].nlargest(REPORT_TOP_WORDS, "mean_attribution")
    words = [{
        "word": row["word"],
        "category": categorize_word(row["word"], index)[0],
        "mean_attribution": _num(row["mean_attribution"]),
        "support": int(row["support"]),
    } for _, row in top.iterrows()]

    inside = in_comparison(technique, results_dir)
    path = Path(rq_dir) / (RQ2_WORD_AGREEMENT_CSV if inside else RQ2_WORD_AGREEMENT_OUTSIDE_CSV).name
    agreement = _read_table(path)
    whole = agreement["member_a"].map(_is_whole) & agreement["member_b"].map(_is_whole)
    agreement = agreement[whole & (agreement["split"] == "test")
                          & ((agreement["technique_a"] == technique) | (agreement["technique_b"] == technique))]
    if agreement.empty:
        raise _stale(path, technique, "tools/word_agreement.py")
    pairs = []
    for _, row in agreement.iterrows():
        pairs.append({
            "other_technique": row["technique_b"] if row["technique_a"] == technique else row["technique_a"],
            "spearman_mean_attribution": _num(row["spearman_mean_attribution"]),
            "jaccard_top_k_positive": _num(row["jaccard_top_k_positive"]),
            "top_k": int(row["top_k"]),
        })
    _require_every_other(technique, [r["other_technique"] for r in pairs], results_dir, path,
                         "tools/word_agreement.py")
    pairs.sort(key=lambda r: (-(r["spearman_mean_attribution"] if r["spearman_mean_attribution"] is not None else -2),
                              r["other_technique"]))
    by_jaccard = sorted(pairs, key=lambda r: (-(r["jaccard_top_k_positive"] if r["jaccard_top_k_positive"] is not None
                                                else -2), r["other_technique"]))
    return {
        "sources": [_rel(Path(results_dir) / technique / "attributions" / f"{run_id}.csv"),
                    "data/lexicons/", _rel(path)],
        "run_id": run_id,
        "n_posts_explained": int(meta["n_posts_explained"]),
        "top_words_by_positive_mean_attribution": words,
        "empty_lexicon_categories": empty_categories,
        "agreement": pairs,
        "most_agreeing_by_spearman": pairs[0]["other_technique"],
        "least_agreeing_by_spearman": pairs[-1]["other_technique"],
        "most_agreeing_by_jaccard": by_jaccard[0]["other_technique"],
        "least_agreeing_by_jaccard": by_jaccard[-1]["other_technique"],
    }


def _shap_check(technique, results_dir):
    """The coefficient check for every whole-model run, read from its committed check file."""
    folder = attributions_dir(technique, results_dir)
    coefficients = folder / COEFFICIENTS_FILENAME
    anchor_ok, anchor_message = _anchor_passes(results_dir)
    base = {"anchor": ANCHOR_TECHNIQUE, "anchor_passes": anchor_ok, "anchor_detail": anchor_message}
    if not coefficients.exists():
        return {**base, "sources": [], "status": "no coefficients", "runs": []}
    by_split, _ = _whole_runs(technique, results_dir)
    runs, sources = [], []
    for split in sorted(by_split):
        for run_id in sorted(by_split[split]):
            path = folder / f"{run_id}__coefficient_check.json"
            if not path.exists():
                raise AuditError(
                    f"{_rel(path)} not found although {_rel(coefficients)} exists; the check is stale. "
                    f"Rerun tools/validate_shap.py --technique {technique} --run-id {run_id}"
                )
            with open(path) as handle:
                check = json.load(handle)
            sources.append(_rel(path))
            runs.append({
                "run_id": run_id,
                "split": split,
                "spearman_mean_attribution_vs_coefficient": _num(check["spearman_mean_attribution_vs_coefficient"]),
                "secondary_min_support": check.get("secondary_min_support"),
                "spearman_support_ge_min": _num(check.get("spearman_support_ge_min")),
                "top_k": int(check["top_k"]),
                "top_k_sign_agreement": _num(check["top_k_sign_agreement"]),
                "passed": bool(check["passed"]),
            })
    if all(r["passed"] for r in runs):
        status = "PASS"
    elif technique in CALIBRATION_POINT_TECHNIQUES:
        status = "calibration point"
    else:
        status = "FAIL"
    return {**base, "sources": sources, "status": status,
            "calibration_reason": CALIBRATION_POINT_TECHNIQUES.get(technique), "runs": runs}


def _rq3(technique, results_dir, rq_dir):
    by_split, _ = _whole_runs(technique, results_dir)
    path = Path(rq_dir) / RQ3_STABILITY_CSV.name
    table = _read_table(path)
    rows = table[table["technique"] == technique]
    if rows.empty:
        raise _stale(path, technique, "tools/compare_reliance.py")
    row = rows.iloc[0]
    return {
        "sources": [_rel(path)],
        "n_runs": int(row["n_runs"]),
        "run_splits": sorted(by_split),
        "mean_pairwise_spearman_abs_attribution": _num(row["mean_pairwise_spearman_abs_attribution"]),
        "mean_pairwise_jaccard_top50": _num(row["mean_pairwise_jaccard_top50"]),
    }


def _rq4(technique, rq_dir):
    path = Path(rq_dir) / RQ4_IDENTITY_FPR_CSV.name
    table = _read_table(path)
    rows = table[(table["technique"] == technique) & (table["split"] == "test")]
    if rows.empty:
        raise _stale(path, technique, "tools/identity_fpr.py")
    row = rows.iloc[0]
    n_with = int(row["n_negative_with_identity_terms"])
    return {
        "sources": [_rel(path)],
        "threshold": _num(row["threshold"]),
        "n_negative": int(row["n_negative"]),
        "n_negative_with_identity_terms": n_with,
        "n_negative_without_identity_terms": int(row["n_negative"]) - n_with,
        "fpr_with_identity_terms": _num(row["fpr_with_identity_terms"]),
        "fpr_without_identity_terms": _num(row["fpr_without_identity_terms"]),
        "fpr_ratio": _num(row["fpr_ratio"]),
        "fisher_p_value": _num(row["fisher_p_value"]),
    }


SENSITIVITY_VARIANT = "exclude_ambiguous_only"


def _missing_number(value):
    if value is None:
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _same_number(a, b, rel_tol=1e-5, abs_tol=1e-9):
    """Equal within tolerance; NaN or None on both sides is equal, on one side is not."""
    if _missing_number(a) or _missing_number(b):
        return _missing_number(a) and _missing_number(b)
    return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)


def _rq4_sensitivity(technique, rq_dir, rq4):
    """RQ4 without the posts whose only identity matches are ambiguous words.

    The tool's full_lexicon row recomputes RQ4; it must agree with the RQ4
    table, or one of the two is stale. The exclude_ambiguous_only row must be
    the full_lexicon row minus its excluded posts: the excluded posts all have
    an identity match, so the counts drop by exactly the excluded numbers and
    the rate on posts without an identity term is unchanged. Recomputing the
    excluded posts from the declared ambiguous list would need the post texts,
    which the report does not read.
    """
    path = Path(rq_dir) / RQ4_IDENTITY_FPR_SENSITIVITY_CSV.name
    table = _read_table(path)
    rows = table[(table["technique"] == technique) & (table["split"] == "test")]
    if rows["variant"].duplicated().any():
        raise AuditError(f"{_rel(path)} has more than one row per variant for {technique}; the table is stale")
    rows = rows.set_index("variant")
    if not {"full_lexicon", SENSITIVITY_VARIANT} <= set(rows.index):
        raise _stale(path, technique, "tools/identity_sensitivity.py fpr")
    full = rows.loc["full_lexicon"]
    same = (int(full["n_negative_with_identity_terms"]) == rq4["n_negative_with_identity_terms"]
            and all(_same_number(full[c], rq4[c])
                    for c in ("threshold", "fpr_with_identity_terms", "fpr_without_identity_terms")))
    if not same:
        raise AuditError(
            f"{_rel(path)} full_lexicon row for {technique} differs from {RQ4_IDENTITY_FPR_CSV.name}; "
            "the table is stale. Rerun tools/identity_sensitivity.py fpr before reporting"
        )
    row = rows.loc[SENSITIVITY_VARIANT]
    n_neg_x, n_pos_x = int(row["n_negative_excluded"]), int(row["n_positive_excluded"])
    consistent = (
        0 <= n_neg_x <= int(full["n_negative_with_identity_terms"])
        and 0 <= n_pos_x <= int(full["n_positive_with_identity_terms"])
        and int(row["n_negative"]) == int(full["n_negative"]) - n_neg_x
        and int(row["n_negative_with_identity_terms"]) == int(full["n_negative_with_identity_terms"]) - n_neg_x
        and int(row["n_positive_with_identity_terms"]) == int(full["n_positive_with_identity_terms"]) - n_pos_x
        and all(_same_number(row[c], full[c]) for c in ("threshold", "fpr_without_identity_terms",
                                                         "recall_without_identity_terms"))
    )
    if not consistent:
        raise AuditError(
            f"{_rel(path)} {SENSITIVITY_VARIANT} row for {technique} is not its full_lexicon row minus the "
            "excluded posts; the table is stale. Rerun tools/identity_sensitivity.py fpr before reporting"
        )
    n_with = int(row["n_negative_with_identity_terms"])
    return {
        "sources": [_rel(path)],
        "variant": SENSITIVITY_VARIANT,
        "n_negative_excluded": n_neg_x,
        "n_positive_excluded": n_pos_x,
        "n_negative_with_identity_terms": n_with,
        "n_negative_without_identity_terms": int(row["n_negative"]) - n_with,
        "n_positive_with_identity_terms": int(row["n_positive_with_identity_terms"]),
        "fpr_with_identity_terms": _num(row["fpr_with_identity_terms"]),
        "fpr_without_identity_terms": _num(row["fpr_without_identity_terms"]),
        "fpr_ratio": _num(row["fpr_ratio"]),
        "fisher_p_value": _num(row["fisher_p_value"]),
    }


def _counterfactual_rows(rows, population=lambda row: row["population"]):
    out = []
    for _, row in rows.sort_values("y_true").iterrows():
        out.append({
            "population": population(row),
            "n_posts": int(row["n_posts"]),
            "mean_delta": _num(row["mean_delta"]),
            "n_flagged_original": _int(row["n_flagged_original"]),
            "n_flip_pos_to_neg": _int(row["n_flip_pos_to_neg"]),
            "n_flip_neg_to_pos": _int(row["n_flip_neg_to_pos"]),
            "wilcoxon_p_holm": _num(row["wilcoxon_p_holm"]),
            "mcnemar_p_holm": _num(row["mcnemar_p_holm"]),
        })
    return out


def _require_locked(rows, technique, locked, path, tool):
    """Every counterfactual row must have been scored at the technique's locked threshold."""
    off = sorted({float(t) for t in rows["threshold"] if not _same_number(t, locked)})
    if off:
        raise AuditError(
            f"{_rel(path)} scored {technique} at threshold {off} but the locked threshold is {locked}; "
            f"the table is stale. Rerun {tool} --technique {technique} before reporting"
        )


def _wp7_unambiguous(technique, rq_dir, locked, sensitivity):
    """The restricted-population remove rows; their n must be the sensitivity table's identity-term counts."""
    path = Path(rq_dir) / RQ4_IDENTITY_COUNTERFACTUAL_SENSITIVITY_CSV.name
    if not path.exists():
        return {"sources": [], "status": "not run", "rows": []}
    table = pd.read_csv(path)
    rows = table[(table["technique"] == technique) & (table["operation"] == "remove")]
    if rows.empty:
        return {"sources": [_rel(path)], "status": "not run", "rows": []}
    tool = "tools/identity_sensitivity.py counterfactual"
    _require_locked(rows, technique, locked, path, tool)
    expected = {"non_extremist": sensitivity["n_negative_with_identity_terms"],
                "extremist": sensitivity["n_positive_with_identity_terms"]}
    found = {row["population"]: int(row["n_posts"]) for _, row in rows.iterrows()}
    if len(rows) != len(found) or found != expected:
        raise AuditError(
            f"{_rel(path)} remove rows for {technique} cover {found} posts but "
            f"{RQ4_IDENTITY_FPR_SENSITIVITY_CSV.name} ({SENSITIVITY_VARIANT}) has {expected} with an "
            f"unambiguous identity match; one of them is stale. Rerun {tool} and identity_sensitivity.py fpr"
        )
    return {"sources": [_rel(path)], "status": "run",
            "rows": _counterfactual_rows(rows, lambda row: f"{row['population']}, unambiguous match")}


def _wp7(technique, rq_dir, locked, sensitivity):
    path = Path(rq_dir) / RQ4_IDENTITY_COUNTERFACTUAL_CSV.name
    payload = {"sources": [], "status": "not run", "rows": []}
    if path.exists():
        table = pd.read_csv(path)
        rows = table[(table["technique"] == technique) & (table["operation"] == "remove")]
        payload["sources"] = [_rel(path)]
        if not rows.empty:
            _require_locked(table[table["technique"] == technique], technique, locked, path,
                            "tools/identity_counterfactual.py")
            payload.update(status="run", rows=_counterfactual_rows(rows))
    payload["unambiguous_population"] = _wp7_unambiguous(technique, rq_dir, locked, sensitivity)
    return payload


FAITHFULNESS_POPULATION = "shared"  # the primary Holm family (faithfulness.py)


def _locked_threshold(technique, results_dir):
    with open(Path(results_dir) / technique / "best_config.json") as handle:
        return float(json.load(handle)["selected_threshold"])


def _faithfulness(technique, results_dir, rq_dir):
    """Deletion faithfulness from faithfulness_test.json and its rows in rq2_faithfulness.csv.

    Neither present: "not run". Either without the other, or the two
    disagreeing, or a threshold other than the locked one: stale, refused.
    """
    json_path = Path(results_dir) / technique / FAITHFULNESS_FILENAME.format(split="test")
    path = Path(rq_dir) / RQ2_FAITHFULNESS_CSV.name
    table = pd.read_csv(path) if path.exists() else None
    rows = table[table["technique"] == technique] if table is not None else pd.DataFrame()
    if not json_path.exists():
        if not rows.empty:
            raise AuditError(f"{_rel(path)} has rows for {technique} but {_rel(json_path)} is missing; the "
                             "table is stale. Rerun tools/faithfulness.py --rebuild-table before reporting")
        return {"sources": [], "status": "not run", "rows": []}
    if rows.empty:
        raise _stale(path, technique, "tools/faithfulness.py --rebuild-table")
    with open(json_path) as handle:
        payload = json.load(handle)
    rebuild = "Rerun tools/faithfulness.py --rebuild-table before reporting"
    rerun_all = "Rerun tools/faithfulness.py --technique T for every comparison technique before reporting"
    comparison = [d.name for d in comparison_technique_dirs(results_dir)]
    inside = technique in comparison

    # The shared population is the posts every comparison technique explained; a changed comparison changes it.
    if list(payload.get("shared_population_from") or []) != comparison:
        raise AuditError(
            f"{_rel(json_path)} took its shared population from {payload.get('shared_population_from')} but the "
            f"comparison is now {comparison}; it is stale. {rerun_all}"
        )
    # Holm runs over every comparison technique's rows; a partial table gives the wrong family.
    if inside:
        absent = sorted(set(comparison) - set(table["technique"]))
        if absent:
            raise AuditError(f"{_rel(path)} has faithfulness rows for some comparison techniques but none for "
                             f"{absent}; the Holm family is incomplete and the table is stale. {rerun_all}")
    sizes = set()
    for other in comparison:
        other_json = Path(results_dir) / other / FAITHFULNESS_FILENAME.format(split="test")
        if other_json.exists():
            with open(other_json) as handle:
                sizes.add(int(json.load(handle)["n_posts_shared"]))
    if len(sizes) != 1:
        raise AuditError(f"the comparison techniques' faithfulness JSONs give {sorted(sizes) or 'no'} shared-set "
                         f"size(s); the shared population is not one set. {rerun_all}")
    n_shared_set = sizes.pop()

    # Every numeric row field, and the derived columns, must be the JSON's.
    from faithfulness import DERIVED_COLUMNS, ROW_COLUMNS, derive_columns

    def key(row):
        return row["population"], int(row["k"])

    table_keys = [key(r) for _, r in rows.iterrows()]
    json_keys = [key(r) for r in payload["rows"]]
    if len(set(table_keys)) != len(table_keys) or set(table_keys) != set(json_keys):
        raise AuditError(f"{_rel(path)} rows for {technique} {sorted(table_keys)} are not the (population, k) "
                         f"rows of {_rel(json_path)} {sorted(json_keys)}; the table is stale. {rebuild}")
    keyed = {key(r): r for _, r in rows.iterrows()}
    expected_rows = derive_columns(pd.DataFrame(payload["rows"]))
    for i, expected in enumerate(payload["rows"]):
        row = keyed[key(expected)]
        columns = [c for c in ROW_COLUMNS if isinstance(expected.get(c), (int, float, type(None)))
                   and not isinstance(expected.get(c), bool)]
        values = [(c, expected.get(c)) for c in columns]
        values += [(c, expected_rows.loc[i, c]) for c in DERIVED_COLUMNS]
        differs = [c for c, value in values if not _same_number(row[c], value, abs_tol=1e-12)]
        if differs:
            raise AuditError(
                f"{_rel(path)} does not match {_rel(json_path)} for {expected['population']} k={expected['k']} "
                f"({', '.join(differs)}); the table is stale. {rebuild}"
            )
    locked = _locked_threshold(technique, results_dir)
    if not math.isclose(float(payload["threshold"]), locked, abs_tol=1e-9):
        raise AuditError(
            f"{_rel(json_path)} was computed at threshold {payload['threshold']} but the locked threshold is "
            f"{locked}; it is stale. Rerun tools/faithfulness.py --technique {technique} before reporting"
        )

    # The stored Holm p must be Holm over the current family: the comparison rows of the same
    # population, or this technique's own rows when it is outside the comparison.
    from mcnemar_pairs import holm_adjust

    def family_of(population):
        block = table[table["population"] == population]
        members = block["technique"].isin(comparison) if inside else block["technique"] == technique
        return block[members]

    for population in sorted(set(rows["population"])):
        block = family_of(population)
        p = pd.to_numeric(block["wilcoxon_p"], errors="coerce")
        recomputed = pd.Series(np.nan, index=block.index)
        if p.notna().any():
            recomputed[p.notna()] = holm_adjust(p[p.notna()].to_numpy())
        own = block["technique"] == technique
        wrong = [int(block.loc[i, "k"]) for i in block.index[own]
                 if not _same_number(block.loc[i, "wilcoxon_p_holm"], recomputed[i], abs_tol=1e-300)]
        if wrong:
            raise AuditError(
                f"{_rel(path)} wilcoxon_p_holm for {technique} ({population}, k={wrong}) is not Holm over the "
                f"current family of {len(block)} rows; the table is stale. {rebuild}"
            )
    family = family_of(FAITHFULNESS_POPULATION)
    shared = rows[rows["population"] == FAITHFULNESS_POPULATION].sort_values("k")
    out = [{
        "k": int(r["k"]),
        "n_posts": int(r["n_posts"]),
        "mean_drop_topk": _num(r["mean_drop_topk"]),
        "mean_drop_random": _num(r["mean_drop_random"]),
        "drop_minus_random": _num(r["drop_minus_random"]),
        "n_flagged": int(r["n_flagged"]),
        "drop_minus_random_flagged": _num(r["drop_minus_random_flagged"]),
        "share_topk_drop_gt_random": _num(r["share_topk_drop_gt_random"]),
        "drop_over_attributed_topk": _num(r["drop_over_attributed_topk"]),
        "wilcoxon_p_holm": _num(r["wilcoxon_p_holm"]),
    } for _, r in shared.iterrows()]
    return {
        "sources": [_rel(json_path), _rel(path)],
        "status": "run",
        "population": FAITHFULNESS_POPULATION,
        "n_posts_shared": int(payload["n_posts_shared"]),
        "n_posts_shared_set": n_shared_set,
        "n_posts_explained": int(payload["n_posts_explained"]),
        "n_random_seeds": len(payload["random_seeds"]),
        "threshold": _num(payload["threshold"]),
        "verification_max_abs_diff": _num(payload["verification"]["max_abs_diff"]),
        "in_comparison": inside,
        "holm_family_size": int(len(family)),
        "rows": out,
    }


def build_report(technique, results_dir=None, rq_dir=None):
    """Assemble the report card payload from committed outputs. Raises AuditError on stale tables."""
    results_dir = RESULTS_DIR if results_dir is None else Path(results_dir)
    rq_dir = RQ_DIR if rq_dir is None else Path(rq_dir)
    if not (results_dir / technique).is_dir():
        raise AuditError(f"no result folder for {technique}")
    payload = {
        "technique": technique,
        "generated_by": "tools/audit.py report",
        "scope": "post-level labels, text-only features, one dataset, 450-row test split",
        "rq1": _rq1(technique, results_dir, rq_dir),
        "rq2": _rq2(technique, results_dir, rq_dir),
        "shap_check": _shap_check(technique, results_dir),
        "faithfulness": _faithfulness(technique, results_dir, rq_dir),
        "rq3": _rq3(technique, results_dir, rq_dir),
        "rq4": _rq4(technique, rq_dir),
    }
    sensitivity = _rq4_sensitivity(technique, rq_dir, payload["rq4"])
    payload["rq4_ambiguous_terms_sensitivity"] = sensitivity
    payload["wp7_counterfactual"] = _wp7(technique, rq_dir, _locked_threshold(technique, results_dir), sensitivity)
    assert_text_free(payload)
    return payload


def assert_text_free(node, where="report"):
    """Refuse row-keyed keys and row ids anywhere in the payload."""
    if isinstance(node, dict):
        for key, value in node.items():
            if str(key).lower() in FORBIDDEN_REPORT_KEYS:
                raise AuditError(f"{where}: key '{key}' would carry per-post data")
            assert_text_free(value, f"{where}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            assert_text_free(value, f"{where}[{i}]")
    elif isinstance(node, str) and ROW_ID_RE.search(node):
        raise AuditError(f"{where}: carries a row id")


def _fmt(value, digits=3):
    if value is None:
        return "n/a"
    if isinstance(value, float) and value != 0 and abs(value) < 10 ** (-digits):
        return f"{value:.2g}"
    return f"{value:.{digits}f}"


def _gap_phrase(technique, pair):
    gap = pair["row_gap"]
    if gap > 0:
        lead = f"{technique} is right on {gap} more test row(s)"
    elif gap < 0:
        lead = f"{pair['other_technique']} is right on {-gap} more test row(s)"
    else:
        lead = "same number of correct test rows"
    if pair["significant_holm"]:
        return f"{lead}; the difference clears Holm-corrected McNemar significance"
    return f"{lead}; not distinguishable at n = {TEST_ROWS}"


def _family_sentence(rq1):
    n = rq1["holm_family_size"]
    if rq1["in_comparison"]:
        return (f"Holm correction runs over the comparison's {n} test pairs; techniques outside the "
                "comparison (best_config.json `in_comparison: false`) are not in that family.")
    return (f"This technique is outside the comparison (best_config.json `in_comparison: false`). Its "
            f"pairs are Holm-corrected over its own {n} pairs only, so it does not move the comparison's "
            "p values.")


def _splits_phrase(splits):
    if not splits:
        return "no whole-model runs found"
    return "samples from the " + " and ".join(splits) + (" split" if len(splits) == 1 else " splits")


def _empty_lexicon_sentence(categories):
    if not categories:
        return ""
    names = ", ".join(c.replace("_", " ") for c in categories)
    return (f" The {names} lexicon is empty (RESEARCH_PLAN WP3), so words of that kind are counted as "
            "topical here.")


def _shap_lines(technique, shap):
    verdict = "passes" if shap["anchor_passes"] else "does NOT pass"
    lines = [f"Pipeline gate: the anchor `{shap['anchor']}` {verdict} its coefficient check "
             f"({shap['anchor_detail']})."]
    if shap["status"] == "no coefficients":
        lines += ["", "No `coefficients.csv`: this technique has no word-level coefficients to check "
                  "against, so its attributions rest on the anchor's pipeline check."]
        return lines
    lines += ["", f"Sources: {', '.join(f'`{s}`' for s in shap['sources'])}.", "",
              f"Status: **{shap['status']}**"
              + (f"; {shap['calibration_reason']}" if shap["status"] == "calibration point" else "")
              + ". The pass rule is Spearman at least 0.80 over all shared words and top-50 sign agreement "
              "at least 0.90; the support-filtered Spearman is a diagnostic only.", "",
              "| Run | Split | Spearman (all shared words) | Spearman (support >= min) | Top-k sign agreement "
              "| Passed |",
              "|---|---|---|---|---|---|"]
    for run in shap["runs"]:
        floor = run["secondary_min_support"]
        secondary = (f"{_fmt(run['spearman_support_ge_min'])} (>= {floor})"
                     if run["spearman_support_ge_min"] is not None else "n/a")
        lines.append(f"| `{run['run_id']}` | {run['split']} | "
                     f"{_fmt(run['spearman_mean_attribution_vs_coefficient'])} | {secondary} | "
                     f"{_fmt(run['top_k_sign_agreement'], 2)} (top {run['top_k']}) | "
                     f"{'yes' if run['passed'] else 'no'} |")
    return lines


def _rq2_reading(shap):
    if not shap["anchor_passes"]:
        return ("The anchor's SHAP check does not pass, so no cross-model attribution claim is licensed; "
                "the words below are reported only.")
    if shap["status"] == "PASS":
        return "This technique's SHAP check passed: its attributions track its own coefficients."
    if shap["status"] == "calibration point":
        return ("This technique's SHAP check is a recorded calibration point: its coefficients do not act "
                "on words, so a weaker match is expected. Read per-word claims with that in mind.")
    if shap["status"] == "FAIL":
        return ("This technique's SHAP check FAILED: its attributions do not match its own coefficients "
                "at the preregistered bar. The words below are reported, but no cross-model claim about "
                "this technique's reliance should rest on them.")
    return "No coefficients to check against; the attributions rest on the anchor's pipeline check."


def _faithfulness_lines(faith):
    lines = ["## RQ2 support. Are the explanations faithful to the model?", ""]
    if faith["status"] != "run":
        return lines + ["Not run for this technique (`tools/faithfulness.py`; needs the restorable predictor and "
                        "the per-post attribution files under `external/`: `audit.py run --with-perturbation`)."]
    if faith["in_comparison"]:
        family = f"over the {faith['holm_family_size']} `{faith['population']}` rows of the comparison techniques"
    else:
        family = (f"over this technique's own {faith['holm_family_size']} `{faith['population']}` rows, "
                  "so it does not move the comparison's p values")
    lines += [
        f"Sources: {', '.join(f'`{s}`' for s in faith['sources'])}.",
        "",
        f"Deletion test on {faith['n_posts_shared']} of the {faith['n_posts_shared_set']} test posts that every "
        f"comparison technique explained (this technique explained {faith['n_posts_explained']} in all). "
        "For each k, the k words the post's own "
        "explanation ranks highest toward EXTREMIST are deleted, against deleting as many random words of the "
        f"same post ({faith['n_random_seeds']} seeds). Drop is p(extremist) before minus after; n counts the "
        "posts with at least one positive word. The restored predictor matched the committed probabilities "
        f"within {_fmt(faith['verification_max_abs_diff'])}.",
        "",
        "| k | n | Drop, top-k | Drop, random | Margin | Margin on flagged posts (n) | Share top-k > random "
        "| Drop / summed SHAP | Wilcoxon Holm p |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in faith["rows"]:
        lines.append(
            f"| {row['k']} | {row['n_posts']} | {_fmt(row['mean_drop_topk'])} | {_fmt(row['mean_drop_random'])} | "
            f"{_fmt(row['drop_minus_random'])} | {_fmt(row['drop_minus_random_flagged'])} ({row['n_flagged']}) | "
            f"{_fmt(row['share_topk_drop_gt_random'], 2)} | {_fmt(row['drop_over_attributed_topk'], 2)} | "
            f"{_fmt(row['wilcoxon_p_holm'])} |"
        )
    lines += [
        "",
        f"Holm runs {family}. A positive margin with a small Holm p means the explanation finds words this "
        "model needs; it does not show the model is right to need them. No test compares margins across "
        "techniques, and the all-post means are capped by each model's probabilities, so compare techniques "
        "on the flagged-post margin, descriptively.",
    ]
    return lines


def _sensitivity_lines(sens):
    return [
        f"Ambiguous identity words (`{sens['sources'][0]}`, variant `{sens['variant']}`): dropping the "
        f"{sens['n_negative_excluded']} non-extremist posts whose only identity matches are ambiguous words "
        "(colour words, interjections; declared in `tools/identity_sensitivity.py`, the lexicon is unchanged) leaves "
        f"{sens['n_negative_with_identity_terms']} with an identity term (FPR "
        f"{_fmt(sens['fpr_with_identity_terms'])}) and {sens['n_negative_without_identity_terms']} without (FPR "
        f"{_fmt(sens['fpr_without_identity_terms'])}): ratio {_fmt(sens['fpr_ratio'], 2)}, Fisher exact p "
        f"{_fmt(sens['fisher_p_value'])}.",
    ]


def _wp7_lines(wp7):
    restricted = wp7["unambiguous_population"]
    if wp7["status"] != "run" and restricted["status"] != "run":
        return ["Not run for this technique (`tools/identity_counterfactual.py` and `identity_sensitivity.py "
                "counterfactual` need `external/`: `audit.py run --with-perturbation`)."]
    lines = []
    if wp7["status"] != "run":
        lines += ["The full-population counterfactual was not run (`tools/identity_counterfactual.py`).", ""]
    sources = [*(wp7["sources"][:1] if wp7["status"] == "run" else []),
               *(restricted["sources"][:1] if restricted["status"] == "run" else [])]
    lines += [
        f"Sources: {', '.join(f'`{s}`' for s in sources)}. Mean delta is p(extremist) before minus after the edit.",
        "",
        "| Population | n | Mean delta | Flagged before | Flips to not flagged | Flips to flagged "
        "| Wilcoxon Holm p | McNemar Holm p |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in [*wp7["rows"], *restricted["rows"]]:
        lines.append(
            f"| {row['population']} | {row['n_posts']} | {_fmt(row['mean_delta'], 4)} | "
            f"{row['n_flagged_original']} | {row['n_flip_pos_to_neg']} | {row['n_flip_neg_to_pos']} | "
            f"{_fmt(row['wilcoxon_p_holm'])} | {_fmt(row['mcnemar_p_holm'])} |"
        )
    if restricted["status"] == "run":
        lines += [
            "",
            "Rows marked `unambiguous match` are the posts that keep at least one identity match outside the "
            "ambiguous list, with every match removed. Their Holm p values are over the sensitivity table's own "
            "rows, not the WP7 family. A smaller population has less power; its effect size is that of the "
            "restricted posts.",
        ]
    return lines


def render_markdown(payload):
    t = payload["technique"]
    rq1, rq2, rq3, rq4, wp7 = (payload[k] for k in ("rq1", "rq2", "rq3", "rq4", "wp7_counterfactual"))
    lines = [
        f"# Report card: {t}",
        "",
        "Generated by `python3 tools/audit.py report` from committed outputs only. Do not edit by hand.",
        f"Scope: {payload['scope']}. The labels mark extremist content in single posts, not accounts "
        "or radicalization over time.",
        "",
        "## RQ1. Accuracy and which gaps are real",
        "",
        f"Sources: {', '.join(f'`{s}`' for s in rq1['sources'])}.",
        "",
        f"Test accuracy {_fmt(rq1['test_accuracy'])} on {rq1['n_test_rows']} rows. A gap of a few rows is "
        "not a difference; only a Holm-significant McNemar test counts.",
        "",
        _family_sentence(rq1),
        "",
        "| Other technique | Its accuracy | Row gap | Discordant | McNemar p | Holm p | Reading |",
        "|---|---|---|---|---|---|---|",
    ]
    for pair in rq1["pairs"]:
        lines.append(
            f"| `{pair['other_technique']}` | {_fmt(pair['other_accuracy'])} | {pair['row_gap']:+d} | "
            f"{pair['n_discordant']} | {_fmt(pair['mcnemar_p'])} | {_fmt(pair['holm_p'])} | "
            f"{_gap_phrase(t, pair)} |"
        )
    shap = payload["shap_check"]
    lines += [
        "",
        "Row gap is this technique's correct rows minus the other's.",
        "",
        "## SHAP pipeline check",
        "",
        *_shap_lines(t, shap),
        "",
        "## RQ2. What the model relies on",
        "",
        f"Sources: {', '.join(f'`{s}`' for s in rq2['sources'])}.",
        "",
        _rq2_reading(shap),
        "",
        f"Top {len(rq2['top_words_by_positive_mean_attribution'])} words with positive mean attribution "
        f"(toward EXTREMIST) in the test run `{rq2['run_id']}` ({rq2['n_posts_explained']} posts "
        "explained). Category comes from the committed lexicons."
        + _empty_lexicon_sentence(rq2["empty_lexicon_categories"]),
        "",
        "| Rank | Word | Category | Mean attribution | Support (posts) |",
        "|---|---|---|---|---|",
    ]
    for rank, word in enumerate(rq2["top_words_by_positive_mean_attribution"], start=1):
        lines.append(f"| {rank} | `{word['word']}` | {word['category']} | "
                     f"{_fmt(word['mean_attribution'], 4)} | {word['support']} |")
    top_k = rq2["agreement"][0]["top_k"]
    lines += [
        "",
        f"Word-level agreement with the other techniques' whole-model test runs (Spearman over shared "
        f"words on mean attribution; Jaccard over the top-{top_k} words by positive attribution). "
        f"By Spearman, most agreeing `{rq2['most_agreeing_by_spearman']}`, least agreeing "
        f"`{rq2['least_agreeing_by_spearman']}`. By top-{top_k} Jaccard, most agreeing "
        f"`{rq2['most_agreeing_by_jaccard']}`, least agreeing `{rq2['least_agreeing_by_jaccard']}`. "
        "The table is sorted by Spearman.",
        "",
        f"| Other technique | Spearman | Top-{top_k} Jaccard |",
        "|---|---|---|",
    ]
    for pair in rq2["agreement"]:
        lines.append(f"| `{pair['other_technique']}` | {_fmt(pair['spearman_mean_attribution'])} | "
                     f"{_fmt(pair['jaccard_top_k_positive'])} |")
    lines += ["", *_faithfulness_lines(payload["faithfulness"])]
    lines += [
        "",
        "## RQ3. Stability of the explanation across post samples",
        "",
        f"Source: `{rq3['sources'][0]}`.",
        "",
        f"Across {rq3['n_runs']} whole-model runs ({_splits_phrase(rq3['run_splits'])}): mean pairwise Spearman "
        f"on mean absolute attribution {_fmt(rq3['mean_pairwise_spearman_abs_attribution'])}, top-50 "
        f"Jaccard {_fmt(rq3['mean_pairwise_jaccard_top50'])}. Low overlap means per-word claims about "
        "this model are fragile.",
        "",
        "## RQ4. Identity terms and false positives",
        "",
        f"Source: `{rq4['sources'][0]}`.",
        "",
        "| Non-extremist test posts | n | False-positive rate |",
        "|---|---|---|",
        f"| with an identity term | {rq4['n_negative_with_identity_terms']} | {_fmt(rq4['fpr_with_identity_terms'])} |",
        f"| without | {rq4['n_negative_without_identity_terms']} | {_fmt(rq4['fpr_without_identity_terms'])} |",
        "",
        f"Ratio {_fmt(rq4['fpr_ratio'], 2)}, Fisher exact p {_fmt(rq4['fisher_p_value'])}, at the locked "
        f"threshold {rq4['threshold']}. This is a correlation; the counterfactual below is the intervention.",
        "",
        *_sensitivity_lines(payload["rq4_ambiguous_terms_sensitivity"]),
        "",
        "## WP7. Identity counterfactual (remove the term)",
        "",
        *_wp7_lines(wp7),
    ]
    lines += [
        "",
        "## Limits",
        "",
        f"Post-level labels and text-only features on one dataset; the test split has {TEST_ROWS} rows. "
        "About 14 rows is the rough floor for an uncorrected McNemar test; after Holm correction over "
        "the family the floor is higher, and only the Holm column decides. Attributions describe the "
        "model, not the language, and are not causal claims.",
        "",
    ]
    return "\n".join(lines)


def write_payload(payload, out_dir=None):
    """Write one card's Markdown and JSON. Returns (md_path, json_path)."""
    assert_text_free(payload)
    technique = payload["technique"]
    out_dir = REPORT_CARDS_DIR if out_dir is None else Path(out_dir)
    markdown = render_markdown(payload)
    if ROW_ID_RE.search(markdown):
        raise AuditError(f"{technique}: rendered report card carries a row id")
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / REPORT_CARD_MD_FILENAME.format(technique=technique)
    json_path = out_dir / REPORT_CARD_JSON_FILENAME.format(technique=technique)
    md_path.write_text(markdown)
    with open(json_path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return md_path, json_path


def write_reports(techniques, results_dir=None, rq_dir=None, out_dir=None):
    """Build every payload before writing any, so a stale table writes nothing."""
    payloads = [build_report(t, results_dir, rq_dir) for t in techniques]
    return [write_payload(payload, out_dir) for payload in payloads]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check-contract", help="report each contract item")
    check.add_argument("--technique", required=True)

    run = sub.add_parser("run", help="check the contract, run the chain, write the report cards")
    run.add_argument("--technique", required=True)
    run.add_argument("--with-perturbation", "--with-counterfactual", dest="with_perturbation",
                     action="store_true",
                     help="also run the stages that restore the predictor from external/<TECHNIQUE>/: "
                          "identity_counterfactual, identity_sensitivity counterfactual, faithfulness")
    run.add_argument("--device", default="auto")
    run.add_argument("--sequential-checkpoints", action="store_true")

    report = sub.add_parser("report", help="write text-free report cards from committed outputs")
    group = report.add_mutually_exclusive_group(required=True)
    group.add_argument("--technique")
    group.add_argument("--all", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "check-contract":
        items = check_contract(args.technique)
        return EXIT_OK if print_contract(args.technique, items) else EXIT_FAILED

    if args.command == "run":
        ctx = Context(args.technique, args.with_perturbation, args.device, args.sequential_checkpoints)
        ok, _ = run_chain(ctx)
        if not ok:
            return EXIT_FAILED
        shortfall = perturbation_shortfall(ctx)
        if shortfall:
            print(f"--with-perturbation was given but {', '.join(n for n, _ in shortfall)} did not run; "
                  f"exit {EXIT_PERTURBATION_SKIPPED}")
            return EXIT_PERTURBATION_SKIPPED
        return EXIT_OK

    techniques = [args.technique] if args.technique else [d.name for d in technique_dirs()]
    try:
        written = write_reports(techniques)
    except AuditError as error:
        print(f"ERROR: {error}; no report card was written", file=sys.stderr)
        return EXIT_FAILED
    except Exception as error:  # noqa: BLE001 - a malformed input must not escape as a traceback
        print(f"ERROR: unexpected {type(error).__name__} while building a report card: {error}; "
              "no report card was written", file=sys.stderr)
        return EXIT_FAILED
    for md_path, json_path in written:
        print(f"wrote {_rel(md_path)} and {_rel(json_path)}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
