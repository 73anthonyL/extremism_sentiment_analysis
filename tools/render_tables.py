#!/usr/bin/env python3
"""Render every documentation results table from results_summary/.

Doc tables are rendered, not edited. The repository keeps the same metrics
table in several documents (README, docs/EXPERIMENTS.md, docs/MODEL_CARD.md,
results_summary/README.md), and hand-transcribing numbers into them is exactly
how published tables drift from the artifacts that back them. This tool makes
results_summary/<TECHNIQUE>/metrics_*.json the single source: it regenerates
each marked table region from those files, so a digit can only change in a doc
because the underlying committed artifact changed.

MARKER CONVENTION
-----------------
A rendered region is delimited by a pair of HTML comments, which survive
Markdown rendering invisibly:

    <!-- RENDERED-TABLE:BEGIN id=<table_id> -->
    ...machine-owned content...
    <!-- RENDERED-TABLE:END id=<table_id> -->

Everything between the markers is owned by this tool and is rewritten
wholesale on --write; hand edits inside a region are deliberately clobbered.
The same table_id may appear in several files and renders identically in all
of them.

TABLE IDS
---------
main-comparison   technique, validation accuracy, test accuracy, test
                  balanced accuracy, test macro F1, test ROC-AUC
test-detail       held-out test metrics: accuracy, positive-class F1 /
                  precision / recall, ROC-AUC, PR-AUC, threshold
confusion-test    test confusion-matrix counts and error rates

Techniques come from repo_paths.technique_dirs(), sorted by id. A technique
folder without metrics_test.json (test not yet unlocked) is skipped with a
warning rather than rendered blank, so tables only ever show completed,
comparable techniques.

USAGE
-----
python3 tools/render_tables.py --check   # exit 1 if any doc table drifts
python3 tools/render_tables.py --write   # regenerate all marked regions
"""

import argparse
import difflib
import json
import re
import sys

import repo_paths

METRIC_DECIMALS = 4

PROVENANCE_LINE = (
    "Rendered by tools/render_tables.py from results_summary/ "
    "— do not edit by hand."
)

MARKER_TOKEN = "RENDERED-TABLE"
REGION_RE = re.compile(
    r"<!-- RENDERED-TABLE:BEGIN id=(?P<id>[A-Za-z0-9_.-]+) -->"
    r"(?P<body>.*?)"
    r"<!-- RENDERED-TABLE:END id=(?P=id) -->",
    re.DOTALL,
)

VALIDATION_METRICS_FILE = "metrics_validation.json"
TEST_METRICS_FILE = "metrics_test.json"


def fmt(value):
    return f"{value:.{METRIC_DECIMALS}f}"


def fmt_threshold(value):
    # Thresholds are exact configuration values (0.45, 0.72, possibly 0.585);
    # fixed-width padding would misstate their precision.
    return format(value, "g")


def macro_f1(metrics):
    """The schema allows either spelling (macro_f1 for classical folders,
    f1_macro for the transformer folder); accept both, refuse neither."""
    for key in ("macro_f1", "f1_macro"):
        if key in metrics:
            return metrics[key]
    raise SystemExit(f"error: metrics file has neither macro_f1 nor f1_macro")


def load_techniques():
    """Load per-technique validation and test metrics from results_summary/.

    Returns a list of (technique_id, validation_metrics, test_metrics)
    sorted by technique id. Folders missing metrics_test.json are skipped
    with a warning: an unlocked-test technique is a legitimate state, but it
    has no place in a held-out comparison table.
    """
    rows = []
    for tech_dir in sorted(repo_paths.technique_dirs(), key=lambda p: p.name):
        val_path = tech_dir / VALIDATION_METRICS_FILE
        test_path = tech_dir / TEST_METRICS_FILE
        if not test_path.is_file() or not val_path.is_file():
            print(
                f"warning: skipping {tech_dir.name}: missing "
                f"{VALIDATION_METRICS_FILE if not val_path.is_file() else TEST_METRICS_FILE}",
                file=sys.stderr,
            )
            continue
        with open(val_path, encoding="utf-8") as f:
            val = json.load(f)
        with open(test_path, encoding="utf-8") as f:
            test = json.load(f)
        rows.append((tech_dir.name, val, test))
    if not rows:
        raise SystemExit("error: no technique with complete metrics found")
    return rows


def render_main_comparison(rows):
    lines = [
        "| Technique | Validation accuracy | Test accuracy "
        "| Test balanced accuracy | Test macro F1 | Test ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for tech, val, test in rows:
        lines.append(
            f"| `{tech}` | {fmt(val['accuracy'])} | {fmt(test['accuracy'])} "
            f"| {fmt(test['balanced_accuracy'])} | {fmt(macro_f1(test))} "
            f"| {fmt(test['roc_auc'])} |"
        )
    return "\n".join(lines)


def render_test_detail(rows):
    lines = [
        "| Technique | Accuracy | Positive F1 | Positive precision "
        "| Positive recall | ROC-AUC | PR-AUC | Threshold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tech, _val, test in rows:
        lines.append(
            f"| `{tech}` | {fmt(test['accuracy'])} | {fmt(test['positive_f1'])} "
            f"| {fmt(test['positive_precision'])} | {fmt(test['positive_recall'])} "
            f"| {fmt(test['roc_auc'])} | {fmt(test['pr_auc'])} "
            f"| {fmt_threshold(test['threshold'])} |"
        )
    return "\n".join(lines)


def render_confusion_test(rows):
    lines = [
        "| Technique | TN | FP | FN | TP | FPR | FNR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for tech, _val, test in rows:
        lines.append(
            f"| `{tech}` | {test['tn']} | {test['fp']} | {test['fn']} "
            f"| {test['tp']} | {fmt(test['false_positive_rate'])} "
            f"| {fmt(test['false_negative_rate'])} |"
        )
    return "\n".join(lines)


TABLE_RENDERERS = {
    "main-comparison": render_main_comparison,
    "test-detail": render_test_detail,
    "confusion-test": render_confusion_test,
}


def documentation_files():
    """The surfaces scanned for rendered regions: the top-level README,
    every docs/*.md, and results_summary/README.md."""
    files = [repo_paths.REPO_ROOT / "README.md"]
    files.extend(sorted(repo_paths.DOCS_DIR.glob("*.md")))
    files.append(repo_paths.RESULTS_DIR / "README.md")
    return [f for f in files if f.is_file()]


def rewrite_content(content, rows, rel_path):
    """Return (new_content, region_ids) with every marked region re-rendered.

    Errors out on an unknown table id or on marker text that the region
    regex did not consume (an unpaired or typo'd marker would otherwise be
    silently ignored — the failure mode this tool exists to prevent).
    """
    region_ids = []

    def replace(match):
        table_id = match.group("id")
        renderer = TABLE_RENDERERS.get(table_id)
        if renderer is None:
            raise SystemExit(
                f"error: {rel_path}: unknown table id {table_id!r} "
                f"(known: {', '.join(sorted(TABLE_RENDERERS))})"
            )
        region_ids.append(table_id)
        body = f"\n{renderer(rows)}\n\n{PROVENANCE_LINE}\n"
        return (
            f"<!-- RENDERED-TABLE:BEGIN id={table_id} -->{body}"
            f"<!-- RENDERED-TABLE:END id={table_id} -->"
        )

    new_content = REGION_RE.sub(replace, content)
    n_markers = content.count(MARKER_TOKEN)
    if n_markers != 2 * len(region_ids):
        raise SystemExit(
            f"error: {rel_path}: found {n_markers} {MARKER_TOKEN} marker(s) "
            f"but matched {len(region_ids)} complete region(s); "
            "check for an unpaired or malformed marker"
        )
    return new_content, region_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check",
        action="store_true",
        help="regenerate in memory and exit 1 if any doc region differs",
    )
    mode.add_argument(
        "--write",
        action="store_true",
        help="rewrite every marked region in place",
    )
    args = parser.parse_args()

    rows = load_techniques()
    drifted = []
    total_regions = 0

    for path in documentation_files():
        rel_path = path.relative_to(repo_paths.REPO_ROOT)
        content = path.read_text(encoding="utf-8")
        new_content, region_ids = rewrite_content(content, rows, rel_path)
        total_regions += len(region_ids)
        if not region_ids or new_content == content:
            continue
        if args.write:
            path.write_text(new_content, encoding="utf-8")
            print(f"rendered {rel_path}: {', '.join(region_ids)}")
        else:
            drifted.append((rel_path, region_ids, content, new_content))

    if total_regions == 0:
        # No markers anywhere means the drift check is checking nothing;
        # that is itself a broken state, not a pass.
        raise SystemExit("error: no RENDERED-TABLE regions found in any document")

    if args.check:
        if drifted:
            print(f"DRIFT: {len(drifted)} file(s) differ from results_summary/")
            for rel_path, region_ids, old, new in drifted:
                print(f"  - {rel_path}: regions {', '.join(region_ids)}")
                diff = difflib.unified_diff(
                    old.splitlines(keepends=True),
                    new.splitlines(keepends=True),
                    fromfile=f"{rel_path} (committed)",
                    tofile=f"{rel_path} (rendered)",
                )
                sys.stdout.writelines(diff)
            return 1
        print(f"OK: {total_regions} rendered region(s) match results_summary/")
        return 0

    print(f"OK: {total_regions} rendered region(s) up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
