"""Audit the research-protocol invariants that make published results comparable.

Each check corresponds to a way the repository has actually drifted or could
drift silently. A check that has no deterministic implementation is reported as
NOT VERIFIABLE rather than assumed to pass, so the gap is visible.

USAGE
-----
    python3 tools/protocol_check.py --all
    python3 tools/protocol_check.py --splits --naming
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from repo_paths import (
    DATASET_VERSION,
    NOTEBOOKS_DIR,
    RESULTS_DIR,
    SPLIT_ASSIGNMENTS_CSV,
    SPLIT_VERSION,
    technique_dirs,
)
from split_protocol import verify_assignments

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_UNVERIFIABLE = "NOT VERIFIABLE"

EXIT_OK = 0
EXIT_FAILED = 1


def check_splits():
    """The committed split mirror must match the frozen assignment exactly."""
    if not SPLIT_ASSIGNMENTS_CSV.exists():
        return STATUS_FAIL, [f"{SPLIT_ASSIGNMENTS_CSV} does not exist"]
    assignments = pd.read_csv(SPLIT_ASSIGNMENTS_CSV)
    ok, problems = verify_assignments(assignments)
    if not ok:
        problems = problems + [
            "Run `python3 tools/repair_split_mirror.py` to inspect, then --write to repair."
        ]
    return (STATUS_PASS if ok else STATUS_FAIL), problems


def check_naming():
    """Notebook filename, results folder, and CONFIG technique_name must agree.

    Every notebook omits the numeric prefix in CONFIG (`LOG-REG_TF-IDF` for
    `01_LOG-REG_TF-IDF.ipynb`), so the prefix is stripped before comparing. What
    remains flags real divergence: token reordering (04, 05) and a technique_name
    that names a different model than its own filename (08).
    """
    problems = []
    for notebook in sorted(NOTEBOOKS_DIR.glob("[0-9][0-9]_*.ipynb")):
        stem = notebook.stem
        _, _, stem_without_prefix = stem.partition("_")
        try:
            payload = json.loads(notebook.read_text())
        except (OSError, json.JSONDecodeError) as error:
            problems.append(f"{notebook.name}: unreadable ({error})")
            continue

        declared = _extract_config_string(payload, "technique_name")
        if declared is None:
            continue
        if declared not in (stem, stem_without_prefix):
            problems.append(
                f"{notebook.name}: CONFIG technique_name is '{declared}' but the filename implies '"
                f"{stem_without_prefix}'"
            )
    return (STATUS_PASS if not problems else STATUS_FAIL), problems


def check_versions():
    """Every notebook must declare the frozen dataset and split version strings."""
    problems = []
    for notebook in sorted(NOTEBOOKS_DIR.glob("[0-9][0-9]_*.ipynb")):
        try:
            payload = json.loads(notebook.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        split_version = _extract_config_string(payload, "split_version")
        if split_version is not None and split_version != SPLIT_VERSION:
            problems.append(
                f"{notebook.name}: split_version is '{split_version}', expected '"
                f"{SPLIT_VERSION}'"
            )
        dataset_version = _extract_config_string(payload, "dataset_version")
        if dataset_version is not None and dataset_version not in (
            DATASET_VERSION,
            "extremism_dataset_v1",
        ):
            problems.append(
                f"{notebook.name}: dataset_version is '{dataset_version}', expected '"
                f"{DATASET_VERSION}'"
            )
    return (STATUS_PASS if not problems else STATUS_FAIL), problems


def check_notebook_outputs():
    """Committed notebooks must not carry saved cell outputs.

    Saved outputs are how dataset text, row ids, and stale metrics enter version
    control. Stripping them is also what makes notebook diffs reviewable.
    """
    problems = []
    for notebook in sorted(NOTEBOOKS_DIR.glob("*.ipynb")):
        try:
            payload = json.loads(notebook.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        with_outputs = sum(
            1
            for cell in payload.get("cells", [])
            if cell.get("cell_type") == "code" and cell.get("outputs")
        )
        if with_outputs:
            problems.append(f"{notebook.name}: {with_outputs} code cell(s) have saved outputs")
    return (STATUS_PASS if not problems else STATUS_FAIL), problems


def check_text_preview_disabled():
    """error_analysis.include_text_preview must be False in every notebook."""
    problems = []
    for notebook in sorted(NOTEBOOKS_DIR.glob("*.ipynb")):
        try:
            source = notebook.read_text()
        except OSError:
            continue
        if '"include_text_preview": True' in source or "'include_text_preview': True" in source:
            problems.append(
                f"{notebook.name}: include_text_preview is True; result artifacts would embed raw dataset text"
            )

    return (STATUS_PASS if not problems else STATUS_FAIL), problems


def check_threshold_policy():
    """No result folder may record the test split as informing selection."""
    problems = []
    for folder in technique_dirs():
        config_path = folder / "best_config.json"
        if not config_path.exists():
            continue
        with open(config_path) as handle:
            config = json.load(handle)
        if config.get("test_set_used_for_selection") is True:
            problems.append(f"{folder.name}: test_set_used_for_selection is true")
        selected_on = config.get("threshold_selected_on_split")
        if selected_on not in (None, "validation"):
            problems.append(
                f"{folder.name}: threshold selected on '{selected_on}', not validation"
            )
    return (STATUS_PASS if not problems else STATUS_FAIL), problems


def check_results_coverage():
    """Every numbered notebook should have a matching results folder, and vice versa."""
    problems = []
    notebook_ids = {
        n.stem for n in NOTEBOOKS_DIR.glob("[0-9][0-9]_*.ipynb") if not n.stem.startswith("00")
    }
    folder_ids = {f.name for f in technique_dirs()}

    for missing in sorted(notebook_ids - folder_ids):
        problems.append(f"notebook '{missing}' has no results_summary/ folder")
    for orphan in sorted(folder_ids - notebook_ids):
        problems.append(f"results folder '{orphan}' has no matching notebook")
    return (STATUS_PASS if not problems else STATUS_FAIL), problems


def _extract_config_string(notebook_payload, key):
    """Pull a quoted string value for `key` out of a notebook's CONFIG source.

    Notebooks are read as text rather than executed, so this is a targeted scan
    for `"key": "value"` inside code cells. Returns None when the key is absent.
    """
    needle = f'"{key}":'
    for cell in notebook_payload.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for line in cell.get("source", []):
            stripped = line.strip()
            if not stripped.startswith(needle):
                continue
            remainder = stripped[len(needle):].strip().rstrip(",")
            if remainder.startswith('"') and remainder.endswith('"') and len(remainder) > 1:
                return remainder[1:-1]
    return None


CHECKS = {
    "splits": ("Split mirror matches the frozen assignment", check_splits),
    "naming": ("Notebook / folder / technique_name agree", check_naming),
    "versions": ("Frozen dataset and split version strings", check_versions),
    "outputs": ("Notebooks carry no saved cell outputs", check_notebook_outputs),
    "text-preview": ("include_text_preview disabled everywhere", check_text_preview_disabled),
    "threshold-policy": ("Thresholds selected on validation only", check_threshold_policy),
    "coverage": ("Notebooks and result folders correspond", check_results_coverage),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="run every check")
    for name in CHECKS:
        parser.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args()

    selected = [name for name in CHECKS if getattr(args, name.replace("-", "_"))]
    if args.all or not selected:
        selected = list(CHECKS)

    failures = 0
    print(f"{'INVARIANT':<46} STATUS")
    print("-" * 70)
    details = []
    for name in selected:
        description, check = CHECKS[name]
        status, problems = check()
        print(f"{description:<46} {status}")
        if status == STATUS_FAIL:
            failures += 1
            details.append((description, problems))

    if details:
        print()
        for description, problems in details:
            print(f"{description}:")
            for problem in problems:
                print(f"  - {problem}")

    print()
    print(f"GATE: {'PASS' if failures == 0 else f'FAIL ({failures} blocking)'}")
    return EXIT_OK if failures == 0 else EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
