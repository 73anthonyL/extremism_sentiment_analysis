"""Repair the committed split mirror from the deterministic frozen assignment.

WHY THIS EXISTS
---------------
`splits/split_assignments.csv` is documented as the canonical split every notebook
must reuse. The committed file is a stale pre-correction artifact: git shows
`db04968e "split assignments"` (2026-06-20) predates
`5b70a7a6 "replace old dataset with corrected dataset"` (2026-06-27). Measured
against the current dataset it disagrees on 728 labels and assigns 1420 of 2999
rows to a different split, and its class counts (test 240/210) contradict the
published `results_summary/foundation/split_label_distribution.csv` (test 280/170).

In that state the repository cannot reproduce its own published metrics.

WHY REPAIR IS NOT "REGENERATING THE SPLIT"
------------------------------------------
The split *assignment* is frozen and derivable: notebook 00 builds it from
`data/dataset.csv` with seed 30 and a fixed stratified 70/15/15 recipe. The
committed CSV is a mirror of that computation. This script recomputes the
assignment and rewrites the mirror only after confirming it reproduces the
published foundation counts exactly. If verification fails, nothing is written --
the reconstruction is rejected rather than the foundation being overwritten.

USAGE
-----
    python3 tools/repair_split_mirror.py            # read-only diff, exit 1 if drifted
    python3 tools/repair_split_mirror.py --write    # rewrite after verification

`--write` backs the current file up to `splits/split_assignments.PRE-REPAIR.csv`.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from repo_paths import SPLIT_ASSIGNMENTS_CSV
from split_protocol import reconstruct_split_assignments, split_counts, verify_assignments

# Column order of the committed mirror. Kept explicit so a repair writes
# exactly the column layout the notebooks and validators already expect,
# even if the reconstruction frame happens to carry extra working columns
# alongside the four the mirror is defined by.
MIRROR_COLUMNS = ["row_id", "label", "text_hash", "split"]
BACKUP_PATH = SPLIT_ASSIGNMENTS_CSV.with_name("split_assignments.PRE-REPAIR.csv")

EXIT_OK = 0
EXIT_DRIFTED = 1
EXIT_RECONSTRUCTION_FAILED = 2


def compare_to_committed(reconstructed):
    """Diff the reconstruction against the committed mirror.

    Returns a summary dict; `joined` may be less than the row count if the two
    files disagree about which row_ids exist at all.
    """
    if not SPLIT_ASSIGNMENTS_CSV.exists():
        return {"exists": False}

    committed = pd.read_csv(SPLIT_ASSIGNMENTS_CSV)
    merged = reconstructed.merge(committed, on="row_id", suffixes=("_new", "_old"))
    return {
        "exists": True,
        "committed_rows": int(len(committed)),
        "joined": int(len(merged)),
        "label_disagreements": int((merged["label_new"] != merged["label_old"]).sum()),
        "split_disagreements": int((merged["split_new"] != merged["split_old"]).sum()),
        "committed_counts": split_counts(committed),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="rewrite the mirror (only if the reconstruction verifies)",
    )
    args = parser.parse_args()

    reconstructed = reconstruct_split_assignments()
    verified, problems = verify_assignments(reconstructed)

    print("Reconstructed split assignment from data/dataset.csv (seed 30)")
    print(f"  counts: {split_counts(reconstructed)}")
    print(f"  verifies against foundation/: {verified}")
    for problem in problems:
        print(f"  PROBLEM: {problem}")

    if not verified:
        print(
            "\nREFUSING TO WRITE: the reconstruction does not reproduce the published"
            " foundation counts. The dataset or the recipe has changed; investigate"
            " before touching the mirror.",
            file=sys.stderr,
        )
        return EXIT_RECONSTRUCTION_FAILED

    diff = compare_to_committed(reconstructed)
    if not diff["exists"]:
        print("\nNo committed mirror present.")
    else:
        print("\nCommitted mirror vs reconstruction:")
        print(f"  committed counts:     {diff['committed_counts']}")
        print(f"  rows joined:          {diff['joined']} of {diff['committed_rows']}")
        print(f"  label disagreements:  {diff['label_disagreements']}")
        print(f"  split disagreements:  {diff['split_disagreements']}")

    drifted = not diff["exists"] or diff["split_disagreements"] > 0 or diff[
        "label_disagreements"
    ] > 0

    if not drifted:
        print("\nMirror already matches the frozen assignment. Nothing to do.")
        return EXIT_OK

    if not args.write:
        print("\nMirror is DRIFTED. Re-run with --write to repair.")
        return EXIT_DRIFTED

    if SPLIT_ASSIGNMENTS_CSV.exists():
        BACKUP_PATH.write_bytes(SPLIT_ASSIGNMENTS_CSV.read_bytes())
        print(f"\nBacked up current mirror to {BACKUP_PATH}")

    reconstructed[MIRROR_COLUMNS].to_csv(SPLIT_ASSIGNMENTS_CSV, index=False)
    print(f"Wrote repaired mirror to {SPLIT_ASSIGNMENTS_CSV}")

    written = pd.read_csv(SPLIT_ASSIGNMENTS_CSV)
    rewritten_ok, rewritten_problems = verify_assignments(written)
    print(f"Post-write verification: {rewritten_ok}")
    for problem in rewritten_problems:
        print(f"  PROBLEM: {problem}")
    return EXIT_OK if rewritten_ok else EXIT_RECONSTRUCTION_FAILED


if __name__ == "__main__":
    sys.exit(main())
