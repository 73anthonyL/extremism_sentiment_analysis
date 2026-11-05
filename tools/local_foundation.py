"""Build the notebook-00 inputs that nk.load_foundation needs, for a local run.

On Kaggle a technique notebook reads notebook 00's `research_foundation/`
output. Running the kit locally (a script or a notebook outside Kaggle), that
folder does not exist, because `processed_dataset.csv` carries post text and
is never committed. This tool rebuilds it from `data/dataset.csv` exactly as
notebook 00 does (`split_protocol.build_processed_dataset`), copies the
committed split mirror and dataset manifest next to it, and checks that the
three agree row for row before writing anything.

The output holds dataset text, so it must live outside the repository or
under `external/` (gitignored). Any other path inside the repository is
refused.

USAGE
-----
    python3 tools/local_foundation.py --out /path/outside/the/repo/kit_inputs
    python3 tools/local_foundation.py --out external/kit_inputs

Then, in the script or notebook:

    ctx = nk.bootstrap(CONFIG, working_root="<scratch>/kit_output",
                       input_roots=["<the --out folder>"])
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from repo_paths import DATASET_MANIFEST_JSON, EXTERNAL_DIR, REPO_ROOT, SPLIT_ASSIGNMENTS_CSV  # noqa: E402

FOUNDATION_FOLDER = "research_foundation"


class LocalFoundationError(RuntimeError):
    pass


def check_out_dir(out_dir):
    """Refuse a destination inside the repository unless it is under external/."""
    out_dir = Path(out_dir).resolve()
    inside_repo = out_dir == REPO_ROOT or REPO_ROOT in out_dir.parents
    under_external = out_dir == EXTERNAL_DIR.resolve() or EXTERNAL_DIR.resolve() in out_dir.parents
    if inside_repo and not under_external:
        raise LocalFoundationError(
            f"{out_dir} is inside the repository; processed_dataset.csv carries post text and may only be "
            f"written outside it or under {EXTERNAL_DIR.relative_to(REPO_ROOT)}/"
        )
    return out_dir


def build(out_dir, split_csv=SPLIT_ASSIGNMENTS_CSV, manifest_json=DATASET_MANIFEST_JSON, processed=None):
    """Write <out_dir>/research_foundation/{processed_dataset.csv, split_assignments.csv, dataset_manifest.json}."""
    from split_protocol import build_processed_dataset

    out_dir = check_out_dir(out_dir)
    processed = build_processed_dataset() if processed is None else processed
    assignments = pd.read_csv(split_csv)
    joined = processed.merge(assignments, on="row_id", how="outer", suffixes=("", "_split"), indicator=True)
    if (joined["_merge"] != "both").any():
        raise LocalFoundationError("the rebuilt processed dataset and the split mirror cover different row ids")
    if (joined["text_hash"] != joined["text_hash_split"]).any() or (joined["label"] != joined["label_split"]).any():
        raise LocalFoundationError("the rebuilt processed dataset disagrees with the split mirror on text hash "
                                   "or label; data/dataset.csv is not the committed version")

    folder = out_dir / FOUNDATION_FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    processed.to_csv(folder / "processed_dataset.csv", index=False)
    shutil.copyfile(split_csv, folder / "split_assignments.csv")
    shutil.copyfile(manifest_json, folder / "dataset_manifest.json")
    return folder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="outside the repository, or under external/")
    args = parser.parse_args(argv)
    try:
        folder = build(args.out)
    except LocalFoundationError as error:
        print(f"ERROR: {error}; nothing was written", file=sys.stderr)
        return 1
    print(f"wrote {folder} (holds post text: never commit it)")
    print(f"nk.bootstrap(CONFIG, working_root=<scratch>/kit_output, input_roots=[{str(folder.parent)!r}])")
    return 0


if __name__ == "__main__":
    sys.exit(main())
