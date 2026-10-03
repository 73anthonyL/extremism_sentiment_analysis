"""Reconcile committed and external artifacts for one technique run.

Each results_summary/<TECHNIQUE>/ carries a run_manifest.json that says where
everything the run produced now lives:

* `source`: the Kaggle notebook and version the run came from, and the git
  commit of the notebook code that ran, so a result can be traced to code.
* `committed_artifacts`: sha256 of every file in the technique folder at the
  time of the last `refresh`, so `check` detects silent edits.
* `external_assets`: weights, checkpoints, training logs, per-post attribution
  files, anything with text or too large for git, each with a location URI,
  hash, size, and a `contains_text` flag. An asset that contains dataset text
  must not point inside the repository.

USAGE
-----
    python3 tools/run_manifest.py init --technique T --run-id R \
        --kaggle-notebook https://www.kaggle.com/code/<user>/<slug> --kaggle-version 7 \
        --git-commit <sha>
    python3 tools/run_manifest.py add-asset --technique T --kind weights \
        --location kaggle://datasets/<user>/<slug>/v3/model.safetensors \
        --sha256 <hex> --size-bytes 498000000
    python3 tools/run_manifest.py add-assets-from --technique T \
        --file external_assets.json --location-prefix kaggle://datasets/<user>/<slug>/v1
    python3 tools/run_manifest.py refresh --technique T     # re-hash committed files
    python3 tools/run_manifest.py check --all               # drift and policy check
    python3 tools/run_manifest.py show --technique T
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from repo_paths import REPO_ROOT, RESULTS_DIR, RUN_MANIFEST_FILENAME, technique_dirs

ASSET_KINDS = (
    "weights",
    "checkpoint",
    "tokenizer",
    "training_log",
    "local_attributions",
    "predictions_with_text",
    "error_review_queue",
    "notebook_export",
    "other",
)
# Location schemes that mean "outside this repository".
EXTERNAL_SCHEMES = ("kaggle://", "gdrive://", "https://", "s3://", "gs://", "hf://")
REPO_SCHEMES = ("github.com", "raw.githubusercontent.com")

EXIT_OK = 0
EXIT_FAILED = 1


class ManifestError(RuntimeError):
    pass


def manifest_path(technique, results_dir=None):
    base = RESULTS_DIR if results_dir is None else Path(results_dir)
    return base / technique / RUN_MANIFEST_FILENAME


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_committed(technique, results_dir=None):
    """{relative path: sha256} for every file in the technique folder except the manifest."""
    base = RESULTS_DIR if results_dir is None else Path(results_dir)
    folder = base / technique
    if not folder.exists():
        raise ManifestError(f"no results folder for {technique}")
    hashes = {}
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        rel = path.relative_to(folder).as_posix()
        if rel == RUN_MANIFEST_FILENAME:
            continue
        hashes[rel] = sha256_file(path)
    return hashes


def read_manifest(technique, results_dir=None):
    path = manifest_path(technique, results_dir)
    if not path.exists():
        raise ManifestError(f"no manifest at {path}; run `init` first")
    with open(path) as handle:
        return json.load(handle)


def write_manifest(technique, manifest, results_dir=None):
    path = manifest_path(technique, results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return path


def _now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def init_manifest(technique, run_id, kaggle_notebook, kaggle_version, git_commit, notes=None,
                  results_dir=None):
    if manifest_path(technique, results_dir).exists():
        raise ManifestError(f"manifest for {technique} already exists; edit it or use refresh/add-asset")
    manifest = {
        "technique": technique,
        "run_id": run_id,
        "created_utc": _now(),
        "source": {
            "kaggle_notebook": kaggle_notebook,
            "kaggle_version": kaggle_version,
            "git_commit": git_commit,
            "notes": notes,
        },
        "committed_artifacts": hash_committed(technique, results_dir),
        "committed_hashed_utc": _now(),
        "external_assets": [],
    }
    return write_manifest(technique, manifest, results_dir)


def add_asset(technique, kind, location, sha256, size_bytes, contains_text=False,
              description=None, results_dir=None):
    if kind not in ASSET_KINDS:
        raise ManifestError(f"kind '{kind}' not in {ASSET_KINDS}")
    problems = asset_problems({"location": location, "contains_text": contains_text, "sha256": sha256})
    if problems:
        raise ManifestError("; ".join(problems))
    manifest = read_manifest(technique, results_dir)
    if any(a["location"] == location for a in manifest["external_assets"]):
        raise ManifestError(f"an asset at {location} is already recorded")
    manifest["external_assets"].append(
        {
            "kind": kind,
            "location": location,
            "sha256": sha256,
            "size_bytes": int(size_bytes),
            "contains_text": bool(contains_text),
            "description": description,
            "added_utc": _now(),
        }
    )
    return write_manifest(technique, manifest, results_dir)


def add_assets_from(technique, inventory_path, location_prefix, results_dir=None):
    """Record every asset listed in a notebook's external_assets.json.

    `tools/notebook_kit.py::finalize` writes that inventory with each file's
    kind, relative path, sha256, size and text flag, so the hashes are never
    retyped. The location of each asset is `location_prefix` joined with its
    relative path. Every entry is validated before any is written: one bad
    entry records nothing. Returns the number of assets added.
    """
    with open(inventory_path) as handle:
        inventory = json.load(handle)
    if inventory.get("technique") != technique:
        raise ManifestError(
            f"inventory is for '{inventory.get('technique')}', not '{technique}'"
        )
    prefix = location_prefix.rstrip("/")
    manifest = read_manifest(technique, results_dir)
    recorded = {a["location"] for a in manifest["external_assets"]}
    pending = []
    for entry in inventory.get("assets", []):
        location = f"{prefix}/{entry['relative_path']}"
        if entry["kind"] not in ASSET_KINDS:
            raise ManifestError(f"{entry['relative_path']}: kind '{entry['kind']}' not in {ASSET_KINDS}")
        problems = asset_problems({"location": location, "sha256": entry["sha256"]})
        if problems:
            raise ManifestError(f"{entry['relative_path']}: " + "; ".join(problems))
        if location in recorded or any(location == p["location"] for p in pending):
            raise ManifestError(f"an asset at {location} is already recorded")
        pending.append(
            {
                "kind": entry["kind"],
                "location": location,
                "sha256": entry["sha256"],
                "size_bytes": int(entry["size_bytes"]),
                "contains_text": bool(entry["contains_text"]),
                "description": entry.get("description"),
                "added_utc": _now(),
            }
        )
    manifest["external_assets"].extend(pending)
    write_manifest(technique, manifest, results_dir)
    return len(pending)


def refresh(technique, results_dir=None):
    manifest = read_manifest(technique, results_dir)
    manifest["committed_artifacts"] = hash_committed(technique, results_dir)
    manifest["committed_hashed_utc"] = _now()
    return write_manifest(technique, manifest, results_dir)


def asset_problems(asset):
    """Policy checks on one external asset record."""
    problems = []
    location = str(asset.get("location", ""))
    if not location.startswith(EXTERNAL_SCHEMES):
        problems.append(f"location '{location}' must use one of {EXTERNAL_SCHEMES}")
    if any(host in location for host in REPO_SCHEMES):
        problems.append(f"location '{location}' points at a git host; assets belong in external storage")
    sha = str(asset.get("sha256", ""))
    if len(sha) != 64 or any(ch not in "0123456789abcdef" for ch in sha.lower()):
        problems.append("sha256 must be 64 hex characters")
    return problems


def check(technique, results_dir=None):
    """Return a list of problems for one technique's manifest (empty = ok)."""
    problems = []
    try:
        manifest = read_manifest(technique, results_dir)
    except ManifestError as error:
        return [str(error)]

    for field in ("kaggle_notebook", "kaggle_version", "git_commit"):
        if not manifest.get("source", {}).get(field):
            problems.append(f"source.{field} is missing")

    current = hash_committed(technique, results_dir)
    recorded = manifest.get("committed_artifacts", {})
    for rel in sorted(set(recorded) - set(current)):
        problems.append(f"{rel}: recorded in manifest but no longer present")
    for rel in sorted(set(current) - set(recorded)):
        problems.append(f"{rel}: present but not recorded; run refresh")
    for rel in sorted(set(current) & set(recorded)):
        if current[rel] != recorded[rel]:
            problems.append(f"{rel}: content changed since the manifest was refreshed")

    for asset in manifest.get("external_assets", []):
        for problem in asset_problems(asset):
            problems.append(f"asset {asset.get('kind')}: {problem}")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init")
    p.add_argument("--technique", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--kaggle-notebook", required=True)
    p.add_argument("--kaggle-version", type=int, required=True)
    p.add_argument("--git-commit", required=True)
    p.add_argument("--notes", default=None)

    p = sub.add_parser("add-asset")
    p.add_argument("--technique", required=True)
    p.add_argument("--kind", required=True, choices=ASSET_KINDS)
    p.add_argument("--location", required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--size-bytes", type=int, required=True)
    p.add_argument("--contains-text", action="store_true")
    p.add_argument("--description", default=None)

    p = sub.add_parser("add-assets-from")
    p.add_argument("--technique", required=True)
    p.add_argument("--file", required=True, help="external_assets.json written by the notebook")
    p.add_argument("--location-prefix", required=True)

    p = sub.add_parser("refresh")
    p.add_argument("--technique", required=True)

    p = sub.add_parser("check")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--technique")
    g.add_argument("--all", action="store_true")

    p = sub.add_parser("show")
    p.add_argument("--technique", required=True)

    args = parser.parse_args()
    try:
        if args.command == "init":
            print(f"wrote {init_manifest(args.technique, args.run_id, args.kaggle_notebook, args.kaggle_version, args.git_commit, args.notes)}")
        elif args.command == "add-asset":
            print(f"wrote {add_asset(args.technique, args.kind, args.location, args.sha256, args.size_bytes, args.contains_text, args.description)}")
        elif args.command == "add-assets-from":
            added = add_assets_from(args.technique, args.file, args.location_prefix)
            print(f"recorded {added} external asset(s) for {args.technique}")
        elif args.command == "refresh":
            print(f"wrote {refresh(args.technique)}")
        elif args.command == "show":
            print(json.dumps(read_manifest(args.technique), indent=2, sort_keys=True))
        elif args.command == "check":
            techniques = [args.technique] if args.technique else [d.name for d in technique_dirs()]
            total = 0
            for technique in techniques:
                if args.all and not manifest_path(technique).exists():
                    print(f"----  {technique}: no manifest")
                    continue
                problems = check(technique)
                total += len(problems)
                print(f"{'PASS' if not problems else 'FAIL'}  {technique}  ({len(problems)} problem(s))")
                for problem in problems:
                    print(f"        {problem}")
            return EXIT_OK if total == 0 else EXIT_FAILED
    except ManifestError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return EXIT_FAILED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
