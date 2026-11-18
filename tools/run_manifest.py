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

`fetch` is the way back: it downloads the recorded assets of a technique into
external/<TECHNIQUE>/, the folder tools/model_loaders.py and the per-post tools
read. A file is kept only if its bytes have the recorded sha256, a local file
with another hash is never overwritten, and the destination is external/ or a
folder outside the repository. `--predictor` selects what the loaders read:
weights, tokenizers and training logs, and for a pooled ensemble only the
components its best_config.json admits. Only kaggle://datasets/ locations can
be fetched; it needs the kaggle package and one Kaggle sign-in per run, with
access to them.

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
    python3 tools/run_manifest.py fetch --all --dry-run     # on disk, missing, different
    python3 tools/run_manifest.py fetch --technique T --predictor   # weights and tokenizers
    python3 tools/run_manifest.py fetch --technique T       # every recorded asset
"""

import argparse
import contextlib
import hashlib
import inspect
import io
import json
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from repo_paths import RESULTS_DIR, RUN_MANIFEST_FILENAME, technique_dirs

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


# ---------------------------------------------------------------------------
# fetch: restore external/<TECHNIQUE>/ from the recorded locations
# ---------------------------------------------------------------------------

# What tools/model_loaders.py reads to rebuild a predictor (the ensemble takes
# its inference precision from training_logs/run_summary.csv). All text-free.
PREDICTOR_KINDS = ("weights", "tokenizer", "training_log")

KAGGLE_DATASET_LOCATION_RE = re.compile(r"^kaggle://datasets/([^/]+)/([^/]+)/v(\d+)/(.+)$")
# A pooled ensemble keeps one weights folder per component and seed, and one
# tokenizer folder per component (tools/model_loaders.py::_member_folders).
ENSEMBLE_COMPONENT_RE = re.compile(r"/(?:weights/member-(.+)-seed\d+|tokenizer/([^/]+))/[^/]+$")
# Each download lands in its own folder with this prefix, inside the technique
# folder, until its hash is checked.
STAGING_PREFIX = ".fetch_staging_"

PRESENT = "present"          # on disk with the recorded sha256
FETCHED = "fetched"          # downloaded, hash checked, moved into place
TO_FETCH = "to fetch"        # dry run: absent locally
DIFFERS = "differs"          # on disk with another sha256; never overwritten
REFUSED = "refused"          # unsafe or ambiguous path, or downloaded bytes with another sha256
UNSUPPORTED = "unsupported"  # a location fetch cannot download
FAILED = "failed"            # the download, or the local file work around it, failed
FETCH_OK = (PRESENT, FETCHED, TO_FETCH)


class FetchAborted(ManifestError):
    """A problem every download would hit (no client, no sign-in): stop instead of repeating it."""


def parse_kaggle_location(location):
    """(owner, slug, version, relative path) of a kaggle://datasets/<owner>/<slug>/v<N>/<path> location."""
    match = KAGGLE_DATASET_LOCATION_RE.match(str(location))
    if not match:
        raise ManifestError(
            f"location '{location}' is not kaggle://datasets/<owner>/<slug>/v<N>/<path>; "
            "fetch cannot download it"
        )
    owner, slug, version, relative = match.groups()
    return owner, slug, int(version), relative


def safe_relative_path(relative):
    """An asset's path below the technique folder; refused if it could leave that folder."""
    text = str(relative)
    parts = text.split("/")
    if text.startswith("/") or "\\" in text or any(part in ("", ".", "..") for part in parts):
        raise ManifestError(f"asset path '{relative}' is not a plain relative path")
    if parts[0].startswith(STAGING_PREFIX):
        raise ManifestError(f"asset path '{relative}' is inside a folder name fetch keeps for its downloads")
    return Path(*parts)


def check_technique_name(technique):
    """A technique is one folder name (NN_NAME), never a path: it is joined onto the destination."""
    from repo_paths import TECHNIQUE_DIR_RE

    name = str(technique)
    if not TECHNIQUE_DIR_RE.match(name) or Path(name).name != name or "\\" in name:
        raise ManifestError(
            f"'{technique}' is not a technique folder name such as 07_TWITTER-ROBERTA_FINE-TUNE"
        )
    return name


def check_destination(dest):
    """Refuse a destination inside the repository unless it is under external/.

    Assets can carry dataset text, and the gitignored external/ is the only
    place for them in the working tree.
    """
    from repo_paths import EXTERNAL_DIR, outside_repo_or_under_external

    dest = Path(dest).resolve()
    if not outside_repo_or_under_external(dest):
        raise ManifestError(
            f"{dest} is inside the repository; external assets may only be written outside it "
            f"or under {EXTERNAL_DIR.name}/"
        )
    return dest


def select_assets(manifest, kinds=None):
    """The manifest's external assets, all of them or those of the given kinds."""
    assets = manifest.get("external_assets", [])
    if kinds is None:
        return list(assets)
    unknown = sorted(set(kinds) - set(ASSET_KINDS))
    if unknown:
        raise ManifestError(f"kind(s) {unknown} not in {ASSET_KINDS}")
    return [asset for asset in assets if asset.get("kind") in kinds]


def _admitted_components(technique, results_dir=None):
    """The components a pooled ensemble's committed best_config.json admits, or None."""
    base = RESULTS_DIR if results_dir is None else Path(results_dir)
    try:
        with open(base / technique / "best_config.json") as handle:
            components = json.load(handle).get("hyperparameters", {}).get("components")
    except (OSError, ValueError, AttributeError):
        return None
    return [str(c) for c in components] if isinstance(components, list) else None


def predictor_assets(technique, manifest, results_dir=None):
    """(assets tools/model_loaders.py reads, ensemble components left out).

    Weights, tokenizers and training logs. For a pooled ensemble, only the
    components its committed best_config.json admits: the loader reads no
    other member, and the excluded ones are a third of notebook 11's upload.
    """
    assets = select_assets(manifest, PREDICTOR_KINDS)
    admitted = _admitted_components(technique, results_dir)
    if admitted is None:
        return assets, []
    kept, left_out = [], set()
    for asset in assets:
        match = ENSEMBLE_COMPONENT_RE.search(str(asset.get("location", "")))
        component = (match.group(1) or match.group(2)) if match else None
        if component is not None and component not in admitted:
            left_out.add(component)
        else:
            kept.append(asset)
    return kept, sorted(left_out)


def kaggle_downloader():
    """A download function bound to one Kaggle client, signed in once, here."""
    try:
        # Imported here: the package signs in when it is imported.
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as error:
        raise FetchAborted("fetch needs the kaggle package (requirements.txt)") from error
    api = KaggleApi()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            api.authenticate()
    except (Exception, SystemExit) as error:
        raise FetchAborted(
            f"Kaggle sign-in failed ({type(error).__name__}: {error}); if the login has expired, "
            "run `kaggle auth login --force` and retry"
        ) from error
    if "unzip" not in inspect.signature(api.dataset_download_file).parameters:
        raise FetchAborted(
            "the installed kaggle client cannot unwrap the archive the server puts around large files "
            "(dataset_download_file has no `unzip` option); upgrade it with `pip install -U kaggle`"
        )

    def download(owner, slug, version, relative_path, staging_dir):
        """Download one file of one dataset version into `staging_dir`; return its path."""
        # The client prints the dataset URL on every call; the caller reports each file itself.
        with contextlib.redirect_stdout(io.StringIO()):
            api.dataset_download_file(
                f"{owner}/{slug}/{version}", relative_path, path=str(staging_dir),
                force=True, quiet=True, unzip=True,
            )
        expected = Path(staging_dir) / relative_path
        if expected.is_file():
            return expected
        produced = [
            path for path in Path(staging_dir).rglob("*")
            if path.is_file() and not path.name.endswith(".kaggle-partial")
        ]
        if len(produced) != 1:
            raise ManifestError(f"the download of {relative_path} left {len(produced)} files, expected one")
        return produced[0]

    return download


class LazyKaggleDownload:
    """Signs in on the first download, not before, and keeps the client for the rest of the run."""

    def __init__(self):
        self._download = None

    def __call__(self, owner, slug, version, relative_path, staging_dir):
        if self._download is None:
            self._download = kaggle_downloader()
        return self._download(owner, slug, version, relative_path, staging_dir)


def _restore_one(folder, target, recorded, source, download):
    """Download one asset beside its target and move it into place if its hash matches."""
    folder.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=STAGING_PREFIX, dir=folder))
    try:
        staged = Path(download(*source, staging))
        actual = sha256_file(staged)
        if actual != recorded:
            return REFUSED, (f"the downloaded bytes have sha256 {actual[:12]}..., the manifest records "
                             f"{recorded[:12]}...; nothing was kept")
        if target.exists():
            return DIFFERS, "the local path appeared during the download and was not overwritten"
        target.parent.mkdir(parents=True, exist_ok=True)
        # A move, not a rename: a subfolder may sit on another volume.
        shutil.move(str(staged), str(target))
        return FETCHED, ""
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def fetch_assets(technique, dest=None, kinds=None, predictor=False, dry_run=False, download=None,
                 results_dir=None, progress=None):
    """Restore a technique's recorded external assets under <dest>/<technique>/.

    Returns one outcome per selected asset: its kind, location, relative path,
    size, text flag, a status and a detail. A file reaches its final path only
    after its bytes match the recorded sha256. A local file with another hash
    is reported and left alone. One failed asset does not stop the others; a
    problem every download would hit (no sign-in) raises FetchAborted. With
    `dry_run` nothing is downloaded or written. `predictor` selects what
    model_loaders.py reads (see predictor_assets) instead of `kinds`.
    `download` is the function that fetches one file (default: the Kaggle
    client); `progress` is called with one line before each download.
    """
    if dest is None:
        from repo_paths import EXTERNAL_DIR

        dest = EXTERNAL_DIR
    folder = check_destination(dest) / check_technique_name(technique)
    manifest = read_manifest(technique, results_dir)
    selected = predictor_assets(technique, manifest, results_dir)[0] if predictor else select_assets(manifest, kinds)
    download = LazyKaggleDownload() if download is None else download

    parsed = []
    for asset in selected:
        try:
            parsed.append(parse_kaggle_location(asset.get("location")))
        except ManifestError as error:
            parsed.append(error)
    shared = {}
    for source in parsed:
        if not isinstance(source, ManifestError):
            shared[source[3]] = shared.get(source[3], 0) + 1

    outcomes = []
    for asset, source in zip(selected, parsed):
        outcome = {
            "kind": asset.get("kind"),
            "location": asset.get("location"),
            "relative_path": None,
            "size_bytes": int(asset.get("size_bytes") or 0),
            "contains_text": bool(asset.get("contains_text")),
            "status": None,
            "detail": "",
        }
        outcomes.append(outcome)
        if isinstance(source, ManifestError):
            outcome.update(status=UNSUPPORTED, detail=str(source))
            continue
        relative = source[3]
        outcome["relative_path"] = relative
        try:
            target = folder / safe_relative_path(relative)
        except ManifestError as error:
            outcome.update(status=REFUSED, detail=str(error))
            continue
        if shared[relative] > 1:
            outcome.update(
                status=REFUSED,
                detail=f"{shared[relative]} recorded assets have this local path (other datasets or "
                       "versions); fetch cannot tell which one the folder should hold",
            )
            continue
        recorded = str(asset.get("sha256", "")).lower()
        try:
            if target.is_file():
                if sha256_file(target) == recorded:
                    outcome["status"] = PRESENT
                else:
                    outcome.update(
                        status=DIFFERS,
                        detail="the local file has another sha256 and was not overwritten; "
                               "move it aside to fetch the recorded one",
                    )
            elif target.exists():
                outcome.update(status=DIFFERS, detail="the local path exists and is not a file")
            elif dry_run:
                outcome["status"] = TO_FETCH
            else:
                if progress is not None:
                    progress(f"  fetching     {relative}  ({_human_bytes(outcome['size_bytes'])})")
                status, detail = _restore_one(folder, target, recorded, source, download)
                outcome.update(status=status, detail=detail)
        except FetchAborted:
            raise
        except (Exception, SystemExit) as error:
            outcome.update(status=FAILED, detail=f"{type(error).__name__}: {error}")
    return outcomes


def _human_bytes(size):
    size = float(size)
    if size < 1000:
        return f"{size:.0f} B"
    for unit in ("kB", "MB"):
        size /= 1000
        if round(size, 1) < 1000:
            return f"{size:.1f} {unit}"
    return f"{size / 1000:.2f} GB"


def fetch_command(techniques, dest=None, kinds=None, predictor=False, dry_run=False, download=None,
                  results_dir=None, all_techniques=False):
    """Run `fetch` for each technique, print the report, and return the exit code.

    `all_techniques` is for `--all`: a technique folder with no manifest is
    listed and passed over, and a technique with nothing of the selected kinds
    is not a failure. A technique named explicitly must have a manifest and at
    least one selected asset.
    """
    download = LazyKaggleDownload() if download is None else download  # one sign-in for the whole run
    failed = 0
    for technique in techniques:
        if all_techniques and not manifest_path(technique, results_dir).exists():
            print(f"----  {technique}: no manifest")
            continue
        print(technique)
        outcomes = fetch_assets(technique, dest=dest, kinds=kinds, predictor=predictor, dry_run=dry_run,
                                download=download, results_dir=results_dir, progress=print)
        for outcome in outcomes:
            if outcome["status"] == PRESENT:
                continue
            name = outcome["relative_path"] or outcome["location"]
            line = f"  {outcome['status']:<12} {name}  ({_human_bytes(outcome['size_bytes'])})"
            print(line + (f": {outcome['detail']}" if outcome["detail"] else ""))
        if predictor:
            left_out = predictor_assets(technique, read_manifest(technique, results_dir), results_dir)[1]
            if left_out:
                print(f"  left out, not in best_config.json components: {', '.join(left_out)}")
        counts = {}
        for outcome in outcomes:
            counts[outcome["status"]] = counts.get(outcome["status"], 0) + 1
        summary = ", ".join(f"{count} {status}" for status, count in sorted(counts.items())) or "nothing selected"
        moved = [o for o in outcomes if o["status"] in (TO_FETCH, FETCHED)]
        if moved:
            summary += (
                f"; {_human_bytes(sum(o['size_bytes'] for o in moved))} "
                f"{'to download' if dry_run else 'downloaded'}, "
                f"{sum(o['contains_text'] for o in moved)} file(s) with dataset text"
            )
        print(f"  {len(outcomes)} asset(s): {summary}")
        failed += sum(outcome["status"] not in FETCH_OK for outcome in outcomes)
        if not outcomes and not all_techniques:
            print("  the manifest records no asset of the selected kinds")
            failed += 1
    return EXIT_OK if failed == 0 else EXIT_FAILED


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

    p = sub.add_parser("fetch")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--technique")
    g.add_argument("--all", action="store_true")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--kind", action="append", choices=ASSET_KINDS,
                   help="repeatable; default: every recorded asset")
    g.add_argument("--predictor", action="store_true",
                   help="weights, tokenizers and training logs: what model_loaders.py reads "
                        "(for an ensemble, only the components in best_config.json)")
    p.add_argument("--dest", default=None,
                   help="default: external/; any other path inside the repository is refused")
    p.add_argument("--dry-run", action="store_true",
                   help="hash what is on disk and list what would be fetched; no download")

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
        elif args.command == "fetch":
            techniques = [args.technique] if args.technique else [d.name for d in technique_dirs()]
            return fetch_command(techniques, dest=args.dest, kinds=args.kind, predictor=args.predictor,
                                 dry_run=args.dry_run, all_techniques=args.all)
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
