"""Adversarial tests for `run_manifest.py fetch`.

Each test hands the command one bad situation (corrupted bytes, a path that
leaves the technique folder, a destination inside the repository, a local file
that differs from the manifest, a location it cannot download) and checks that
it is refused and that nothing unverified is left at a final path. No test
touches the network: the download function is replaced by a fake that serves
bytes from a dict.
"""

import hashlib
import sys
import types
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIR))

import run_manifest  # noqa: E402
from repo_paths import EXTERNAL_DIR, REPO_ROOT, RESULTS_DIR, technique_dirs  # noqa: E402
from run_manifest import (  # noqa: E402
    DIFFERS,
    EXIT_FAILED,
    EXIT_OK,
    FAILED,
    FETCHED,
    PREDICTOR_KINDS,
    PRESENT,
    REFUSED,
    STAGING_DIRNAME,
    TO_FETCH,
    UNSUPPORTED,
    ManifestError,
    fetch_assets,
    fetch_command,
)

TECHNIQUE = "07_T"
PREFIX = "kaggle://datasets/u/slug/v3"


def asset(relative, content, kind="weights", contains_text=False, location=None):
    """A manifest entry whose recorded hash is the hash of `content`."""
    return {
        "kind": kind,
        "location": f"{PREFIX}/{relative}" if location is None else location,
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
        "contains_text": contains_text,
        "description": None,
        "added_utc": "2026-11-06T00:00:00+00:00",
    }


class FakeRemote:
    """Serves bytes by relative path and records every download it is asked for."""

    def __init__(self, files, errors=None):
        self.files = dict(files)
        self.errors = dict(errors or {})
        self.calls = []

    def __call__(self, owner, slug, version, relative, staging_dir):
        self.calls.append((owner, slug, version, relative))
        if relative in self.errors:
            raise self.errors[relative]
        path = Path(staging_dir) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.files[relative])
        return path


@pytest.fixture
def write_manifest(results_dir):
    def _write(assets):
        run_manifest.write_manifest(
            TECHNIQUE,
            {"technique": TECHNIQUE, "run_id": "r", "source": {}, "committed_artifacts": {},
             "external_assets": list(assets)},
            results_dir,
        )

    return _write


@pytest.fixture
def dest(tmp_path):
    return tmp_path / "external"


def files_under(folder):
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()) if folder.exists() else []


def test_a_corrupted_download_is_refused_and_nothing_is_kept(results_dir, write_manifest, dest):
    write_manifest([asset("weights/model.safetensors", b"the recorded weights")])
    remote = FakeRemote({"weights/model.safetensors": b"other bytes under the same name"})

    [outcome] = fetch_assets(TECHNIQUE, dest=dest, download=remote, results_dir=results_dir)

    assert outcome["status"] == REFUSED and "sha256" in outcome["detail"]
    assert files_under(dest) == []
    assert not (dest / TECHNIQUE / STAGING_DIRNAME).exists()
    assert fetch_command([TECHNIQUE], dest=dest, download=remote, results_dir=results_dir) == EXIT_FAILED


@pytest.mark.parametrize(
    "relative",
    ["../../escape.bin", "weights/../../escape.bin", "/escape.bin", "weights//escape.bin", "./escape.bin",
     "weights\\..\\escape.bin"],
)
def test_a_path_leaving_the_technique_folder_is_refused_before_any_download(
    results_dir, write_manifest, dest, tmp_path, relative
):
    write_manifest([asset(relative, b"payload")])
    remote = FakeRemote({relative: b"payload"})

    [outcome] = fetch_assets(TECHNIQUE, dest=dest, download=remote, results_dir=results_dir)

    assert outcome["status"] == REFUSED and "plain relative path" in outcome["detail"]
    assert remote.calls == []
    assert [p.name for p in tmp_path.rglob("escape.bin")] == []


@pytest.mark.parametrize("dry_run", [True, False], ids=["dry-run", "download"])
@pytest.mark.parametrize("inside", [REPO_ROOT, RESULTS_DIR / "downloads", REPO_ROOT / "tools"])
def test_a_destination_inside_the_repository_is_refused(results_dir, write_manifest, inside, dry_run):
    write_manifest([asset("predictions/predictions_test.csv", b"text", "predictions_with_text", True)])
    remote = FakeRemote({"predictions/predictions_test.csv": b"text"})

    with pytest.raises(ManifestError, match="inside the repository"):
        fetch_assets(TECHNIQUE, dest=inside, dry_run=dry_run, download=remote, results_dir=results_dir)
    assert remote.calls == []
    assert not (Path(inside) / TECHNIQUE).exists()


def test_external_and_outside_destinations_are_allowed(tmp_path):
    assert run_manifest.check_destination(tmp_path) == tmp_path.resolve()
    run_manifest.check_destination(EXTERNAL_DIR)
    run_manifest.check_destination(EXTERNAL_DIR / "07_T")


def test_a_different_local_file_is_never_overwritten(results_dir, write_manifest, dest):
    write_manifest([asset("weights/pipeline.joblib", b"the recorded pipeline")])
    local = dest / TECHNIQUE / "weights" / "pipeline.joblib"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"a newer local pipeline")
    remote = FakeRemote({"weights/pipeline.joblib": b"the recorded pipeline"})

    [outcome] = fetch_assets(TECHNIQUE, dest=dest, download=remote, results_dir=results_dir)

    assert outcome["status"] == DIFFERS
    assert local.read_bytes() == b"a newer local pipeline"
    assert remote.calls == []
    assert fetch_command([TECHNIQUE], dest=dest, download=remote, results_dir=results_dir) == EXIT_FAILED


def test_a_matching_local_file_is_not_downloaded_again(results_dir, write_manifest, dest):
    write_manifest([asset("weights/pipeline.joblib", b"the recorded pipeline")])
    local = dest / TECHNIQUE / "weights" / "pipeline.joblib"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"the recorded pipeline")
    remote = FakeRemote({})

    [outcome] = fetch_assets(TECHNIQUE, dest=dest, download=remote, results_dir=results_dir)

    assert outcome["status"] == PRESENT and remote.calls == []
    assert fetch_command([TECHNIQUE], dest=dest, download=remote, results_dir=results_dir) == EXIT_OK


@pytest.mark.parametrize(
    "location",
    [
        "gdrive://folder/weights/model.bin",
        "https://example.org/weights/model.bin",
        "kaggle://datasets/u/slug/weights/model.bin",   # no version
        "kaggle://models/u/model/v1/weights/model.bin",  # not a dataset
    ],
)
def test_a_location_fetch_cannot_download_is_reported_not_skipped(results_dir, write_manifest, dest, location):
    write_manifest([asset("weights/model.bin", b"weights", location=location)])
    remote = FakeRemote({"weights/model.bin": b"weights"})

    [outcome] = fetch_assets(TECHNIQUE, dest=dest, download=remote, results_dir=results_dir)

    assert outcome["status"] == UNSUPPORTED and "fetch cannot download it" in outcome["detail"]
    assert remote.calls == [] and files_under(dest) == []
    assert fetch_command([TECHNIQUE], dest=dest, download=remote, results_dir=results_dir) == EXIT_FAILED


def test_the_predictor_selection_leaves_text_bearing_files_alone(results_dir, write_manifest, dest):
    contents = {
        "weights/transformer_model/model.safetensors": (b"weights", "weights", False),
        "tokenizer/tokenizer.json": (b"tokenizer", "tokenizer", False),
        "training_logs/run_summary.csv": (b"log", "training_log", False),
        "predictions/predictions_test.csv": (b"posts", "predictions_with_text", True),
        "attributions_local/shap-partition_test_seed30.csv": (b"tokens", "local_attributions", True),
    }
    write_manifest([asset(rel, data, kind, text) for rel, (data, kind, text) in contents.items()])
    remote = FakeRemote({rel: data for rel, (data, _, _) in contents.items()})

    outcomes = fetch_assets(TECHNIQUE, dest=dest, kinds=PREDICTOR_KINDS, download=remote, results_dir=results_dir)

    assert [o["status"] for o in outcomes] == [FETCHED] * 3
    assert files_under(dest / TECHNIQUE) == [
        "tokenizer/tokenizer.json", "training_logs/run_summary.csv", "weights/transformer_model/model.safetensors"]
    assert not any(o["contains_text"] for o in outcomes)


def test_an_unknown_kind_is_refused(results_dir, write_manifest, dest):
    write_manifest([asset("weights/model.bin", b"weights")])
    with pytest.raises(ManifestError, match="not in"):
        fetch_assets(TECHNIQUE, dest=dest, kinds=["everything"], download=FakeRemote({}), results_dir=results_dir)


def test_a_dry_run_downloads_and_writes_nothing(results_dir, write_manifest, dest, capsys):
    write_manifest([asset("weights/model.bin", b"weights"),
                    asset("predictions/predictions_test.csv", b"posts", "predictions_with_text", True)])
    remote = FakeRemote({"weights/model.bin": b"weights", "predictions/predictions_test.csv": b"posts"})

    outcomes = fetch_assets(TECHNIQUE, dest=dest, dry_run=True, download=remote, results_dir=results_dir)
    code = fetch_command([TECHNIQUE], dest=dest, dry_run=True, download=remote, results_dir=results_dir)

    assert [o["status"] for o in outcomes] == [TO_FETCH, TO_FETCH]
    assert remote.calls == [] and not dest.exists()
    assert code == EXIT_OK
    assert "1 file(s) with dataset text" in capsys.readouterr().out


@pytest.mark.parametrize("error", [RuntimeError("404 not found"), SystemExit(1)])
def test_one_failed_download_does_not_stop_the_rest(results_dir, write_manifest, dest, error):
    write_manifest([asset("weights/a.bin", b"first"), asset("weights/b.bin", b"second")])
    remote = FakeRemote({"weights/b.bin": b"second"}, errors={"weights/a.bin": error})

    outcomes = fetch_assets(TECHNIQUE, dest=dest, download=remote, results_dir=results_dir)

    assert [o["status"] for o in outcomes] == [FAILED, FETCHED]
    assert files_under(dest / TECHNIQUE) == ["weights/b.bin"]
    assert fetch_command([TECHNIQUE], dest=dest, download=remote, results_dir=results_dir) == EXIT_FAILED


def test_fetched_weights_have_the_layout_the_loader_expects(results_dir, write_manifest, dest):
    from model_loaders import detect_family

    write_manifest([asset("weights/pipeline.joblib", b"a pipeline")])
    remote = FakeRemote({"weights/pipeline.joblib": b"a pipeline"})

    code = fetch_command([TECHNIQUE], dest=dest, download=remote, results_dir=results_dir)

    assert code == EXIT_OK
    assert remote.calls == [("u", "slug", 3, "weights/pipeline.joblib")]
    assert (dest / TECHNIQUE / "weights" / "pipeline.joblib").read_bytes() == b"a pipeline"
    assert detect_family(dest / TECHNIQUE) == "sklearn_pipeline"


def test_a_named_technique_must_have_a_manifest(results_dir, dest, capsys):
    (results_dir / TECHNIQUE).mkdir()
    with pytest.raises(ManifestError, match="no manifest"):
        fetch_command([TECHNIQUE], dest=dest, download=FakeRemote({}), results_dir=results_dir)
    code = fetch_command([TECHNIQUE], dest=dest, download=FakeRemote({}), results_dir=results_dir,
                         skip_without_manifest=True)
    assert code == EXIT_OK and "no manifest" in capsys.readouterr().out


def test_every_committed_location_can_be_fetched():
    """The eight committed manifests hold only locations and paths fetch accepts."""
    checked = 0
    for folder in technique_dirs():
        if not run_manifest.manifest_path(folder.name).exists():
            continue
        for entry in run_manifest.read_manifest(folder.name)["external_assets"]:
            _, _, version, relative = run_manifest.parse_kaggle_location(entry["location"])
            run_manifest.safe_relative_path(relative)
            assert version >= 1
            if entry["kind"] in PREDICTOR_KINDS:
                assert entry["contains_text"] is False, entry["location"]
            checked += 1
    assert checked > 0


class TestKaggleDownload:
    """The adapter around the Kaggle client, with the client replaced by a fake."""

    @pytest.fixture
    def client(self, monkeypatch):
        state = {"calls": [], "sign_in_fails": False, "writes": None}

        class FakeApi:
            def authenticate(self):
                if state["sign_in_fails"]:
                    raise SystemExit(1)

            def dataset_download_file(self, dataset, file_name, path, force=False, quiet=False, unzip=False):
                state["calls"].append({"dataset": dataset, "file_name": file_name, "path": path,
                                       "force": force, "quiet": quiet, "unzip": unzip})
                print("Dataset URL: https://www.kaggle.com/datasets/u/slug")
                for name in ([file_name] if state["writes"] is None else state["writes"]):
                    out = Path(path) / name
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(b"bytes")

        package, api, extended = (types.ModuleType(name) for name in
                                  ("kaggle", "kaggle.api", "kaggle.api.kaggle_api_extended"))
        setattr(extended, "KaggleApi", FakeApi)
        for module in (package, api, extended):
            monkeypatch.setitem(sys.modules, module.__name__, module)
        return state

    def test_it_asks_for_the_recorded_version_and_returns_the_file(self, client, tmp_path, capsys):
        path = run_manifest.kaggle_download("u", "slug", 3, "weights/model.safetensors", tmp_path)

        assert path == tmp_path / "weights" / "model.safetensors" and path.read_bytes() == b"bytes"
        assert client["calls"] == [{"dataset": "u/slug/3", "file_name": "weights/model.safetensors",
                                    "path": str(tmp_path), "force": True, "quiet": True, "unzip": True}]
        assert capsys.readouterr().out == ""

    def test_a_failed_sign_in_names_the_login_command(self, client, tmp_path):
        client["sign_in_fails"] = True
        with pytest.raises(ManifestError, match="kaggle auth login"):
            run_manifest.kaggle_download("u", "slug", 3, "weights/model.safetensors", tmp_path)
        assert client["calls"] == []

    def test_a_download_that_leaves_several_files_is_refused(self, client, tmp_path):
        client["writes"] = ["a.bin", "b.bin"]
        with pytest.raises(ManifestError, match="left 2 files"):
            run_manifest.kaggle_download("u", "slug", 3, "weights/model.safetensors", tmp_path)
