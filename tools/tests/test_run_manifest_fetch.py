"""Adversarial tests for `run_manifest.py fetch`.

Each test hands the command one bad situation (corrupted bytes, a path that
leaves the technique folder, a technique name that is a path, a destination
inside the repository, a local file that differs from the manifest, a location
it cannot download, a failed sign-in) and checks that it is refused and that
nothing unverified is left at a final path. No test touches the network: the
download function is replaced by a fake that serves bytes from a dict, and
the tests that need a repository to refuse build a stand-in under tmp_path, so
a regression cannot write into the real one.
"""

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIR))

import repo_paths  # noqa: E402
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
    STAGING_PREFIX,
    TO_FETCH,
    UNSUPPORTED,
    FetchAborted,
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


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    """A stand-in repository: the guards read repo_paths at call time."""
    root = tmp_path / "Repo"
    for name in ("results_summary", "tools", "external"):
        (root / name).mkdir(parents=True)
    monkeypatch.setattr(repo_paths, "REPO_ROOT", root)
    monkeypatch.setattr(repo_paths, "EXTERNAL_DIR", root / "external")
    return root


def files_under(folder):
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()) if folder.exists() else []


def staging_folders(folder):
    return sorted(p.name for p in folder.glob(STAGING_PREFIX + "*")) if folder.exists() else []


# --- the bytes ---------------------------------------------------------------


def test_a_corrupted_download_is_refused_and_nothing_is_kept(results_dir, write_manifest, dest):
    write_manifest([asset("weights/model.safetensors", b"the recorded weights")])
    remote = FakeRemote({"weights/model.safetensors": b"other bytes under the same name"})

    [outcome] = fetch_assets(TECHNIQUE, dest=dest, download=remote, results_dir=results_dir)

    assert outcome["status"] == REFUSED and "sha256" in outcome["detail"]
    assert files_under(dest) == []
    assert staging_folders(dest / TECHNIQUE) == []
    assert fetch_command([TECHNIQUE], dest=dest, download=remote, results_dir=results_dir) == EXIT_FAILED


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


def test_a_file_that_appears_during_the_download_is_not_overwritten(results_dir, write_manifest, dest):
    write_manifest([asset("weights/pipeline.joblib", b"the recorded pipeline")])
    local = dest / TECHNIQUE / "weights" / "pipeline.joblib"
    remote = FakeRemote({"weights/pipeline.joblib": b"the recorded pipeline"})

    def download_while_another_run_writes(*args):
        local.parent.mkdir(parents=True)
        local.write_bytes(b"written by another run")
        return remote(*args)

    [outcome] = fetch_assets(TECHNIQUE, dest=dest, download=download_while_another_run_writes,
                             results_dir=results_dir)

    assert outcome["status"] == DIFFERS and "appeared during the download" in outcome["detail"]
    assert local.read_bytes() == b"written by another run"


def test_a_matching_local_file_is_not_downloaded_again(results_dir, write_manifest, dest):
    write_manifest([asset("weights/pipeline.joblib", b"the recorded pipeline")])
    local = dest / TECHNIQUE / "weights" / "pipeline.joblib"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"the recorded pipeline")
    remote = FakeRemote({})

    [outcome] = fetch_assets(TECHNIQUE, dest=dest, download=remote, results_dir=results_dir)

    assert outcome["status"] == PRESENT and remote.calls == []
    assert fetch_command([TECHNIQUE], dest=dest, download=remote, results_dir=results_dir) == EXIT_OK


def test_another_runs_staging_folder_is_left_alone(results_dir, write_manifest, dest):
    write_manifest([asset("weights/a.bin", b"first")])
    other = dest / TECHNIQUE / (STAGING_PREFIX + "other_run") / "weights" / "big.bin"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"half of another run's download")

    [outcome] = fetch_assets(TECHNIQUE, dest=dest, download=FakeRemote({"weights/a.bin": b"first"}),
                             results_dir=results_dir)

    assert outcome["status"] == FETCHED
    assert other.read_bytes() == b"half of another run's download"
    assert staging_folders(dest / TECHNIQUE) == [STAGING_PREFIX + "other_run"]


# --- the paths ---------------------------------------------------------------


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


def test_a_path_inside_a_staging_folder_name_is_refused(results_dir, write_manifest, dest):
    relative = STAGING_PREFIX + "x/model.bin"
    write_manifest([asset(relative, b"payload")])
    remote = FakeRemote({relative: b"payload"})

    [outcome] = fetch_assets(TECHNIQUE, dest=dest, download=remote, results_dir=results_dir)

    assert outcome["status"] == REFUSED and remote.calls == []
    assert fetch_command([TECHNIQUE], dest=dest, download=remote, results_dir=results_dir) == EXIT_FAILED


@pytest.mark.parametrize(
    "name", ["../results_summary/07_T", "07_T/..", "07_T/weights", "results_summary", "..", "07_T\\x"]
)
def test_a_technique_given_as_a_path_is_refused(results_dir, dest, name):
    remote = FakeRemote({})
    with pytest.raises(ManifestError, match="not a technique folder name"):
        fetch_assets(name, dest=dest, dry_run=True, download=remote, results_dir=results_dir)
    with pytest.raises(ManifestError, match="not a technique folder name"):
        fetch_assets(str(results_dir / TECHNIQUE), dest=dest, download=remote, results_dir=results_dir)
    assert remote.calls == [] and not dest.exists()


@pytest.mark.parametrize("dry_run", [True, False], ids=["dry-run", "download"])
@pytest.mark.parametrize("inside", ["", "results_summary/downloads", "tools"])
def test_a_destination_inside_the_repository_is_refused(results_dir, write_manifest, fake_repo, inside, dry_run):
    write_manifest([asset("predictions/predictions_test.csv", b"text", "predictions_with_text", True)])
    remote = FakeRemote({"predictions/predictions_test.csv": b"text"})

    with pytest.raises(ManifestError, match="inside the repository"):
        fetch_assets(TECHNIQUE, dest=fake_repo / inside, dry_run=dry_run, download=remote, results_dir=results_dir)
    assert remote.calls == []
    assert [p.name for p in fake_repo.rglob("predictions_test.csv")] == []


def test_another_spelling_of_a_repository_folder_is_refused(fake_repo, tmp_path):
    link = tmp_path / "link_to_results"
    link.symlink_to(fake_repo / "results_summary", target_is_directory=True)
    with pytest.raises(ManifestError, match="inside the repository"):
        run_manifest.check_destination(link / "downloads")

    other_case = tmp_path / "rEPO"
    if not other_case.exists():
        pytest.skip("this volume is case-sensitive: another letter case is another folder")
    with pytest.raises(ManifestError, match="inside the repository"):
        run_manifest.check_destination(other_case / "RESULTS_SUMMARY" / "downloads")
    run_manifest.check_destination(other_case / "EXTERNAL" / "downloads")


def test_external_and_outside_destinations_are_allowed(tmp_path):
    assert run_manifest.check_destination(tmp_path) == tmp_path.resolve()
    run_manifest.check_destination(EXTERNAL_DIR)
    run_manifest.check_destination(EXTERNAL_DIR / "07_T")
    for inside in (REPO_ROOT, RESULTS_DIR / "downloads", REPO_ROOT / "tools"):
        with pytest.raises(ManifestError, match="inside the repository"):
            run_manifest.check_destination(inside)


def test_two_recorded_assets_with_one_local_path_are_refused(results_dir, write_manifest, dest):
    write_manifest([
        asset("weights/m.bin", b"version one", location="kaggle://datasets/u/slug/v1/weights/m.bin"),
        asset("weights/m.bin", b"version two", location="kaggle://datasets/u/slug/v2/weights/m.bin"),
    ])
    remote = FakeRemote({"weights/m.bin": b"version one"})

    outcomes = fetch_assets(TECHNIQUE, dest=dest, download=remote, results_dir=results_dir)

    assert [o["status"] for o in outcomes] == [REFUSED, REFUSED]
    assert all("2 recorded assets have this local path" in o["detail"] for o in outcomes)
    assert remote.calls == [] and files_under(dest) == []


# --- the locations and the selection ------------------------------------------


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

    outcomes = fetch_assets(TECHNIQUE, dest=dest, predictor=True, download=remote, results_dir=results_dir)

    assert [o["status"] for o in outcomes] == [FETCHED] * 3
    assert files_under(dest / TECHNIQUE) == [
        "tokenizer/tokenizer.json", "training_logs/run_summary.csv", "weights/transformer_model/model.safetensors"]
    assert not any(o["contains_text"] for o in outcomes)


def test_the_predictor_selection_leaves_out_components_the_ensemble_does_not_admit(
    results_dir, write_manifest, dest, capsys
):
    contents = {
        "weights/member-A_anchor-seed17/model.safetensors": (b"a17", "weights"),
        "weights/member-A_anchor-seed30/model.safetensors": (b"a30", "weights"),
        "weights/member-D_deberta-seed17/model.safetensors": (b"d17", "weights"),
        "tokenizer/A_anchor/tokenizer.json": (b"tok a", "tokenizer"),
        "tokenizer/D_deberta/tokenizer.json": (b"tok d", "tokenizer"),
        "training_logs/run_summary.csv": (b"log", "training_log"),
    }
    write_manifest([asset(rel, data, kind) for rel, (data, kind) in contents.items()])
    (results_dir / TECHNIQUE / "best_config.json").write_text(
        json.dumps({"hyperparameters": {"components": ["A_anchor"]}}))
    remote = FakeRemote({rel: data for rel, (data, _) in contents.items()})

    code = fetch_command([TECHNIQUE], dest=dest, predictor=True, download=remote, results_dir=results_dir)

    assert code == EXIT_OK
    assert files_under(dest / TECHNIQUE) == [
        "tokenizer/A_anchor/tokenizer.json", "training_logs/run_summary.csv",
        "weights/member-A_anchor-seed17/model.safetensors", "weights/member-A_anchor-seed30/model.safetensors"]
    assert "left out, not in best_config.json components: D_deberta" in capsys.readouterr().out
    # Without the predictor selection every recorded file is wanted.
    everything = fetch_assets(TECHNIQUE, dest=dest, dry_run=True, download=remote, results_dir=results_dir)
    assert sorted(o["status"] for o in everything) == [PRESENT] * 4 + [TO_FETCH] * 2


def test_an_unknown_kind_is_refused(results_dir, write_manifest, dest):
    write_manifest([asset("weights/model.bin", b"weights")])
    with pytest.raises(ManifestError, match="not in"):
        fetch_assets(TECHNIQUE, dest=dest, kinds=["everything"], download=FakeRemote({}), results_dir=results_dir)


def test_a_named_technique_with_nothing_selected_is_a_failure(results_dir, write_manifest, dest, capsys):
    write_manifest([asset("predictions/predictions_test.csv", b"posts", "predictions_with_text", True)])
    remote = FakeRemote({})

    assert fetch_command([TECHNIQUE], dest=dest, predictor=True, download=remote, results_dir=results_dir) == EXIT_FAILED
    assert "records no asset of the selected kinds" in capsys.readouterr().out
    assert fetch_command([TECHNIQUE], dest=dest, predictor=True, download=remote, results_dir=results_dir,
                         all_techniques=True) == EXIT_OK


def test_a_named_technique_must_have_a_manifest(results_dir, dest, capsys):
    (results_dir / TECHNIQUE).mkdir()
    with pytest.raises(ManifestError, match="no manifest"):
        fetch_command([TECHNIQUE], dest=dest, download=FakeRemote({}), results_dir=results_dir)
    code = fetch_command([TECHNIQUE], dest=dest, download=FakeRemote({}), results_dir=results_dir,
                         all_techniques=True)
    assert code == EXIT_OK and "no manifest" in capsys.readouterr().out


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


# --- failures ----------------------------------------------------------------


@pytest.mark.parametrize("error", [RuntimeError("404 not found"), SystemExit(1), OSError("disk full")])
def test_one_failed_download_does_not_stop_the_rest(results_dir, write_manifest, dest, error):
    write_manifest([asset("weights/a.bin", b"first"), asset("weights/b.bin", b"second")])
    remote = FakeRemote({"weights/b.bin": b"second"}, errors={"weights/a.bin": error})

    outcomes = fetch_assets(TECHNIQUE, dest=dest, download=remote, results_dir=results_dir)

    assert [o["status"] for o in outcomes] == [FAILED, FETCHED]
    assert files_under(dest / TECHNIQUE) == ["weights/b.bin"]
    assert fetch_command([TECHNIQUE], dest=dest, download=remote, results_dir=results_dir) == EXIT_FAILED


def test_a_technique_folder_that_cannot_be_written_is_a_failure_not_a_crash(results_dir, write_manifest, dest):
    write_manifest([asset("weights/a.bin", b"first"), asset("weights/b.bin", b"second")])
    dest.mkdir()
    (dest / TECHNIQUE).write_text("a file where the technique folder should be")
    remote = FakeRemote({"weights/a.bin": b"first", "weights/b.bin": b"second"})

    outcomes = fetch_assets(TECHNIQUE, dest=dest, download=remote, results_dir=results_dir)

    assert [o["status"] for o in outcomes] == [FAILED, FAILED]
    assert remote.calls == []


def test_a_failed_sign_in_stops_the_run_at_the_first_download(results_dir, write_manifest, dest):
    write_manifest([asset("weights/a.bin", b"first"), asset("weights/b.bin", b"second")])
    calls = []

    def no_sign_in(*args):
        calls.append(args)
        raise FetchAborted("Kaggle sign-in failed")

    with pytest.raises(FetchAborted, match="sign-in failed"):
        fetch_command([TECHNIQUE, TECHNIQUE], dest=dest, download=no_sign_in, results_dir=results_dir)
    assert len(calls) == 1
    assert staging_folders(dest / TECHNIQUE) == []


# --- the happy path and the committed manifests ---------------------------------


def test_fetched_weights_have_the_layout_the_loader_expects(results_dir, write_manifest, dest):
    from model_loaders import detect_family

    write_manifest([asset("weights/pipeline.joblib", b"a pipeline")])
    remote = FakeRemote({"weights/pipeline.joblib": b"a pipeline"})

    code = fetch_command([TECHNIQUE], dest=dest, download=remote, results_dir=results_dir)

    assert code == EXIT_OK
    assert remote.calls == [("u", "slug", 3, "weights/pipeline.joblib")]
    assert (dest / TECHNIQUE / "weights" / "pipeline.joblib").read_bytes() == b"a pipeline"
    assert detect_family(dest / TECHNIQUE) == "sklearn_pipeline"


def test_every_committed_location_can_be_fetched():
    """The committed manifests hold only locations and paths fetch accepts."""
    checked = 0
    for folder in technique_dirs():
        if not run_manifest.manifest_path(folder.name).exists():
            continue
        run_manifest.check_technique_name(folder.name)
        relatives = []
        for entry in run_manifest.read_manifest(folder.name)["external_assets"]:
            _, _, version, relative = run_manifest.parse_kaggle_location(entry["location"])
            run_manifest.safe_relative_path(relative)
            relatives.append(relative)
            assert version >= 1
            if entry["kind"] in PREDICTOR_KINDS:
                assert entry["contains_text"] is False, entry["location"]
            checked += 1
        assert len(relatives) == len(set(relatives)), folder.name
    assert checked > 0


def test_the_predictor_selection_covers_what_the_committed_ensemble_admits():
    """For every committed pooled ensemble: each admitted component has weights
    and a tokenizer in the selection, and no other component is in it."""
    seen = 0
    for folder in technique_dirs():
        admitted = run_manifest._admitted_components(folder.name)
        if admitted is None or not run_manifest.manifest_path(folder.name).exists():
            continue
        seen += 1
        kept, left_out = run_manifest.predictor_assets(folder.name, run_manifest.read_manifest(folder.name))
        locations = [entry["location"] for entry in kept]
        assert not set(left_out) & set(admitted)
        for component in admitted:
            assert any(f"/weights/member-{component}-seed" in loc for loc in locations), component
            assert any(f"/tokenizer/{component}/" in loc for loc in locations), component
        for component in left_out:
            assert not any(f"member-{component}-seed" in loc or f"/tokenizer/{component}/" in loc for loc in locations)
    assert seen >= 1


@pytest.mark.parametrize(
    "size, text",
    [(911, "911 B"), (4500, "4.5 kB"), (999_960, "1.0 MB"), (498_600_000, "498.6 MB"),
     (999_999_999, "1.00 GB"), (6_140_000_000, "6.14 GB")],
)
def test_sizes_are_never_printed_as_a_thousand_of_the_smaller_unit(size, text):
    assert run_manifest._human_bytes(size) == text


# --- the Kaggle client ---------------------------------------------------------


class TestKaggleDownload:
    """The adapter around the Kaggle client, with the client replaced by a fake."""

    @pytest.fixture
    def client(self, monkeypatch):
        state = {"calls": [], "sign_ins": 0, "sign_in_fails": False, "writes": None, "old_client": False}

        class FakeApi:
            def authenticate(self):
                state["sign_ins"] += 1
                if state["sign_in_fails"]:
                    print("a screen of sign-in help")
                    raise SystemExit(1)

            def dataset_download_file(self, dataset, file_name, path, force=False, quiet=False, unzip=False):
                state["calls"].append({"dataset": dataset, "file_name": file_name, "path": path,
                                       "force": force, "quiet": quiet, "unzip": unzip})
                print("Dataset URL: https://www.kaggle.com/datasets/u/slug")
                for name in ([file_name] if state["writes"] is None else state["writes"]):
                    out = Path(path) / name
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(b"bytes")

        class OldFakeApi(FakeApi):
            def dataset_download_file(self, dataset, file_name, path, force=False, quiet=False):  # type: ignore[override]
                raise AssertionError("an old client must be refused before any download")

        def make_api():
            return OldFakeApi() if state["old_client"] else FakeApi()

        package, api, extended = (types.ModuleType(name) for name in
                                  ("kaggle", "kaggle.api", "kaggle.api.kaggle_api_extended"))
        setattr(extended, "KaggleApi", make_api)
        for module in (package, api, extended):
            monkeypatch.setitem(sys.modules, module.__name__, module)
        return state

    def test_it_asks_for_the_recorded_version_and_returns_the_file(self, client, tmp_path, capsys):
        path = run_manifest.kaggle_downloader()("u", "slug", 3, "weights/model.safetensors", tmp_path)

        assert path == tmp_path / "weights" / "model.safetensors" and path.read_bytes() == b"bytes"
        assert client["calls"] == [{"dataset": "u/slug/3", "file_name": "weights/model.safetensors",
                                    "path": str(tmp_path), "force": True, "quiet": True, "unzip": True}]
        assert capsys.readouterr().out == ""

    def test_it_signs_in_once_for_a_whole_run(self, client, results_dir, write_manifest, dest):
        write_manifest([asset("weights/a.bin", b"bytes"), asset("weights/b.bin", b"bytes")])

        code = fetch_command([TECHNIQUE], dest=dest, results_dir=results_dir)

        assert code == EXIT_OK and len(client["calls"]) == 2
        assert client["sign_ins"] == 1

    def test_nothing_signs_in_when_nothing_is_downloaded(self, client, results_dir, write_manifest, dest):
        write_manifest([asset("weights/a.bin", b"bytes")])
        assert fetch_command([TECHNIQUE], dest=dest, dry_run=True, results_dir=results_dir) == EXIT_OK
        assert client["sign_ins"] == 0

    def test_a_failed_sign_in_stops_the_run_and_names_the_login_command(
        self, client, results_dir, write_manifest, dest, capsys
    ):
        client["sign_in_fails"] = True
        write_manifest([asset("weights/a.bin", b"bytes"), asset("weights/b.bin", b"bytes")])

        with pytest.raises(FetchAborted, match="kaggle auth login"):
            fetch_command([TECHNIQUE], dest=dest, results_dir=results_dir)
        assert client["sign_ins"] == 1 and client["calls"] == []
        assert "sign-in help" not in capsys.readouterr().out

    def test_a_client_without_the_unzip_option_is_refused(self, client):
        client["old_client"] = True
        with pytest.raises(FetchAborted, match="pip install -U kaggle"):
            run_manifest.kaggle_downloader()

    def test_a_download_that_leaves_several_files_is_refused(self, client, tmp_path):
        client["writes"] = ["a.bin", "b.bin"]
        with pytest.raises(ManifestError, match="left 2 files"):
            run_manifest.kaggle_downloader()("u", "slug", 3, "weights/model.safetensors", tmp_path)
