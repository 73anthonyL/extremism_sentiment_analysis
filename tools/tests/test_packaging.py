"""Packaging: the study root is found, a bad one is refused, and pyproject points at real code."""

import importlib
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent
PYPROJECT = tomllib.loads((TOOLS.parent / "pyproject.toml").read_text())


def _root_seen_by_a_fresh_interpreter(env_value, cwd):
    env = {k: v for k, v in os.environ.items() if k != "ESA_STUDY_ROOT"}
    if env_value is not None:
        env["ESA_STUDY_ROOT"] = env_value
    return subprocess.run(
        [sys.executable, "-c", "import repo_paths; print(repo_paths.REPO_ROOT)"],
        cwd=cwd, env={**env, "PYTHONPATH": str(TOOLS)}, capture_output=True, text=True,
    )


def test_named_study_root_is_used(tmp_path):
    study = tmp_path / "study"
    study.mkdir()
    done = _root_seen_by_a_fresh_interpreter(str(study), tmp_path)
    assert done.returncode == 0 and Path(done.stdout.strip()) == study.resolve()


def test_named_study_root_that_is_not_a_folder_is_refused(tmp_path):
    done = _root_seen_by_a_fresh_interpreter(str(tmp_path / "missing"), tmp_path)
    assert done.returncode != 0 and "ESA_STUDY_ROOT is not a folder" in done.stderr


def test_checkout_is_the_root_when_nothing_is_named(tmp_path):
    done = _root_seen_by_a_fresh_interpreter(None, tmp_path)
    assert Path(done.stdout.strip()) == TOOLS.parent.resolve()


@pytest.mark.parametrize("command", sorted(PYPROJECT["project"]["scripts"]))
def test_console_commands_point_at_callables(command):
    module, _, function = PYPROJECT["project"]["scripts"][command].partition(":")
    assert module.startswith("esa_audit.")
    source = (TOOLS / (module.split(".", 1)[1] + ".py")).read_text()
    assert f"def {function}(" in source


def test_package_maps_to_tools_and_ships_the_page():
    setuptools = PYPROJECT["tool"]["setuptools"]
    assert setuptools["package-dir"] == {"esa_audit": "tools"}
    patterns = PYPROJECT["tool"]["setuptools"]["package-data"]["esa_audit"]
    for pattern in patterns:
        assert list(TOOLS.glob(pattern)), f"package-data pattern matches nothing: {pattern}"


def test_not_publishable_until_a_licence_is_chosen():
    assert "Private :: Do Not Upload" in PYPROJECT["project"]["classifiers"]
    assert "license" not in PYPROJECT["project"]


def test_importing_the_package_makes_sibling_modules_importable():
    importlib.import_module("tools")
    assert str(TOOLS) in sys.path
