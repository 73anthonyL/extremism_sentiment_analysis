"""Adversarial tests for tools/local_foundation.py."""

import sys
from pathlib import Path

import pandas as pd
import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIR))

import local_foundation  # noqa: E402
from repo_paths import REPO_ROOT, RESULTS_DIR  # noqa: E402


@pytest.mark.parametrize("inside", [REPO_ROOT, RESULTS_DIR / "kit_inputs", REPO_ROOT / "notebooks" / "x"])
def test_a_destination_inside_the_repo_is_refused(inside):
    with pytest.raises(local_foundation.LocalFoundationError, match="inside the repository"):
        local_foundation.check_out_dir(inside)


def test_external_and_outside_destinations_are_allowed(tmp_path):
    assert local_foundation.check_out_dir(tmp_path) == tmp_path.resolve()
    local_foundation.check_out_dir(REPO_ROOT / "external" / "kit_inputs")


def _split_and_processed(tmp_path):
    assignments = pd.DataFrame({"row_id": ["ex_000000", "ex_000001"], "label": [0, 1],
                                "text_hash": ["h0", "h1"], "split": ["train", "test"]})
    split_csv = tmp_path / "split.csv"
    assignments.to_csv(split_csv, index=False)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    processed = pd.DataFrame({"row_id": ["ex_000000", "ex_000001"], "text": ["a", "b"], "label": [0, 1],
                              "text_hash": ["h0", "h1"]})
    return split_csv, manifest, processed


def test_a_processed_dataset_disagreeing_with_the_split_is_refused(tmp_path):
    split_csv, manifest, processed = _split_and_processed(tmp_path)
    processed.loc[1, "text_hash"] = "edited"
    out = tmp_path / "out"
    with pytest.raises(local_foundation.LocalFoundationError, match="disagrees"):
        local_foundation.build(out, split_csv, manifest, processed=processed)
    assert not out.exists()


def test_writes_the_three_inputs_under_research_foundation(tmp_path):
    split_csv, manifest, processed = _split_and_processed(tmp_path)
    folder = local_foundation.build(tmp_path / "out", split_csv, manifest, processed=processed)
    assert folder.name == "research_foundation"
    assert sorted(p.name for p in folder.iterdir()) == [
        "dataset_manifest.json", "processed_dataset.csv", "split_assignments.csv"]


def test_the_committed_dataset_matches_the_committed_split(tmp_path):
    folder = local_foundation.build(tmp_path / "out")
    assert len(pd.read_csv(folder / "processed_dataset.csv")) == len(pd.read_csv(folder / "split_assignments.csv"))
