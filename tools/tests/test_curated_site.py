"""Tests for tools/curated_site.py.

Synthetic text only. The synthetic explain outputs of test_curated_plots are
given the run records a real run writes (explainer, threshold, verification,
serving) and a committed-style best_config.json per technique, drawn once into
a gallery, and each test copies that gallery and builds a site from it. The
method page is checked against those records. Adversarial tests prove the tool
refuses, before writing anything: a folder without a gallery, a gallery older
than the explain outputs, techniques explained with different settings, a
technique without its best_config.json, a page that links to an image the site
would not contain, a link that leaves the site, and a destination holding
files the tool did not write.
"""

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIR))

pytest.importorskip("shap")
pytest.importorskip("matplotlib")

import curated_examples as ce  # noqa: E402
import curated_plots as cp  # noqa: E402
import curated_site as cs  # noqa: E402
from test_curated_plots import TECHNIQUES, TEXTS, write_outputs  # noqa: E402

EXPLAINER = {"algorithm": "partition", "mask_token": "...", "masker": "\\W+", "max_evals": 60, "seed": 30}
MAX_DIFFS = {TECHNIQUES[0]: 0.0, TECHNIQUES[1]: 0.0012}


def add_run_records(out, results_dir):
    """What a real explain run records, plus each technique's committed best_config.json."""
    for technique in TECHNIQUES:
        path = out / technique / ce.RUN_FILENAME
        record = json.loads(path.read_text())
        sequential = technique == TECHNIQUES[1]
        record.update({
            "explainer": dict(EXPLAINER),
            "threshold": 0.5,
            "verification": {"n_rows": 16, "tolerance": 0.01, "max_abs_diff": MAX_DIFFS[technique],
                             "passed": True},
            "serving": {"mode": "sequential_checkpoints", "rounds": 3} if sequential else {"mode": "resident"},
        })
        path.write_text(json.dumps(record))
        config = results_dir / technique / "best_config.json"
        config.parent.mkdir(parents=True)
        config.write_text(json.dumps({
            "model_family": "logistic_regression" if not sequential else "a_new_family",
            "feature_family": "tfidf_word",
            "external_pretraining": "Twitter text" if sequential else None,
            "threshold_metric": "accuracy",
        }))


@pytest.fixture(scope="module")
def drawn(tmp_path_factory):
    root = tmp_path_factory.mktemp("drawn")
    out = write_outputs(root / "out")
    add_run_records(out, root / "results")
    cp.run_plots(out, log=lambda *_: None)
    return root


@pytest.fixture
def world(drawn, tmp_path):
    shutil.copytree(drawn, tmp_path / "world")
    return tmp_path / "world"


def build(world, site=None):
    site = world / "site" if site is None else site
    return cs.run_site(world / "out", site, results_dir=world / "results", log=lambda *_: None)


class TestSite:
    def test_every_page_and_image_but_not_the_copies(self, world):
        site = build(world)
        assert (site / "index.html").is_file() and (site / cs.METHOD_FILENAME).is_file()
        assert len(list((site / "examples").glob("*.html"))) == len(TEXTS)
        assert len(list((site / "models").glob("*.html"))) == len(TECHNIQUES)
        assert len(list((site / "by_model").rglob("*.png"))) == len(TECHNIQUES) * (len(TEXTS) + 1)
        assert not (site / "by_example").exists() and not (site / cp.CHECKS_FILENAME).exists()
        for page in site.rglob("*.html"):
            text = page.read_text(encoding="utf-8")
            assert text.count(cs.ROBOTS_META) == 1
            prefix = "../" * len(page.relative_to(site).parts[:-1])
            assert f'href="{prefix}{cs.METHOD_FILENAME}"' in text
        assert (site / ".nojekyll").exists() and (site / cs.MARKER_FILENAME).exists()
        assert f"{len(TEXTS)} hand-written curated texts" in (site / "README.md").read_text(encoding="utf-8")

    def test_a_rebuild_keeps_git_and_replaces_the_rest(self, world):
        site = build(world)
        (site / ".git").mkdir()
        (site / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
        (site / "stale.html").write_text("old page")
        build(world, site)
        assert (site / ".git" / "HEAD").read_text() == "ref: refs/heads/main\n"
        assert not (site / "stale.html").exists() and (site / "index.html").exists()


class TestMethodPage:
    def test_numbers_come_from_the_records(self, world):
        facts = cs.method_facts(world / "out", world / "results")
        assert [r["technique"] for r in facts["techniques"]] == list(TECHNIQUES)
        assert [r["max_abs_diff"] for r in facts["techniques"]] == [MAX_DIFFS[t] for t in TECHNIQUES]
        assert facts["n_waterfalls"] == len(TECHNIQUES) * len(TEXTS)
        assert facts["max_additivity_gap"] < 1e-9
        # The synthetic curated_03 ends with "." and carries a 0.02 raw-text gap under both techniques.
        assert facts["raw_gap_notes"] == {t: 1 for t in TECHNIQUES}
        assert facts["expected"] == {cp.POSITIVE_CLASS_NAME: 2, cp.NEGATIVE_CLASS_NAME: 1}

        page = (build(world) / cs.METHOD_FILENAME).read_text(encoding="utf-8")
        assert f"{len(TEXTS)} posts &times; {len(TECHNIQUES)} models = {len(TEXTS) * len(TECHNIQUES)} waterfalls" in page
        assert "0.0012" in page and "max_evals=60" in page and "within 0.01" in page
        assert f"{TECHNIQUES[1]} was explained one checkpoint at a time" in page and "(3 rounds)" in page
        assert "a new family" not in page and "a_new_family" in page  # an unknown family is shown as written
        assert "maximising validation accuracy" in page

    def test_different_explainer_settings_are_refused(self, world):
        path = world / "out" / TECHNIQUES[0] / ce.RUN_FILENAME
        record = json.loads(path.read_text())
        record["explainer"]["max_evals"] = 500
        path.write_text(json.dumps(record))
        with pytest.raises(cs.CuratedSiteError, match="different"):
            build(world)
        assert not (world / "site").exists()

    def test_a_gallery_older_than_the_outputs_is_refused(self, world):
        checks = world / "out" / cp.PLOTS_DIRNAME / cp.CHECKS_FILENAME
        frame = pd.read_csv(checks)
        frame[frame["technique"] != TECHNIQUES[1]].to_csv(checks, index=False)
        with pytest.raises(cs.CuratedSiteError, match="older than the outputs"):
            build(world)

    def test_a_technique_without_best_config_is_refused(self, world):
        (world / "results" / TECHNIQUES[0] / "best_config.json").unlink()
        with pytest.raises(cs.CuratedSiteError, match="best_config"):
            build(world)


class TestRefusals:
    def test_a_folder_without_a_gallery_is_refused(self, tmp_path):
        with pytest.raises(cs.CuratedSiteError, match="no gallery"):
            cs.run_site(tmp_path, tmp_path / "site", log=lambda *_: None)

    def test_a_page_linking_to_a_missing_image_is_refused(self, world):
        (world / "out" / cp.PLOTS_DIRNAME / "by_model" / TECHNIQUES[0] / "waterfall_curated_02.png").unlink()
        with pytest.raises(cs.CuratedSiteError, match="would not contain"):
            build(world)
        assert not (world / "site").exists()

    def test_a_link_that_leaves_the_site_is_refused(self):
        pages = {"examples/a.html": '<a href="../../secret.html">x</a>'}
        with pytest.raises(cs.CuratedSiteError, match="leave the site"):
            cs.check_links(pages, {"examples/a.html"})
        cs.check_links({"examples/a.html": '<a href="https://example.org/">x</a><a href="#top">y</a>'},
                       {"examples/a.html"})

    def test_a_folder_the_tool_did_not_write_is_refused(self, world):
        site = world / "site"
        site.mkdir()
        (site / "notes.txt").write_text("keep me")
        with pytest.raises(cs.CuratedSiteError, match="did not write"):
            build(world, site)
        assert (site / "notes.txt").read_text() == "keep me"
