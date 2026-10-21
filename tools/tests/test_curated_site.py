"""Tests for tools/curated_site.py.

Synthetic text only. The synthetic explain outputs of test_curated_plots are
drawn once into a gallery, and each test copies that gallery and builds a
site from it. Adversarial tests prove the tool refuses, before writing
anything: a folder without a gallery, a page that links to an image the site
would not contain, a link that leaves the site, and a destination holding
files the tool did not write.
"""

import shutil
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIR))

pytest.importorskip("shap")
pytest.importorskip("matplotlib")

import curated_plots as cp  # noqa: E402
import curated_site as cs  # noqa: E402
from test_curated_plots import TECHNIQUES, TEXTS, write_outputs  # noqa: E402


@pytest.fixture(scope="module")
def drawn(tmp_path_factory):
    out = write_outputs(tmp_path_factory.mktemp("drawn") / "out")
    cp.run_plots(out, log=lambda *_: None)
    return out


@pytest.fixture
def outputs(drawn, tmp_path):
    out = tmp_path / "out"
    shutil.copytree(drawn, out)
    return out


def build(outputs, site):
    return cs.run_site(outputs, site, log=lambda *_: None)


class TestSite:
    def test_every_page_and_image_but_not_the_copies(self, outputs, tmp_path):
        site = build(outputs, tmp_path / "site")
        assert (site / "index.html").is_file()
        assert len(list((site / "examples").glob("*.html"))) == len(TEXTS)
        assert len(list((site / "models").glob("*.html"))) == len(TECHNIQUES)
        assert len(list((site / "by_model").rglob("*.png"))) == len(TECHNIQUES) * (len(TEXTS) + 1)
        assert not (site / "by_example").exists() and not (site / cp.CHECKS_FILENAME).exists()
        for page in site.rglob("*.html"):
            assert page.read_text(encoding="utf-8").count(cs.ROBOTS_META) == 1
        assert (site / ".nojekyll").exists() and (site / cs.MARKER_FILENAME).exists()
        assert f"{len(TEXTS)} hand-written curated texts" in (site / "README.md").read_text(encoding="utf-8")

    def test_a_rebuild_keeps_git_and_replaces_the_rest(self, outputs, tmp_path):
        site = build(outputs, tmp_path / "site")
        (site / ".git").mkdir()
        (site / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
        (site / "stale.html").write_text("old page")
        build(outputs, site)
        assert (site / ".git" / "HEAD").read_text() == "ref: refs/heads/main\n"
        assert not (site / "stale.html").exists() and (site / "index.html").exists()


class TestRefusals:
    def test_a_folder_without_a_gallery_is_refused(self, tmp_path):
        with pytest.raises(cs.CuratedSiteError, match="no gallery"):
            build(tmp_path, tmp_path / "site")

    def test_a_page_linking_to_a_missing_image_is_refused(self, outputs, tmp_path):
        (outputs / cp.PLOTS_DIRNAME / "by_model" / TECHNIQUES[0] / "waterfall_curated_02.png").unlink()
        with pytest.raises(cs.CuratedSiteError, match="would not contain"):
            build(outputs, tmp_path / "site")
        assert not (tmp_path / "site").exists()

    def test_a_link_that_leaves_the_site_is_refused(self):
        pages = {"examples/a.html": '<a href="../../secret.html">x</a>'}
        with pytest.raises(cs.CuratedSiteError, match="leave the site"):
            cs.check_links(pages, {"examples/a.html"})
        cs.check_links({"examples/a.html": '<a href="https://example.org/">x</a><a href="#top">y</a>'},
                       {"examples/a.html"})

    def test_a_folder_the_tool_did_not_write_is_refused(self, outputs, tmp_path):
        site = tmp_path / "site"
        site.mkdir()
        (site / "notes.txt").write_text("keep me")
        with pytest.raises(cs.CuratedSiteError, match="did not write"):
            build(outputs, site)
        assert (site / "notes.txt").read_text() == "keep me"
