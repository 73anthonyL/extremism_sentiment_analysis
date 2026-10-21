"""A static copy of the curated SHAP gallery that GitHub Pages can serve as is.

Reads the gallery `tools/curated_plots.py` wrote under `<out>/plots/` and
writes a site folder with `index.html` as the front page:

    index.html, examples/, models/   the gallery pages
    by_model/                        the beeswarm and waterfall images (the by_example/
                                     copies are left out; no page links to them)
    README.md, .nojekyll             what the site is; serve the files without Jekyll
    .curated_site                    marks the folder as this tool's output

Every page gains a robots `noindex` meta, since the pages hold the curated
texts. Nothing is redrawn: the pages and images are the gallery's own.

The site holds the curated texts, so it belongs in its own folder under
`external/` (untracked here) and is published from there as a separate,
deliberate step. A rebuild replaces everything in the folder except `.git`,
so the folder can be its own repository.

The tool refuses, before writing anything: a plots folder without the
gallery, a page that links to a file the site would not contain, and a
destination that holds files this tool did not write.

Usage:

    python3 tools/curated_site.py --out external/curated_examples_34 --site external/curated_shap_site
"""

import argparse
import posixpath
import re
import shutil
import sys
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))

from curated_examples import DEFAULT_OUT  # noqa: E402
from curated_plots import PLOTS_DIRNAME  # noqa: E402
from repo_paths import REPO_ROOT  # noqa: E402

DEFAULT_SITE = REPO_ROOT / "external" / "curated_shap_site"
MARKER_FILENAME = ".curated_site"
MARKER_TEXT = "Written by tools/curated_site.py. A rebuild replaces everything here except .git.\n"
KEPT_ON_REBUILD = frozenset({".git"})
PAGE_FOLDERS = ("examples", "models")
IMAGE_FOLDER = "by_model"
VIEWPORT_META = '<meta name="viewport" content="width=device-width, initial-scale=1">'
ROBOTS_META = '<meta name="robots" content="noindex, nofollow">'
LINK_PATTERN = re.compile(r'(?:href|src)="([^"]*)"')
SCHEME_PATTERN = re.compile(r"^[a-z][a-z0-9+.-]*:", re.I)
PROJECT_URL = "https://github.com/73anthonyL/extremism_sentiment_analysis"


class CuratedSiteError(ValueError):
    """The gallery cannot be turned into a faithful static site."""


def adapt_page(page, name):
    """One gallery page with the robots meta added."""
    if page.count(VIEWPORT_META) != 1:
        raise CuratedSiteError(f"{name}: not a curated_plots.py page (expected one viewport meta)")
    return page.replace(VIEWPORT_META, VIEWPORT_META + ROBOTS_META)


def check_links(pages, files):
    """Refuse a relative href/src that leaves the site or names a file it will not contain.

    `pages` maps a site-relative page path to its HTML; `files` is every
    site-relative path the site will hold.
    """
    broken = []
    for name, text in pages.items():
        base = PurePosixPath(name).parent
        for target in LINK_PATTERN.findall(text):
            target = target.split("#", 1)[0]
            if not target or SCHEME_PATTERN.match(target):
                continue
            resolved = posixpath.normpath((base / target).as_posix())
            if resolved.startswith("..") or resolved not in files:
                broken.append(f"{name} -> {target}")
    if broken:
        raise CuratedSiteError(
            f"{len(broken)} link(s) leave the site or point to a file it would not contain, "
            f"e.g. {', '.join(broken[:3])}"
        )


def readme(n_texts, n_techniques):
    return (
        "# Curated SHAP gallery\n\n"
        f"SHAP waterfall and beeswarm plots for {n_texts} hand-written curated texts under each of "
        f"the {n_techniques} techniques of [extremism_sentiment_analysis]({PROJECT_URL}). "
        "Open `index.html`.\n\n"
        "Built by `tools/curated_site.py` from the gallery of `tools/curated_plots.py`. The curated "
        "texts are not dataset rows; they illustrate model behaviour and support no accuracy claim. "
        "This is not a moderation tool.\n"
    )


def _prepare_destination(site_dir):
    """Empty the destination except .git, refusing a folder this tool did not write."""
    if site_dir.exists():
        leftovers = [p for p in site_dir.iterdir() if p.name not in KEPT_ON_REBUILD]
        if leftovers and not (site_dir / MARKER_FILENAME).is_file():
            raise CuratedSiteError(
                f"{site_dir} holds {len(leftovers)} entries this tool did not write (no "
                f"{MARKER_FILENAME}); choose a new or empty --site folder"
            )
        for path in leftovers:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    site_dir.mkdir(parents=True, exist_ok=True)


def run_site(out_dir=DEFAULT_OUT, site_dir=DEFAULT_SITE, log=print):
    """Check the gallery, then write the site folder. Returns the site folder."""
    out_dir, site_dir = Path(out_dir), Path(site_dir)
    plots_dir = out_dir / PLOTS_DIRNAME
    if not (plots_dir / "index.html").is_file():
        raise CuratedSiteError(f"{plots_dir}: no gallery; run `curated_plots.py --out {out_dir}` first")

    sources = [plots_dir / "index.html"]
    for folder in PAGE_FOLDERS:
        sources += sorted((plots_dir / folder).glob("*.html"))
    pages = {}
    for source in sources:
        name = source.relative_to(plots_dir).as_posix()
        pages[name] = adapt_page(source.read_text(encoding="utf-8"), name)
    images = sorted(p for p in (plots_dir / IMAGE_FOLDER).rglob("*") if p.is_file())
    extras = {
        "README.md": readme(len(list((plots_dir / "examples").glob("*.html"))),
                            len(list((plots_dir / "models").glob("*.html")))),
        ".nojekyll": "",
        MARKER_FILENAME: MARKER_TEXT,
    }
    files = set(pages) | set(extras) | {p.relative_to(plots_dir).as_posix() for p in images}
    check_links(pages, files)

    _prepare_destination(site_dir)
    for name, text in {**pages, **extras}.items():
        target = site_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    for image in images:
        target = site_dir / image.relative_to(plots_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(image, target)
    log(f"wrote {len(pages)} pages and {len(images)} images to {site_dir}")
    return site_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="an output folder of curated_examples.py explain")
    parser.add_argument("--site", default=str(DEFAULT_SITE), help="the site folder to (re)write")
    args = parser.parse_args(argv)
    try:
        run_site(args.out, args.site)
    except CuratedSiteError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
