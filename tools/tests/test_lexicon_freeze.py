"""The WP3 freeze: the three lexicons must match the version note in data/lexicons/README.md.

Changing a lexicon changes RQ2 and RQ4. The freeze rule (RESEARCH_PLAN section 5,
"Lexicons") allows a change only with a full re-run and a new version note, so a
silent edit to a list, or a note edited without the list, must fail here.
"""

import hashlib
import re
from pathlib import Path

import pytest

from lexicons import read_lexicon
from repo_paths import LEXICON_DIR, LEXICON_FILES

README = LEXICON_DIR / "README.md"
ROW = re.compile(r"^\| `(?P<file>[a-z_]+\.txt)` \| (?P<n>\d+) \| `(?P<sha>[0-9a-f]{64})` \|$")


def version_note(path=README):
    """{file name: (entry count, sha256)} from the version-note table."""
    text = Path(path).read_text(encoding="utf-8")
    if "## Version note: frozen at WP3" not in text:
        raise ValueError("no WP3 version note in the lexicon README")
    note = text.split("## Version note: frozen at WP3", 1)[1]
    rows = {}
    for line in note.splitlines():
        match = ROW.match(line.strip())
        if match:
            rows[match["file"]] = (int(match["n"]), match["sha"])
    return rows


def problems(note, files):
    """Every mismatch between the note and the files on disk."""
    found = []
    for path in files:
        path = Path(path)
        if path.name not in note:
            found.append(f"{path.name}: not in the version note")
            continue
        n, sha = note[path.name]
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha != sha:
            found.append(f"{path.name}: sha256 {actual_sha} differs from the note")
        if len(read_lexicon(path)) != n:
            found.append(f"{path.name}: {len(read_lexicon(path))} entries, the note says {n}")
    return found


def test_committed_lexicons_match_the_version_note():
    note = version_note()
    assert set(note) == {Path(p).name for p in LEXICON_FILES.values()}
    assert problems(note, LEXICON_FILES.values()) == []


def test_an_added_entry_is_refused(tmp_path):
    note = version_note()
    edited = tmp_path / "identity_terms.txt"
    edited.write_bytes(LEXICON_FILES["identity_term"].read_bytes() + b"newgroup\n")
    found = problems(note, [edited])
    assert any("sha256" in p for p in found) and any("entries" in p for p in found)


def test_a_comment_only_edit_is_still_refused(tmp_path):
    """A comment does not change the entries, but the frozen file is byte-pinned."""
    note = version_note()
    edited = tmp_path / "slurs.txt"
    edited.write_bytes(LEXICON_FILES["slur"].read_bytes() + b"# a note\n")
    found = problems(note, [edited])
    assert any("sha256" in p for p in found)
    assert not any("entries" in p for p in found)


def test_a_note_without_the_freeze_heading_is_refused(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# Lexicons\n\nno version note\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no WP3 version note"):
        version_note(readme)


def test_dropped_term_stays_out_of_the_identity_list():
    """Author decision 2026-11-05: aliens was test-derived and is not an identity term."""
    from lexicons import load_lexicons

    lexicons = load_lexicons()
    assert "aliens" not in lexicons["identity_term"]
    assert version_note()["identity_terms.txt"][0] == len(lexicons["identity_term"]) == 96
