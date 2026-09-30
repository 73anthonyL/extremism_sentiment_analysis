"""Load the word lexicons that categorize attributions and find identity terms.

Format: one entry per line, '#' starts a comment, blank lines ignored,
matching is case-insensitive. Entries are normalized to lowercase with
whitespace collapsed. Multi-word entries are kept for phrase matching against
post text (RQ4) but cannot match a word-level attribution artifact (RQ2); the
categorizer reports how many it skipped so the gap is visible.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from repo_paths import CATEGORY_PRECEDENCE, LEXICON_FILES

WHITESPACE_RE = re.compile(r"\s+")


def normalize_entry(entry):
    return WHITESPACE_RE.sub(" ", entry.strip().lower())


def read_lexicon(path):
    """Return the ordered, de-duplicated entries of one lexicon file."""
    path = Path(path)
    if not path.exists():
        return []
    seen = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0]
        entry = normalize_entry(line)
        if entry and entry not in seen:
            seen.append(entry)
    return seen


def load_lexicons(files=None):
    """Return {category: [entries]} for every category in precedence order."""
    files = LEXICON_FILES if files is None else files
    return {category: read_lexicon(files[category]) for category in CATEGORY_PRECEDENCE}


def split_by_arity(entries):
    """Partition entries into (single_words, multi_word_phrases)."""
    singles = [e for e in entries if " " not in e]
    phrases = [e for e in entries if " " in e]
    return singles, phrases


def phrase_pattern(entries):
    """One compiled regex matching any entry at word boundaries, or None.

    Entries are escaped, so lexicon punctuation is literal. Word boundaries are
    approximated by lookarounds on non-word characters so entries starting or
    ending with punctuation (e.g. '/pol/') still anchor correctly.
    """
    entries = [e for e in entries if e]
    if not entries:
        return None
    alternation = "|".join(re.escape(e) for e in sorted(entries, key=len, reverse=True))
    return re.compile(r"(?<![\w])(?:" + alternation + r")(?![\w])", re.IGNORECASE)
