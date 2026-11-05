"""Load the word lexicons that categorize attributions and find identity terms.

Format: one entry per line, '#' starts a comment, blank lines ignored,
matching is case-insensitive. Entries are normalized to lowercase with
whitespace collapsed. Entries the word-level masker would split (a space, a
hyphen or any other non-word character: 'white genocide', 'anti-muslim') are
kept for phrase matching against post text (RQ4) but cannot match a
word-level attribution artifact (RQ2); the categorizer reports how many it
skipped so the gap is visible.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from repo_paths import CATEGORY_PRECEDENCE, LEXICON_FILES

WHITESPACE_RE = re.compile(r"\s+")

# The word-level text masker's split pattern. It must equal
# notebook_kit.MASKER_PATTERN (pinned by tools/tests/test_lexicon_coverage.py);
# it is repeated here because notebook_kit deliberately imports nothing from
# the repository and this module should not import the kit's dependencies.
MASKER_PATTERN = r"\W+"


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


def masker_tokens(entry):
    """The word tokens the SHAP text masker would cut `entry` into."""
    return [piece for piece in re.split(MASKER_PATTERN, entry) if piece]


def is_word_level(entry):
    """True if `entry` survives the masker as exactly one token, itself.

    Only such entries can equal a word of a word-level attribution artifact.
    A space, a hyphen or any other non-word character splits an entry
    ('white genocide', 'anti-muslim', '9/11'), and an entry of punctuation
    only yields no token at all; none of these can match.
    """
    return masker_tokens(entry) == [entry]


def split_by_arity(entries):
    """Partition entries into (word_level, unmatchable_at_word_level).

    The second list holds every entry the masker would split or erase, not
    only entries with a space: they can still match post text (RQ4) but can
    never match a word-level attribution artifact (RQ2).
    """
    singles = [e for e in entries if is_word_level(e)]
    phrases = [e for e in entries if not is_word_level(e)]
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
