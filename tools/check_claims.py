"""Verify every quantitative claim in docs/paper/CLAIMS.md against its committed artifact.

The paper draft (`docs/paper/RESULTS_DRAFT.md`) cites each number through a
claim id, singly (`C012`) or as a range (`C001–C056`). `CLAIMS.md` holds one markdown table with a row per claim:

    | id | artifact | selector | column | value | decimals |

id
    `C` followed by three digits, unique. The draft cites it, and this tool
    checks that every id in the table is cited and every cited id exists.
artifact
    A repository-relative path to a committed CSV or JSON file.
selector
    CSV: `key=value;key=value` (all must hold; values compared as the raw
    strings in the file), or `*` for every row. JSON: `-`.
column
    CSV: a column name, which needs the selector to match exactly one row,
    or an aggregate over the matching rows: `count()`, `min(<column>)`,
    `max(<column>)`, `sum(<column>)`.
    JSON: a dotted path. A segment is a key, a list index, or
    `[key=value]`, which picks the one element of a list whose `key` equals
    `value` (as a string).
value
    The value as written in the draft.
decimals
    The rounding the draft uses. An integer d accepts |artifact - value| <=
    0.5 x 10^-d. `sN` accepts N significant figures (for p values such as
    1.5e-5). `-` compares strings exactly (booleans case-insensitively).

The tool refuses, and exits 1, on a value outside its tolerance, a missing
or unreadable artifact, a path outside the repository, an unknown column or
selector key, a selector that does not pick exactly one row, a duplicate id,
an empty table, or an id mismatch with the draft. It reads committed files
only and writes nothing.

    python3 tools/check_claims.py
    python3 tools/check_claims.py --claims docs/paper/CLAIMS.md --draft docs/paper/RESULTS_DRAFT.md
    python3 tools/check_claims.py --no-draft
"""

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from repo_paths import REPO_ROOT  # noqa: E402

DEFAULT_CLAIMS = REPO_ROOT / "docs" / "paper" / "CLAIMS.md"
DEFAULT_DRAFT = REPO_ROOT / "docs" / "paper" / "RESULTS_DRAFT.md"
HEADER = ("id", "artifact", "selector", "column", "value", "decimals")
ID_PATTERN = re.compile(r"^C\d{3}$")
CITED_ID_PATTERN = re.compile(r"\bC\d{3}\b")
# A cited range such as "C001–C056" (en dash or hyphen) cites every id in it.
CITED_RANGE_PATTERN = re.compile(r"\bC(\d{3})\s*[\u2013-]\s*C(\d{3})\b")
AGGREGATES = ("min", "max", "sum")
AGGREGATE_PATTERN = re.compile(r"(\w+)\(([^()]*)\)")


class ClaimError(ValueError):
    """A claim that cannot be verified, or whose value does not match."""


def _cell(text):
    text = text.strip()
    if len(text) >= 2 and text.startswith("`") and text.endswith("`"):
        text = text[1:-1].strip()
    return text


def parse_claims(path):
    """Return the claim rows of the one table whose header is HEADER."""
    path = Path(path)
    if not path.is_file():
        raise ClaimError(f"claims file not found: {path}")
    rows, in_table = [], False
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped.startswith("|"):
            in_table = False
            continue
        cells = [_cell(c) for c in stripped.strip("|").split("|")]
        if tuple(c.lower() for c in cells) == HEADER:
            in_table = True
            continue
        if not in_table or all(set(c) <= set("-: ") for c in cells):
            continue
        if len(cells) != len(HEADER):
            raise ClaimError(f"line {line_no}: expected {len(HEADER)} cells, found {len(cells)}")
        rows.append(dict(zip(HEADER, cells), line=line_no))
    if not rows:
        raise ClaimError(f"no claim rows under a header {' | '.join(HEADER)} in {path}")
    seen = set()
    for row in rows:
        if not ID_PATTERN.match(row["id"]):
            raise ClaimError(f"line {row['line']}: id {row['id']!r} is not C followed by three digits")
        if row["id"] in seen:
            raise ClaimError(f"line {row['line']}: duplicate id {row['id']}")
        seen.add(row["id"])
    return rows


def resolve_artifact(relative, root=REPO_ROOT):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ClaimError(f"artifact {relative!r} is outside the repository")
    if not path.is_file():
        raise ClaimError(f"artifact {relative!r} does not exist")
    return path


class ArtifactCache:
    def __init__(self, root=REPO_ROOT):
        self.root = root
        self._cache = {}

    def load(self, relative):
        if relative not in self._cache:
            path = resolve_artifact(relative, self.root)
            if path.suffix == ".csv":
                with open(path, newline="") as handle:
                    reader = csv.DictReader(handle)
                    self._cache[relative] = ("csv", list(reader), reader.fieldnames or [])
            elif path.suffix == ".json":
                try:
                    self._cache[relative] = ("json", json.loads(path.read_text()), None)
                except json.JSONDecodeError as error:
                    raise ClaimError(f"artifact {relative!r} is not valid JSON: {error}") from error
            else:
                raise ClaimError(f"artifact {relative!r} is neither .csv nor .json")
        return self._cache[relative]


def _parse_selector(selector):
    if selector == "*":
        return []
    pairs = []
    for part in selector.split(";"):
        if "=" not in part:
            raise ClaimError(f"selector part {part!r} is not key=value")
        key, value = part.split("=", 1)
        pairs.append((key.strip(), value.strip()))
    return pairs


def _as_float(raw, where):
    try:
        number = float(raw)
    except (TypeError, ValueError) as error:
        raise ClaimError(f"{where}: {raw!r} is not a number") from error
    if math.isnan(number):
        raise ClaimError(f"{where}: value is NaN")
    return number


def csv_value(rows, fieldnames, selector, column):
    pairs = _parse_selector(selector)
    for key, _ in pairs:
        if key not in fieldnames:
            raise ClaimError(f"selector key {key!r} is not a column")
    matched = [r for r in rows if all(r[k] == v for k, v in pairs)]
    aggregate = AGGREGATE_PATTERN.fullmatch(column)
    if aggregate:
        how, target = aggregate.groups()
        if how == "count":
            if target:
                raise ClaimError("count() takes no column")
            return len(matched)
        if how not in AGGREGATES:
            raise ClaimError(f"unknown aggregate {how!r}")
        if target not in fieldnames:
            raise ClaimError(f"column {target!r} is not in the artifact")
        if not matched:
            raise ClaimError(f"selector {selector!r} matches no rows")
        values = [_as_float(r[target], target) for r in matched]
        return {"min": min, "max": max, "sum": sum}[how](values)
    if column not in fieldnames:
        raise ClaimError(f"column {column!r} is not in the artifact")
    if len(matched) != 1:
        raise ClaimError(f"selector {selector!r} matches {len(matched)} rows, not exactly one")
    return matched[0][column]


_SEGMENT = re.compile(r"\[([^\]=]+)=([^\]]*)\]|([^.\[\]]+)")


def json_value(payload, selector, path):
    if selector != "-":
        raise ClaimError("a JSON artifact takes selector '-'")
    node = payload
    for match in _SEGMENT.finditer(path):
        key, value, plain = match.groups()
        if plain is not None:
            if isinstance(node, list):
                if not plain.isdigit() or int(plain) >= len(node):
                    raise ClaimError(f"path {path!r}: no list element {plain!r}")
                node = node[int(plain)]
            elif isinstance(node, dict) and plain in node:
                node = node[plain]
            else:
                raise ClaimError(f"path {path!r}: no key {plain!r}")
        else:
            if not isinstance(node, list):
                raise ClaimError(f"path {path!r}: [{key}={value}] needs a list")
            hits = [e for e in node if isinstance(e, dict) and str(e.get(key)) == value]
            if len(hits) != 1:
                raise ClaimError(f"path {path!r}: [{key}={value}] matches {len(hits)} elements")
            node = hits[0]
    if isinstance(node, (dict, list)):
        raise ClaimError(f"path {path!r} ends at a {type(node).__name__}, not a value")
    return node


def tolerance(decimals, claimed):
    if re.fullmatch(r"\d+", decimals):
        return 0.5 * 10 ** (-int(decimals))
    match = re.fullmatch(r"s(\d+)", decimals)
    if match:
        figures = int(match.group(1))
        if figures < 1:
            raise ClaimError("significant figures must be at least 1")
        if claimed == 0:
            return 0.0
        return 0.5 * 10 ** (math.floor(math.log10(abs(claimed))) - figures + 1)
    raise ClaimError(f"decimals {decimals!r} is not an integer, sN or -")


def compare(actual, claimed, decimals):
    """Raise ClaimError unless `actual` rounds to `claimed` at `decimals`."""
    if decimals == "-":
        a, c = str(actual).strip(), claimed.strip()
        if a.lower() in ("true", "false") or c.lower() in ("true", "false"):
            a, c = a.lower(), c.lower()
        if a != c:
            raise ClaimError(f"artifact has {actual!r}, claim says {claimed!r}")
        return
    claimed_number = _as_float(claimed, "claimed value")
    if isinstance(actual, bool):
        raise ClaimError("artifact value is a boolean; use decimals '-'")
    actual_number = _as_float(actual, "artifact value")
    allowed = tolerance(decimals, claimed_number)
    if abs(actual_number - claimed_number) > allowed + 1e-12 * max(1.0, abs(claimed_number)):
        raise ClaimError(f"artifact has {actual_number!r}, claim says {claimed} (tolerance {allowed:g})")


def verify_claim(claim, cache):
    kind, payload, fieldnames = cache.load(claim["artifact"])
    if kind == "csv":
        actual = csv_value(payload, fieldnames, claim["selector"], claim["column"])
    else:
        actual = json_value(payload, claim["selector"], claim["column"])
    compare(actual, claim["value"], claim["decimals"])
    return actual


def check_draft_ids(claims, draft_path):
    text = Path(draft_path).read_text()
    cited = set(CITED_ID_PATTERN.findall(text))
    for start, end in CITED_RANGE_PATTERN.findall(text):
        if int(end) < int(start):
            raise ClaimError(f"cited range C{start}-C{end} runs backwards")
        cited.update(f"C{n:03d}" for n in range(int(start), int(end) + 1))
    listed = {c["id"] for c in claims}
    problems = []
    for missing in sorted(listed - cited):
        problems.append(f"{missing} is in the claims table but not cited in {Path(draft_path).name}")
    for unknown in sorted(cited - listed):
        problems.append(f"{unknown} is cited in {Path(draft_path).name} but not in the claims table")
    return problems


def run(claims_path=DEFAULT_CLAIMS, draft_path=DEFAULT_DRAFT, root=REPO_ROOT, check_draft=True):
    """Return (n_verified, list of failure messages)."""
    claims = parse_claims(claims_path)
    cache = ArtifactCache(root)
    failures = []
    for claim in claims:
        try:
            verify_claim(claim, cache)
        except ClaimError as error:
            failures.append(f"{claim['id']} ({claim['artifact']}): {error}")
    if check_draft:
        if not Path(draft_path).is_file():
            failures.append(f"draft not found: {draft_path}")
        else:
            failures.extend(check_draft_ids(claims, draft_path))
    return len(claims) - sum(1 for f in failures if re.match(r"C\d{3} \(", f)), failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--claims", default=str(DEFAULT_CLAIMS))
    parser.add_argument("--draft", default=str(DEFAULT_DRAFT))
    parser.add_argument("--no-draft", action="store_true", help="skip the id cross-check against the draft")
    parser.add_argument("--root", default=str(REPO_ROOT), help="directory artifact paths are relative to")
    args = parser.parse_args(argv)
    try:
        verified, failures = run(args.claims, args.draft, root=args.root, check_draft=not args.no_draft)
    except ClaimError as error:
        print(f"REFUSED: {error}")
        return 1
    for failure in failures:
        print(f"FAIL  {failure}")
    if failures:
        print(f"REFUSED: {len(failures)} problem(s); {verified} claim(s) verified")
        return 1
    print(f"OK: {verified} claim(s) verified against their artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
