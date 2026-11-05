"""Verify every quantitative claim in docs/paper/CLAIMS.md against its artifact and the draft.

`CLAIMS.md` holds one markdown table with a row per claim:

    | id | artifact | selector | column | value | decimals |

id
    `C` followed by three digits, unique.
artifact
    A repository-relative path to a committed CSV or JSON file.
selector
    CSV: `key=value` or `key!=value` conditions joined by `;`, all of which
    must hold (values are compared as the raw strings in the file; every
    value named must occur in its column, so a misspelt value is refused,
    not silently matched by nothing), or `*` for every row. The selection
    must match at least one row. JSON: `-`.
column
    CSV: a column name, which needs the selection to be exactly one row; or
    an aggregate over the selected rows. `count()` takes no column and its
    value is the row count. `min(col)@N`, `max(col)@N`, `sum(col)@N` and
    `nunique(col)@N` must state N, the number of rows the selection is
    expected to match, so an under-specified selector that picks up extra
    rows is refused.
    JSON: a dotted path whose segments are keys (`[A-Za-z0-9_-]+`), list
    indices (digits) or `[key=value]`, which picks the one list element whose
    `key` equals `value` as a string (none or several is refused).
value
    The value as the draft writes it.
decimals
    The rounding the draft uses, which must agree with how `value` is
    written. An integer d: `value` is a plain decimal with exactly d digits
    after the point, and the artifact must lie within 0.5 x 10^-d. `sN`:
    `value` has N significant figures (for p values such as 1.5e-5). `-`: an
    exact string comparison (booleans case-insensitively); a numeric value
    may not use `-`.

Binding to the draft. Every claim id the draft mentions must be written as
`<value> [Cnnn]`, the claim's value string, one space, the id in brackets,
for example `0.873 [C100]`. Every claim must be bound at least once, every
bound value must equal its claim's value, and an id that appears in any other
form is refused. So a number changed in the prose, or in the table, fails.

The tool exits 1 on any problem and writes nothing.

    python3 tools/check_claims.py
    python3 tools/check_claims.py --claims docs/paper/CLAIMS.md --draft docs/paper/RESULTS_DRAFT.md
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
ANY_ID_PATTERN = re.compile(r"\bC\d{3}\b")
BOUND_PATTERN = re.compile(r"(?P<value>[^\s(|\[\]]+) \[(?P<id>C\d{3})\]")
AGGREGATE_PATTERN = re.compile(r"(?P<how>\w+)\((?P<target>[^()]*)\)(?:@(?P<n>\d+))?")
ROW_AGGREGATES = {"min": min, "max": max, "sum": sum}
JSON_SEGMENT = r"(?:[A-Za-z0-9_-]+|\[[A-Za-z0-9_-]+=[^\]]*\])"
JSON_PATH_PATTERN = re.compile(rf"{JSON_SEGMENT}(?:\.{JSON_SEGMENT})*")
PLAIN_DECIMAL = re.compile(r"-?\d+(?:\.(\d+))?")
NUMBER = re.compile(r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


class ClaimError(ValueError):
    """A claim that cannot be verified, or whose value does not match."""


def _cell(text):
    text = text.strip()
    if len(text) >= 2 and text.startswith("`") and text.endswith("`"):
        text = text[1:-1].strip()
    return text


def parse_claims(path):
    """Return the claim rows of the table whose header is HEADER."""
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
    conditions = []
    for part in selector.split(";"):
        match = re.fullmatch(r"\s*([^=!]+?)\s*(!=|=)\s*(.*?)\s*", part)
        if not match:
            raise ClaimError(f"selector part {part!r} is not key=value or key!=value")
        conditions.append(match.groups())
    return conditions


def _as_float(raw, where):
    if isinstance(raw, bool):
        raise ClaimError(f"{where}: {raw!r} is a boolean, not a number")
    try:
        number = float(raw)
    except (TypeError, ValueError) as error:
        raise ClaimError(f"{where}: {raw!r} is not a number") from error
    if math.isnan(number):
        raise ClaimError(f"{where}: value is NaN")
    return number


def select_rows(rows, fieldnames, selector):
    conditions = _parse_selector(selector)
    for key, _, value in conditions:
        if key not in fieldnames:
            raise ClaimError(f"selector key {key!r} is not a column")
        if not any(r[key] == value for r in rows):
            raise ClaimError(f"selector value {value!r} does not occur in column {key!r}")
    matched = [r for r in rows if all((r[k] == v) == (op == "=") for k, op, v in conditions)]
    if not matched:
        raise ClaimError(f"selector {selector!r} matches no rows")
    return matched


def csv_value(rows, fieldnames, selector, column):
    matched = select_rows(rows, fieldnames, selector)
    aggregate = AGGREGATE_PATTERN.fullmatch(column)
    if aggregate:
        how, target, expected = aggregate.group("how"), aggregate.group("target"), aggregate.group("n")
        if how == "count":
            if target or expected:
                raise ClaimError("count() takes no column and no @N")
            return len(matched)
        if how not in ROW_AGGREGATES and how != "nunique":
            raise ClaimError(f"unknown aggregate {how!r}")
        if expected is None:
            raise ClaimError(f"{how}() needs @N, the number of rows the selector must match")
        if target not in fieldnames:
            raise ClaimError(f"column {target!r} is not in the artifact")
        if len(matched) != int(expected):
            raise ClaimError(f"selector {selector!r} matches {len(matched)} rows, not the stated {expected}")
        if how == "nunique":
            return len({r[target] for r in matched})
        return ROW_AGGREGATES[how]([_as_float(r[target], target) for r in matched])
    if column not in fieldnames:
        raise ClaimError(f"column {column!r} is not in the artifact")
    if len(matched) != 1:
        raise ClaimError(f"selector {selector!r} matches {len(matched)} rows, not exactly one")
    return matched[0][column]


def json_value(payload, selector, path):
    if selector != "-":
        raise ClaimError("a JSON artifact takes selector '-'")
    if not JSON_PATH_PATTERN.fullmatch(path):
        raise ClaimError(f"path {path!r} is not a dotted path of keys, indices and [key=value]")
    node = payload
    for segment in re.findall(JSON_SEGMENT, path):
        if segment.startswith("["):
            key, value = segment[1:-1].split("=", 1)
            if not isinstance(node, list):
                raise ClaimError(f"path {path!r}: {segment} needs a list")
            hits = [e for e in node if isinstance(e, dict) and str(e.get(key)) == value]
            if len(hits) != 1:
                raise ClaimError(f"path {path!r}: {segment} matches {len(hits)} elements, not exactly one")
            node = hits[0]
        elif isinstance(node, list):
            if not segment.isdigit() or int(segment) >= len(node):
                raise ClaimError(f"path {path!r}: no list element {segment!r}")
            node = node[int(segment)]
        elif isinstance(node, dict) and segment in node:
            node = node[segment]
        else:
            raise ClaimError(f"path {path!r}: no key {segment!r}")
    if isinstance(node, (dict, list)):
        raise ClaimError(f"path {path!r} ends at a {type(node).__name__}, not a value")
    return node


def _significant_figures(value):
    mantissa = re.split(r"[eE]", value.lstrip("-"))[0]
    digits = mantissa.replace(".", "").lstrip("0")
    return len(digits)


def check_precision(value, decimals):
    """Refuse a decimals entry that disagrees with how the value is written."""
    numeric = NUMBER.fullmatch(value) is not None
    if decimals == "-":
        if numeric:
            raise ClaimError(f"value {value!r} is numeric; give its decimals instead of '-'")
        return
    if not numeric:
        raise ClaimError(f"value {value!r} is not a number but decimals is {decimals!r}")
    if re.fullmatch(r"\d+", decimals):
        plain = PLAIN_DECIMAL.fullmatch(value)
        if not plain:
            raise ClaimError(f"value {value!r} is not a plain decimal; use sN")
        written = len(plain.group(1) or "")
        if written != int(decimals):
            raise ClaimError(f"value {value!r} has {written} decimal(s), decimals says {decimals}")
        return
    match = re.fullmatch(r"s(\d+)", decimals)
    if not match or int(match.group(1)) < 1:
        raise ClaimError(f"decimals {decimals!r} is not an integer, sN (N >= 1) or -")
    if _significant_figures(value) != int(match.group(1)):
        raise ClaimError(f"value {value!r} does not have {match.group(1)} significant figure(s)")


def tolerance(decimals, claimed):
    if re.fullmatch(r"\d+", decimals):
        return 0.5 * 10 ** (-int(decimals))
    figures = int(decimals[1:])
    if claimed == 0:
        return 0.0
    return 0.5 * 10 ** (math.floor(math.log10(abs(claimed))) - figures + 1)


def compare(actual, claimed, decimals):
    """Raise ClaimError unless `actual` rounds to `claimed` at `decimals`."""
    check_precision(claimed, decimals)
    if decimals == "-":
        a, c = str(actual).strip(), claimed.strip()
        if a.lower() in ("true", "false") and c.lower() in ("true", "false"):
            a, c = a.lower(), c.lower()
        if a != c:
            raise ClaimError(f"artifact has {actual!r}, claim says {claimed!r}")
        return
    claimed_number = _as_float(claimed, "claimed value")
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


def check_draft_binding(claims, draft_path):
    """Every id in the draft is written `<value> [Cnnn]` with its claim's value."""
    text = Path(draft_path).read_text()
    name = Path(draft_path).name
    values = {c["id"]: c["value"] for c in claims}
    problems, bound = [], set()
    for match in BOUND_PATTERN.finditer(text):
        cid, written = match.group("id"), match.group("value")
        if cid not in values:
            problems.append(f"{cid} is cited in {name} but not in the claims table")
        elif written != values[cid]:
            problems.append(f"{cid} is written {written!r} in {name}, the claim says {values[cid]!r}")
        else:
            bound.add(cid)
    loose = len(ANY_ID_PATTERN.findall(text)) - len(BOUND_PATTERN.findall(text))
    if loose:
        problems.append(f"{loose} claim id(s) in {name} are not written as '<value> [Cnnn]'")
    for cid in sorted(set(values) - bound):
        if not any(p.startswith(f"{cid} is written") for p in problems):
            problems.append(f"{cid} is in the claims table but not bound to its value in {name}")
    return problems


def run(claims_path=DEFAULT_CLAIMS, draft_path=DEFAULT_DRAFT, root=REPO_ROOT, check_draft=True):
    """Return (n_verified, list of failure messages)."""
    claims = parse_claims(claims_path)
    cache = ArtifactCache(root)
    failures, verified = [], 0
    for claim in claims:
        try:
            verify_claim(claim, cache)
            verified += 1
        except ClaimError as error:
            failures.append(f"{claim['id']} ({claim['artifact']}): {error}")
    if check_draft:
        if not Path(draft_path).is_file():
            failures.append(f"draft not found: {draft_path}")
        else:
            failures.extend(check_draft_binding(claims, draft_path))
    return verified, failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--claims", default=str(DEFAULT_CLAIMS))
    parser.add_argument("--draft", default=str(DEFAULT_DRAFT))
    parser.add_argument("--no-draft", action="store_true", help="skip the binding check against the draft")
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
    bound = "not checked against the draft" if args.no_draft else "bound to their values in the draft"
    print(f"OK: {verified} claim(s) verified against their artifacts and {bound}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
