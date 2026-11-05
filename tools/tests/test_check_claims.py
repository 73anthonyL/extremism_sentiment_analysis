"""Behavioural and adversarial tests for tools/check_claims.py on synthetic artifacts."""

import json
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIR))

import check_claims  # noqa: E402
from check_claims import ClaimError, check_precision, compare, parse_claims, run  # noqa: E402

HEADER = "| id | artifact | selector | column | value | decimals |\n|---|---|---|---|---|---|\n"
PAIRS = "`results_summary/rq/pairs.csv`"
CARD = "`results_summary/card.json`"


@pytest.fixture
def repo(tmp_path):
    rq = tmp_path / "results_summary" / "rq"
    rq.mkdir(parents=True)
    (rq / "pairs.csv").write_text(
        "technique_a,technique_b,holm_p,significant_holm,note\n"
        "01_A,02_B,1.0,False,x\n"
        "01_A,03_C,0.049116,True,\n"
        "02_B,03_C,0.0000152,True,nan\n"
    )
    (tmp_path / "results_summary" / "card.json").write_text(json.dumps(
        {"rq1": {"test_accuracy": 0.855556, "passed": True,
                 "pairs": [{"other": "02_B", "row_gap": 1}, {"other": "03_C", "row_gap": -24},
                           {"other": "03_C", "row_gap": -25}]}}))
    return tmp_path


def _row(cid, artifact, selector, column, value, decimals):
    return f"{cid} | {artifact} | `{selector}` | `{column}` | {value} | {decimals}"


GOOD = [
    _row("C001", PAIRS, "technique_a=01_A;technique_b=03_C", "holm_p", "0.049", 3),
    _row("C002", PAIRS, "significant_holm=True", "count()", "2", 0),
    _row("C003", PAIRS, "*", "count()", "3", 0),
    _row("C004", PAIRS, "*", "max(holm_p)@3", "1.0", 1),
    _row("C005", PAIRS, "technique_a=02_B;technique_b=03_C", "holm_p", "1.5e-5", "s2"),
    _row("C006", PAIRS, "technique_a=01_A;technique_b=02_B", "significant_holm", "false", "-"),
    _row("C007", CARD, "-", "rq1.test_accuracy", "0.856", 3),
    _row("C008", CARD, "-", "rq1.pairs.[other=02_B].row_gap", "1", 0),
    _row("C009", PAIRS, "technique_a!=02_B", "min(holm_p)@2", "0.049", 3),
    _row("C010", PAIRS, "*", "nunique(technique_a)@3", "2", 0),
]


def _value(row):
    return row.split("|")[4].strip()


def _write(tmp_path, rows, draft=None):
    claims = tmp_path / "CLAIMS.md"
    claims.write_text("# Claims\n\n" + HEADER + "".join(f"| {r} |\n" for r in rows))
    path = tmp_path / "DRAFT.md"
    if draft is None:
        draft = "Results: " + "; ".join(f"{_value(r)} [{r.split('|')[0].strip()}]" for r in rows) + ".\n"
    path.write_text(draft)
    return claims, path


def _fails(repo, rows, message, draft=None):
    claims, draft_path = _write(repo, rows, draft)
    _, failures = run(claims, draft_path, root=repo)
    assert len(failures) == 1 and message in failures[0], failures
    return failures[0]


def test_good_claims_verify(repo):
    claims, draft = _write(repo, GOOD)
    verified, failures = run(claims, draft, root=repo)
    assert failures == [] and verified == len(GOOD)


# --- values and artifacts ---------------------------------------------------------------------

def test_altered_value_is_refused(repo):
    """Adversarial: one digit changed in a claimed value (and in the draft, so only the artifact disagrees)."""
    altered = list(GOOD)
    altered[0] = altered[0].replace("| 0.049 |", "| 0.059 |")
    failure = _fails(repo, altered, "0.059")
    assert failure.startswith("C001")


def test_altered_value_fails_main(repo, capsys):
    claims, draft = _write(repo, [GOOD[6].replace("0.856", "0.866")])
    assert check_claims.main(["--claims", str(claims), "--draft", str(draft), "--root", str(repo)]) == 1
    assert "REFUSED" in capsys.readouterr().out
    claims, draft = _write(repo, [GOOD[6]])
    assert check_claims.main(["--claims", str(claims), "--draft", str(draft), "--root", str(repo)]) == 0


def test_missing_artifact_is_refused(repo):
    """Adversarial: a claim pointing at an artifact that does not exist."""
    _fails(repo, [_row("C001", "`results_summary/rq/not_there.csv`", "*", "count()", "3", 0)], "does not exist")


def test_path_outside_repository_is_refused(repo, tmp_path_factory):
    outside = tmp_path_factory.mktemp("elsewhere") / "x.csv"
    outside.write_text("a\n1\n")
    _fails(repo, [_row("C001", f"`../{outside.parent.name}/x.csv`", "*", "count()", "1", 0)], "outside")


# --- the draft binding --------------------------------------------------------------------------

def test_changed_prose_number_is_refused(repo):
    """Adversarial: the artifact and the claims table agree, but the draft prints another number."""
    rows = [_row("C100", CARD, "-", "rq1.test_accuracy", "0.856", 3)]
    _fails(repo, rows, "written '0.956'", draft="Accuracy is 0.956 [C100].\n")


def test_changed_prose_number_in_committed_draft_is_refused(tmp_path):
    """Adversarial on the real files: 0.873 [C100] changed to 0.973 [C100] in a copy of the draft."""
    if not check_claims.DEFAULT_CLAIMS.is_file():
        pytest.skip("no committed CLAIMS.md")
    text = check_claims.DEFAULT_DRAFT.read_text()
    assert "0.873 [C100]" in text
    copy = tmp_path / "RESULTS_DRAFT.md"
    copy.write_text(text.replace("0.873 [C100]", "0.973 [C100]"))
    _, failures = run(check_claims.DEFAULT_CLAIMS, copy)
    assert failures == ["C100 is written '0.973' in RESULTS_DRAFT.md, the claim says '0.873'"]


def test_unbound_and_loose_ids_are_refused(repo):
    rows = GOOD[:2]
    _fails(repo, rows, "not bound", draft="Only 0.049 [C001].\n")
    claims, draft = _write(repo, rows, draft="0.049 [C001] and 2 [C002], see C002.\n")
    _, failures = run(claims, draft, root=repo)
    assert failures == ["1 claim id(s) in DRAFT.md are not written as '<value> [Cnnn]'"]


def test_unknown_cited_id_is_refused(repo):
    _fails(repo, GOOD[:1], "C077 is cited", draft="0.049 [C001] and 5 [C077].\n")


# --- selectors -----------------------------------------------------------------------------------

@pytest.mark.parametrize("row, message", [
    (_row("C001", PAIRS, "technique_a=01_A", "holm_p", "1.0", 1), "exactly one"),
    (_row("C001", PAIRS, "nope=1", "count()", "0", 0), "not a column"),
    (_row("C001", PAIRS, "technique_a=01_Z", "count()", "0", 0), "does not occur"),
    (_row("C001", PAIRS, "technique_a!=01_Z", "count()", "3", 0), "does not occur"),
    (_row("C001", PAIRS, "technique_a=02_B;technique_b=02_B", "count()", "0", 0), "matches no rows"),
    (_row("C001", PAIRS, "technique_a", "count()", "3", 0), "not key=value"),
    (_row("C001", PAIRS, "-", "count()", "3", 0), "not key=value"),
])
def test_bad_selectors_are_refused(repo, row, message):
    _fails(repo, [row], message)


# --- columns and aggregates ----------------------------------------------------------------------

@pytest.mark.parametrize("row, message", [
    (_row("C001", PAIRS, "*", "max(nope)@3", "0", 0), "not in the artifact"),
    (_row("C001", PAIRS, "*", "median(holm_p)@3", "0", 0), "unknown aggregate"),
    (_row("C001", PAIRS, "*", "max(holm_p)", "1.0", 1), "needs @N"),
    (_row("C001", PAIRS, "*", "max(holm_p)@2", "1.0", 1), "not the stated 2"),
    (_row("C001", PAIRS, "*", "count(holm_p)", "3", 0), "takes no column"),
    (_row("C001", PAIRS, "*", "count()@3", "3", 0), "takes no column"),
    (_row("C001", PAIRS, "technique_a=01_A;technique_b=02_B", "nope", "1.0", 1), "not in the artifact"),
    (_row("C001", PAIRS, "technique_a=01_A;technique_b=02_B", "significant_holm", "0", 0), "not a number"),
    (_row("C001", PAIRS, "technique_a=01_A;technique_b=03_C", "note", "0", 0), "not a number"),
    (_row("C001", PAIRS, "technique_a=02_B;technique_b=03_C", "note", "0", 0), "NaN"),
])
def test_bad_columns_are_refused(repo, row, message):
    _fails(repo, [row], message)


def test_column_named_like_an_aggregate_is_read_as_a_column(repo):
    (repo / "results_summary" / "rq" / "dist.csv").write_text("split,count\ntest,280\nvalidation,281\n")
    dist = "`results_summary/rq/dist.csv`"
    rows = [_row("C001", dist, "split=test", "count", "280", 0), _row("C002", dist, "*", "count()", "2", 0)]
    claims, draft = _write(repo, rows)
    assert run(claims, draft, root=repo)[1] == []


# --- JSON paths ----------------------------------------------------------------------------------

@pytest.mark.parametrize("row, message", [
    (_row("C001", CARD, "-", "rq1.nope", "0", 0), "no key"),
    (_row("C001", CARD, "-", "rq1.pairs.[other=09_Z].row_gap", "0", 0), "matches 0 elements"),
    (_row("C001", CARD, "-", "rq1.pairs.[other=03_C].row_gap", "-24", 0), "matches 2 elements"),
    (_row("C001", CARD, "-", "rq1", "0", 0), "not a value"),
    (_row("C001", CARD, "*", "rq1.test_accuracy", "0.856", 3), "selector '-'"),
    (_row("C001", CARD, "-", "rq1..test_accuracy", "0.856", 3), "not a dotted path"),
    (_row("C001", CARD, "-", "rq1.pairs[other=02_B].row_gap", "1", 0), "not a dotted path"),
    (_row("C001", CARD, "-", "rq1.pairs.7.row_gap", "1", 0), "no list element"),
    (_row("C001", CARD, "-", "rq1.passed", "1", 0), "boolean"),
])
def test_bad_json_paths_are_refused(repo, row, message):
    _fails(repo, [row], message)


# --- ids and the table ---------------------------------------------------------------------------

def test_duplicate_id_is_refused(repo):
    claims, _ = _write(repo, [GOOD[0], GOOD[1].replace("C002", "C001")])
    with pytest.raises(ClaimError, match="duplicate"):
        parse_claims(claims)


@pytest.mark.parametrize("bad_id", ["C1", "C0001", "X001", "c001"])
def test_malformed_id_is_refused(repo, bad_id):
    claims, _ = _write(repo, [GOOD[0].replace("C001", bad_id)])
    with pytest.raises(ClaimError, match="three digits"):
        parse_claims(claims)


def test_empty_table_is_refused(tmp_path):
    path = tmp_path / "CLAIMS.md"
    path.write_text("# Claims\n\n" + HEADER)
    with pytest.raises(ClaimError, match="no claim rows"):
        parse_claims(path)


# --- precision and comparison --------------------------------------------------------------------

@pytest.mark.parametrize("value, decimals", [
    ("0.873", "2"), ("0.87", "3"), ("12", "1"), ("1.5e-5", "5"), ("1.5e-5", "s3"), ("0.0023", "s3"),
    ("0.5", "-"), ("PASS", "3"), ("0.5", "x"), ("0.5", "s0"),
])
def test_decimals_inconsistent_with_value_is_refused(value, decimals):
    with pytest.raises(ClaimError):
        check_precision(value, decimals)


@pytest.mark.parametrize("value, decimals", [
    ("0.873", "3"), ("-24", "0"), ("1.00", "2"), ("1.5e-5", "s2"), ("0.0023", "s2"), ("0.0023", "4"),
    ("PASS", "-"), ("false", "-"),
])
def test_decimals_consistent_with_value_passes(value, decimals):
    check_precision(value, decimals)


def test_rounding_tolerance():
    compare("0.826362", "0.83", "2")
    compare("0.826362", "0.826", "3")
    with pytest.raises(ClaimError):
        compare("0.826362", "0.82", "2")
    compare("1.50e-05", "1.5e-5", "s2")
    with pytest.raises(ClaimError):
        compare("1.6e-05", "1.5e-5", "s2")
    compare("12", "12", "0")
    with pytest.raises(ClaimError):
        compare("13", "12", "0")
    with pytest.raises(ClaimError):
        compare("", "0.5", "1")
    with pytest.raises(ClaimError):
        compare(True, "1", "0")


def test_string_compare_is_case_insensitive_for_booleans_only():
    compare("True", "true", "-")
    compare(False, "FALSE", "-")
    compare("PASS", "PASS", "-")
    with pytest.raises(ClaimError):
        compare("PASS", "pass", "-")
    with pytest.raises(ClaimError):
        compare("True", "false", "-")


def test_committed_claims_verify():
    """The committed CLAIMS.md matches the committed artifacts and is bound in the draft."""
    if not check_claims.DEFAULT_CLAIMS.is_file():
        pytest.skip("no committed CLAIMS.md")
    verified, failures = run()
    assert failures == [] and verified > 0
