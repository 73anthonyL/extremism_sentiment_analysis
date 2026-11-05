"""Behavioural and adversarial tests for tools/check_claims.py on synthetic artifacts."""

import json
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIR))

import check_claims  # noqa: E402
from check_claims import ClaimError, compare, parse_claims, run  # noqa: E402

HEADER = "| id | artifact | selector | column | value | decimals |\n|---|---|---|---|---|---|\n"


@pytest.fixture
def repo(tmp_path):
    rq = tmp_path / "results_summary" / "rq"
    rq.mkdir(parents=True)
    (rq / "pairs.csv").write_text(
        "technique_a,technique_b,holm_p,significant_holm\n"
        "01_A,02_B,1.0,False\n"
        "01_A,03_C,0.049116,True\n"
        "02_B,03_C,0.0000152,True\n"
    )
    (tmp_path / "results_summary" / "card.json").write_text(json.dumps(
        {"rq1": {"test_accuracy": 0.855556,
                 "pairs": [{"other": "02_B", "row_gap": 1}, {"other": "03_C", "row_gap": -24}]}}))
    return tmp_path


def _write(tmp_path, rows, draft_ids=None):
    claims = tmp_path / "CLAIMS.md"
    claims.write_text("# Claims\n\n" + HEADER + "".join(f"| {r} |\n" for r in rows))
    draft = tmp_path / "DRAFT.md"
    ids = draft_ids if draft_ids is not None else [r.split("|")[0].strip() for r in rows]
    draft.write_text("Results " + " ".join(f"[{i}]" for i in ids) + "\n")
    return claims, draft


GOOD = [
    "C001 | `results_summary/rq/pairs.csv` | `technique_a=01_A;technique_b=03_C` | holm_p | 0.049 | 3",
    "C002 | `results_summary/rq/pairs.csv` | `significant_holm=True` | count() | 2 | 0",
    "C003 | `results_summary/rq/pairs.csv` | `*` | count() | 3 | 0",
    "C004 | `results_summary/rq/pairs.csv` | `*` | max(holm_p) | 1.0 | 1",
    "C005 | `results_summary/rq/pairs.csv` | `technique_a=02_B;technique_b=03_C` | holm_p | 1.5e-5 | s2",
    "C006 | `results_summary/rq/pairs.csv` | `technique_a=01_A;technique_b=02_B` | significant_holm | false | -",
    "C007 | `results_summary/card.json` | - | rq1.test_accuracy | 0.856 | 3",
    "C008 | `results_summary/card.json` | - | rq1.pairs.[other=03_C].row_gap | -24 | 0",
]


def test_good_claims_verify(repo):
    claims, draft = _write(repo, GOOD)
    verified, failures = run(claims, draft, root=repo)
    assert failures == [] and verified == len(GOOD)


def test_altered_value_is_refused(repo):
    """Adversarial: one digit changed in a claimed value."""
    altered = list(GOOD)
    altered[0] = altered[0].replace("| 0.049 |", "| 0.059 |")
    claims, draft = _write(repo, altered)
    verified, failures = run(claims, draft, root=repo)
    assert len(failures) == 1 and failures[0].startswith("C001") and "0.059" in failures[0]
    assert verified == len(GOOD) - 1


def test_altered_value_fails_main(repo, capsys):
    """Adversarial, through the CLI: an altered JSON value exits 1 and says REFUSED."""
    claims, draft = _write(repo, [GOOD[6].replace("0.856", "0.866")])
    assert check_claims.main(["--claims", str(claims), "--draft", str(draft), "--root", str(repo)]) == 1
    assert "REFUSED" in capsys.readouterr().out
    claims, draft = _write(repo, [GOOD[6]])
    assert check_claims.main(["--claims", str(claims), "--draft", str(draft), "--root", str(repo)]) == 0


def test_missing_artifact_is_refused(repo):
    """Adversarial: a claim pointing at an artifact that does not exist."""
    rows = GOOD[:1] + ["C009 | `results_summary/rq/not_there.csv` | `*` | count() | 3 | 0"]
    claims, draft = _write(repo, rows)
    _, failures = run(claims, draft, root=repo)
    assert len(failures) == 1 and "C009" in failures[0] and "does not exist" in failures[0]


def test_path_outside_repository_is_refused(repo, tmp_path_factory):
    outside = tmp_path_factory.mktemp("elsewhere") / "x.csv"
    outside.write_text("a\n1\n")
    rows = [f"C001 | `../{outside.parent.name}/x.csv` | `*` | count() | 1 | 0"]
    claims, draft = _write(repo, rows)
    _, failures = run(claims, draft, root=repo)
    assert len(failures) == 1 and ("outside" in failures[0] or "does not exist" in failures[0])


@pytest.mark.parametrize("row, message", [
    ("C001 | `results_summary/rq/pairs.csv` | `technique_a=01_A` | holm_p | 1.0 | 1", "exactly one"),
    ("C001 | `results_summary/rq/pairs.csv` | `nope=1` | count() | 0 | 0", "not a column"),
    ("C001 | `results_summary/rq/pairs.csv` | `*` | max(nope) | 0 | 0", "not in the artifact"),
    ("C001 | `results_summary/rq/pairs.csv` | `*` | median(holm_p) | 0 | 0", "unknown aggregate"),
    ("C001 | `results_summary/card.json` | - | rq1.nope | 0 | 0", "no key"),
    ("C001 | `results_summary/card.json` | - | rq1.pairs.[other=09_Z].row_gap | 0 | 0", "matches 0"),
    ("C001 | `results_summary/card.json` | - | rq1 | 0 | 0", "not a value"),
    ("C001 | `results_summary/rq/pairs.csv` | `technique_a=01_A;technique_b=02_B` | significant_holm | 0 | 0",
     "not a number"),
])
def test_malformed_claims_are_refused(repo, row, message):
    claims, draft = _write(repo, [row])
    _, failures = run(claims, draft, root=repo)
    assert len(failures) == 1 and message in failures[0]


def test_duplicate_id_is_refused(repo):
    claims, _ = _write(repo, [GOOD[0], GOOD[1].replace("C002", "C001")])
    with pytest.raises(ClaimError, match="duplicate"):
        parse_claims(claims)


def test_empty_table_is_refused(tmp_path):
    path = tmp_path / "CLAIMS.md"
    path.write_text("# Claims\n\n" + HEADER)
    with pytest.raises(ClaimError, match="no claim rows"):
        parse_claims(path)


def test_draft_id_mismatch_is_refused(repo):
    claims, draft = _write(repo, GOOD[:2], draft_ids=["C001", "C077"])
    _, failures = run(claims, draft, root=repo)
    assert any("C002 is in the claims table but not cited" in f for f in failures)
    assert any("C077 is cited" in f for f in failures)


def test_column_named_like_an_aggregate_is_read_as_a_column(repo):
    (repo / "results_summary" / "rq" / "dist.csv").write_text("split,count\ntest,280\nvalidation,281\n")
    rows = ["C001 | `results_summary/rq/dist.csv` | `split=test` | count | 280 | 0",
            "C002 | `results_summary/rq/dist.csv` | `*` | count() | 2 | 0"]
    claims, draft = _write(repo, rows)
    assert run(claims, draft, root=repo)[1] == []


def test_cited_range_covers_every_id(repo):
    claims, draft = _write(repo, GOOD, draft_ids=["C001–C008"])
    assert run(claims, draft, root=repo)[1] == []
    claims, draft = _write(repo, GOOD, draft_ids=["C001-C007"])
    _, failures = run(claims, draft, root=repo)
    assert failures == ["C008 is in the claims table but not cited in DRAFT.md"]


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
    compare("True", "true", "-")
    with pytest.raises(ClaimError):
        compare("", "0.5", "2")


def test_committed_claims_verify():
    """The committed CLAIMS.md matches the committed artifacts and the draft."""
    if not check_claims.DEFAULT_CLAIMS.is_file():
        pytest.skip("no committed CLAIMS.md")
    verified, failures = run()
    assert failures == [] and verified > 0
