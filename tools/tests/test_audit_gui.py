"""Adversarial tests for tools/audit_gui.py, the local page for the auditing framework.

The page must not become a way round the rules the tools enforce. Each test
hands it one bad request or one bad file and checks that it is refused: a
technique name that is a path, a card that carries a row id, a run request
from another site, a file outside the page's two assets. The rest pin what the
page promises: slurs masked unless asked for, one audit at a time, exactly the
command a user would type, and nothing that ranks techniques or takes text.
All fixtures are synthetic and text-free; the slur stand-in is a made-up word.
"""

import json
import re
import socket
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIR))

import audit  # noqa: E402
import audit_gui  # noqa: E402
from audit_gui import Busy, JobRunner, NotFound, NotTextFree, Workbench  # noqa: E402
from conftest import FRAMING_WORDS, IDENTITY_WORDS, TOPICAL_WORDS  # noqa: E402
from repo_paths import CATEGORIES  # noqa: E402

WORDS = FRAMING_WORDS + IDENTITY_WORDS + TOPICAL_WORDS + ["river", "garden", "window", "pencil"]
INSIDE = ("01_A", "02_B")
OUTSIDE = "15_MINE"
SLUR = "zzmadeup"  # a stand-in; no real slur appears in a test
HOST = {"Host": "127.0.0.1"}


def write_card(rq_dir, technique, payload=None, markdown=None):
    folder = rq_dir / "report_cards"
    folder.mkdir(parents=True, exist_ok=True)
    payload = payload if payload is not None else {
        "technique": technique,
        "rq2": {"top_words_by_positive_mean_attribution": [
            {"word": "river", "category": "topical", "mean_attribution": 0.2, "support": 9},
            {"word": SLUR, "category": "slur", "mean_attribution": 0.18, "support": 2},
            {"word": f"{SLUR}land", "category": "topical", "mean_attribution": 0.1, "support": 3},
        ]},
        "rq4": {"threshold": 0.5, "fpr_with_identity_terms": 0.16, "fpr_without_identity_terms": 0.08,
                "fpr_ratio": 2.0, "fisher_p_value": 0.07, "n_negative": 280, "n_negative_with_identity_terms": 67},
    }
    markdown = markdown if markdown is not None else (
        f"# Report card: {technique}\n\n## RQ2. What the model relies on\n\n"
        f"| Rank | Word | Category |\n|---|---|---|\n| 1 | `river` | topical |\n| 2 | `{SLUR}` | slur |\n"
        f"| 3 | `{SLUR}land` | topical |\n\nThe word {SLUR}ish in prose is not a listed word.\n")
    (folder / f"{technique}.json").write_text(json.dumps(payload))
    (folder / f"{technique}.md").write_text(markdown)
    return folder / f"{technique}.json", folder / f"{technique}.md"


@pytest.fixture
def bench(tmp_path, make_probs, make_run, results_dir):
    """Two techniques in the comparison and one outside it, each meeting the contract."""
    for index, technique in enumerate((*INSIDE, OUTSIDE)):
        make_probs(technique, seed=index + 1)
        folder = results_dir / technique
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "best_config.json").write_text(json.dumps(
            {"selected_threshold": 0.5, "in_comparison": technique != OUTSIDE, "model_family": "linear"}))
        (folder / "metrics_test.json").write_text(json.dumps({"accuracy": 0.8 + index / 100, "roc_auc": float("nan")}))
        make_run(technique, "shap_test", WORDS, seed=index + 1, split="test")
        make_run(technique, "shap_validation", WORDS, seed=index + 11, split="validation")

    rq_dir = tmp_path / "rq"
    rq_dir.mkdir()
    shares = []
    for technique in (*INSIDE, OUTSIDE):
        for split in ("test", "validation"):
            for category, share in zip(CATEGORIES, (0.02, 0.03, 0.15, 0.80)):
                shares.append({"technique": technique, "run_id": f"shap_{split}", "member": "", "split": split,
                               "top_k": 50, "category": category, "n_words": 5, "share_of_positive_mass": share})
    # A member run of an ensemble: never a whole-model share.
    shares.append({"technique": "01_A", "run_id": "shap_test_member-x", "member": "x", "split": "test", "top_k": 50,
                   "category": "slur", "n_words": 50, "share_of_positive_mass": 0.99})
    pd.DataFrame(shares).to_csv(rq_dir / "rq2_category_shares.csv", index=False)
    pd.DataFrame({
        "technique": [*INSIDE, OUTSIDE], "n_runs": 2,
        **{f"std_share_of_positive_mass__{category}": 0.01 for category in CATEGORIES},
        "mean_pairwise_spearman_abs_attribution": [0.7, 0.4, 0.5], "mean_pairwise_jaccard_top50": [0.4, 0.2, 0.3],
    }).to_csv(rq_dir / "rq3_stability.csv", index=False)
    pd.DataFrame({
        "technique": [*INSIDE, OUTSIDE], "split": "test", "threshold": 0.5, "n_negative": 280,
        "n_negative_with_identity_terms": 67, "fpr_with_identity_terms": [0.16, 0.13, 0.2],
        "fpr_without_identity_terms": [0.08, 0.07, 0.1], "fpr_ratio": [2.0, 1.9, 2.0],
        "fisher_p_value": [0.07, 0.13, None],
    }).to_csv(rq_dir / "rq4_identity_fpr.csv", index=False)
    pair = {"split": "test", "n_rows": 450, "n_discordant": 20, "mcnemar_p": 0.5, "holm_p": 1.0}
    pd.DataFrame([{"technique_a": "01_A", "technique_b": "02_B", "accuracy_gap_rows": -4,
                   "significant_holm": False, **pair}]).to_csv(rq_dir / "rq1_mcnemar_pairs.csv", index=False)
    pd.DataFrame([{"technique_a": OUTSIDE, "technique_b": other, "accuracy_gap_rows": 20,
                   "significant_holm": True, **pair} for other in INSIDE]).to_csv(
        rq_dir / "rq1_mcnemar_pairs_outside_comparison.csv", index=False)
    write_card(rq_dir, "01_A")

    started = []

    def command_for(technique):
        started.append(technique)
        return [sys.executable, "-c", "print('OK    validate_results_folder: 0 problem(s)')"]

    workbench = Workbench(results_dir=results_dir, rq_dir=rq_dir, external_dir=tmp_path / "external",
                          runner=JobRunner(command_for=command_for, cwd=tmp_path, snapshot=dict))
    workbench.started = started
    return workbench


def get(bench, target, headers=HOST):
    status, _, body = bench.handle("GET", target, headers)
    return status, json.loads(body) if body[:1] in (b"{", b"[") else body.decode()


def post_run(bench, technique, headers=None, body=None):
    headers = {**HOST, audit_gui.TOKEN_HEADER: bench.token, **(headers or {})}
    payload = json.dumps({"technique": technique}).encode() if body is None else body
    status, _, answer = bench.handle("POST", "/api/run", headers, payload)
    return status, json.loads(answer)


# --- technique names ------------------------------------------------------------


@pytest.mark.parametrize("name", ["../01_A", "01_A/..", "01_A/../02_B", "/etc/passwd", "rq", "foundation", "", "01_a"])
def test_a_name_that_is_not_a_technique_folder_is_not_found(bench, name):
    with pytest.raises(NotFound):
        bench.technique(name)
    with pytest.raises(NotFound):
        bench.start_run(name)
    assert get(bench, f"/api/technique?name={name}")[0] == 404
    assert post_run(bench, name)[0] == 404
    assert bench.started == []


def test_a_missing_name_is_not_found(bench):
    assert get(bench, "/api/technique")[0] == 404
    assert post_run(bench, None)[0] == 404
    assert bench.started == []


# --- nothing per post is served ------------------------------------------------------


def test_a_card_carrying_a_row_id_is_refused_not_served(bench):
    write_card(bench.rq_dir, "01_A", payload={"technique": "01_A", "note": "worst post was ex_000123"})
    with pytest.raises(NotTextFree, match="row id"):
        bench.technique("01_A")
    status, answer = get(bench, "/api/technique?name=01_A")
    assert status == 500 and "refused" in answer["error"] and "ex_000123" not in json.dumps(answer)


@pytest.mark.parametrize("key", ["row_id", "row_ids", "text", "text_hash"])
def test_a_card_with_a_per_post_key_is_refused(bench, key):
    write_card(bench.rq_dir, "01_A", payload={"technique": "01_A", "rq2": {key: ["anything"]}})
    with pytest.raises(NotTextFree, match="per-post"):
        bench.technique("01_A")


def test_a_markdown_card_carrying_a_row_id_is_refused(bench):
    write_card(bench.rq_dir, "01_A", markdown="# Report card\n\nSee ex_000123.\n")
    with pytest.raises(NotTextFree, match="row id"):
        bench.technique("01_A")


def test_a_row_id_in_run_output_is_withheld(bench, tmp_path):
    bench.runner = JobRunner(
        command_for=lambda technique: [sys.executable, "-c", "print('OK    mcnemar_pairs: saw ex_000123')"],
        cwd=tmp_path, snapshot=dict)
    bench.start_run("01_A")
    assert bench.runner.wait()
    job = bench.runner.snapshot()["job"]
    assert job["lines"] == ["OK    mcnemar_pairs: saw [row id withheld]"]
    assert not audit.ROW_ID_RE.search(json.dumps(job))


def test_every_payload_passes_the_text_check_on_the_way_out(bench, monkeypatch):
    """overview, technique, comparison and template each go through assert_text_free."""
    seen = []
    real = audit_gui.assert_text_free
    monkeypatch.setattr(audit_gui, "assert_text_free", lambda payload, where="response": seen.append(1) or real(payload))
    for build in (bench.overview, lambda: bench.technique("01_A"), bench.comparison,
                  lambda: bench.template("classical")):
        build()
    assert len(seen) == 4


# --- slurs ---------------------------------------------------------------------------


def test_slurs_are_masked_unless_asked_for(bench):
    masked = bench.technique("01_A")
    words = [entry["word"] for entry in masked["card"]["rq2"]["top_words_by_positive_mean_attribution"]]
    assert words == ["river", "z*******", f"{SLUR}land"]
    assert masked["slurs_masked"] == 1 and masked["verbatim"] is False
    assert f"`{SLUR}`" not in masked["card_markdown"] and "`z*******`" in masked["card_markdown"]
    # Only the listed slur is touched: a longer listed word and the prose keep their letters.
    assert f"`{SLUR}land`" in masked["card_markdown"] and f"{SLUR}ish in prose" in masked["card_markdown"]

    verbatim = bench.technique("01_A", verbatim=True)
    assert verbatim["card"]["rq2"]["top_words_by_positive_mean_attribution"][1]["word"] == SLUR
    assert f"`{SLUR}`" in verbatim["card_markdown"] and verbatim["slurs_masked"] == 0


def test_masking_is_the_default_over_http_and_leaves_the_committed_card_alone(bench):
    json_path, md_path = write_card(bench.rq_dir, "01_A")
    before = json_path.read_bytes(), md_path.read_bytes()
    _, default = get(bench, "/api/technique?name=01_A")
    _, other = get(bench, "/api/technique?name=01_A&verbatim=0")
    _, asked = get(bench, "/api/technique?name=01_A&verbatim=1")
    assert SLUR not in json.dumps(default["card"]["rq2"]["top_words_by_positive_mean_attribution"][1])
    assert other["slurs_masked"] == 1 and asked["slurs_masked"] == 0
    assert (json_path.read_bytes(), md_path.read_bytes()) == before


def test_the_committed_cards_show_no_slur_unmasked_by_default():
    bench = Workbench()
    checked = 0
    for name in bench.technique_names():
        shown, written = bench.technique(name), bench.technique(name, verbatim=True)
        if not shown["card"]:
            continue
        pairs = zip(shown["card"]["rq2"]["top_words_by_positive_mean_attribution"],
                    written["card"]["rq2"]["top_words_by_positive_mean_attribution"])
        for masked, original in pairs:
            if original["category"] == "slur":
                checked += 1
                assert masked["word"] == audit_gui.mask_word(original["word"]) != original["word"]
                assert f"`{original['word']}`" not in shown["card_markdown"]
            else:
                assert masked["word"] == original["word"]
    assert checked >= 1


# --- requests -------------------------------------------------------------------------


@pytest.mark.parametrize("host", ["evil.example", "127.0.0.1.evil.example", "", "192.168.1.5"])
def test_a_request_for_another_host_is_refused(bench, host):
    assert get(bench, "/api/overview", {"Host": host})[0] == 403
    assert post_run(bench, "01_A", headers={"Host": host})[0] == 403
    assert bench.started == []


def test_starting_a_run_needs_the_pages_token(bench):
    assert post_run(bench, "01_A", headers={audit_gui.TOKEN_HEADER: ""})[0] == 403
    assert post_run(bench, "01_A", headers={audit_gui.TOKEN_HEADER: "guess"})[0] == 403
    status, _, _ = bench.handle("POST", "/api/run", HOST, json.dumps({"technique": "01_A"}).encode())
    assert status == 403 and bench.started == []


def test_a_run_cannot_be_started_from_another_site(bench):
    assert post_run(bench, "01_A", headers={"Origin": "https://evil.example"})[0] == 403
    assert bench.started == []
    status, answer = post_run(bench, "01_A", headers={"Origin": "http://127.0.0.1"})
    assert status == 200 and answer["job"]["technique"] == "01_A"
    assert bench.runner.wait() and bench.started == ["01_A"]


@pytest.mark.parametrize("body", [b"not json", b"[1, 2]", b'"01_A"'])
def test_a_malformed_run_request_is_refused(bench, body):
    assert post_run(bench, "01_A", body=body)[0] in (400, 404)
    assert bench.started == []


@pytest.mark.parametrize("target", ["/static/../audit_gui.py", "/static/index.html", "/static/app.js/../../audit.py",
                                    "/audit_gui.py", "/static/", "/static/%2e%2e/audit.py", "/api/nothing"])
def test_only_the_pages_own_files_are_served(bench, target):
    assert get(bench, target)[0] == 404


def test_the_page_is_served_with_its_token_and_its_assets(bench):
    status, page = get(bench, "/")
    assert status == 200 and bench.token in page and audit_gui.TOKEN_PLACEHOLDER not in page
    for name, content_type in audit_gui.STATIC_FILES.items():
        status, served_as, _ = bench.handle("GET", f"/static/{name}", HOST)
        assert status == 200 and served_as == content_type


def test_other_methods_are_refused(bench):
    assert bench.handle("PUT", "/api/run", HOST)[0] == 400
    assert bench.handle("POST", "/api/overview", HOST)[0] == 400


# --- runs -------------------------------------------------------------------------------


def test_the_one_command_is_the_standard_chain():
    command = audit_gui.audit_command("01_A")
    assert command == [sys.executable, "-u", str(TOOLS_DIR / "audit.py"), "run", "--technique", "01_A"]
    assert not any("perturbation" in part or "counterfactual" in part for part in command)


def test_one_audit_at_a_time(bench, tmp_path):
    gate = tmp_path / "gate"
    # Waits for the gate, and gives up by itself so a failing test leaves no process behind.
    waiting = ("import pathlib, sys, time\nend = time.time() + 20\n"
               "while not pathlib.Path(sys.argv[1]).exists() and time.time() < end: time.sleep(0.02)")
    bench.runner = JobRunner(command_for=lambda technique: [sys.executable, "-c", waiting, str(gate)],
                             cwd=tmp_path, snapshot=dict)
    assert post_run(bench, "01_A")[0] == 200
    with pytest.raises(Busy, match="still running"):
        bench.start_run("02_B")
    assert post_run(bench, "02_B")[0] == 409
    gate.write_text("go")
    assert bench.runner.wait()
    assert post_run(bench, "02_B")[0] == 200
    assert bench.runner.wait()


def test_stages_and_the_stop_are_read_from_the_output(bench, tmp_path):
    script = ("print('contract for 01_A')\nprint('OK    validate_results_folder: 0 problem(s)')\n"
              "print('SKIP  faithfulness: not requested (--with-perturbation)')\nprint('FAIL  compare_reliance')\n"
              "print('STOPPED at stage compare_reliance: the table is stale')\nraise SystemExit(1)")
    bench.runner = JobRunner(command_for=lambda technique: [sys.executable, "-c", script], cwd=tmp_path, snapshot=dict)
    bench.start_run("01_A")
    assert bench.runner.wait()
    job = bench.runner.snapshot()["job"]
    stages = {stage["name"]: stage for stage in job["stages"]["stages"]}
    assert [stage["name"] for stage in job["stages"]["stages"]] == list(audit.STAGE_ORDER)
    assert stages["validate_results_folder"]["status"] == "ok"
    assert stages["faithfulness"]["status"] == "skip"
    assert stages["compare_reliance"] == {"name": "compare_reliance", "status": "fail", "message": "the table is stale"}
    assert stages["report"]["status"] == "pending"
    assert job["stages"]["stopped_at"] == "compare_reliance"
    assert job["exit_code"] == 1 and "stopped" in job["outcome"]


def test_a_run_reports_exactly_the_files_it_changed(bench, tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "kept.csv").write_text("same")
    (outputs / "table.csv").write_text("old")
    (outputs / "gone.csv").write_text("x")
    script = ("import pathlib, sys\nroot = pathlib.Path(sys.argv[1])\n(root / 'table.csv').write_text('new')\n"
              "(root / 'card.md').write_text('card')\n(root / 'gone.csv').unlink()")
    bench.runner = JobRunner(command_for=lambda technique: [sys.executable, "-c", script, str(outputs)], cwd=tmp_path,
                             snapshot=lambda: audit_gui.snapshot_outputs([outputs]))
    bench.start_run("01_A")
    assert bench.runner.wait()
    changed = {entry["path"]: entry["state"] for entry in bench.runner.snapshot()["job"]["changed"]}
    assert changed == {"card.md": "new", "gone.csv": "removed", "table.csv": "changed"}


def test_a_command_that_cannot_start_is_a_finished_job_with_a_message(bench, tmp_path):
    bench.runner = JobRunner(command_for=lambda technique: [str(tmp_path / "no_such_program")], cwd=tmp_path, snapshot=dict)
    bench.start_run("01_A")
    assert bench.runner.wait()
    job = bench.runner.snapshot()["job"]
    assert job["state"] == "finished" and job["exit_code"] == -1 and "could not start" in job["lines"][0]


# --- the tables -------------------------------------------------------------------------


def test_a_technique_outside_the_comparison_stays_out_of_its_family(bench):
    data = bench.comparison()
    assert data["in_comparison"] == list(INSIDE) and data["outside_comparison"] == [OUTSIDE]
    assert [(pair["a"], pair["b"]) for pair in data["rq1"]["pairs"]] == [("01_A", "02_B")]
    assert {pair["a"] for pair in data["rq1"]["pairs_outside"]} == {OUTSIDE}
    flags = {row["technique"]: row["in_comparison"] for row in data["rq4"]["fpr"]}
    assert flags == {"01_A": True, "02_B": True, OUTSIDE: False}
    overview = {t["name"]: t["in_comparison"] for t in bench.overview()["techniques"]}
    assert overview[OUTSIDE] is False and bench.technique(OUTSIDE)["in_comparison"] is False


def test_shares_are_the_four_categories_of_the_whole_model_run(bench):
    shares = bench.technique("01_A")["shares"]
    assert [category["key"] for category in shares["categories"]] == list(CATEGORIES)
    assert shares["split"] == "test"
    assert shares["categories"][0]["share"] == pytest.approx(0.02)  # not the member run's 0.99
    assert [category["key"] for category in bench.overview()["categories"]] == list(CATEGORIES)
    assert set(audit_gui.CATEGORY_LABELS) == set(CATEGORIES) and len(CATEGORIES) == 4


def test_a_missing_number_is_absent_not_a_nan(bench):
    assert "roc_auc" not in bench.technique("01_A")["metrics"]
    outside = next(row for row in bench.comparison()["rq4"]["fpr"] if row["technique"] == OUTSIDE)
    assert outside["fisher_p"] is None
    for target in ("/api/overview", "/api/compare", "/api/technique?name=01_A"):
        status, _, body = bench.handle("GET", target, HOST)
        assert status == 200 and b"NaN" not in body


def test_a_technique_without_a_card_or_tables_still_has_a_page(bench, tmp_path):
    empty = Workbench(results_dir=bench.results_dir, rq_dir=tmp_path / "no_rq", external_dir=tmp_path / "external")
    page = empty.technique("02_B")
    assert page["card"] is None and page["card_markdown"] is None and page["shares"] is None
    assert page["contract_met"] is True
    data = empty.comparison()
    assert data["rq1"]["pairs"] == [] and data["rq4"]["fpr"] == []


def test_a_broken_folder_shows_as_a_failed_contract_not_a_crash(bench):
    (bench.results_dir / "02_B" / "best_config.json").write_text("{not json")
    page = bench.technique("02_B")
    assert page["contract_met"] is False
    assert get(bench, "/api/overview")[0] == 200


def test_only_the_three_notebook_roles_have_a_template(bench):
    assert "## 1." in bench.template("classical")["outline"]
    for role in ("probe", "foundation", "../x", "", None):
        with pytest.raises(NotFound):
            bench.template(role)


# --- the committed repository ------------------------------------------------------------


def test_the_committed_repository_loads():
    bench = Workbench()
    overview = bench.overview()
    data = bench.comparison()
    assert len(data["in_comparison"]) >= 8 and all(t["contract_met"] for t in overview["techniques"])
    assert len(data["rq1"]["pairs"]) == len(data["in_comparison"]) * (len(data["in_comparison"]) - 1) // 2
    for name in bench.technique_names():
        page = bench.technique(name)
        assert page["card"] and page["card_markdown"] and len(page["contract"]) == 7
        assert str(audit_gui.REPO_ROOT) not in json.dumps(page["contract"])


# --- the server ---------------------------------------------------------------------------


def test_the_server_listens_on_the_loopback_address_only(bench):
    server = audit_gui.make_server(bench, 0)
    # Serve before asserting anything: shutdown() waits for a loop that has run.
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        assert server.server_address[0] == "127.0.0.1"
        base = f"http://127.0.0.1:{server.server_port}"
        with urllib.request.urlopen(f"{base}/") as response:
            page = response.read().decode()
            assert response.status == 200 and bench.token in page
            policy = response.headers["Content-Security-Policy"]
            assert "script-src 'self'" in policy and "unsafe" not in policy
        with urllib.request.urlopen(f"{base}/api/overview") as response:
            assert len(json.loads(response.read())["techniques"]) == 3
        request = urllib.request.Request(f"{base}/api/overview", headers={"Host": "evil.example"})
        with pytest.raises(urllib.error.HTTPError) as refused:
            urllib.request.urlopen(request)
        assert refused.value.code == 403
        # The port is part of the address: this server's name on another port is not this server.
        request = urllib.request.Request(f"{base}/api/overview", headers={"Host": "127.0.0.1:1"})
        with pytest.raises(urllib.error.HTTPError) as refused:
            urllib.request.urlopen(request)
        assert refused.value.code == 403
    finally:
        server.shutdown()
        server.server_close()


def test_a_port_in_use_is_an_error_exit_not_a_crash(capsys):
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen(1)
        assert audit_gui.main(["--port", str(taken.getsockname()[1]), "--no-browser"]) == 1
    assert "--port 0" in capsys.readouterr().err


# --- the page itself -----------------------------------------------------------------------


def page_sources():
    return {name: (audit_gui.STATIC_DIR / name).read_text() for name in ("index.html", "app.js", "style.css")}


def test_the_page_does_not_rank_techniques_or_overclaim():
    """The prose rules of the project hold on the page: no 'better', no 'outperforms',
    no claim that the task is unexplored, no champion."""
    banned = re.compile(r"\b(better|best|outperform\w*|unexplored|champion\w*|state[- ]of[- ]the[- ]art)\b", re.IGNORECASE)
    for name, source in page_sources().items():
        assert not banned.findall(source), name


def test_the_page_takes_no_text_to_classify():
    sources = page_sources()
    assert "<textarea" not in sources["index.html"] and "textarea" not in sources["app.js"]
    inputs = re.findall(r"<input\b[^>]*>", sources["index.html"])
    assert inputs and all('type="checkbox"' in tag for tag in inputs)
    assert not re.search(r'h\(\s*"input"', sources["app.js"])


def test_file_content_is_never_placed_as_markup():
    sources = page_sources()
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert sink not in sources["app.js"], sink
    assert "<style" not in sources["index.html"] and " style=" not in sources["index.html"]
    assert re.findall(r"<script\b[^>]*>", sources["index.html"]) == ['<script src="/static/app.js">']


def test_the_page_names_exactly_the_four_categories():
    styles = page_sources()["style.css"]
    assert set(re.findall(r"--cat-([a-z_]+):", styles)) == set(CATEGORIES)
    assert set(re.findall(r"^\.c-([a-z_]+) \{", styles, re.MULTILINE)) == set(CATEGORIES)


# --- the page works offline -----------------------------------------------------------------


def test_the_page_asks_no_other_host_for_anything():
    """Fonts, styles and scripts all come from this server: the page must work with no network."""
    allowed = {"http://www.w3.org/2000/svg"}  # the SVG namespace, a name and not a request
    for name, source in page_sources().items():
        found = set(re.findall(r"https?://[^\s\"')]+", source)) - allowed
        assert not found, (name, found)
        assert "@import" not in source, name


def test_every_typeface_the_page_names_is_served_from_here(bench):
    styles = page_sources()["style.css"]
    faces = re.findall(r'url\("/static/(fonts/[^"]+)"\)', styles)
    assert len(faces) == 3 and set(faces) <= set(audit_gui.STATIC_FILES)
    for face in faces:
        status, content_type, body = bench.handle("GET", f"/static/{face}", HOST)
        assert status == 200 and content_type == "font/woff2" and body[:4] == b"wOF2"
    # Each face ships with its licence, which is kept beside it and is not part of the page.
    licences = sorted(path.name for path in (audit_gui.STATIC_DIR / "fonts").glob("OFL-*.txt"))
    assert licences == ["OFL-STIX-Two-Text.txt", "OFL-Source-Sans-3.txt"]
    assert get(bench, "/static/fonts/OFL-STIX-Two-Text.txt")[0] == 404
    assert get(bench, "/static/fonts/../../audit.py")[0] == 404


def test_the_policy_allows_fonts_from_this_server_only():
    assert "font-src 'self'" in audit_gui.CSP and "unsafe" not in audit_gui.CSP and "https:" not in audit_gui.CSP
