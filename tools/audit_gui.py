"""A local web page for the auditing framework (docs/FRAMEWORK.md).

For a researcher who wants to audit their own classifier next to the
techniques in this repository, without learning the tool chain first. It shows,
for every technique folder under results_summary/:

* the contract check (`audit.py check-contract`), item by item;
* the committed report card, with the figures it names;
* the cross-technique tables, one view per research question;

and it can run the audit chain (`audit.py run --technique T`) and show each
stage as it finishes. It adds no analysis and writes no artifact of its own:
every number on the page is read from a committed file, and the only command
it starts is the one a user would type.

What it does not do, on purpose. It does not classify text: the framework
audits classifiers, it is not moderation tooling (RESEARCH_PLAN.md, section 6).
It does not run the perturbation stages, which load model weights; the page
gives the command instead. It shows no post text, row id or text hash, and a
payload carrying one is refused rather than served. Words the lexicons mark as
slurs are masked unless the reader asks for them verbatim.

The server listens on the loopback address only. A request whose Host header is
not that address is refused, and starting a run needs the token the page was
served with, so another site open in the same browser cannot start one.

USAGE
-----
    python3 tools/audit_gui.py                      # opens http://127.0.0.1:8765
    python3 tools/audit_gui.py --port 0 --no-browser
"""

import argparse
import csv
import hashlib
import io
import json
import re
import secrets
import subprocess
import sys
import threading
import time
import webbrowser
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit  # noqa: E402
import check_notebook  # noqa: E402
from repo_paths import (  # noqa: E402
    CATEGORIES,
    EXPECTED_SPLIT_COUNTS,
    EXTERNAL_DIR,
    REPO_ROOT,
    REPORT_CARD_JSON_FILENAME,
    REPORT_CARD_MD_FILENAME,
    RESULTS_DIR,
    RQ1_MCNEMAR_PAIRS_CSV,
    RQ1_MCNEMAR_PAIRS_OUTSIDE_CSV,
    RQ2_CATEGORY_SHARES_CSV,
    RQ3_STABILITY_CSV,
    RQ4_IDENTITY_FPR_CSV,
    RQ_DIR,
    TOOLS_DIR,
    in_comparison,
    technique_dirs,
)

LOOPBACK = "127.0.0.1"
DEFAULT_PORT = 8765
STATIC_DIR = TOOLS_DIR / "audit_gui_static"
# The only files the page is made of. Nothing else under tools/ is served.
STATIC_FILES = {
    "app.js": "text/javascript; charset=utf-8",
    "style.css": "text/css; charset=utf-8",
    # The two typefaces, served from here so the page needs no network (SIL OFL 1.1; licences beside them).
    "fonts/stix-two-text-roman.woff2": "font/woff2",
    "fonts/stix-two-text-italic.woff2": "font/woff2",
    "fonts/source-sans-3-roman.woff2": "font/woff2",
}
TOKEN_PLACEHOLDER = "{{AUDIT_TOKEN}}"
TOKEN_HEADER = "X-Audit-Token"

CATEGORY_LABELS = {
    "slur": "Slur",
    "extremist_framing": "Extremist framing",
    "identity_term": "Identity term",
    "topical": "Topical",
}
# What the page shows from a result folder. Anything else in those files stays there.
CONFIG_KEYS = ("selected_threshold", "model_family", "feature_family", "run_id", "kit_version",
               "threshold_selected_on_split")
METRIC_KEYS = ("accuracy", "balanced_accuracy", "f1_macro", "roc_auc", "pr_auc")
NOTEBOOK_ROLES = ("classical", "transformer", "ensemble")
MAX_LOG_LINES = 2000
STAGE_LINE_RE = re.compile(r"^(OK|SKIP|FAIL)\s+([a-z0-9_]+)(?::\s*(.*))?$")
STOPPED_RE = re.compile(r"^STOPPED at stage ([a-z0-9_-]+)(?::\s*(.*))?$")
CSP = "default-src 'none'; script-src 'self'; style-src 'self'; font-src 'self'; connect-src 'self'; img-src 'self' data:; " \
      "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"


class GuiError(RuntimeError):
    """A request the page cannot serve. `status` is the HTTP status it is answered with."""

    status = 400


class NotFound(GuiError):
    status = 404


class Forbidden(GuiError):
    status = 403


class Busy(GuiError):
    status = 409


class NotTextFree(GuiError):
    status = 500


# ---------------------------------------------------------------------------
# Reading committed files
# ---------------------------------------------------------------------------
def _read_json(path):
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _read_rows(path):
    try:
        with open(path, newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def _number(value):
    """A finite float, or None for a blank, a non-number, NaN or infinity."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _integer(value):
    number = _number(value)
    return None if number is None else int(round(number))


def _rel(path):
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return Path(path).name


def _short(text):
    """`text` with the repository's absolute path removed, so messages read as repository paths."""
    return str(text).replace(f"{REPO_ROOT}/", "")


def mask_word(word):
    """`word` with every character after the first replaced, so its length stays readable."""
    text = str(word)
    return text[:1] + "*" * max(len(text) - 1, 0)


def mask_slurs(card, markdown):
    """(card, markdown, n_masked) with the words the card labels as slurs masked in both.

    Only words the card itself lists under the `slur` category are touched, so
    ordinary prose in the card is never rewritten. The committed card is the
    audit record and stays verbatim on disk; this is the figure-side redaction
    of author decision 8.
    """
    words = []
    for entry in (card or {}).get("rq2", {}).get("top_words_by_positive_mean_attribution", []) or []:
        if entry.get("category") == "slur" and entry.get("word"):
            words.append(str(entry["word"]))
            entry["word"] = mask_word(entry["word"])
            entry["masked"] = True
    if markdown:
        for word in sorted(set(words), key=len, reverse=True):
            markdown = markdown.replace(f"`{word}`", f"`{mask_word(word)}`")
    return card, markdown, len(set(words))


def assert_text_free(payload, where="response"):
    """Refuse a payload with a per-post key or a row id anywhere in it (audit.assert_text_free)."""
    try:
        audit.assert_text_free(payload, where)
    except audit.AuditError as error:
        raise NotTextFree(f"refused to serve it: {error}") from error
    return payload


# ---------------------------------------------------------------------------
# Running the audit chain
# ---------------------------------------------------------------------------
def audit_command(technique):
    """The one command the page starts: the standard chain, without the perturbation stages."""
    return [sys.executable, "-u", str(TOOLS_DIR / "audit.py"), "run", "--technique", technique]


def output_roots():
    """Where the audit chain writes: the result folders, and the rendered tables in the docs."""
    return [RESULTS_DIR, REPO_ROOT / "docs", REPO_ROOT / "README.md"]


def snapshot_outputs(roots=None):
    """{repository path: sha256} for every file the chain could write."""
    digests = {}
    for root in output_roots() if roots is None else roots:
        root = Path(root)
        files = [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file())
        for path in files:
            try:
                digests[_rel(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                continue
    return digests


def changed_between(before, after):
    """What a run did to the files it could write: changed, new or removed, by path."""
    changes = []
    for path in sorted(set(before) | set(after)):
        if path not in before:
            changes.append({"state": "new", "path": path})
        elif path not in after:
            changes.append({"state": "removed", "path": path})
        elif before[path] != after[path]:
            changes.append({"state": "changed", "path": path})
    return changes


class JobRunner:
    """Runs one audit at a time and keeps its output for the page to poll."""

    def __init__(self, command_for=audit_command, cwd=REPO_ROOT, snapshot=snapshot_outputs):
        self._command_for = command_for
        self._cwd = cwd
        self._snapshot = snapshot
        self._lock = threading.Lock()
        self._job = None
        self._counter = 0

    def start(self, technique):
        with self._lock:
            if self._job is not None and self._job["state"] == "running":
                raise Busy(f"an audit of {self._job['technique']} is still running")
            self._counter += 1
            job = {
                "id": self._counter, "technique": technique, "state": "running", "exit_code": None,
                "command": " ".join(["python3", "tools/audit.py", "run", "--technique", technique]),
                "lines": [], "started": time.time(), "finished": None, "changed": None, "truncated": False,
            }
            self._job = job
        thread = threading.Thread(target=self._run, args=(job, self._command_for(technique)), daemon=True)
        thread.start()
        return self.snapshot()

    def _run(self, job, command):
        exit_code = None
        before = self._snapshot()
        try:
            process = subprocess.Popen(command, cwd=self._cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1)
            for line in process.stdout or ():
                self._append(job, line.rstrip("\n"))
            exit_code = process.wait()
        except OSError as error:
            self._append(job, f"could not start the audit: {error}")
            exit_code = -1
        changes = changed_between(before, self._snapshot())
        with self._lock:
            job.update(state="finished", exit_code=exit_code, finished=time.time(), changed=changes)

    def _append(self, job, line):
        # A stage message is text-free by construction; a row id in it would be a bug upstream,
        # and the log is still worth reading, so it is withheld rather than refused.
        line = _short(audit.ROW_ID_RE.sub("[row id withheld]", line))
        with self._lock:
            if len(job["lines"]) >= MAX_LOG_LINES:
                job["truncated"] = True
            else:
                job["lines"].append(line)

    def snapshot(self):
        with self._lock:
            if self._job is None:
                return {"job": None}
            job = {key: (list(value) if isinstance(value, list) else value) for key, value in self._job.items()}
        job["stages"] = parse_stages(job["lines"])
        job["outcome"] = describe_exit(job["exit_code"]) if job["state"] == "finished" else None
        return {"job": job}

    def wait(self, timeout=60.0):
        """Block until the current job finishes (tests and scripts)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._job is None or self._job["state"] != "running":
                    return True
            time.sleep(0.02)
        return False


def parse_stages(lines):
    """One entry per stage of the chain, in order, from the lines `audit.py run` printed."""
    seen = {}
    stopped = None
    for line in lines:
        match = STAGE_LINE_RE.match(line)
        if match:
            status, name, message = match.groups()
            seen[name] = {"status": status.lower(), "message": message or ""}
            continue
        match = STOPPED_RE.match(line)
        if match:
            stopped = {"stage": match.group(1), "message": match.group(2) or ""}
    stages = []
    for name in audit.STAGE_ORDER:
        entry = seen.get(name, {"status": "pending", "message": ""})
        if stopped and stopped["stage"] == name and stopped["message"]:
            entry = {"status": "fail", "message": stopped["message"]}
        stages.append({"name": name, **entry})
    return {"stages": stages, "stopped_at": stopped["stage"] if stopped else None,
            "stopped_message": stopped["message"] if stopped else None}


def describe_exit(code):
    if code == audit.EXIT_OK:
        return "The chain finished."
    if code == audit.EXIT_PERTURBATION_SKIPPED:
        return "The chain finished, but a perturbation stage that was asked for did not run."
    if code == audit.EXIT_FAILED:
        return "The chain stopped at a stage that failed. Nothing after it ran."
    return f"The audit process ended with exit code {code}."


# ---------------------------------------------------------------------------
# The payloads
# ---------------------------------------------------------------------------
@dataclass
class Workbench:
    """Everything the page can ask for, read from one set of folders."""

    results_dir: Path = RESULTS_DIR
    rq_dir: Path = RQ_DIR
    external_dir: Path = EXTERNAL_DIR
    static_dir: Path = STATIC_DIR
    token: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    runner: JobRunner = field(default_factory=JobRunner)

    # -- techniques ----------------------------------------------------------
    def technique_names(self):
        return [folder.name for folder in technique_dirs(self.results_dir)]

    def require_technique(self, name):
        """`name` if it is exactly one of the technique folders; anything else is not found."""
        if not isinstance(name, str) or name not in self.technique_names():
            raise NotFound(f"no technique folder named {name!r} under {_rel(self.results_dir)}/")
        return name

    def _contract(self, name):
        try:
            items = audit.check_contract(name, results_dir=self.results_dir, external_dir=self.external_dir)
        except Exception as error:  # noqa: BLE001 - a broken folder must show as a failed check, not a blank page
            return [{"name": "contract check", "status": audit.INVALID, "required": True,
                     "detail": f"the check itself failed: {type(error).__name__}: {error}"}], False
        listed = [{"name": item.name, "status": item.status, "detail": _short(item.detail), "required": item.required}
                  for item in items]
        return listed, not any(item.failed for item in items)

    def _card_paths(self, name):
        folder = Path(self.rq_dir) / "report_cards"
        return (folder / REPORT_CARD_JSON_FILENAME.format(technique=name),
                folder / REPORT_CARD_MD_FILENAME.format(technique=name))

    def overview(self):
        techniques = []
        for name in self.technique_names():
            contract, met = self._contract(name)
            metrics = _read_json(Path(self.results_dir) / name / "metrics_test.json") or {}
            techniques.append({
                "name": name,
                "in_comparison": in_comparison(name, self.results_dir),
                "contract_met": met,
                "required_failed": sum(1 for item in contract if item["required"] and item["status"] != audit.OK),
                "has_card": self._card_paths(name)[0].is_file(),
                "test_accuracy": _number(metrics.get("accuracy")),
            })
        return assert_text_free({
            "techniques": techniques,
            "categories": [{"key": key, "label": CATEGORY_LABELS[key]} for key in CATEGORIES],
            "split": {name: counts[0] for name, counts in EXPECTED_SPLIT_COUNTS.items()},
            "anchor": audit.ANCHOR_TECHNIQUE,
            "notebook_roles": list(NOTEBOOK_ROLES),
            "results_dir": _rel(self.results_dir),
        })

    def technique(self, name, verbatim=False):
        name = self.require_technique(name)
        folder = Path(self.results_dir) / name
        contract, met = self._contract(name)
        config = _read_json(folder / "best_config.json") or {}
        metrics = _read_json(folder / "metrics_test.json") or {}
        json_path, md_path = self._card_paths(name)
        card = _read_json(json_path)
        markdown = md_path.read_text() if md_path.is_file() else None
        masked = 0
        if not verbatim:
            card, markdown, masked = mask_slurs(card, markdown)
        return assert_text_free({
            "name": name,
            "in_comparison": in_comparison(name, self.results_dir),
            "config": {key: config.get(key) for key in CONFIG_KEYS if config.get(key) is not None},
            "metrics": {key: _number(metrics.get(key)) for key in METRIC_KEYS if _number(metrics.get(key)) is not None},
            "contract": contract,
            "contract_met": met,
            "card": card,
            "card_markdown": markdown,
            "card_source": _rel(md_path) if markdown else None,
            "slurs_masked": masked,
            "verbatim": bool(verbatim),
            "shares": self._shares(name),
            "commands": {
                "check": f"python3 tools/audit.py check-contract --technique {name}",
                "run": f"python3 tools/audit.py run --technique {name}",
                "perturbation": f"python3 tools/audit.py run --technique {name} --with-perturbation",
            },
        })

    # -- the cross-technique tables -------------------------------------------
    def _share_rows(self):
        rows = _read_rows(Path(self.rq_dir) / RQ2_CATEGORY_SHARES_CSV.name)
        return [row for row in rows if not (row.get("member") or "").strip()]

    def _stability(self):
        return {row["technique"]: row for row in _read_rows(Path(self.rq_dir) / RQ3_STABILITY_CSV.name)}

    def _shares(self, name, rows=None, stability=None):
        """Share of positive attribution mass per category, test run first, with the across-run SD."""
        rows = self._share_rows() if rows is None else rows
        stability = self._stability() if stability is None else stability
        by_split = {}
        for row in rows:
            if row.get("technique") == name:
                by_split.setdefault(row.get("split"), {})[row.get("category")] = row
        split = "test" if "test" in by_split else next(iter(sorted(by_split)), None)
        if split is None:
            return None
        std = stability.get(name, {})
        return {
            "split": split,
            "top_k": _integer(next(iter(by_split[split].values())).get("top_k")),
            "categories": [
                {
                    "key": key,
                    "share": _number(by_split[split].get(key, {}).get("share_of_positive_mass")),
                    "n_words": _integer(by_split[split].get(key, {}).get("n_words")),
                    "std": _number(std.get(f"std_share_of_positive_mass__{key}")),
                }
                for key in CATEGORIES
            ],
        }

    def comparison(self):
        names = self.technique_names()
        inside = [name for name in names if in_comparison(name, self.results_dir)]
        outside = [name for name in names if name not in inside]
        share_rows, stability = self._share_rows(), self._stability()

        accuracy = []
        for name in names:
            metrics = _read_json(Path(self.results_dir) / name / "metrics_test.json") or {}
            if _number(metrics.get("accuracy")) is not None:
                accuracy.append({"technique": name, "accuracy": _number(metrics["accuracy"]),
                                 "in_comparison": name in inside})

        def pairs(path):
            return [
                {
                    "a": row.get("technique_a"), "b": row.get("technique_b"),
                    "row_gap": _integer(row.get("accuracy_gap_rows")),
                    "n_discordant": _integer(row.get("n_discordant")),
                    "mcnemar_p": _number(row.get("mcnemar_p")), "holm_p": _number(row.get("holm_p")),
                    "significant": str(row.get("significant_holm")).strip().lower() == "true",
                }
                for row in _read_rows(Path(self.rq_dir) / path.name)
            ]

        fpr = {row["technique"]: row for row in _read_rows(Path(self.rq_dir) / RQ4_IDENTITY_FPR_CSV.name)}
        payload = {
            "in_comparison": inside,
            "outside_comparison": outside,
            "n_test_rows": EXPECTED_SPLIT_COUNTS["test"][0],
            "rq1": {
                "accuracy": accuracy,
                "pairs": pairs(RQ1_MCNEMAR_PAIRS_CSV),
                "pairs_outside": pairs(RQ1_MCNEMAR_PAIRS_OUTSIDE_CSV),
                "sources": [f"{_rel(self.results_dir)}/<TECHNIQUE>/metrics_test.json",
                            _rel(Path(self.rq_dir) / RQ1_MCNEMAR_PAIRS_CSV.name)],
            },
            "rq2": {
                "shares": [{"technique": name, "in_comparison": name in inside,
                            **(self._shares(name, share_rows, stability) or {"categories": None})}
                           for name in names],
                "sources": [_rel(Path(self.rq_dir) / RQ2_CATEGORY_SHARES_CSV.name),
                            _rel(Path(self.rq_dir) / RQ3_STABILITY_CSV.name)],
            },
            "rq3": {
                "stability": [
                    {"technique": name, "in_comparison": name in inside,
                     "n_runs": _integer(stability[name].get("n_runs")),
                     "spearman": _number(stability[name].get("mean_pairwise_spearman_abs_attribution")),
                     "jaccard": _number(stability[name].get("mean_pairwise_jaccard_top50"))}
                    for name in names if name in stability
                ],
                "sources": [_rel(Path(self.rq_dir) / RQ3_STABILITY_CSV.name)],
            },
            "rq4": {
                "fpr": [
                    {"technique": name, "in_comparison": name in inside,
                     "threshold": _number(fpr[name].get("threshold")),
                     "with_identity": _number(fpr[name].get("fpr_with_identity_terms")),
                     "without_identity": _number(fpr[name].get("fpr_without_identity_terms")),
                     "ratio": _number(fpr[name].get("fpr_ratio")),
                     "fisher_p": _number(fpr[name].get("fisher_p_value")),
                     "n_with": _integer(fpr[name].get("n_negative_with_identity_terms")),
                     "n_negative": _integer(fpr[name].get("n_negative"))}
                    for name in names if name in fpr
                ],
                "sources": [_rel(Path(self.rq_dir) / RQ4_IDENTITY_FPR_CSV.name)],
            },
        }
        return assert_text_free(payload)

    # -- the notebook template --------------------------------------------------
    def template(self, role):
        if role not in NOTEBOOK_ROLES:
            raise NotFound(f"no notebook role {role!r}; choose one of {', '.join(NOTEBOOK_ROLES)}")
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            check_notebook.print_template(role)
        return assert_text_free({
            "role": role,
            "command": f"python3 tools/check_notebook.py --print-template {role}",
            "outline": buffer.getvalue(),
        })

    # -- runs -------------------------------------------------------------------
    def start_run(self, name):
        name = self.require_technique(name)
        return self.runner.start(name)

    # -- HTTP -------------------------------------------------------------------
    def handle(self, method, target, headers, body=b"", port=None):
        """Answer one request: (status, content type, body bytes). Never raises."""
        try:
            return self._route(method, target, {key.lower(): value for key, value in dict(headers).items()},
                               body, port)
        except GuiError as error:
            return _json_response({"error": str(error)}, error.status)
        except Exception as error:  # noqa: BLE001 - the page shows the message instead of a dead request
            return _json_response({"error": f"unexpected {type(error).__name__}: {error}"}, 500)

    def _route(self, method, target, headers, body, port):
        allowed_hosts = {LOOPBACK, "localhost"} if port is None else {f"{LOOPBACK}:{port}", f"localhost:{port}"}
        if headers.get("host", "") not in allowed_hosts:
            raise Forbidden("this page is served on the loopback address only")
        parts = urlsplit(target)
        path, query = parts.path, parse_qs(parts.query)

        def one(key, default=None):
            return query.get(key, [default])[0]

        if method == "GET":
            if path == "/":
                page = (self.static_dir / "index.html").read_text().replace(TOKEN_PLACEHOLDER, self.token)
                return 200, "text/html; charset=utf-8", page.encode()
            if path.startswith("/static/"):
                name = path[len("/static/"):]
                if name not in STATIC_FILES:
                    raise NotFound("no such file")
                return 200, STATIC_FILES[name], (self.static_dir / name).read_bytes()
            if path == "/api/overview":
                return _json_response(self.overview())
            if path == "/api/technique":
                return _json_response(self.technique(one("name"), verbatim=one("verbatim") == "1"))
            if path == "/api/compare":
                return _json_response(self.comparison())
            if path == "/api/template":
                return _json_response(self.template(one("role")))
            if path == "/api/job":
                return _json_response(self.runner.snapshot())
            raise NotFound("no such page")
        if method == "POST" and path == "/api/run":
            if not secrets.compare_digest(headers.get(TOKEN_HEADER.lower(), ""), self.token):
                raise Forbidden("starting a run needs the token this page was served with; reload the page")
            origin = headers.get("origin")
            if origin and urlsplit(origin).netloc not in allowed_hosts:
                raise Forbidden("a run can only be started from this page")
            try:
                request = json.loads(body or b"{}")
            except ValueError as error:
                raise GuiError("the request body is not JSON") from error
            if not isinstance(request, dict):
                raise GuiError("the request body must be a JSON object")
            return _json_response(self.start_run(request.get("technique")))
        raise GuiError("unsupported request")


def _json_response(payload, status=200):
    return status, "application/json; charset=utf-8", json.dumps(payload, allow_nan=False).encode()


# ---------------------------------------------------------------------------
# The server
# ---------------------------------------------------------------------------
def make_server(workbench, port=DEFAULT_PORT):
    """An HTTP server for `workbench` on the loopback address. Port 0 picks a free one."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "AuditWorkbench"

        def _answer(self, method):
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            status, content_type, payload = workbench.handle(
                method, self.path, self.headers, body, port=self.server.server_port)
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", CSP)
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):  # noqa: N802 - http.server's naming
            self._answer("GET")

        def do_POST(self):  # noqa: N802
            self._answer("POST")

        def log_message(self, format, *args):  # noqa: A002 - http.server's signature
            return

    return ThreadingHTTPServer((LOOPBACK, port), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"port on {LOOPBACK} (default {DEFAULT_PORT}; 0 picks a free one)")
    parser.add_argument("--no-browser", action="store_true", help="do not open the page in a browser")
    args = parser.parse_args(argv)

    try:
        server = make_server(Workbench(), args.port)
    except OSError as error:
        print(f"ERROR: could not listen on {LOOPBACK}:{args.port} ({error}); try --port 0", file=sys.stderr)
        return 1
    url = f"http://{LOOPBACK}:{server.server_port}/"
    print(f"Audit workbench at {url}  (Ctrl-C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
