#!/usr/bin/env python3
"""Static check that a notebook follows the template. Never executes anything.

Notebooks run on Kaggle and are never executed locally, so this tool is how a
notebook is verified before it is uploaded and before its outputs are
committed. It parses the notebook JSON and each code cell's syntax tree and
checks, in order:

1. Filename: `NN_NAME.ipynb`, no claim (BEST, FINAL, ...) in the name.
2. Structure: cell 0 is the markdown title block; then the numbered sections
   of the notebook's role, as `## N. Title` markdown headers in the fixed
   order, each followed by code; the last cell packages the outputs.
3. CONFIG literals: `technique_name` equals the filename stem, `split_version`
   and `dataset_version` are the frozen strings, `include_text_preview` is
   False, and notebook 00 does not overwrite the split.
4. Pipeline calls, each in its own section: the kit is loaded with the
   canonical bootstrap cell; the foundation is loaded and the frozen split
   asserted; the threshold is selected once, on validation; the test split is
   evaluated once; the probability artifact, at least two attribution runs
   and the result folder are exported.
5. Nothing that would publish dataset text: no `.head(`, no per-post SHAP
   plots, no `print`/`display`/`raise` that names a text or row-id variable,
   no shortcut that skips training, and no test data before section 8.
6. Soundness: every code cell parses, and every name it uses is defined in a
   cell at or before it (or inside the function that uses it).
7. Outputs. With `--primed` (code ready to upload, not yet run) every output
   must be empty. Without it (after a Kaggle run) there must be no error
   output and no row id in any output.

USAGE
-----
    python3 tools/check_notebook.py --all --primed
    python3 tools/check_notebook.py --notebook notebooks/07_TWITTER-ROBERTA_FINE-TUNE.ipynb
    python3 tools/check_notebook.py --print-template classical
"""

import argparse
import ast
import builtins
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from notebook_kit import (
    CLAIM_TOKENS,
    DATASET_VERSION,
    FOUNDATION_TECHNIQUE,
    ROLES,
    SPLIT_VERSION,
    TECHNIQUE_RE,
)
from repo_paths import NOTEBOOKS_DIR

EXIT_OK = 0
EXIT_FAILED = 1

MODEL_SECTIONS = (
    "Bootstrap",
    "Configuration",
    "Load foundation and assert the frozen split",
    "Features and model",
    "Ablation on validation",
    "Final fit",
    "Threshold on validation",
    "Single test evaluation",
    "Probability artifacts",
    "Word-level SHAP attribution runs",
    "Result folder",
    "External assets",
    "Interpretation",
    "Package outputs",
)
FOUNDATION_SECTIONS = (
    "Bootstrap",
    "Configuration",
    "Load raw dataset",
    "Build processed dataset",
    "Create and verify the split",
    "Foundation summaries",
    "Package outputs",
)
ROLE_SECTIONS = {
    "foundation": FOUNDATION_SECTIONS,
    "classical": MODEL_SECTIONS,
    "transformer": MODEL_SECTIONS,
    "ensemble": MODEL_SECTIONS,
}
# Sections that may consist of prose alone.
PROSE_ONLY_SECTIONS = ("Interpretation",)

# The first code cell of every notebook, verbatim. It finds the kit among the
# Kaggle inputs (or in the repository when the notebook is read locally) and
# loads it as `nk` without touching sys.path.
BOOTSTRAP_SNIPPET = '''import importlib.util
import sys
from pathlib import Path

_kit_candidates = (
    sorted(Path("/kaggle/input").rglob("notebook_kit.py"))
    if Path("/kaggle/input").exists()
    else []
)
_kit_candidates += [p / "tools" / "notebook_kit.py" for p in (Path.cwd(), *Path.cwd().parents)]
_kit_path = next((p for p in _kit_candidates if p.is_file()), None)
if _kit_path is None:
    raise FileNotFoundError(
        "notebook_kit.py not found. Attach the repository (or its tools/ folder) "
        "to this kernel as an input dataset."
    )
_kit_spec = importlib.util.spec_from_file_location("notebook_kit", _kit_path)
nk = importlib.util.module_from_spec(_kit_spec)
sys.modules["notebook_kit"] = nk
_kit_spec.loader.exec_module(nk)
'''

HEADER_RE = re.compile(r"^## (\d+)\. (.+?)\s*$")
ROW_ID_RE = re.compile(r"\bex_\d{6}\b")

# Words that make a claim the 450-row test split cannot support, or that
# misdescribe the field. Checked in every markdown cell.
BANNED_PROSE_RE = re.compile(
    r"\b(outperform\w*|unexplored|champion\w*|state[- ]of[- ]the[- ]art)\b", re.IGNORECASE
)
BANNED_TITLE_RE = re.compile(r"\bbest\b", re.IGNORECASE)

# Source patterns that publish rows, skip training, or reach around the kit.
FORBIDDEN_SOURCE_PATTERNS = (
    (re.compile(r"\.head\("), "`.head(` shows rows; use counts or nk.top_words"),
    (re.compile(r"\.tail\("), "`.tail(` shows rows"),
    (re.compile(r"\.sample\("), "`.sample(` shows or subsamples rows"),
    (re.compile(r"\.to_string\("), "`.to_string(` prints a frame"),
    (re.compile(r"\.to_markdown\("), "`.to_markdown(` prints a frame"),
    (re.compile(r"shap\.plots\.(text|waterfall|force)"), "per-post SHAP plots show dataset text"),
    (re.compile(r"sys\.path\.(append|insert)"), "sys.path is not edited; the kit is loaded by path"),
    (re.compile(r"^\s*%%", re.MULTILINE), "cell magics are not used"),
    (
        re.compile(r"\b(fast_path|fast_mode|smoke_test|debug_mode|quick_run|dry_run|max_rows)\b", re.IGNORECASE),
        "shortcut flags that skip or shrink training are not allowed",
    ),
    (
        re.compile(r"\b(train|val|test)_df\s*=\s*\1_df(\.iloc)?\["),
        "split frames are never sliced",
    ),
)
TEST_REFERENCE_RE = re.compile(r"\btest_(df|prob|probs|texts|loader|logits)\b")
# Sections (by title) in which the test split must not be referenced at all.
NO_TEST_SECTIONS = ("Features and model", "Ablation on validation", "Threshold on validation")

# Identifiers and column names that hold dataset text or row-level identity.
TEXT_IDENTIFIERS = frozenset(
    {
        "text",
        "texts",
        "raw_text",
        "model_input_text",
        "text_preview",
        "message",
        "Original_Message",
        "text_hash",
        "row_id",
        "row_ids",
        "missing_ids",
    }
)
TEXT_COLUMNS = frozenset(
    {"text", "raw_text", "model_input_text", "text_preview", "Original_Message", "text_hash", "row_id"}
)
# Frames that carry a text column; displaying one shows posts.
ROW_FRAMES = frozenset(
    {
        "frame",
        "raw_df",
        "processed",
        "train_df",
        "val_df",
        "test_df",
        "predictions",
        "predictions_df",
        "review_queue",
        "local_attributions",
    }
)
# Calls whose argument may name a text variable because they return a count.
COUNTING_CALLS = frozenset({"len"})

KERNEL_METADATA_FORBIDDEN_WHEN_PRIMED = ("papermill", "widgets", "colab")

IPYTHON_NAMES = frozenset({"display", "get_ipython", "__file__", "__name__"})
BUILTIN_NAMES = frozenset(dir(builtins)) | IPYTHON_NAMES


# ---------------------------------------------------------------------------
# Notebook parsing
# ---------------------------------------------------------------------------
def cell_source(cell):
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def strip_line_magics(source):
    """Replace `%magic` and `!shell` lines so the cell parses as Python."""
    lines = []
    for line in source.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("%") or stripped.startswith("!"):
            lines.append(line[: len(line) - len(stripped)] + "pass")
        else:
            lines.append(line)
    return "\n".join(lines)


class Section:
    def __init__(self, number, title, header_index):
        self.number = number
        self.title = title
        self.header_index = header_index
        self.code_indices = []


def parse_sections(cells):
    """Split the notebook into its numbered sections. Returns (sections, stray_code)."""
    sections = []
    stray_code = []
    for index, cell in enumerate(cells):
        source = cell_source(cell)
        if cell.get("cell_type") == "markdown":
            match = HEADER_RE.match(source.split("\n", 1)[0])
            if match:
                sections.append(Section(int(match.group(1)), match.group(2), index))
        elif cell.get("cell_type") == "code":
            if sections:
                sections[-1].code_indices.append(index)
            else:
                stray_code.append(index)
    return sections, stray_code


def parse_cells(cells):
    """{cell index: ast.Module} for every code cell that parses, plus syntax problems."""
    trees = {}
    problems = []
    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        try:
            trees[index] = ast.parse(strip_line_magics(cell_source(cell)))
        except SyntaxError as error:
            problems.append(f"cell {index}: does not parse ({error.msg}, line {error.lineno})")
    return trees, problems


# ---------------------------------------------------------------------------
# Kit calls
# ---------------------------------------------------------------------------
def kit_calls(tree):
    """Every `nk.<name>(...)` call in a cell, as (name, ast.Call)."""
    calls = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "nk"
        ):
            calls.append((node.func.attr, node))
    return calls


def call_has_constant(call, value):
    for arg in list(call.args) + [kw.value for kw in call.keywords]:
        if isinstance(arg, ast.Constant) and arg.value == value:
            return True
    return False


def call_has_keyword(call, name):
    return any(kw.arg == name for kw in call.keywords)


class CallIndex:
    """Where each kit call appears: {name: [(section_title, cell_index, call)]}."""

    def __init__(self, sections, trees):
        self.by_name = {}
        for section in sections:
            for index in section.code_indices:
                if index not in trees:
                    continue
                for name, call in kit_calls(trees[index]):
                    self.by_name.setdefault(name, []).append((section.title, index, call))

    def calls(self, name):
        return self.by_name.get(name, [])

    def in_section(self, name, title):
        return [c for c in self.calls(name) if c[0] == title]


def require_once_in(index, name, title, problems, label=None):
    label = label or f"nk.{name}"
    found = index.calls(name)
    if len(found) != 1:
        problems.append(f"{label} must be called exactly once (found {len(found)})")
    for section_title, cell_index, _ in found:
        if section_title != title:
            problems.append(
                f"cell {cell_index}: {label} belongs in section '{title}', not '{section_title}'"
            )


# ---------------------------------------------------------------------------
# Text and row-id exposure
# ---------------------------------------------------------------------------
def _returns_an_aggregate(call):
    """len(...) and every kit function return counts or word-level tables, never rows."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id in COUNTING_CALLS
    return (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "nk"
    )


def _exposed_names(node, include_frames=False):
    """Text-bearing names an expression would show, ignoring calls that only aggregate.

    With `include_frames`, a frame that carries a text column counts as well:
    displaying it, or anything sliced from it, shows posts.
    """
    found = set()

    def visit(current):
        if isinstance(current, ast.Call) and _returns_an_aggregate(current):
            return
        if isinstance(current, ast.Name):
            if current.id in TEXT_IDENTIFIERS or (include_frames and current.id in ROW_FRAMES):
                found.add(current.id)
        if isinstance(current, ast.Subscript):
            key = current.slice
            if isinstance(key, ast.Constant) and key.value in TEXT_COLUMNS:
                found.add(f'["{key.value}"]')
        for child in ast.iter_child_nodes(current):
            visit(child)

    visit(node)
    return found


def exposure_problems(index, tree):
    """print/display/raise statements that would put text or row ids into an output."""
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in ("print", "display"):
                exposed = set()
                for arg in list(node.args) + [kw.value for kw in node.keywords]:
                    exposed |= _exposed_names(arg, include_frames=node.func.id == "display")
                if exposed:
                    problems.append(
                        f"cell {index}: {node.func.id}(...) references {sorted(exposed)}; "
                        "outputs are committed, so show counts, metrics or word-level tables"
                    )
        elif isinstance(node, ast.Raise) and node.exc is not None:
            exposed = _exposed_names(node.exc)
            if exposed:
                problems.append(
                    f"cell {index}: an error message references {sorted(exposed)}; "
                    "report counts, not rows"
                )
    return problems


# ---------------------------------------------------------------------------
# Undefined names
# ---------------------------------------------------------------------------
_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _bindings(nodes):
    """Names bound by `nodes` in their own scope (nested scopes contribute only their name)."""
    bound = set()
    declared_global = set()

    def visit(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
            for decorator in node.decorator_list:
                visit(decorator)
            return
        if isinstance(node, ast.Lambda):
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.Global):
            declared_global.update(node.names)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            bound.add(node.name)
        for child in ast.iter_child_nodes(node):
            visit(child)

    for node in nodes:
        visit(node)
    return bound, declared_global


def _scope_arguments(node):
    if isinstance(node, ast.ClassDef):
        return set()
    args = node.args
    names = {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs}
    if args.vararg:
        names.add(args.vararg.arg)
    if args.kwarg:
        names.add(args.kwarg.arg)
    return names


def _global_declarations(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            names.update(node.names)
    return names


def undefined_name_problems(trees):
    """Names used in a cell that no reachable scope defines.

    Module-level code may only use names bound in the same or an earlier
    cell. Code inside a function runs later, so it may also use any name the
    notebook binds at module level anywhere.
    """
    ordered = sorted(trees)
    module_bound = {}
    all_module_names = set()
    for index in ordered:
        bound, _ = _bindings(trees[index].body)
        bound |= _global_declarations(trees[index])
        module_bound[index] = bound
        all_module_names |= bound

    problems = []
    reported = set()
    available = set()

    def check_scope(index, nodes, scopes, at_module_level):
        def visit(node):
            if isinstance(node, _SCOPE_NODES):
                if not isinstance(node, ast.Lambda):
                    for decorator in node.decorator_list:
                        visit(decorator)
                if isinstance(node, ast.ClassDef):
                    for base in node.bases:
                        visit(base)
                    body = node.body
                else:
                    for default in node.args.defaults + [d for d in node.args.kw_defaults if d]:
                        visit(default)
                    body = node.body if isinstance(node.body, list) else [node.body]
                local, _ = _bindings(body)
                local |= _scope_arguments(node)
                check_scope(index, body, scopes + [local], False)
                return
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                name = node.id
                known = (
                    name in BUILTIN_NAMES
                    or any(name in scope for scope in scopes)
                    or (name in available if at_module_level else name in all_module_names)
                )
                if not known and (index, name) not in reported:
                    reported.add((index, name))
                    problems.append(f"cell {index}: name '{name}' is used but never defined")
            for child in ast.iter_child_nodes(node):
                visit(child)

        for node in nodes:
            visit(node)

    for index in ordered:
        available |= module_bound[index]
        check_scope(index, trees[index].body, [], True)
    return problems


# ---------------------------------------------------------------------------
# CONFIG literals
# ---------------------------------------------------------------------------
def dict_literal_values(trees, key):
    """Every value given to `key` in any dict literal, as Python constants where possible."""
    values = []
    for tree in trees.values():
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for dict_key, dict_value in zip(node.keys, node.values):
                    if isinstance(dict_key, ast.Constant) and dict_key.value == key:
                        if isinstance(dict_value, ast.Constant):
                            values.append(dict_value.value)
                        else:
                            values.append(ast.unparse(dict_value))
    return values


def single_config_value(trees, key, problems, label="CONFIG"):
    values = dict_literal_values(trees, key)
    if not values:
        problems.append(f'{label} has no literal "{key}"')
        return None
    if len(set(map(repr, values))) > 1:
        problems.append(f'"{key}" is given {len(values)} different literal values: {values}')
    return values[0]


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------
def output_text(cell):
    parts = []
    for output in cell.get("outputs", []):
        text = output.get("text")
        if text:
            parts.append("".join(text) if isinstance(text, list) else str(text))
        for value in (output.get("data") or {}).values():
            if isinstance(value, list):
                parts.append("".join(str(v) for v in value))
            elif isinstance(value, str):
                parts.append(value)
        for line in output.get("traceback", []) or []:
            parts.append(str(line))
    return "\n".join(parts)


def check_filename(path, problems):
    stem = path.stem
    if stem == FOUNDATION_TECHNIQUE:
        return
    if not TECHNIQUE_RE.match(stem):
        problems.append(
            f"filename '{path.name}' is not NN_NAME.ipynb in upper case with '-' and '_' only"
        )
    claims = [token for token in CLAIM_TOKENS if token in re.split(r"[-_]", stem)]
    if claims:
        problems.append(f"filename encodes a claim {claims}; name the technique for what it is")


def check_notebook(path, primed=False):
    """Return a list of problems for one notebook (empty means it conforms)."""
    path = Path(path)
    problems = []
    check_filename(path, problems)

    try:
        with open(path) as handle:
            notebook = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        return problems + [f"cannot read notebook JSON: {error}"]
    cells = notebook.get("cells", [])
    if not cells:
        return problems + ["notebook has no cells"]

    stem = path.stem
    title_cell = cells[0]
    title_source = cell_source(title_cell)
    if title_cell.get("cell_type") != "markdown" or not title_source.startswith(f"# {stem}"):
        problems.append(f"cell 0 must be a markdown title block starting '# {stem}'")
    elif BANNED_TITLE_RE.search(title_source):
        problems.append("cell 0: the title block must not call anything 'best'")

    for index, cell in enumerate(cells):
        if cell.get("cell_type") == "markdown":
            match = BANNED_PROSE_RE.search(cell_source(cell))
            if match:
                problems.append(f"cell {index}: prose uses '{match.group(0)}'")

    trees, syntax_problems = parse_cells(cells)
    problems.extend(syntax_problems)
    sections, stray_code = parse_sections(cells)
    if stray_code:
        problems.append(f"code cell(s) {stray_code} come before the first '## 1.' section header")

    role = single_config_value(trees, "role", problems)
    if role not in ROLES:
        problems.append(f'CONFIG "role" is {role!r}; must be one of {ROLES}')
        return problems
    if (role == "foundation") != (stem == FOUNDATION_TECHNIQUE):
        problems.append(f"role '{role}' does not fit notebook '{stem}'")

    expected = ROLE_SECTIONS[str(role)]
    actual = [(s.number, s.title) for s in sections]
    wanted = [(i + 1, title) for i, title in enumerate(expected)]
    if actual != wanted:
        problems.append(
            "section headers must be exactly, in order: "
            + "; ".join(f"## {n}. {t}" for n, t in wanted)
            + f" (found {[f'{n}. {t}' for n, t in actual]})"
        )
        return problems
    for section in sections:
        if not section.code_indices and section.title not in PROSE_ONLY_SECTIONS:
            problems.append(f"section '{section.title}' has no code cell")
    by_title = {s.title: s for s in sections}

    # --- CONFIG literals ---------------------------------------------------
    config_trees = {i: trees[i] for i in by_title["Configuration"].code_indices if i in trees}
    technique = single_config_value(config_trees, "technique_name", problems)
    if technique is not None and technique != stem:
        problems.append(
            f'CONFIG "technique_name" is {technique!r}; it must equal the filename stem {stem!r}'
        )
    split_version = single_config_value(config_trees, "split_version", problems)
    if split_version is not None and split_version != SPLIT_VERSION:
        problems.append(
            f'CONFIG "split_version" is {split_version!r}; it must be the full frozen string '
            f"{SPLIT_VERSION!r}"
        )
    dataset_version = single_config_value(config_trees, "dataset_version", problems)
    if dataset_version is not None and dataset_version != DATASET_VERSION:
        problems.append(f'CONFIG "dataset_version" is {dataset_version!r}; must be {DATASET_VERSION!r}')

    previews = dict_literal_values(trees, "include_text_preview")
    if any(value is not False for value in previews):
        problems.append('"include_text_preview" must be False everywhere')
    if role != "foundation" and not previews:
        problems.append('CONFIG must declare "include_text_preview": False')
    if role == "foundation":
        overwrite = dict_literal_values(config_trees, "overwrite_existing_split")
        if overwrite != [False]:
            problems.append('CONFIG must declare "overwrite_existing_split": False')

    # --- pipeline calls ----------------------------------------------------
    index = CallIndex(sections, trees)
    bootstrap_cells = by_title["Bootstrap"].code_indices
    if not bootstrap_cells or BOOTSTRAP_SNIPPET.strip() not in cell_source(cells[bootstrap_cells[0]]):
        problems.append(
            "section 'Bootstrap' must begin with the canonical kit-loading cell "
            "(see --print-template)"
        )
    require_once_in(index, "bootstrap", "Configuration", problems)
    require_once_in(index, "finalize", "Package outputs", problems)
    last = cells[-1]
    if last.get("cell_type") != "code" or "nk.finalize(" not in cell_source(last):
        problems.append("the last cell must be the code cell that calls nk.finalize(ctx)")

    if role == "foundation":
        if not index.in_section("find_input", "Load raw dataset"):
            problems.append("section 'Load raw dataset' must locate the dataset with nk.find_input")
        require_once_in(index, "assert_split_counts", "Create and verify the split", problems)
        require_once_in(index, "compare_with_committed_split", "Create and verify the split", problems)
    else:
        require_once_in(index, "load_foundation", "Load foundation and assert the frozen split", problems)
        require_once_in(index, "split_frames", "Load foundation and assert the frozen split", problems)
        require_once_in(index, "select_threshold", "Threshold on validation", problems)
        require_once_in(index, "export_probabilities", "Probability artifacts", problems)
        require_once_in(index, "export_results_folder", "Result folder", problems)

        test_evaluations = [c for c in index.calls("evaluate") if call_has_constant(c[2], "test")]
        if len(test_evaluations) != 1:
            problems.append(
                f"the test split must be evaluated exactly once (found {len(test_evaluations)} "
                'nk.evaluate(..., "test") call(s))'
            )
        for section_title, cell_index, _ in test_evaluations:
            if section_title != "Single test evaluation":
                problems.append(
                    f"cell {cell_index}: the test evaluation belongs in 'Single test evaluation', "
                    f"not '{section_title}'"
                )
        for section_title, cell_index, call in index.calls("select_threshold"):
            first = call.args[0] if call.args else None
            if isinstance(first, ast.Name) and first.id.startswith("test"):
                problems.append(f"cell {cell_index}: a threshold is selected on test rows")

        shap_title = "Word-level SHAP attribution runs"
        runs = index.calls("export_attribution_run")
        if len(runs) < 2:
            problems.append(
                f"at least two attribution runs are exported (test and validation); found {len(runs)}"
            )
        for section_title, cell_index, _ in runs + index.calls("explain"):
            if section_title != shap_title:
                problems.append(f"cell {cell_index}: attribution code belongs in '{shap_title}'")
        if not index.calls("explain"):
            problems.append("no nk.explain(...) call; attributions come from the kit's explainer")
        if True in dict_literal_values(config_trees, "per_member_runs"):
            if not any(call_has_keyword(c[2], "member") for c in runs):
                problems.append(
                    '"per_member_runs" is True but no nk.export_attribution_run(..., member=...)'
                )
        wants_coefficients = True in dict_literal_values(config_trees, "export_coefficients")
        has_coefficients = bool(index.calls("export_coefficients"))
        if wants_coefficients != has_coefficients:
            problems.append(
                '"export_coefficients" in CONFIG and the nk.export_coefficients call disagree'
            )

        model_section = by_title["Features and model"]
        if not any(
            isinstance(node, ast.FunctionDef) and node.name == "predict_proba_texts"
            for i in model_section.code_indices
            if i in trees
            for node in ast.walk(trees[i])
        ):
            problems.append(
                "section 'Features and model' must define predict_proba_texts(texts), the one "
                "function every explainer run calls"
            )
        for title in NO_TEST_SECTIONS:
            for cell_index in by_title[title].code_indices:
                match = TEST_REFERENCE_RE.search(cell_source(cells[cell_index]))
                if match:
                    problems.append(
                        f"cell {cell_index}: '{match.group(0)}' is referenced in section '{title}'; "
                        "the test split is not touched before section 8"
                    )

    # --- source patterns and exposure --------------------------------------
    for cell_index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        source = cell_source(cell)
        for pattern, message in FORBIDDEN_SOURCE_PATTERNS:
            if pattern.search(source):
                problems.append(f"cell {cell_index}: {message}")
        if cell_index in trees:
            problems.extend(exposure_problems(cell_index, trees[cell_index]))

    problems.extend(undefined_name_problems(trees))

    # --- outputs -----------------------------------------------------------
    if primed:
        with_outputs = [
            i
            for i, cell in enumerate(cells)
            if cell.get("cell_type") == "code"
            and (cell.get("outputs") or cell.get("execution_count") is not None)
        ]
        if with_outputs:
            problems.append(
                f"--primed: {len(with_outputs)} code cell(s) still carry outputs or execution counts"
            )
        stale = [k for k in KERNEL_METADATA_FORBIDDEN_WHEN_PRIMED if k in notebook.get("metadata", {})]
        if stale:
            problems.append(f"--primed: notebook metadata still carries {stale}")
    else:
        for cell_index, cell in enumerate(cells):
            if cell.get("cell_type") != "code":
                continue
            if any(o.get("output_type") == "error" for o in cell.get("outputs", [])):
                problems.append(f"cell {cell_index}: output is an error traceback")
            row_ids = ROW_ID_RE.findall(output_text(cell))
            if row_ids:
                problems.append(f"cell {cell_index}: output shows {len(row_ids)} row id(s)")

    return problems


def notebook_paths(directory=None):
    directory = NOTEBOOKS_DIR if directory is None else Path(directory)
    return sorted(directory.glob("*.ipynb"))


def check_all(directory=None, primed=False):
    """{path: problems} for every notebook, plus cross-notebook problems under key None."""
    results = {}
    techniques = {}
    for path in notebook_paths(directory):
        results[path] = check_notebook(path, primed=primed)
        try:
            with open(path) as handle:
                cells = json.load(handle).get("cells", [])
            trees, _ = parse_cells(cells)
            for value in dict_literal_values(trees, "technique_name"):
                techniques.setdefault(value, []).append(path.name)
        except (OSError, json.JSONDecodeError):
            pass
    shared = [
        f"technique_name {name!r} is declared by {files}"
        for name, files in sorted(techniques.items())
        if len(set(files)) > 1
    ]
    if shared:
        results[None] = shared
    return results


def print_template(role):
    print(f"Sections for role '{role}':\n")
    print("  cell 0   markdown title block: '# <FILENAME STEM>' then role, inputs, outputs")
    for number, title in enumerate(ROLE_SECTIONS[role], start=1):
        print(f"  ## {number}. {title}")
    print("\nFirst code cell of section 1, verbatim:\n")
    print(BOOTSTRAP_SNIPPET)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="check every notebook in notebooks/")
    group.add_argument("--notebook", help="check one notebook")
    group.add_argument("--print-template", choices=sorted(ROLE_SECTIONS), help="print a role's sections")
    parser.add_argument(
        "--primed",
        action="store_true",
        help="the notebook is code ready to upload: require empty outputs",
    )
    args = parser.parse_args()

    if args.print_template:
        print_template(args.print_template)
        return EXIT_OK

    if args.notebook:
        results = {Path(args.notebook): check_notebook(args.notebook, primed=args.primed)}
    else:
        results = check_all(primed=args.primed)

    total = 0
    for path, problems in results.items():
        name = path.name if path is not None else "(across notebooks)"
        print(f"{'PASS' if not problems else 'FAIL'}  {name}  ({len(problems)} problem(s))")
        for problem in problems:
            print(f"        {problem}")
        total += len(problems)
    print(f"\n{total} problem(s) across {len([p for p in results if p is not None])} notebook(s)")
    return EXIT_OK if total == 0 else EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
