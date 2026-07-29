#!/usr/bin/env python3
"""Scan every git-tracked file for dataset text that should never be committed.

The release policy (docs/RELEASE_CHECKLIST.md, docs/RESULTS_SCHEMA.md) is that
raw dataset text lives in exactly one committed place: data/dataset.csv.
Everything else — result folders, notebooks, probability artifacts — must refer
to rows only by row_id / text_hash. That policy is what makes the repository
publishable at all for a dataset of extremist text: a single pasted cell output
quietly re-publishes source messages and defeats the sanitization that
eval_from_probs and probs_artifact enforce downstream. This scanner makes the
policy checkable instead of aspirational.

Two separate leak classes are reported, because they are differently severe:

* TEXT leak — a dataset message (whitespace-normalized exactly the way
  notebook 00 normalizes before hashing) appears in a committed file, either
  verbatim or as its JSON-escaped form. This is never acceptable outside
  data/dataset.csv. Notebook .ipynb files are additionally scanned through
  their decoded JSON string content, so text hidden behind escape sequences
  or split across saved output lines is still found.
* HASH leak — a 16-char sha256 text_hash from splits/split_assignments.csv
  appears in a committed file. Hashes are the *sanctioned* row identifier, so
  a small allowlist of artifacts legitimately carries them (see
  HASH_ALLOWED_PATTERNS); anywhere else, a hash means row-level information
  escaped its designated artifacts.

Exit status is nonzero if any leak is found outside the allowlist. As of
2026-07 the nine committed notebooks are KNOWN to carry dataset text in saved
cell outputs (a documented open item), so this tool is expected to FAIL until
those outputs are stripped.

USAGE
-----
python3 tools/scan_text_leakage.py
"""

import argparse
import csv
import fnmatch
import hashlib
import json
import re
import subprocess
import sys

import repo_paths

# Column in data/dataset.csv that carries the raw message text.
DATASET_TEXT_COLUMN = "Original_Message"

# Texts shorter than this (after normalization) are not searched for: strings
# like "no" or "they are wrong" occur in ordinary prose and documentation by
# coincidence, so matching them would drown real leaks in false positives.
# 20 characters is comfortably above common-phrase length while still catching
# every realistic pasted message.
MIN_TEXT_MATCH_LENGTH = 20

# Length of the short sha256 prefix used as text_hash throughout the repo
# (see sha256_text in notebook 00).
TEXT_HASH_LENGTH = 16

# Cheap per-file screen before the expensive substring search: a text can only
# occur in a file if its longest alphanumeric token does. Tokens shorter than
# this are too common to discriminate, so texts without a long token skip the
# screen and are always searched in full.
SIGNATURE_TOKEN_MIN_LENGTH = 4

WHITESPACE_RE = re.compile(r"\s+")
SIGNATURE_TOKEN_RE = re.compile(r"[a-z0-9]{%d,}" % SIGNATURE_TOKEN_MIN_LENGTH)
HEX_RUN_RE = re.compile(r"[0-9a-f]{%d,}" % TEXT_HASH_LENGTH)

# Files that legitimately carry row_id / text_hash values. Each entry must
# justify itself; anything not listed here fails the hash check.
HASH_ALLOWED_PATTERNS = (
    # The canonical frozen split assignment: text_hash IS its join key.
    "splits/split_assignments.csv",
    # Forensic snapshot of the stale pre-repair mirror (untracked today, kept
    # on disk per CLAUDE.md; allowlisted in case it is ever committed).
    "splits/split_assignments.PRE-REPAIR.csv",
    # Committed probability artifacts are keyed by row_id/text_hash by design;
    # probs_artifact.load_probs already rejects any text column in them.
    "research_loop/probs/*.csv",
    # The duplicate report is keyed by text_hash (its text column must stay
    # empty — a populated text column is caught by the TEXT check instead).
    "results_summary/foundation/duplicate_text_report.csv",
)

# Number of leading bytes inspected to classify a file as binary.
BINARY_SNIFF_BYTES = 8192

# Output layout mirrors protocol_check: name column width, then STATUS.
STATUS_COLUMN = 46
RULE_WIDTH = 70


def normalize_text(text):
    """Collapse runs of whitespace to single spaces and strip, exactly as
    notebook 00's clean_text does before hashing. Matching against this form
    means a leaked text is found even when the leaking file re-wrapped it."""
    return WHITESPACE_RE.sub(" ", text).strip()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:TEXT_HASH_LENGTH]


def load_candidates():
    """Build the list of searchable dataset texts.

    Returns (candidates, n_skipped_short) where each candidate is a tuple
    (normalized_text, json_escaped_variant_or_None, signature_token_or_None,
    row_id_or_None). The row_id comes from joining the normalized text's hash
    against splits/split_assignments.csv, so reports can cite the leaked row
    without printing its text.
    """
    hash_to_row = {}
    with open(repo_paths.SPLIT_ASSIGNMENTS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            hash_to_row[row["text_hash"]] = row["row_id"]

    candidates = []
    n_skipped_short = 0
    with open(repo_paths.DATASET_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = normalize_text(row[DATASET_TEXT_COLUMN])
            if len(text) < MIN_TEXT_MATCH_LENGTH:
                n_skipped_short += 1
                continue
            # ensure_ascii=True so the variant covers \" \\ and \uXXXX forms
            # a JSON writer may have produced. Notebooks are scanned decoded,
            # so this variant only needs to cover plain JSON/text files.
            escaped = json.dumps(text, ensure_ascii=True)[1:-1]
            if escaped == text:
                escaped = None
            tokens = SIGNATURE_TOKEN_RE.findall(text.lower())
            signature = max(tokens, key=len) if tokens else None
            candidates.append(
                (text, escaped, signature, hash_to_row.get(sha256_text(text)))
            )
    return candidates, hash_to_row, n_skipped_short


def tracked_files():
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_paths.REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [p for p in out.split("\0") if p]


def notebook_string_content(raw):
    """Reassemble all string content of a notebook's JSON.

    nbformat stores cell source and output text as lists of line strings, so a
    dataset text can straddle list elements and hide behind JSON escapes. The
    decoded, joined form is what a reader of the rendered notebook would see,
    which is exactly the form leaks take.
    """
    parts = []

    def walk(node):
        if isinstance(node, str):
            parts.append(node)
        elif isinstance(node, list):
            if node and all(isinstance(item, str) for item in node):
                parts.append("".join(node))
            else:
                for item in node:
                    walk(item)
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)

    walk(json.loads(raw))
    return "\n".join(parts)


def find_leaked_texts(scan_text, candidates):
    """Return (count, example_row_id) for dataset texts present in scan_text.

    scan_text must already be whitespace-normalized. The token-set screen
    keeps the pass over clean files near O(1) per candidate; only candidates
    whose rarest long token appears in the file pay for a substring search.
    """
    token_set = set(SIGNATURE_TOKEN_RE.findall(scan_text.lower()))
    count = 0
    example_row_id = None
    for text, escaped, signature, row_id in candidates:
        if signature is not None and signature not in token_set:
            continue
        if text in scan_text or (escaped is not None and escaped in scan_text):
            count += 1
            if example_row_id is None and row_id is not None:
                example_row_id = row_id
    return count, example_row_id


def find_leaked_hashes(raw_content, hash_set):
    """Return the set of known text_hash values embedded in raw_content.

    Every 16-char window of each long hex run is tested so a hash is found
    even when it is embedded in a longer digest string.
    """
    found = set()
    for match in HEX_RUN_RE.finditer(raw_content):
        run = match.group(0)
        for i in range(len(run) - TEXT_HASH_LENGTH + 1):
            window = run[i : i + TEXT_HASH_LENGTH]
            if window in hash_set:
                found.add(window)
    return found


def hash_allowed(rel_path):
    return any(fnmatch.fnmatch(rel_path, pat) for pat in HASH_ALLOWED_PATTERNS)


def scan_repository():
    """Scan all tracked files; returns (text_leaks, hash_leaks, stats).

    text_leaks: {rel_path: (count, example_row_id)}
    hash_leaks: {rel_path: hash_count}   (allowlisted files excluded)
    """
    candidates, hash_to_row, n_skipped_short = load_candidates()
    hash_set = set(hash_to_row)
    dataset_rel = str(repo_paths.DATASET_CSV.relative_to(repo_paths.REPO_ROOT))

    text_leaks = {}
    hash_leaks = {}
    n_scanned = 0
    n_binary = 0

    for rel in tracked_files():
        if rel == dataset_rel:
            continue
        raw_bytes = (repo_paths.REPO_ROOT / rel).read_bytes()
        if b"\0" in raw_bytes[:BINARY_SNIFF_BYTES]:
            n_binary += 1
            continue
        raw = raw_bytes.decode("utf-8", errors="replace")
        n_scanned += 1

        if rel.endswith(".ipynb"):
            try:
                scan_text = normalize_text(notebook_string_content(raw))
            except json.JSONDecodeError:
                # A notebook that is not valid JSON still gets the plain scan.
                scan_text = normalize_text(raw)
        else:
            scan_text = normalize_text(raw)

        count, example_row_id = find_leaked_texts(scan_text, candidates)
        if count:
            text_leaks[rel] = (count, example_row_id)

        leaked_hashes = find_leaked_hashes(raw, hash_set)
        if leaked_hashes and not hash_allowed(rel):
            hash_leaks[rel] = len(leaked_hashes)

    stats = {
        "files_scanned": n_scanned,
        "binary_skipped": n_binary,
        "texts_searched": len(candidates),
        "texts_skipped_short": n_skipped_short,
    }
    return text_leaks, hash_leaks, stats


def report(text_leaks, hash_leaks, stats):
    """Print the protocol_check-style table plus per-file leak counts;
    return the number of blocking failures."""
    checks = [
        ("Dataset text confined to data/dataset.csv", not text_leaks),
        ("Row hashes confined to allowlisted artifacts", not hash_leaks),
    ]
    print(f"{'CHECK':<{STATUS_COLUMN}} STATUS")
    print("-" * RULE_WIDTH)
    for name, passed in checks:
        print(f"{name:<{STATUS_COLUMN}} {'PASS' if passed else 'FAIL'}")
    print()
    print(
        f"Scanned {stats['files_scanned']} tracked files "
        f"({stats['binary_skipped']} binary skipped) for "
        f"{stats['texts_searched']} dataset texts "
        f"({stats['texts_skipped_short']} below {MIN_TEXT_MATCH_LENGTH} chars "
        f"not searched) and {TEXT_HASH_LENGTH}-char text_hash values."
    )

    if text_leaks:
        print()
        print("Dataset text confined to data/dataset.csv:")
        for rel in sorted(text_leaks):
            count, example_row_id = text_leaks[rel]
            example = f" (e.g. {example_row_id})" if example_row_id else ""
            print(f"  - {rel}: {count} dataset texts in committed content{example}")
    if hash_leaks:
        print()
        print("Row hashes confined to allowlisted artifacts:")
        for rel in sorted(hash_leaks):
            print(f"  - {rel}: {hash_leaks[rel]} known text_hash values")

    return sum(1 for _, passed in checks if not passed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    text_leaks, hash_leaks, stats = scan_repository()
    n_failed = report(text_leaks, hash_leaks, stats)
    print()
    if n_failed:
        print(f"GATE: FAIL ({n_failed} blocking)")
        return 1
    print("GATE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
