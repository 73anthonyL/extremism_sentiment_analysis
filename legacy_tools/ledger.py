"""Append-only, hash-chained ledger of held-out test evaluations.

Every look at the test split is an inferential cost. With n=450 and a champion at
~0.889 accuracy, a handful of unrecorded peeks is enough to manufacture a
"winner" from noise. The ledger makes that cost explicit and non-erasable:

* one entry per technique, ever -- a second unlock for the same technique is
  refused, so re-rolling a result requires registering a new technique with a new
  preregistration, which lengthens the multiple-comparison family;
* each entry carries the hash of the previous entry, so deleting or editing
  history breaks the chain and `verify` reports it;
* `promote` refuses unless the corresponding decision file says PROMOTE, so the
  champion pointer cannot be moved by assertion -- only by a recorded verdict.

The chain is the input to the Holm correction in compare_techniques.py. Shopping
for a winner therefore penalizes itself automatically.

USAGE
-----
    python3 tools/ledger.py verify
    python3 tools/ledger.py append --technique X --cycle 00 --prereg-sha <sha>
    python3 tools/ledger.py promote --cycle 00
    python3 tools/ledger.py show
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from repo_paths import DECISIONS_DIR, LOOP_DIR, REGISTRY_JSON, RESULTS_DIR, TEST_LEDGER_JSONL

# The first entry's prev_hash anchors to this all-zero value. Anchoring to a
# constant (rather than accepting whatever the first entry claims) means the
# chain cannot be silently truncated from the front: dropping entry 0 makes the
# new first entry's prev_hash fail to match the genesis value. Sixty-four hex
# zeros mirrors the width of the SHA-256 digests that form the rest of the
# chain, so every link is checked the same way.
GENESIS_HASH = "0000000000000000000000000000000000000000000000000000000000000000"
VERDICT_PROMOTE = "PROMOTE"

EXIT_OK = 0
EXIT_FAILED = 1


class LedgerError(RuntimeError):
    """Raised when an operation would violate append-only or one-unlock rules."""


def _entry_hash(entry):
    """Hash an entry's content plus its predecessor, forming the chain link."""
    payload = {key: value for key, value in entry.items() if key != "entry_hash"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_entries():
    """Read all ledger entries in order, or [] when the ledger does not exist."""
    if not TEST_LEDGER_JSONL.exists():
        return []
    entries = []
    with open(TEST_LEDGER_JSONL) as handle:
        for line in handle:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def verify_chain(entries=None):
    """Recompute the hash chain. Returns (ok, problems)."""
    entries = read_entries() if entries is None else entries
    problems = []
    previous = GENESIS_HASH
    seen_techniques = {}

    for index, entry in enumerate(entries):
        if entry.get("prev_hash") != previous:
            problems.append(
                f"entry {index} ({entry.get('technique')}): prev_hash mismatch -- history was edited or reordered"
            )

        recomputed = _entry_hash(entry)
        if entry.get("entry_hash") != recomputed:
            problems.append(
                f"entry {index} ({entry.get('technique')}): content hash mismatch -- entry was modified after being written"
            )

        technique = entry.get("technique")
        if technique in seen_techniques:
            problems.append(
                f"entry {index}: technique '{technique}' unlocked twice (first at entry "
                f"{seen_techniques[technique]})"
            )
        else:
            seen_techniques[technique] = index
        previous = entry.get("entry_hash", recomputed)

    return not problems, problems


def append_entry(technique, cycle, prereg_sha=None, note=None, timestamp=None):
    """Append one test-unlock entry, reading the metrics from results_summary/.

    Metrics are read from the committed artifact rather than accepted as
    arguments, so a ledger entry cannot record a number that does not exist on
    disk.
    """
    entries = read_entries()
    ok, problems = verify_chain(entries)
    if not ok:
        raise LedgerError("refusing to append to a broken chain:\n  " + "\n  ".join(problems))

    if any(entry.get("technique") == technique for entry in entries):
        raise LedgerError(
            f"technique '{technique}' already has a test evaluation. The test split is "
            "unlocked once per technique. Register a new technique id with its own "
            "preregistration if a further evaluation is genuinely required."
        )

    metrics_path = RESULTS_DIR / technique / "metrics_test.json"
    if not metrics_path.exists():
        raise LedgerError(f"no committed test metrics at {metrics_path}")
    with open(metrics_path) as handle:
        metrics = json.load(handle)

    entry = {
        "cycle": cycle,
        "technique": technique,
        "timestamp_utc": timestamp
        or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "prereg_sha256": prereg_sha,
        "accuracy": metrics.get("accuracy"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "positive_f1": metrics.get("positive_f1"),
        "pr_auc": metrics.get("pr_auc"),
        "false_positive_rate": metrics.get("false_positive_rate", metrics.get("fpr")),
        "tn": metrics.get("tn"),
        "fp": metrics.get("fp"),
        "fn": metrics.get("fn"),
        "tp": metrics.get("tp"),
        "note": note,
        "prev_hash": entries[-1]["entry_hash"] if entries else GENESIS_HASH,
    }
    entry["entry_hash"] = _entry_hash(entry)

    LOOP_DIR.mkdir(parents=True, exist_ok=True)
    with open(TEST_LEDGER_JSONL, "a") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def family_size():
    """Number of recorded test evaluations -- the multiple-comparison family size."""
    return len(read_entries())


def promote(cycle):
    """Move the champion pointer, only on a PROMOTE verdict from the judge.

    This is the sole write path to registry.json's champion field; there is no
    other route to it.
    """
    decision_path = DECISIONS_DIR / f"CYCLE-{cycle}.json"
    if not decision_path.exists():
        raise LedgerError(f"no decision file at {decision_path}")
    with open(decision_path) as handle:
        decision = json.load(handle)

    verdict = decision.get("verdict")
    if verdict != VERDICT_PROMOTE:
        raise LedgerError(
            f"decision for cycle {cycle} is '{verdict}', not {VERDICT_PROMOTE}. The champion pointer is unchanged."
        )

    # The verdict is PROMOTE; the pointer may move.
    with open(REGISTRY_JSON) as handle:
        registry = json.load(handle)

    registry["champion"] = {
        "technique": decision["candidate"],
        "cycle": cycle,
        "test_accuracy": decision["candidate_metrics"]["accuracy"],
        "promoted_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    with open(REGISTRY_JSON, "w") as handle:
        json.dump(registry, handle, indent=2, sort_keys=True)
    return registry["champion"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("verify", help="recompute the hash chain")
    sub.add_parser("show", help="print the ledger")

    append_parser = sub.add_parser("append", help="record a test evaluation")
    append_parser.add_argument("--technique", required=True)
    append_parser.add_argument("--cycle", required=True)
    append_parser.add_argument("--prereg-sha", default=None)
    append_parser.add_argument("--note", default=None)

    promote_parser = sub.add_parser("promote", help="move the champion pointer")
    promote_parser.add_argument("--cycle", required=True)

    args = parser.parse_args()

    try:
        if args.command == "verify":
            ok, problems = verify_chain()
            entries = read_entries()
            print(f"Ledger entries: {len(entries)}")
            print(f"Chain intact:   {ok}")
            for problem in problems:
                print(f"  PROBLEM: {problem}")
            return EXIT_OK if ok else EXIT_FAILED

        if args.command == "show":
            for entry in read_entries():
                print(
                    f"{entry['cycle']:>4}  {entry['technique']:<40} acc="
                    f"{entry['accuracy']:.4f}  {entry['timestamp_utc']}"
                )
            return EXIT_OK

        if args.command == "append":
            entry = append_entry(
                args.technique, args.cycle, prereg_sha=args.prereg_sha, note=args.note
            )
            print(f"Appended test evaluation for {entry['technique']}")
            print(f"  accuracy    {entry['accuracy']:.4f}")
            print(f"  family size {family_size()}")
            return EXIT_OK

        if args.command == "promote":
            champion = promote(args.cycle)
            print(f"Champion is now {champion['technique']} (accuracy {champion['test_accuracy']:.4f})")
            return EXIT_OK

    except LedgerError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return EXIT_FAILED

    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
