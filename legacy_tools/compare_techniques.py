"""The judge: decide whether a candidate technique beats the champion.

No language model is in the path of this decision. Agents may read the verdict
string this script writes; nothing else may set it. That separation exists
because the failure this repository already suffered -- a validation PR-AUC of a
rejected candidate published as headline test accuracy -- is precisely what
happens when a numeric claim passes through a step that can paraphrase.

WHY PAIRED TESTS
----------------
Every technique is scored on the identical 450 test rows, so the comparison is
paired. McNemar's exact test conditions on the discordant pairs (rows where the
two techniques disagree) and is substantially more powerful than treating the two
accuracies as independent samples.

THE POWER REALITY
-----------------
At n=450 with ~40 discordant pairs, significance needs roughly |b-c| >= 13, i.e.
about +3 accuracy points. The current champion/runner-up gap is 5 rows, whose
best attainable exact p is 0.0625. INCONCLUSIVE is therefore the expected
outcome of honest work here, and it is treated as a first-class result rather
than a failure: the technique is still integrated and reported, it simply does
not take the champion pointer or earn a superlative.

MULTIPLICITY
------------
The Holm-Bonferroni family is the entire ledger, not the current cycle. Trying
twenty variants to find one winner multiplies the correction by twenty, so
shopping for a result pays for itself.

USAGE
-----
    python3 tools/compare_techniques.py --candidate X --champion Y --cycle 00
    python3 tools/compare_techniques.py --candidate X --champion Y --cycle 00 --dry-run
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics_core import compute_binary_metrics
from probs_artifact import align_pair, load_technique_probs
from repo_paths import DECISIONS_DIR, RESULTS_DIR

BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 30
CONFIDENCE_LEVEL = 0.95
ALPHA = 0.05

# A candidate can buy headline accuracy by drifting toward the majority class.
# The non-inferiority margin below is the largest balanced-accuracy regression
# tolerated before that trade is refused outright.
BALANCED_ACCURACY_MARGIN = 0.005

# In this domain a false positive labels a person's speech extremist. An FPR
# increase larger than this threshold does not change the verdict by itself,
# but it is flagged in the decision record and requires a human to acknowledge
# it before the result is integrated.
FPR_FLAG_THRESHOLD = 0.02

VERDICT_PROMOTE = "PROMOTE"
VERDICT_REJECT = "REJECT"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"
VERDICT_INVALID = "INVALID"

EVALUATION_SPLIT = "test"


def _load_threshold(technique):
    """Read the locked decision threshold from a technique's best_config.json."""
    config_path = RESULTS_DIR / technique / "best_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing {config_path}")
    with open(config_path) as handle:
        return float(json.load(handle)["selected_threshold"])


def mcnemar_exact(correct_a, correct_b):
    """Exact McNemar test on paired correctness vectors.

    `b` counts rows the champion got right and the candidate got wrong; `c` the
    reverse. Under the null the discordant pairs split 50/50, so the exact test
    is a two-sided binomial test on b of b+c. Returns (b, c, p_value).
    """
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(binomtest(b, b + c, 0.5).pvalue)


def paired_bootstrap_difference(y_true, prob_a, prob_b, threshold_a, threshold_b, metric="accuracy"):
    """Bootstrap CI for the candidate-minus-champion difference in `metric`.

    Resamples rows (not techniques), keeping both techniques' predictions for a
    resampled row together, which preserves the pairing that makes the comparison
    powerful.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(y_true)
    differences = np.empty(BOOTSTRAP_RESAMPLES)

    for i in range(BOOTSTRAP_RESAMPLES):
        idx = rng.integers(0, n, n)
        metrics_a = compute_binary_metrics(y_true[idx], prob_a[idx], threshold_a)
        metrics_b = compute_binary_metrics(y_true[idx], prob_b[idx], threshold_b)
        differences[i] = metrics_b[metric] - metrics_a[metric]

    tail = (1.0 - CONFIDENCE_LEVEL) / 2.0
    return {
        "mean": float(np.mean(differences)),
        "ci_low": float(np.quantile(differences, tail)),
        "ci_high": float(np.quantile(differences, 1.0 - tail)),
    }


def bootstrap_metric_ci(y_true, y_prob, threshold, metric="accuracy"):
    """Bootstrap CI for a single technique's metric, for honest reporting."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(y_true)
    values = np.empty(BOOTSTRAP_RESAMPLES)
    for i in range(BOOTSTRAP_RESAMPLES):
        idx = rng.integers(0, n, n)
        values[i] = compute_binary_metrics(y_true[idx], y_prob[idx], threshold)[metric]
    tail = (1.0 - CONFIDENCE_LEVEL) / 2.0
    return {
        "point": float(compute_binary_metrics(y_true, y_prob, threshold)[metric]),
        "ci_low": float(np.quantile(values, tail)),
        "ci_high": float(np.quantile(values, 1.0 - tail)),
    }


def holm_adjusted_p(p_value, family_size):
    """Holm-Bonferroni adjustment for the most significant test in a family.

    For the smallest p in a family of `family_size`, the Holm-adjusted value is
    min(1, p * family_size). Using the full ledger as the family is what makes
    repeated attempts self-penalizing.
    """
    return min(1.0, p_value * max(1, family_size))


def compare(candidate, champion, cycle, prereg_sha=None, family_size=None, dry_run=False):
    """Run all gates and emit the decision record."""
    from ledger import family_size as ledger_family_size

    if family_size is None:
        family_size = max(1, ledger_family_size())

    candidate_probs = load_technique_probs(candidate, EVALUATION_SPLIT)
    champion_probs = load_technique_probs(champion, EVALUATION_SPLIT)
    merged = align_pair(champion_probs, candidate_probs)

    threshold_champion = _load_threshold(champion)
    threshold_candidate = _load_threshold(candidate)

    y_true = merged["y_true"].values
    prob_champion = merged["y_prob_a"].values
    prob_candidate = merged["y_prob_b"].values

    metrics_champion = compute_binary_metrics(y_true, prob_champion, threshold_champion)
    metrics_candidate = compute_binary_metrics(y_true, prob_candidate, threshold_candidate)

    correct_champion = (prob_champion >= threshold_champion).astype(int) == y_true
    correct_candidate = (prob_candidate >= threshold_candidate).astype(int) == y_true

    b, c, p_raw = mcnemar_exact(correct_champion, correct_candidate)
    p_adjusted = holm_adjusted_p(p_raw, family_size)

    delta_accuracy = metrics_candidate["accuracy"] - metrics_champion["accuracy"]
    delta_balanced = metrics_candidate["balanced_accuracy"] - metrics_champion["balanced_accuracy"]
    delta_fpr = metrics_candidate["false_positive_rate"] - metrics_champion["false_positive_rate"]

    gate_registration = prereg_sha is not None
    gate_balanced = delta_balanced >= -BALANCED_ACCURACY_MARGIN
    gate_statistical = p_adjusted < ALPHA
    fpr_flag = delta_fpr > FPR_FLAG_THRESHOLD

    if not gate_registration:
        verdict = VERDICT_INVALID
        rationale = (
            "No preregistration hash supplied. The run is not admissible evidence, "
            "though it still counts toward the multiple-comparison family."
        )
    elif delta_accuracy <= 0:
        verdict = VERDICT_REJECT
        rationale = f"Test accuracy did not improve (delta = {delta_accuracy:+.4f})."
    elif not gate_balanced:
        verdict = VERDICT_REJECT
        rationale = (
            f"Balanced accuracy regressed by {-delta_balanced:.4f}, beyond the "
            f"{BALANCED_ACCURACY_MARGIN} margin: the accuracy gain came from "
            "drifting toward the majority class."
        )
    elif gate_statistical:
        verdict = VERDICT_PROMOTE
        rationale = (
            f"Test accuracy improved by {delta_accuracy:+.4f} and the paired difference survives Holm correction over "
            f"{family_size} test evaluations (adjusted p = "
            f"{p_adjusted:.4f})."
        )
    else:
        verdict = VERDICT_INCONCLUSIVE
        rationale = (
            f"Test accuracy is higher by {delta_accuracy:+.4f} ({b} rows the champion got right and the candidate missed, "
            f"{c} the reverse), but the difference is not distinguishable from noise at n="
            f"{len(y_true)} (McNemar exact p = "
            f"{p_raw:.4f}, Holm-adjusted over {family_size} evaluations = "
            f"{p_adjusted:.4f})."
        )

    decision = {
        "cycle": cycle,
        "candidate": candidate,
        "champion": champion,
        "verdict": verdict,
        "rationale": rationale,
        "decided_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "prereg_sha256": prereg_sha,
        "family_size": family_size,
        "n_test": int(len(y_true)),
        "gates": {
            "G0_registration": gate_registration,
            "G1_accuracy_delta": delta_accuracy,
            "G2_balanced_accuracy_non_inferior": gate_balanced,
            "G3_fpr_flag": fpr_flag,
            "G4_statistically_significant": gate_statistical,
        },
        "mcnemar": {
            "champion_right_candidate_wrong": b,
            "candidate_right_champion_wrong": c,
            "p_value_raw": p_raw,
            "p_value_holm_adjusted": p_adjusted,
            "alpha": ALPHA,
        },
        "deltas": {
            "accuracy": delta_accuracy,
            "balanced_accuracy": delta_balanced,
            "false_positive_rate": delta_fpr,
        },
        "candidate_metrics": metrics_candidate,
        "champion_metrics": metrics_champion,
        "thresholds": {"candidate": threshold_candidate, "champion": threshold_champion},
    }

    if not dry_run:
        decision["accuracy_ci"] = {
            "candidate": bootstrap_metric_ci(y_true, prob_candidate, threshold_candidate),
            "champion": bootstrap_metric_ci(y_true, prob_champion, threshold_champion),
            "paired_difference": paired_bootstrap_difference(
                y_true, prob_champion, prob_candidate, threshold_champion, threshold_candidate
            ),
        }
        DECISIONS_DIR.mkdir(parents=True, exist_ok=True)
        path = DECISIONS_DIR / f"CYCLE-{cycle}.json"
        with open(path, "w") as handle:
            json.dump(decision, handle, indent=2, sort_keys=True)
        print(f"Wrote {path}")

    print(f"\nVERDICT: {verdict}")
    print(f"  {rationale}")
    print(f"\n  {champion:<42} accuracy {metrics_champion['accuracy']:.4f}")
    print(f"  {candidate:<42} accuracy {metrics_candidate['accuracy']:.4f}")
    print(f"  delta {delta_accuracy:+.4f}   McNemar b={b} c={c} p={p_raw:.4f} (Holm over "
          f"{family_size}: {p_adjusted:.4f})")
    if fpr_flag:
        print(f"  FPR FLAG: false positive rate rose {delta_fpr:+.4f}; requires human acknowledgement")

    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--champion", required=True)
    parser.add_argument("--cycle", required=True)
    parser.add_argument("--prereg-sha", default=None)
    parser.add_argument("--family-size", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    compare(
        args.candidate,
        args.champion,
        args.cycle,
        prereg_sha=args.prereg_sha,
        family_size=args.family_size,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
