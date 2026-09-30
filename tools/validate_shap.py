"""Validate the SHAP pipeline against logistic-regression coefficients.

Before any cross-model attribution claim, the pipeline must be shown to
recover what a linear model demonstrably relies on. For a logistic regression
over TF-IDF, a word's SHAP value on posts containing it has the sign of its
coefficient and grows with |coefficient|, so:

* the sign of `mean_attribution` must agree with the coefficient's sign for
  the top-K words by |coefficient| (default: at least 90 percent), and
* Spearman rank correlation between `mean_attribution` and the coefficient over
  all shared words must be high (default: at least 0.8).

Inputs, in results_summary/<TECHNIQUE>/attributions/:
    coefficients.csv   word, coefficient   (exported by the notebook)
    <run_id>.csv/.json the attribution run to check

Writes <run_id>__coefficient_check.json and exits 1 if either bar is missed.

USAGE
-----
    python3 tools/validate_shap.py --technique 01_LOG-REG_TF-IDF
    python3 tools/validate_shap.py --technique 01_LOG-REG_TF-IDF --run-id <run> --top-k 100
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attributions import AttributionArtifactError, attributions_dir, list_runs, load_run
from repo_paths import COEFFICIENTS_FILENAME

DEFAULT_MIN_SPEARMAN = 0.8
DEFAULT_MIN_SIGN_AGREEMENT = 0.9
DEFAULT_TOP_K = 50
COEFFICIENT_COLUMNS = ("word", "coefficient")


def load_coefficients(path):
    path = Path(path)
    if not path.exists():
        raise AttributionArtifactError(f"coefficient file not found: {path}")
    frame = pd.read_csv(path)
    missing = [c for c in COEFFICIENT_COLUMNS if c not in frame.columns]
    if missing:
        raise AttributionArtifactError(f"{path.name}: missing columns {missing}")
    extra = [c for c in frame.columns if c not in COEFFICIENT_COLUMNS]
    if extra:
        raise AttributionArtifactError(f"{path.name}: unexpected columns {extra}")
    frame = frame.copy()
    frame["word"] = frame["word"].astype(str)
    frame["coefficient"] = pd.to_numeric(frame["coefficient"], errors="coerce")
    if not np.isfinite(frame["coefficient"]).all():
        raise AttributionArtifactError(f"{path.name}: non-finite coefficients")
    if frame["word"].duplicated().any():
        raise AttributionArtifactError(f"{path.name}: duplicate words")
    return frame


def coefficient_check(attributions, coefficients, top_k=DEFAULT_TOP_K):
    """Compute the agreement statistics between one run and the coefficients."""
    merged = coefficients.merge(attributions, on="word", how="inner")
    if len(merged) < 3:
        raise AttributionArtifactError(
            f"only {len(merged)} words shared between coefficients and attributions"
        )
    rho, _ = spearmanr(merged["coefficient"], merged["mean_attribution"])

    ranked = coefficients.reindex(coefficients["coefficient"].abs().sort_values(ascending=False).index)
    top = ranked.head(top_k)
    top_present = top.merge(attributions, on="word", how="inner")
    coverage = len(top_present) / max(len(top), 1)
    if len(top_present):
        agree = np.sign(top_present["coefficient"]) == np.sign(top_present["mean_attribution"])
        sign_agreement = float(agree.mean())
        disagreeing = sorted(top_present.loc[~agree, "word"].tolist())
    else:
        sign_agreement = 0.0
        disagreeing = []

    return {
        "n_shared_words": int(len(merged)),
        "spearman_mean_attribution_vs_coefficient": float(rho),
        "top_k": int(top_k),
        "top_k_coverage": float(coverage),
        "top_k_sign_agreement": sign_agreement,
        "top_k_sign_disagreements": disagreeing,
    }


def validate(technique, run_id=None, top_k=DEFAULT_TOP_K, min_spearman=DEFAULT_MIN_SPEARMAN,
             min_sign_agreement=DEFAULT_MIN_SIGN_AGREEMENT, results_dir=None):
    folder = attributions_dir(technique, results_dir)
    runs = list_runs(technique, results_dir)
    if run_id is None:
        if len(runs) != 1:
            raise AttributionArtifactError(
                f"{technique} has {len(runs)} runs; pass --run-id to choose one"
            )
        run_id = runs[0]
    frame, meta = load_run(technique, run_id, results_dir)
    coefficients = load_coefficients(folder / COEFFICIENTS_FILENAME)

    stats = coefficient_check(frame, coefficients, top_k=top_k)
    stats["passed"] = (
        stats["spearman_mean_attribution_vs_coefficient"] >= min_spearman
        and stats["top_k_sign_agreement"] >= min_sign_agreement
    )
    stats["thresholds"] = {"min_spearman": min_spearman, "min_sign_agreement": min_sign_agreement}
    stats["technique"] = technique
    stats["run_id"] = run_id
    stats["explainer"] = meta["explainer"]

    out = folder / f"{run_id}__coefficient_check.json"
    with open(out, "w") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)
    return stats, out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--technique", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--min-spearman", type=float, default=DEFAULT_MIN_SPEARMAN)
    parser.add_argument("--min-sign-agreement", type=float, default=DEFAULT_MIN_SIGN_AGREEMENT)
    args = parser.parse_args()

    try:
        stats, out = validate(
            args.technique, args.run_id, args.top_k, args.min_spearman, args.min_sign_agreement
        )
    except AttributionArtifactError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"wrote {out}")
    print(f"  shared words        {stats['n_shared_words']}")
    print(f"  spearman            {stats['spearman_mean_attribution_vs_coefficient']:.4f}")
    print(f"  top-{stats['top_k']} sign agreement  {stats['top_k_sign_agreement']:.4f} "
          f"(coverage {stats['top_k_coverage']:.2f})")
    if stats["top_k_sign_disagreements"]:
        print(f"  disagreeing words   {', '.join(stats['top_k_sign_disagreements'])}")
    print(f"  PIPELINE CHECK: {'PASS' if stats['passed'] else 'FAIL'}")
    return 0 if stats["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
