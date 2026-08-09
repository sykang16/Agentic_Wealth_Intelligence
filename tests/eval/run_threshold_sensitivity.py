"""Threshold sensitivity analysis for the No-Recommendation Threshold.

Reviewer critique: the 0.55 threshold is justified by a single ``worst-case
minimum'' calculation (A=0.7, R=0.5, D=0 -> composite 0.445). This is
post-hoc; a sweep is needed to show that 0.55 is a defensible knee point
rather than an arbitrary number.

Fixed WealthNexus weights (0.40 / 0.35 / 0.25) throughout. Sweep the composite
threshold across {0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70} on the same
108 synthetic cases and report Risk Violation Rate (RVR) and Suppression Rate
(SR) at each threshold.

Usage:
    python tests/eval/run_threshold_sensitivity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.recommendation.engine.ranker import RecommendationRanker

from tests.eval.run_scoring_ablation import (  # noqa: E402
    _composite,
    build_test_cases,
)

WEIGHTS = (0.40, 0.35, 0.25)
THRESHOLD_GRID = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
RESULTS_PATH = SCRIPT_DIR / "fixtures" / "threshold_sensitivity_results.json"


def evaluate_threshold(cases: list[dict], threshold: float) -> dict:
    rv_pass, rv_total = 0, 0
    sr_supp, sr_total = 0, 0
    sr_by_type = {t: [0, 0] for t in ("valid_suboptimal", "valid_adjacent", "valid_perfect")}

    for c in cases:
        score = _composite(c["rel"], c["risk_align"], c["div"], WEIGHTS)
        passes = score >= threshold
        if c["scenario"] == "risk_violation":
            rv_total += 1
            if passes:
                rv_pass += 1
        else:
            sr_total += 1
            sr_by_type[c["scenario"]][1] += 1
            if not passes:
                sr_supp += 1
                sr_by_type[c["scenario"]][0] += 1

    return {
        "threshold": threshold,
        "rvr": rv_pass / rv_total if rv_total else 0.0,
        "sr": sr_supp / sr_total if sr_total else 0.0,
        "sr_by_type": {
            t: (v[0] / v[1] if v[1] else 0.0) for t, v in sr_by_type.items()
        },
    }


def main() -> int:
    ranker = RecommendationRanker()
    cases = build_test_cases(ranker)
    print(f"Loaded {len(cases)} synthetic cases (weights fixed at "
          f"{WEIGHTS[0]}/{WEIGHTS[1]}/{WEIGHTS[2]})")

    rows = [evaluate_threshold(cases, t) for t in THRESHOLD_GRID]

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({"weights": WEIGHTS, "grid": rows}, indent=2), encoding="utf-8")

    print("\n=== THRESHOLD SENSITIVITY ===")
    print(f"{'threshold':>10}  {'RVR':>6}  {'SR':>6}  "
          f"{'SR-sub':>7} {'SR-adj':>7} {'SR-perf':>8}")
    print("-" * 60)
    for r in rows:
        mark = " <-- default" if abs(r["threshold"] - 0.55) < 1e-6 else ""
        print(f"{r['threshold']:>10.2f}  "
              f"{r['rvr']:>5.1%}  {r['sr']:>5.1%}  "
              f"{r['sr_by_type']['valid_suboptimal']:>6.1%} "
              f"{r['sr_by_type']['valid_adjacent']:>6.1%} "
              f"{r['sr_by_type']['valid_perfect']:>7.1%}{mark}")

    print(f"\nResults saved to {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
