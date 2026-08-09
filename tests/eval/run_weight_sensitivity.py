"""Weight sensitivity analysis for the composite scoring formula.

Extends `run_scoring_ablation.py`'s 4-point discrete comparison to a 5x5 grid
sweep of (w_rel, w_risk) with w_div = 1 - w_rel - w_risk. Purpose is to answer
the reviewer critique that four hand-picked configurations do not establish
whether (0.40, 0.35, 0.25) is a robust choice or an idiosyncratic tuning point.

Grid:
    w_rel  in {0.30, 0.35, 0.40, 0.45, 0.50}
    w_risk in {0.25, 0.30, 0.35, 0.40, 0.45}
    w_div  = 1 - w_rel - w_risk   (kept only if in [0.05, 0.45])

For each valid configuration we compute RVR and SR on the same 108 synthetic
cases as the original ablation. We then report the ``stability neighbourhood''
of the WealthNexus configuration -- the fraction of the grid where both
(RVR, SR) are within +/- 5 percentage points of the reference point.

Usage:
    python tests/eval/run_weight_sensitivity.py
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
    THRESHOLD,
    _composite,
    build_test_cases,
)

W_REL_GRID = [0.30, 0.35, 0.40, 0.45, 0.50]
W_RISK_GRID = [0.25, 0.30, 0.35, 0.40, 0.45]
MIN_W_DIV, MAX_W_DIV = 0.05, 0.45

WEALTHNEXUS_POINT = (0.40, 0.35, 0.25)

RESULTS_PATH = SCRIPT_DIR / "fixtures" / "weight_sensitivity_results.json"


def evaluate_config(cases: list[dict], w_rel: float, w_risk: float, w_div: float) -> dict:
    """RVR/SR/breakdown for one (w_rel, w_risk, w_div) tuple."""
    rv_pass, rv_total = 0, 0
    sr_supp, sr_total = 0, 0
    sr_by_type = {t: [0, 0] for t in ("valid_suboptimal", "valid_adjacent", "valid_perfect")}

    for c in cases:
        score = _composite(c["rel"], c["risk_align"], c["div"], (w_rel, w_risk, w_div))
        passes = score >= THRESHOLD
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
        "weights": {"rel": w_rel, "risk": w_risk, "div": w_div},
        "rvr": rv_pass / rv_total if rv_total else 0.0,
        "sr": sr_supp / sr_total if sr_total else 0.0,
        "sr_by_type": {
            t: (v[0] / v[1] if v[1] else 0.0) for t, v in sr_by_type.items()
        },
    }


def main() -> int:
    ranker = RecommendationRanker()
    cases = build_test_cases(ranker)
    print(f"Loaded {len(cases)} synthetic cases")

    grid: list[dict] = []
    for w_rel in W_REL_GRID:
        for w_risk in W_RISK_GRID:
            w_div = round(1.0 - w_rel - w_risk, 4)
            if not (MIN_W_DIV <= w_div <= MAX_W_DIV):
                continue
            grid.append(evaluate_config(cases, w_rel, w_risk, w_div))

    # Find the WealthNexus reference row
    ref = next(g for g in grid
               if abs(g["weights"]["rel"] - WEALTHNEXUS_POINT[0]) < 1e-6
               and abs(g["weights"]["risk"] - WEALTHNEXUS_POINT[1]) < 1e-6)
    ref_rvr, ref_sr = ref["rvr"], ref["sr"]

    # Neighborhood: configs whose (RVR, SR) are within 5 pp of the reference.
    def within(g: dict, tol: float = 0.05) -> bool:
        return abs(g["rvr"] - ref_rvr) <= tol and abs(g["sr"] - ref_sr) <= tol

    stable = [g for g in grid if within(g)]
    stable_frac = len(stable) / len(grid) if grid else 0.0

    summary = {
        "n_grid_points": len(grid),
        "reference": {
            "weights": ref["weights"],
            "rvr": ref_rvr,
            "sr": ref_sr,
        },
        "stable_neighborhood": {
            "definition": "|RVR - ref.RVR| <= 0.05 and |SR - ref.SR| <= 0.05",
            "n_points": len(stable),
            "frac_of_grid": stable_frac,
        },
        "grid": grid,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== WEIGHT SENSITIVITY GRID ===")
    print(f"{'w_rel':>6} {'w_risk':>7} {'w_div':>6}   {'RVR':>6}  {'SR':>6}  "
          f"{'SR-sub':>7} {'SR-adj':>7} {'SR-perf':>8}")
    print("-" * 65)
    for g in grid:
        w = g["weights"]
        mark = " <-- WealthNexus" if (w["rel"] == 0.40 and w["risk"] == 0.35) else ""
        print(f"{w['rel']:>6.2f} {w['risk']:>7.2f} {w['div']:>6.2f}   "
              f"{g['rvr']:>5.1%}  {g['sr']:>5.1%}  "
              f"{g['sr_by_type']['valid_suboptimal']:>6.1%} "
              f"{g['sr_by_type']['valid_adjacent']:>6.1%} "
              f"{g['sr_by_type']['valid_perfect']:>7.1%}{mark}")

    print(f"\nReference RVR = {ref_rvr:.1%}, SR = {ref_sr:.1%}")
    print(f"Stable neighbourhood (both within +/-5pp of reference): "
          f"{len(stable)}/{len(grid)} = {stable_frac:.1%}")
    print(f"\nResults saved to {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
