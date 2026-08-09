"""
Recommendation Scoring Ablation: Composite Weight Sensitivity Analysis.

Compares four weight configurations for the composite scoring formula:
  composite = w_rel * relevance + w_risk * risk_alignment + w_div * diversification

Configurations:
  A  Equal weights:    0.333 / 0.333 / 0.333
  B  Relevance-only:   1.000 / 0.000 / 0.000
  C  Risk-first:       0.200 / 0.600 / 0.200
  D  WealthNexus:      0.400 / 0.350 / 0.250  (proposed)

Test set: 108 synthetic cases covering four risk-scenario types x
          3 priority levels x 3 confidence levels x 3 diversification levels.

Metrics:
  Risk Violation Rate (RVR) -- fraction of inappropriate-risk items (conservative
      user, HIGH-risk rec) that pass the 0.55 composite threshold (false positives).
  Suppression Rate (SR)     -- fraction of valid items (risk-compatible, post-filter)
      that are suppressed by the threshold (false negatives, by sub-type).

Usage:
    python tests/eval/run_scoring_ablation.py
"""

from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.src.recommendation.engine.ranker import RecommendationRanker
from backend.src.recommendation.engine.schemas import (
    AggregatedContext,
    ConfidenceLevel,
    Recommendation,
    RecommendationCategory,
    RecommendationPriority,
    RiskLevel,
)

THRESHOLD = 0.55
RESULTS_PATH = (
    project_root / "tests" / "eval" / "fixtures" / "scoring_ablation_results.json"
)

WEIGHT_CONFIGS: dict[str, tuple[float, float, float]] = {
    "A_equal":      (0.333, 0.333, 0.333),
    "B_rel_only":   (1.000, 0.000, 0.000),
    "C_risk_first": (0.200, 0.600, 0.200),
    "D_wealthnexus":(0.400, 0.350, 0.250),
}

CONFIG_LABELS = {
    "A_equal":      "A: Equal (0.33/0.33/0.33)",
    "B_rel_only":   "B: Relevance-only (1.0/0/0)",
    "C_risk_first": "C: Risk-first (0.20/0.60/0.20)",
    "D_wealthnexus":"D: WealthNexus (0.40/0.35/0.25)",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_rec(
    priority: RecommendationPriority,
    confidence: ConfidenceLevel,
    risk_level: RiskLevel,
) -> Recommendation:
    return Recommendation(
        id="abl_001",
        category=RecommendationCategory.BUY,
        title="Test recommendation",
        summary="Ablation test item.",
        detailed_rationale="Testing composite weight sensitivity.",
        suggested_action="Take no action (test only).",
        risk_level=risk_level,
        priority=priority,
        confidence=confidence,
    )


def _composite(rel: float, risk_align: float, div: float,
               weights: tuple[float, float, float]) -> float:
    w_r, w_k, w_d = weights
    return w_r * rel + w_k * risk_align + w_d * div


# ─────────────────────────────────────────────────────────────────────────────
# Test-case generation
# ─────────────────────────────────────────────────────────────────────────────

def build_test_cases(ranker: RecommendationRanker) -> list[dict]:
    """
    Four scenario types (27 cases each, 108 total):

    1. risk_violation  — conservative user, HIGH-risk rec  (risk_align = 0.3)
       These items SHOULD be suppressed.  If they pass, it is a risk violation.

    2. valid_suboptimal — aggressive user, LOW-risk rec  (risk_align = 0.3)
       Items are risk-compatible (pass hard filter) but sub-optimal fit.
       Being too conservative with weights (Config C) causes false suppression.

    3. valid_adjacent  — moderate user, LOW-risk rec  (risk_align = 0.7)
       Items are risk-compatible, adjacent risk.  Should generally pass.

    4. valid_perfect   — moderate user, MODERATE-risk rec  (risk_align = 1.0)
       Ideal items.  Should almost always pass under any config.
    """
    cases: list[dict] = []

    scenario_map = {
        "risk_violation":  (RiskLevel.HIGH,  0.3),  # conservative+HIGH → mismatch
        "valid_suboptimal": (RiskLevel.LOW,   0.3),  # aggressive+LOW → sub-optimal
        "valid_adjacent":  (RiskLevel.LOW,   0.7),  # moderate+LOW → adjacent
        "valid_perfect":   (RiskLevel.MODERATE, 1.0),  # moderate+MODERATE → perfect
    }

    priorities   = [RecommendationPriority.HIGH,
                    RecommendationPriority.MEDIUM,
                    RecommendationPriority.LOW]
    confidences  = [ConfidenceLevel.HIGH,
                    ConfidenceLevel.MEDIUM,
                    ConfidenceLevel.LOW]
    div_levels   = [0.2, 0.5, 0.8]  # poor / neutral / good diversification

    for scenario, (risk_level, risk_align) in scenario_map.items():
        for priority, confidence, div in product(priorities, confidences, div_levels):
            rec = _make_rec(priority, confidence, risk_level)
            rel = ranker._score_relevance(rec)
            cases.append({
                "scenario":   scenario,
                "priority":   priority.value,
                "confidence": confidence.value,
                "div":        div,
                "rel":        rel,
                "risk_align": risk_align,
            })

    return cases


# ─────────────────────────────────────────────────────────────────────────────
# Run ablation
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(cases: list[dict]) -> dict:
    results: dict[str, dict] = {}

    for key, weights in WEIGHT_CONFIGS.items():
        rv_pass   = 0   # risk violations (should suppress, but passed)
        rv_total  = 0
        sr_supp   = 0   # valid items suppressed
        sr_total  = 0

        # By sub-type for suppression
        sr_by_type: dict[str, dict] = {
            "valid_suboptimal": {"suppressed": 0, "total": 0},
            "valid_adjacent":   {"suppressed": 0, "total": 0},
            "valid_perfect":    {"suppressed": 0, "total": 0},
        }

        # Per-priority breakdown for risk violations
        rv_by_priority: dict[str, dict] = {
            "high":   {"pass": 0, "total": 0},
            "medium": {"pass": 0, "total": 0},
            "low":    {"pass": 0, "total": 0},
        }

        all_composites: list[float] = []

        for case in cases:
            score = _composite(case["rel"], case["risk_align"],
                               case["div"], weights)
            passes = score >= THRESHOLD
            all_composites.append(score)

            if case["scenario"] == "risk_violation":
                rv_total += 1
                prio = case["priority"]
                rv_by_priority[prio]["total"] += 1
                if passes:
                    rv_pass += 1
                    rv_by_priority[prio]["pass"] += 1
            else:
                sr_total += 1
                sr_by_type[case["scenario"]]["total"] += 1
                if not passes:
                    sr_supp += 1
                    sr_by_type[case["scenario"]]["suppressed"] += 1

        rvr = rv_pass  / rv_total  if rv_total  > 0 else 0.0
        sr  = sr_supp  / sr_total  if sr_total  > 0 else 0.0

        sr_by_type_rate = {
            t: (v["suppressed"] / v["total"] if v["total"] > 0 else 0.0)
            for t, v in sr_by_type.items()
        }
        rv_by_priority_rate = {
            p: (v["pass"] / v["total"] if v["total"] > 0 else 0.0)
            for p, v in rv_by_priority.items()
        }

        avg_composite = sum(all_composites) / len(all_composites)

        results[key] = {
            "label":             CONFIG_LABELS[key],
            "weights":           {"rel": weights[0], "risk": weights[1], "div": weights[2]},
            "rvr":               round(rvr, 4),
            "sr":                round(sr,  4),
            "rv_pass":           rv_pass,
            "rv_total":          rv_total,
            "sr_suppressed":     sr_supp,
            "sr_total":          sr_total,
            "sr_by_type":        {t: round(v, 4) for t, v in sr_by_type_rate.items()},
            "rv_by_priority":    {p: round(v, 4) for p, v in rv_by_priority_rate.items()},
            "avg_composite":     round(avg_composite, 4),
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ranker = RecommendationRanker()

    print("Building test cases...")
    cases = build_test_cases(ranker)
    print(f"  Total cases: {len(cases)}")
    by_scenario = {}
    for c in cases:
        by_scenario[c["scenario"]] = by_scenario.get(c["scenario"], 0) + 1
    for s, n in by_scenario.items():
        print(f"  {s}: {n}")

    print("\nRunning ablation...")
    results = run_ablation(cases)

    print("\n=== RESULTS ===")
    header = f"{'Config':<38} {'RVR':>6}  {'SR':>6}  {'SR-subopt':>10}  {'SR-adj':>8}  {'SR-perf':>8}"
    print(header)
    print("-" * len(header))
    for key, r in results.items():
        print(
            f"{r['label']:<38} "
            f"{r['rvr']:>5.1%}  "
            f"{r['sr']:>5.1%}  "
            f"{r['sr_by_type']['valid_suboptimal']:>9.1%}  "
            f"{r['sr_by_type']['valid_adjacent']:>7.1%}  "
            f"{r['sr_by_type']['valid_perfect']:>7.1%}"
        )

    print("\n  RV by priority (risk violations that pass threshold):")
    for key, r in results.items():
        rv = r["rv_by_priority"]
        print(
            f"  {r['label']:<38} "
            f"HIGH={rv['high']:.1%}  MED={rv['medium']:.1%}  LOW={rv['low']:.1%}"
        )

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
