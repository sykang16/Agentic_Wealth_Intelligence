"""Analyze results from Experiment 4a (RAG similarity-threshold gating).

Reads an exp4a run directory and produces:
    1. Gating summary: how often each system gated the RAG context.
    2. Pairwise confirmed-win table with binomial 95% CIs and sign-test p-values.
    3. Per-criterion winner distributions.
    4. Per-session breakdown showing where gating changed the outcome.
    5. Markdown report suitable for pasting into the manuscript revision.

Usage:
    python tests/eval/llm_quality/analyze_experiment_4a.py \\
        --run-dir tests/eval/llm_quality/results/runs/exp4a_full_v1

Outputs (in --run-dir):
    exp4a_analysis.json
    exp4a_analysis.md
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend" / "src"))

CRITERIA = [
    "C1_personalization",
    "C2_risk_alignment",
    "C3_factual_grounding",
    "C4_actionability",
    "C5_diversification",
    "C6_safety_compliance",
]

logger = logging.getLogger("analyze_4a")


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def two_sided_binomial_pvalue(wins: int, n: int, p0: float = 0.5) -> float:
    """Exact two-sided binomial test against H0: p = p0.

    Computes P(X = k) exactly using log-space and sums the two tails.
    """
    if n == 0:
        return 1.0
    # Compute all P(X = k) exactly using log gamma.
    log_probs = []
    log_choose_cache = math.lgamma(n + 1)
    for k in range(n + 1):
        log_binom_coef = log_choose_cache - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        log_p = log_binom_coef + k * math.log(p0) + (n - k) * math.log(1 - p0)
        log_probs.append(log_p)
    probs = [math.exp(lp) for lp in log_probs]
    observed = probs[wins]
    # Two-sided: sum all outcomes as extreme or more extreme
    pval = sum(p for p in probs if p <= observed + 1e-12)
    return min(1.0, pval)


def _resolve_pair_winner(order1_parsed: dict | None, order2_parsed: dict | None,
                         sys_a: str, sys_b: str) -> str:
    """Return winner among {sys_a, sys_b, "tie", "unavailable"}.

    An "unavailable" verdict is returned when either order's judgment failed to
    parse; this is distinct from a "tie" (both orders judged but disagreed or
    both said tie), and callers should typically exclude "unavailable" from
    confirmed-win counts.
    """
    if not order1_parsed or not order2_parsed:
        return "unavailable"
    o1 = order1_parsed.get("overall_winner")
    o2 = order2_parsed.get("overall_winner")
    o1_sys = sys_a if o1 == "A" else (sys_b if o1 == "B" else "tie")
    o2_sys = sys_b if o2 == "A" else (sys_a if o2 == "B" else "tie")
    if o1_sys == o2_sys and o1_sys != "tie":
        return o1_sys
    return "tie"


def load_pairwise(run_dir: Path) -> dict:
    """Group pairwise judgment files by (session, pair, judge, order)."""
    pw_dir = run_dir / "pairwise"
    grouped: dict[tuple, dict[int, dict | None]] = {}
    per_session_metadata: dict[str, dict] = {}
    for fp in pw_dir.glob("*.json"):
        d = json.loads(fp.read_text(encoding="utf-8"))
        key = (d["session_id"], d["sys_a"], d["sys_b"], d["judge_label"])
        grouped.setdefault(key, {})[d["order"]] = d.get("parsed")
        per_session_metadata[d["session_id"]] = {
            "persona_id": d.get("persona_id"),
            "query_type": d.get("query_type"),
            "seed": d.get("seed"),
        }
    return {"grouped": grouped, "per_session": per_session_metadata}


def load_gating_stats(run_dir: Path) -> dict:
    """Compute per-system gating stats from session files."""
    sessions_dir = run_dir / "sessions"
    per_system: dict[str, dict] = {}
    per_session_gating: dict[tuple[str, str], bool] = {}
    for fp in sessions_dir.glob("*.json"):
        d = json.loads(fp.read_text(encoding="utf-8"))
        sys_label = d.get("system")
        if not sys_label:
            continue
        entry = per_system.setdefault(sys_label, {
            "n_sessions": 0, "n_gated": 0, "top_scores": [],
        })
        entry["n_sessions"] += 1
        rag = d.get("rag_diagnostics") or {}
        gated = bool(rag.get("gated"))
        if gated:
            entry["n_gated"] += 1
        top = rag.get("top_score", 0.0)
        if top is not None:
            entry["top_scores"].append(float(top))
        per_session_gating[(d["session_id"], sys_label)] = gated

    for sys_label, e in per_system.items():
        scores = e.pop("top_scores")
        e["gated_frac"] = e["n_gated"] / e["n_sessions"] if e["n_sessions"] else 0.0
        e["top_score_mean"] = sum(scores) / len(scores) if scores else 0.0
    return {
        "per_system": per_system,
        "per_session": {f"{k[0]}||{k[1]}": v for k, v in per_session_gating.items()},
    }


def compute_pair_analysis(pw_grouped: dict, sys_a: str, sys_b: str) -> dict:
    """For one pair, compute confirmed wins, CIs, p-value, and per-criterion wins.

    Verdicts where either order failed to parse are counted as ``unavailable`` and
    excluded from both the confirmed-win count and the per-criterion rollup so
    that judge failures do not silently inflate the tie rate.
    """
    confirmed = {sys_a: 0, sys_b: 0, "tie": 0, "unavailable": 0}
    per_crit: dict[str, dict[str, int]] = {c: {sys_a: 0, sys_b: 0, "tie": 0} for c in CRITERIA}

    seen_keys = set()
    for (sess_id, s_a, s_b, judge), orders in pw_grouped.items():
        if (s_a, s_b) != (sys_a, sys_b):
            continue
        winner = _resolve_pair_winner(orders.get(1), orders.get(2), sys_a, sys_b)
        confirmed[winner] += 1
        seen_keys.add((sess_id, judge))

        # Skip per-criterion aggregation for verdicts with missing orders --
        # the confirmed-win protocol requires both orders to be valid.
        if winner == "unavailable":
            continue

        for crit in CRITERIA:
            for order, parsed in orders.items():
                if not parsed:
                    continue
                wpc = parsed.get("winner_per_criterion", {}).get(crit)
                if wpc is None:
                    continue
                if order == 1:
                    decoded = sys_a if wpc == "A" else (sys_b if wpc == "B" else "tie")
                else:
                    decoded = sys_b if wpc == "A" else (sys_a if wpc == "B" else "tie")
                per_crit[crit][decoded] += 1

    n_available = confirmed[sys_a] + confirmed[sys_b] + confirmed["tie"]
    n_decided = confirmed[sys_a] + confirmed[sys_b]
    n_unavailable = confirmed["unavailable"]
    a_rate_of_available = confirmed[sys_a] / n_available if n_available else 0.0
    a_rate_of_decided = confirmed[sys_a] / n_decided if n_decided else 0.0

    ci_available = wilson_ci(confirmed[sys_a], n_available)
    ci_decided = wilson_ci(confirmed[sys_a], n_decided)
    pval_decided = two_sided_binomial_pvalue(confirmed[sys_a], n_decided, p0=0.5) if n_decided else 1.0

    return {
        "sys_a": sys_a,
        "sys_b": sys_b,
        "n_confirmed_verdicts": n_available,
        "n_unavailable": n_unavailable,
        "confirmed_wins": confirmed,
        "a_rate_of_total": a_rate_of_available,      # over available verdicts
        "a_rate_of_decided": a_rate_of_decided,
        "wilson_ci_a_of_total": ci_available,
        "wilson_ci_a_of_decided": ci_decided,
        "sign_test_pvalue_decided": pval_decided,
        "per_criterion": per_crit,
    }


def load_baseline_ab2(baseline_run: Path | None) -> dict | None:
    """Load pre-existing A vs B2 pairwise results from the original Phase B full run."""
    if baseline_run is None or not baseline_run.exists():
        return None
    pw_dir = baseline_run / "pairwise"
    if not pw_dir.exists():
        return None
    grouped: dict[tuple, dict[int, dict | None]] = {}
    for fp in pw_dir.glob("*.json"):
        d = json.loads(fp.read_text(encoding="utf-8"))
        sys_a = d.get("sys_a")
        sys_b = d.get("sys_b")
        if not (sys_a == "A_wealthnexus" and sys_b == "B2_rag_blind"):
            continue
        key = (d["session_id"], sys_a, sys_b, d["judge_label"])
        grouped.setdefault(key, {})[d["order"]] = d.get("parsed")
    if not grouped:
        return None
    return compute_pair_analysis(grouped, "A_wealthnexus", "B2_rag_blind")


def format_wilson_ci_pct(ci: tuple[float, float]) -> str:
    return f"[{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]"


def render_markdown(analysis: dict) -> str:
    lines = ["# Experiment 4a: RAG similarity-threshold gating -- analysis", ""]

    lines.append("## Run summary")
    lines.append("")
    lines.append(f"- Run dir: `{analysis['run_dir']}`")
    lines.append(f"- Baseline (A vs B2 original): "
                 f"{'loaded' if analysis.get('baseline_ab2') else 'not provided'}")
    lines.append("")

    lines.append("## Gating rates per system")
    lines.append("")
    lines.append("| system | sessions | gated (n) | gated (%) | mean top-hit sim. |")
    lines.append("|---|---|---|---|---|")
    for sys_label, e in analysis["gating"]["per_system"].items():
        lines.append(f"| {sys_label} | {e['n_sessions']} | {e['n_gated']} | "
                     f"{e['gated_frac']*100:.1f}% | {e['top_score_mean']:.3f} |")
    lines.append("")

    lines.append("## Pairwise confirmed-win results")
    lines.append("")
    lines.append("Rates and Wilson 95% CIs are computed over available verdicts (i.e., "
                 "verdicts where both order permutations were successfully judged). "
                 "``Unavail.`` counts verdicts dropped because at least one of the two "
                 "order-permutation judgments failed to parse. `p` is the two-sided exact "
                 "binomial p-value testing H0: p(A wins) = p(B wins) over decided verdicts "
                 "(ties excluded).")
    lines.append("")
    lines.append("| Pair (A vs B) | A/B/Tie | Unavail. | A rate (of avail.) | 95% CI | p (decided) |")
    lines.append("|---|---|---|---|---|---|")
    for p in analysis["pairs"]:
        row = (
            f"| {p['sys_a']} vs {p['sys_b']} "
            f"| {p['confirmed_wins'][p['sys_a']]}/"
            f"{p['confirmed_wins'][p['sys_b']]}/"
            f"{p['confirmed_wins']['tie']} "
            f"| {p.get('n_unavailable', 0)} "
            f"| {p['a_rate_of_total']*100:.1f}% "
            f"| {format_wilson_ci_pct(p['wilson_ci_a_of_total'])} "
            f"| {p['sign_test_pvalue_decided']:.3f} |"
        )
        lines.append(row)
    if analysis.get("baseline_ab2"):
        b = analysis["baseline_ab2"]
        lines.append(
            f"| **baseline** {b['sys_a']} vs {b['sys_b']} (original Phase B run) "
            f"| {b['confirmed_wins'][b['sys_a']]}/"
            f"{b['confirmed_wins'][b['sys_b']]}/"
            f"{b['confirmed_wins']['tie']} "
            f"| {b.get('n_unavailable', 0)} "
            f"| {b['a_rate_of_total']*100:.1f}% "
            f"| {format_wilson_ci_pct(b['wilson_ci_a_of_total'])} "
            f"| {b['sign_test_pvalue_decided']:.3f} |"
        )
    lines.append("")

    lines.append("## Per-criterion winner distribution")
    lines.append("")
    for p in analysis["pairs"]:
        lines.append(f"### {p['sys_a']} vs {p['sys_b']}")
        lines.append("")
        lines.append("| Criterion | A wins | B wins | Tie |")
        lines.append("|---|---|---|---|")
        for c, dist in p["per_criterion"].items():
            lines.append(f"| {c} | {dist[p['sys_a']]} | {dist[p['sys_b']]} | {dist['tie']} |")
        lines.append("")

    lines.append("## Interpretation guide")
    lines.append("")
    lines.append("For each candidate manuscript story:")
    lines.append("")
    lines.append("- **Gate closes the A-B2 gap**: A_gated_XX vs B2 should be closer to 50/50 "
                 "or A-favouring, compared to the baseline A vs B2 (36-session pre-experiment).")
    lines.append("- **Gate does not hurt vs full RAG**: A_gated_XX vs A_wealthnexus should be "
                 "close to 50/50 (many ties) or A_gated-favouring; a strong A_wealthnexus victory "
                 "would suggest gating discards useful context.")
    lines.append("- **Dose-response**: as the threshold rises 0.40 -> 0.45, gating rate rises "
                 "and, if the mechanism is retrieval-quality-driven, the A_gated_XX vs B2 rate "
                 "should shift toward B2 (equivalent to disabling RAG).")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="Path to exp4a run directory")
    parser.add_argument("--baseline-run", default=None,
                        help="Optional: path to original Phase B pairwise run "
                             "(e.g. 20260508_065754_full_pw) to include the pre-experiment "
                             "A vs B2 baseline in the report.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        logger.error("Run directory not found: %s", run_dir)
        return 1

    gating = load_gating_stats(run_dir)
    pw = load_pairwise(run_dir)
    pw_grouped = pw["grouped"]

    # Enumerate the pairs actually present in the data (in case future runs add more)
    pair_set = set()
    for (sess_id, sys_a, sys_b, judge), _ in pw_grouped.items():
        pair_set.add((sys_a, sys_b))

    pair_analyses = [compute_pair_analysis(pw_grouped, a, b) for (a, b) in sorted(pair_set)]

    baseline_ab2 = None
    if args.baseline_run:
        baseline_ab2 = load_baseline_ab2(Path(args.baseline_run).resolve())

    analysis = {
        "run_dir": str(run_dir),
        "gating": gating,
        "pairs": pair_analyses,
        "baseline_ab2": baseline_ab2,
    }

    (run_dir / "exp4a_analysis.json").write_text(
        json.dumps(analysis, indent=2, default=str), encoding="utf-8"
    )
    md = render_markdown(analysis)
    (run_dir / "exp4a_analysis.md").write_text(md, encoding="utf-8")
    logger.info("Wrote %s and %s", run_dir / "exp4a_analysis.json", run_dir / "exp4a_analysis.md")

    print("\n=== ANALYSIS REPORT ===")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
