"""Aggregate Phase B results into a single publication-ready summary JSON.

Consumes four run directories:
    --abs-run         : results/runs/<dir> from run_recommendation_judge.py --full
    --pw-run          : results/runs/<dir> from run_pairwise_baseline.py --full
    --postfilter-run  : same path as --abs-run; the postfilter summary lives there
    --calib-dir       : results/calibration/<dir> with agreement_report.json

Writes phase_b_summary.json into the absolute-scoring run dir. This single artifact is
the source of truth for the manuscript draft (draft_5_5.md placeholders) and the HTML
scaffold (docs/llm_quality_evaluation_results.html).

Usage:
    python tests/eval/llm_quality/aggregate_phase_b_results.py \\
        --abs-run tests/eval/llm_quality/results/runs/<abs_dir> \\
        --pw-run tests/eval/llm_quality/results/runs/<pw_dir> \\
        --postfilter-run tests/eval/llm_quality/results/runs/<abs_dir> \\
        --calib-dir tests/eval/llm_quality/results/calibration/<calib_dir>

Any of the four can be omitted with --skip-<x> to aggregate partial state.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger("aggregate")

CRITERIA = [
    "C1_personalization",
    "C2_risk_alignment",
    "C3_factual_grounding",
    "C4_actionability",
    "C5_diversification",
    "C6_safety_compliance",
]


def aggregate_absolute(run_dir: Path) -> dict:
    """Pull per-criterion means from absolute-scoring run dir."""
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {"present": False, "reason": "no summary.json"}
    raw = json.loads(summary_path.read_text(encoding="utf-8"))

    judges = list(raw.get("mean_per_criterion_per_judge", {}).keys())
    overall = raw.get("mean_per_criterion_overall", {})
    per_judge = raw.get("mean_per_criterion_per_judge", {})

    cross_judge_gap = {}
    if len(judges) == 2:
        a, b = judges
        for c in CRITERIA:
            va = per_judge.get(a, {}).get(c)
            vb = per_judge.get(b, {}).get(c)
            if va is not None and vb is not None:
                cross_judge_gap[c] = round(abs(va - vb), 3)

    composite_overall = (
        round(sum(v for v in overall.values() if v is not None) / 6, 3)
        if all(overall.get(c) is not None for c in CRITERIA) else None
    )

    return {
        "present": True,
        "run_dir": str(run_dir),
        "n_sessions": raw.get("n_sessions"),
        "n_judgments_total": raw.get("n_judgments_total"),
        "n_parse_failures": raw.get("n_parse_failures"),
        "n_errors": raw.get("n_errors"),
        "n_critical_failures": raw.get("n_critical_failures"),
        "judges": judges,
        "mean_per_criterion_overall": overall,
        "mean_per_criterion_per_judge": per_judge,
        "cross_judge_gap_per_criterion": cross_judge_gap,
        "composite_mean_overall": composite_overall,
    }


def aggregate_pairwise(run_dir: Path) -> dict:
    """Pull per-pair confirmed wins and per-criterion winner distributions."""
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {"present": False, "reason": "no summary.json"}
    raw = json.loads(summary_path.read_text(encoding="utf-8"))

    confirmed = raw.get("confirmed_wins_per_pair", {})
    criterion = raw.get("criterion_winners_per_pair", {})

    # Compute win rates
    win_rates = {}
    for pair, counts in confirmed.items():
        total = sum(counts.values())
        if total == 0:
            continue
        # Pair label like "A_wealthnexus__VS__B1_profile_blind"
        try:
            sys_a, sys_b = pair.split("__VS__")
        except ValueError:
            continue
        a_wins = counts.get(sys_a, 0)
        b_wins = counts.get(sys_b, 0)
        ties = counts.get("tie", 0)
        win_rates[pair] = {
            "sys_a": sys_a,
            "sys_b": sys_b,
            "a_wins": a_wins,
            "b_wins": b_wins,
            "ties": ties,
            "total": total,
            "a_win_rate": round(a_wins / total, 3),
            "b_win_rate": round(b_wins / total, 3),
            "tie_rate": round(ties / total, 3),
        }

    return {
        "present": True,
        "run_dir": str(run_dir),
        "n_pairwise_files": raw.get("n_pairwise_files"),
        "confirmed_wins_per_pair": confirmed,
        "win_rates_per_pair": win_rates,
        "criterion_winners_per_pair": criterion,
    }


def aggregate_postfilter(run_dir: Path) -> dict:
    """Pull pre/post-filter rates from postfilter_summary.json."""
    summary_path = run_dir / "postfilter_summary.json"
    if not summary_path.exists():
        return {"present": False, "reason": "no postfilter_summary.json"}
    raw = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "present": True,
        "run_dir": str(run_dir),
        "n_recs_pre_filter": raw.get("n_recs_pre_filter"),
        "n_recs_post_filter": raw.get("n_recs_post_filter"),
        "n_recs_dropped_by_filter": raw.get("n_recs_dropped_by_filter"),
        "structural_violation_rate_pre": raw.get("structural_violation_rate_pre"),
        "structural_violation_rate_post": raw.get("structural_violation_rate_post"),
        "content_overriding_rate_pre": raw.get("content_overriding_rate_pre"),
        "content_overriding_rate_post": raw.get("content_overriding_rate_post"),
        "raw_counts": raw.get("raw_counts"),
        "dropped_examples_sample": raw.get("dropped_examples_sample", [])[:10],
    }


def aggregate_calibration(calib_dir: Path) -> dict:
    """Pull alpha + Pearson r from agreement_report.json."""
    rep_path = calib_dir / "agreement_report.json"
    if not rep_path.exists():
        return {"present": False, "reason": "no agreement_report.json"}
    raw = json.loads(rep_path.read_text(encoding="utf-8"))
    return {
        "present": True,
        "calib_dir": str(calib_dir),
        "n_recs": raw.get("n_recs"),
        "n_annotators": raw.get("n_annotators"),
        "annotators": raw.get("annotators"),
        "krippendorff_alpha_per_criterion": raw.get("krippendorff_alpha_interval_per_criterion"),
        "judge_human_correlation_per_criterion": raw.get("judge_human_correlation_per_criterion"),
        "composite_pearson_r": raw.get("composite_pearson_r"),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--abs-run", help="Absolute-scoring run dir")
    p.add_argument("--pw-run", help="Pairwise run dir")
    p.add_argument("--postfilter-run", help="Run dir containing postfilter_summary.json (typically same as --abs-run)")
    p.add_argument("--calib-dir", help="Human calibration dir with agreement_report.json")
    p.add_argument("--out", help="Output path; defaults to <abs-run>/phase_b_summary.json")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()

    out: dict = {"sections": {}}

    if args.abs_run:
        out["sections"]["absolute"] = aggregate_absolute(Path(args.abs_run).resolve())
    else:
        out["sections"]["absolute"] = {"present": False, "reason": "not provided"}

    if args.pw_run:
        out["sections"]["pairwise"] = aggregate_pairwise(Path(args.pw_run).resolve())
    else:
        out["sections"]["pairwise"] = {"present": False, "reason": "not provided"}

    if args.postfilter_run:
        out["sections"]["postfilter"] = aggregate_postfilter(Path(args.postfilter_run).resolve())
    else:
        out["sections"]["postfilter"] = {"present": False, "reason": "not provided"}

    if args.calib_dir:
        out["sections"]["calibration"] = aggregate_calibration(Path(args.calib_dir).resolve())
    else:
        out["sections"]["calibration"] = {"present": False, "reason": "not provided"}

    # Headline numbers — pre-extracted for easy templating into manuscript / HTML
    headline: dict = {}
    abs_data = out["sections"]["absolute"]
    if abs_data.get("present"):
        headline["composite_mean"] = abs_data.get("composite_mean_overall")
        headline["mean_per_criterion"] = abs_data.get("mean_per_criterion_overall")
        headline["cross_judge_max_gap_criterion"] = None
        gaps = abs_data.get("cross_judge_gap_per_criterion") or {}
        if gaps:
            top = max(gaps.items(), key=lambda kv: kv[1])
            headline["cross_judge_max_gap_criterion"] = {"criterion": top[0], "gap": top[1]}
        headline["critical_failure_rate"] = (
            (abs_data.get("n_critical_failures") or 0) / (abs_data.get("n_judgments_total") or 1)
        ) if abs_data.get("n_judgments_total") else None

    pw_data = out["sections"]["pairwise"]
    if pw_data.get("present"):
        headline["pairwise_win_rates"] = pw_data.get("win_rates_per_pair")

    pf_data = out["sections"]["postfilter"]
    if pf_data.get("present"):
        headline["structural_violation_rate_pre"] = pf_data.get("structural_violation_rate_pre")
        headline["structural_violation_rate_post"] = pf_data.get("structural_violation_rate_post")
        headline["content_overriding_rate_pre"] = pf_data.get("content_overriding_rate_pre")
        headline["content_overriding_rate_post"] = pf_data.get("content_overriding_rate_post")

    cal_data = out["sections"]["calibration"]
    if cal_data.get("present"):
        headline["krippendorff_alpha"] = cal_data.get("krippendorff_alpha_per_criterion")
        headline["composite_pearson_r"] = cal_data.get("composite_pearson_r")

    out["headline"] = headline

    if args.out:
        out_path = Path(args.out).resolve()
    elif args.abs_run:
        out_path = Path(args.abs_run).resolve() / "phase_b_summary.json"
    else:
        logger.error("--out is required when --abs-run is not provided")
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_path)

    print("\n=== HEADLINE NUMBERS ===")
    print(json.dumps(headline, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
