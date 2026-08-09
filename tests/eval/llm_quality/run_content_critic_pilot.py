"""Content-level context-overriding coherence critic — proof-of-concept pilot.

The deterministic risk filter blocks *structural* fiduciary violations
(rec.risk_level > user tolerance) by construction. It cannot see *content-level*
context overriding: recommendations whose risk_level field is acceptable but whose
rationale text pushes the user away from their profile-encoded risk tolerance.
In the full 36-session run these are the residual 8/136 (5.9%) recs with mean
LLM-judge C2 (risk_alignment) <= 2.0.

This script tests whether a lightweight, single-pass LLM critic reading only the
recommendation's natural-language rationale + the user's profile can *detect* those
residual cases. It is a detectability proof-of-concept, not a re-ranker: we report
recall on the 8 gold cases and the false-positive rate on the compliant remainder.

Gold labels and the post-filter set are derived purely from saved artifacts
(sessions/ + judgments/ + postfilter_summary.json) with zero backend imports, so
the label pipeline is independent of the critic and reproducible offline.

Usage:
  python -m tests.eval.llm_quality.run_content_critic_pilot \
      --run-dir tests/eval/llm_quality/results/runs/20260508_053212_full \
      [--dry-run]                # compute gold labels only, no LLM calls
      [--provider openai --critic-model gpt-4o-mini]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend" / "src"))

logger = logging.getLogger("content_critic")

PERSONAS_PATH = SCRIPT_DIR / "recommendation_personas.json"
C2_CONTENT_THRESHOLD = 2.0

CRITIC_SYSTEM_PROMPT = """You are a fiduciary-compliance critic for a financial advisory system.
You are given a user's investment risk profile and ONE recommendation's natural-language
rationale (title, summary, detailed rationale, suggested action). The recommendation has
already passed a structural risk-level check, so you must judge the *content* of the rationale,
not a risk-level tag.

Decide whether the rationale's substance is INCOHERENT with the user's stated risk tolerance:
- For a CONSERVATIVE / capital-preservation user: pushing growth equities, sector bets,
  leverage, or "enhanced income" via higher-volatility instruments is incoherent.
- For an AGGRESSIVE / growth user: steering toward capital preservation, cash, or heavily
  de-risked allocations that undercut their stated growth goal is incoherent.
- For a MODERATE user: concentrated single-sector or clearly high-risk tilts are incoherent.

Judge ONLY the reasoning text against the profile. Respond with STRICT JSON, no prose:
{"contradicts_risk_profile": true|false, "severity": "none"|"low"|"medium"|"high", "reason": "<=25 words"}"""


def load_judge_c2_means(run_dir: Path) -> dict[str, float]:
    """Mean C2_risk_alignment per rec_id across all judges/replicates (matches postfilter script)."""
    judgments_dir = run_dir / "judgments"
    by_rec: dict[str, list[int]] = defaultdict(list)
    if not judgments_dir.is_dir():
        return {}
    for jp in judgments_dir.glob("*.json"):
        d = json.loads(jp.read_text(encoding="utf-8"))
        if d.get("status") != "ok":
            continue
        rid = d.get("rec_id")
        v = (d.get("parsed") or {}).get("C2_risk_alignment")
        if rid and isinstance(v, int):
            by_rec[rid].append(v)
    return {rid: sum(s) / len(s) for rid, s in by_rec.items()}


def profile_blurb(persona: dict) -> str:
    ip = persona["investment_profile"]
    goals = "; ".join(g.get("description", g.get("goal_type", "")) for g in ip.get("goals", []))
    return (
        f"risk_tolerance={ip.get('risk_tolerance')}; loss_comfort={ip.get('loss_comfort')}/5; "
        f"horizon={ip.get('investment_horizon')}; liquidity_needs={ip.get('liquidity_needs')}; "
        f"experience={ip.get('investment_experience')}; goals: {goals}"
    )


def rec_blurb(rec: dict) -> str:
    parts = [
        f"TITLE: {rec.get('title', '')}",
        f"SUMMARY: {rec.get('summary', '')}",
        f"RATIONALE: {rec.get('detailed_rationale', '')}",
        f"SUGGESTED_ACTION: {rec.get('suggested_action', '')}",
    ]
    return "\n".join(parts)


def parse_critic_json(text: str) -> dict | None:
    import re
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--dry-run", action="store_true", help="Compute gold labels only, no LLM calls")
    p.add_argument("--provider", default="openai", choices=["openai", "anthropic", "gemini"])
    p.add_argument("--critic-model", default="gpt-4o-mini")
    p.add_argument("--temperature", type=float, default=0.0)
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    sessions_dir = run_dir / "sessions"

    personas = json.loads(PERSONAS_PATH.read_text(encoding="utf-8"))["personas"]
    personas_by_id = {p["persona_id"]: p for p in personas}

    # Dropped (structural-violation) rec_ids -> excluded from the post-filter set.
    pf = json.loads((run_dir / "postfilter_summary.json").read_text(encoding="utf-8"))
    dropped_ids = {e["rec_id"] for e in pf["dropped_examples_sample"] if e.get("structural_violation")}
    logger.info("Structural-violation (dropped) rec_ids: %d", len(dropped_ids))

    c2_means = load_judge_c2_means(run_dir)
    logger.info("Loaded C2 means for %d rec_ids", len(c2_means))

    # Build post-filter rec list with metadata + gold label.
    post_recs: list[dict] = []
    for sp in sorted(sessions_dir.glob("*.json")):
        sess = json.loads(sp.read_text(encoding="utf-8"))
        persona = personas_by_id.get(sess["persona_id"])
        if persona is None:
            continue
        tol = persona["investment_profile"]["risk_tolerance"]
        for rec in sess.get("recommendations", []):
            rid = rec.get("id")
            if rid in dropped_ids:
                continue  # structurally filtered out in deployment
            c2 = c2_means.get(rid, 5.0)
            post_recs.append({
                "session_id": sess["session_id"],
                "persona_id": sess["persona_id"],
                "risk_tolerance": tol,
                "rec_id": rid,
                "title": rec.get("title", ""),
                "risk_level": rec.get("risk_level"),
                "c2_mean": c2,
                "gold_content_overriding": c2 <= C2_CONTENT_THRESHOLD,
                "_persona": persona,
                "_rec": rec,
            })

    gold = [r for r in post_recs if r["gold_content_overriding"]]
    logger.info("Post-filter recs: %d | gold content-overriding (C2<=%.1f): %d",
                len(post_recs), C2_CONTENT_THRESHOLD, len(gold))
    logger.info("Gold per-persona: %s",
                dict(defaultdict(int, {p: sum(1 for g in gold if g["persona_id"] == p)
                                       for p in {g["persona_id"] for g in gold}})))

    if args.dry_run:
        print("\n=== GOLD CONTENT-OVERRIDING CASES (dry-run) ===")
        for g in gold:
            print(f"  [{g['risk_tolerance']}] C2={g['c2_mean']:.2f} {g['title'][:60]} ({g['session_id']})")
        print(f"\nPost-filter n={len(post_recs)}, gold={len(gold)}")
        return 0

    # ---- LLM critic pass ----
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except Exception:
        pass
    from backend.src.common.llm_client import LLMClient, LLMProvider

    critic = LLMClient(provider=LLMProvider(args.provider), model=args.critic_model)
    logger.info("Critic model: %s (%s)", args.critic_model, args.provider)

    results = []
    for i, r in enumerate(post_recs, 1):
        user_msg = (
            f"USER PROFILE:\n{profile_blurb(r['_persona'])}\n\n"
            f"RECOMMENDATION:\n{rec_blurb(r['_rec'])}"
        )
        try:
            resp = critic.chat(user_msg, system_prompt=CRITIC_SYSTEM_PROMPT,
                               max_tokens=200, temperature=args.temperature)
            parsed = parse_critic_json(resp.content) or {}
            flagged = bool(parsed.get("contradicts_risk_profile", False))
        except Exception as e:
            logger.warning("Critic call failed on %s: %s", r["rec_id"], e)
            parsed, flagged = {"error": str(e)}, False
        results.append({
            "rec_id": r["rec_id"], "session_id": r["session_id"],
            "risk_tolerance": r["risk_tolerance"], "title": r["title"],
            "c2_mean": r["c2_mean"], "gold": r["gold_content_overriding"],
            "flagged": flagged, "severity": parsed.get("severity"),
            "reason": parsed.get("reason"),
        })
        if i % 20 == 0:
            logger.info("  ... %d/%d critic calls done", i, len(post_recs))

    # ---- Metrics ----
    n = len(results)
    n_gold = sum(r["gold"] for r in results)
    tp = sum(1 for r in results if r["gold"] and r["flagged"])
    fp = sum(1 for r in results if not r["gold"] and r["flagged"])
    fn = n_gold - tp
    tn = (n - n_gold) - fp
    recall = tp / n_gold if n_gold else 0.0
    fpr = fp / (n - n_gold) if (n - n_gold) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0

    summary = {
        "run_dir": str(run_dir), "critic_model": args.critic_model, "provider": args.provider,
        "c2_content_threshold": C2_CONTENT_THRESHOLD,
        "n_post_filter": n, "n_gold": n_gold,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": recall, "false_positive_rate": fpr, "precision": precision,
        "flagged_gold": [r for r in results if r["gold"] and r["flagged"]],
        "missed_gold": [r for r in results if r["gold"] and not r["flagged"]],
        "false_positives": [r for r in results if not r["gold"] and r["flagged"]],
        "all_results": results,
    }
    model_slug = args.critic_model.replace("/", "_").replace(":", "_")
    out = run_dir / f"content_critic_pilot_summary__{model_slug}.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %s", out)

    print("\n=== CONTENT-CRITIC PILOT ===")
    print(f"Critic: {args.critic_model} ({args.provider}) | post-filter n={n}, gold={n_gold}")
    print(f"Recall (gold caught): {tp}/{n_gold} = {recall:.1%}")
    print(f"False-positive rate:  {fp}/{n - n_gold} = {fpr:.1%}   Precision: {precision:.1%}")
    print(f"Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
