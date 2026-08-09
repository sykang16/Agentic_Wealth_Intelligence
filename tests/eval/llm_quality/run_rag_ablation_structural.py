"""RAG-ablation of STRUCTURAL fiduciary violations (reviewer M2 causal test).

Question: do the structural risk violations the LLM attempts (rec.risk_level > user
tolerance) depend on the retrieval context? If context overriding is real at the
structural level, removing RAG (B2) should REDUCE structural violations relative to
the full pipeline (A). If violations are unchanged, they are RAG-independent LLM risk
misclassification, and the "context overriding" label does not apply to them.

Reads the pairwise run directory (which stores per-config generated recommendations
for A_wealthnexus, B2_rag_blind, ...) and counts structural violations per config from
the saved risk_level tags. Zero LLM calls.

Usage:
  python -m tests.eval.llm_quality.run_rag_ablation_structural \
      --run-dir tests/eval/llm_quality/results/runs/20260508_065754_full_pw
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
PERSONAS_PATH = SCRIPT_DIR / "recommendation_personas.json"

_RISK_RANK = {"low": 0, "moderate": 1, "high": 2}
_TOL_RANK = {"conservative": 0, "moderate": 1, "aggressive": 2}

CONFIGS = {
    "A_wealthnexus": "A (full, RAG on)",
    "B2_rag_blind": "B2 (RAG blind)",
}


def is_structural_violation(risk_level: str | None, tol: str | None) -> bool:
    if risk_level is None or tol is None:
        return False
    rl = _RISK_RANK.get(str(risk_level).lower())
    tr = _TOL_RANK.get(str(tol).lower())
    if rl is None or tr is None:
        return False
    return rl > tr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    sessions_dir = run_dir / "sessions"

    personas = json.loads(PERSONAS_PATH.read_text(encoding="utf-8"))["personas"]
    tol_by_id = {p["persona_id"]: p["investment_profile"]["risk_tolerance"] for p in personas}

    per_config = {c: {"n_recs": 0, "n_viol": 0, "n_sessions": 0,
                      "viol_by_persona": defaultdict(int), "examples": []}
                  for c in CONFIGS}

    for sp in sorted(sessions_dir.glob("*.json")):
        suffix = None
        for c in CONFIGS:
            if sp.stem.endswith("__" + c):
                suffix = c
                break
        if suffix is None:
            continue
        sess = json.loads(sp.read_text(encoding="utf-8"))
        pid = sess["persona_id"]
        tol = tol_by_id.get(pid)
        agg = per_config[suffix]
        agg["n_sessions"] += 1
        for rec in sess.get("recommendations", []):
            agg["n_recs"] += 1
            rl = rec.get("risk_level")
            if is_structural_violation(rl, tol):
                agg["n_viol"] += 1
                agg["viol_by_persona"][pid] += 1
                agg["examples"].append({
                    "session": sess.get("session_id", sp.stem),
                    "persona": pid, "tol": tol,
                    "title": rec.get("title", "")[:60], "risk_level": rl,
                })

    print("\n=== RAG-ABLATION: STRUCTURAL FIDUCIARY VIOLATIONS (pre-filter, raw LLM) ===")
    print(f"Run: {run_dir.name}\n")
    print(f"{'Config':22s} {'Sessions':>9s} {'Recs':>6s} {'Struct.Viol':>12s} {'Rate':>8s}")
    summary = {}
    for c, label in CONFIGS.items():
        a = per_config[c]
        rate = a["n_viol"] / a["n_recs"] if a["n_recs"] else 0.0
        print(f"{label:22s} {a['n_sessions']:>9d} {a['n_recs']:>6d} {a['n_viol']:>12d} {rate:>7.1%}")
        summary[c] = {"label": label, "n_sessions": a["n_sessions"], "n_recs": a["n_recs"],
                      "n_viol": a["n_viol"], "rate": rate,
                      "viol_by_persona": dict(a["viol_by_persona"]),
                      "examples": a["examples"]}

    print("\nViolations by persona:")
    for c, label in CONFIGS.items():
        print(f"  {label}: {dict(per_config[c]['viol_by_persona'])}")

    out = run_dir / "rag_ablation_structural_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
