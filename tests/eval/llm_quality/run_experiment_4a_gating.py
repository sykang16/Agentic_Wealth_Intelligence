"""Experiment 4a: RAG similarity-threshold gating.

Tests whether a similarity-threshold gate on the RAG context mitigates the
context-overriding effect observed in the Phase B pairwise study (A vs B2:
RAG-blind wins 26 vs A's 15 on 72 confirmed comparisons).

Systems compared in this experiment:
    A_wealthnexus     : full pipeline (profile + portfolio + RAG). Reference.
    A_gated_040       : same as A, but RAG context dropped when top_hit similarity < 0.40.
    A_gated_045       : same as A, but RAG context dropped when top_hit similarity < 0.45.
    B2_rag_blind      : RAG disabled entirely. Baseline for the A vs B2 asymmetry.

Pairwise comparisons (72 confirmed verdicts each = 36 sessions x 2 judges):
    A_gated_040 vs A_wealthnexus   -- does gating help vs full pipeline?
    A_gated_045 vs A_wealthnexus   -- more aggressive gating?
    A_gated_040 vs B2_rag_blind    -- does gating close the A-vs-B2 gap?
    A_gated_045 vs B2_rag_blind    -- ditto at higher gating rate.

Cost estimate: 144 generation calls (~$3-5) + 576 judge calls (~$20-25).

Resume-safe: re-running with the same --out reuses existing sessions/judgments.

Usage:
    python tests/eval/llm_quality/run_experiment_4a_gating.py --full \\
        --out tests/eval/llm_quality/results/runs/exp4a_gating_v1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend" / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from backend.src.common.llm_client import LLMClient, LLMProvider
from backend.src.recommendation.engine.generator import RecommendationGenerator
from backend.src.recommendation.engine.schemas import Recommendation

from tests.eval.llm_quality.run_recommendation_judge import (  # noqa: E402
    CRITERIA,
    DEFAULT_JUDGE_ANTHROPIC,
    DEFAULT_JUDGE_OPENAI,
    DEFAULT_SYSTEM_MODEL,
    PERSONAS_PATH,
    PILOT_PERSONAS,
    PILOT_QUERIES,
    PILOT_SEEDS,
    QUERY_TYPES,
    RESULTS_ROOT,
    RUBRIC_PATH,
    build_aggregated_context,
    build_portfolio_context,
    build_profile_context,
    hash_file,
    session_id,
)

from tests.eval.llm_quality.run_pairwise_baseline import (  # noqa: E402
    JUDGE_MAX_TOKENS,
    JUDGE_TEMPERATURE,
    _resolve_pair_winner,
    call_pairwise_judge,
    parse_pairwise_response,
)

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

THRESHOLDS = {
    "A_gated_040": 0.40,
    "A_gated_045": 0.45,
}

SYSTEMS = ["A_wealthnexus", "A_gated_040", "A_gated_045", "B2_rag_blind"]

PAIRS = [
    ("A_wealthnexus", "B2_rag_blind"),   # fresh baseline under identical retrieval/model state
    ("A_gated_040", "A_wealthnexus"),
    ("A_gated_045", "A_wealthnexus"),
    ("A_gated_040", "B2_rag_blind"),
    ("A_gated_045", "B2_rag_blind"),
]

GEN_TEMPERATURE = 0.4
GEN_MAX_TOKENS = 4096

logger = logging.getLogger("experiment_4a_gating")


# ---------------------------------------------------------------------------
# Generation dispatcher
# ---------------------------------------------------------------------------

def generate_for_system(
    system_label: str,
    persona: dict,
    query: str,
    generator: RecommendationGenerator,
    rag_initializer,
    max_recs: int,
    dry_run: bool,
) -> tuple[list[Recommendation], dict]:
    """Generate recommendations for one of the four systems.

    Returns (recommendations, rag_diagnostics). The rag_diagnostics dict is
    populated with top-hit similarity and gating flag when RAG is consulted.
    """
    rag_diag: dict = {"top_score": 0.0, "num_hits": 0, "gated": False, "threshold": None}

    if dry_run:
        return [], rag_diag

    if system_label == "A_wealthnexus":
        ctx = build_aggregated_context(persona, query, rag_initializer,
                                       similarity_threshold=None,
                                       rag_diagnostics=rag_diag)
    elif system_label == "B2_rag_blind":
        ctx = build_aggregated_context(persona, query, rag_initializer=None,
                                       rag_diagnostics=rag_diag)
    elif system_label in THRESHOLDS:
        ctx = build_aggregated_context(persona, query, rag_initializer,
                                       similarity_threshold=THRESHOLDS[system_label],
                                       rag_diagnostics=rag_diag)
    else:
        raise ValueError(f"Unknown system: {system_label}")

    recs = generator.generate(context=ctx, query=query, max_recommendations=max_recs)
    return recs, rag_diag


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_pairwise_summary(run_dir: Path) -> dict:
    """Compute per-pair confirmed wins + per-criterion winner counts."""
    pw_dir = run_dir / "pairwise"
    files = list(pw_dir.glob("*.json"))

    by_session_pair_judge: dict[tuple[str, str, str, str], dict[int, dict | None]] = {}
    for fp in files:
        data = json.loads(fp.read_text(encoding="utf-8"))
        key = (data["session_id"], data["sys_a"], data["sys_b"], data["judge_label"])
        order = data["order"]
        by_session_pair_judge.setdefault(key, {})[order] = data.get("parsed")

    confirmed_wins: dict[str, dict[str, int]] = {}
    criterion_wins: dict[str, dict[str, dict[str, int]]] = {}

    for (sess_id, sys_a, sys_b, judge), orders in by_session_pair_judge.items():
        pair_label = f"{sys_a}__VS__{sys_b}"
        confirmed_wins.setdefault(pair_label, {sys_a: 0, sys_b: 0, "tie": 0})
        criterion_wins.setdefault(pair_label, {})

        winner = _resolve_pair_winner(orders.get(1), orders.get(2), sys_a, sys_b)
        confirmed_wins[pair_label][winner] += 1

        for crit in CRITERIA:
            criterion_wins[pair_label].setdefault(crit, {sys_a: 0, sys_b: 0, "tie": 0})
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
                criterion_wins[pair_label][crit][decoded] += 1

    return {
        "n_pairwise_files": len(files),
        "confirmed_wins_per_pair": confirmed_wins,
        "criterion_winners_per_pair": criterion_wins,
    }


def aggregate_gating_stats(run_dir: Path) -> dict:
    """Summarize how often the gate fired, per system."""
    sessions_dir = run_dir / "sessions"
    per_system: dict[str, dict] = {}
    for fp in sessions_dir.glob("*.json"):
        data = json.loads(fp.read_text(encoding="utf-8"))
        sys_label = data.get("system")
        if not sys_label:
            continue
        entry = per_system.setdefault(sys_label, {
            "n_sessions": 0, "n_gated": 0, "top_scores": [],
        })
        entry["n_sessions"] += 1
        rag = data.get("rag_diagnostics") or {}
        if rag.get("gated"):
            entry["n_gated"] += 1
        if rag.get("top_score") is not None:
            entry["top_scores"].append(float(rag["top_score"]))

    for sys_label, e in per_system.items():
        scores = e.pop("top_scores")
        e["gated_frac"] = e["n_gated"] / e["n_sessions"] if e["n_sessions"] else 0.0
        e["top_score_mean"] = sum(scores) / len(scores) if scores else 0.0
        e["top_score_min"] = min(scores) if scores else 0.0
        e["top_score_max"] = max(scores) if scores else 0.0
    return per_system


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--full", action="store_true")

    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--max-recs", type=int, default=4)
    p.add_argument("--system-model", default=DEFAULT_SYSTEM_MODEL)
    p.add_argument("--judge-anthropic", default=DEFAULT_JUDGE_ANTHROPIC)
    p.add_argument("--judge-openai", default=DEFAULT_JUDGE_OPENAI)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out", default=None,
                   help="Existing run dir to resume, or new name (relative to results/runs).")
    p.add_argument("--skip-judging", action="store_true",
                   help="Only generate recommendations; do not run judges yet.")
    return p.parse_args()


def init_run_dir(args) -> Path:
    label = "exp4a_pilot" if args.pilot else "exp4a_full"
    if args.out:
        out_path = Path(args.out)
        if out_path.is_absolute():
            run_dir = out_path
        else:
            # If the arg already sits under RESULTS_ROOT (e.g. copy-pasted full
            # path), reuse it. Otherwise treat as a folder name under RESULTS_ROOT.
            candidate = (Path.cwd() / out_path).resolve()
            try:
                candidate.relative_to(RESULTS_ROOT.resolve())
                run_dir = candidate
            except ValueError:
                run_dir = RESULTS_ROOT / out_path.name
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = RESULTS_ROOT / f"{ts}_{label}"
    (run_dir / "sessions").mkdir(parents=True, exist_ok=True)
    (run_dir / "pairwise").mkdir(parents=True, exist_ok=True)

    config = {
        "experiment": "4a_rag_similarity_gating",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": label,
        "system_model": args.system_model,
        "judge_anthropic": args.judge_anthropic,
        "judge_openai": args.judge_openai,
        "seeds": args.seeds,
        "max_recs": args.max_recs,
        "thresholds": THRESHOLDS,
        "systems": SYSTEMS,
        "pairs": [list(p) for p in PAIRS],
        "gen_temperature": GEN_TEMPERATURE,
        "judge_temperature": JUDGE_TEMPERATURE,
        "dry_run": args.dry_run,
        "personas_sha": hash_file(PERSONAS_PATH),
        "rubric_sha": hash_file(RUBRIC_PATH),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return run_dir


def build_session_plan(args, personas: list[dict]) -> list[dict]:
    if args.pilot:
        persona_subset = [p for p in personas if p["persona_id"] in PILOT_PERSONAS]
        query_subset = PILOT_QUERIES
        seeds = PILOT_SEEDS
    else:
        persona_subset = personas
        query_subset = QUERY_TYPES
        seeds = list(range(1, args.seeds + 1))

    plan = []
    for persona in persona_subset:
        for qtype in query_subset:
            for seed in seeds:
                plan.append({
                    "session_id": session_id(persona["persona_id"], qtype, seed),
                    "persona": persona,
                    "query_type": qtype,
                    "query": persona["queries"][qtype],
                    "seed": seed,
                })
    return plan


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    personas = json.loads(PERSONAS_PATH.read_text(encoding="utf-8"))["personas"]
    rubric_text = RUBRIC_PATH.read_text(encoding="utf-8")

    run_dir = init_run_dir(args)
    logger.info("Run directory: %s", run_dir)

    if args.dry_run:
        sys_client = None
        opus_client = None
        gpt_client = None
        rag = None
    else:
        sys_client = LLMClient(provider=LLMProvider.ANTHROPIC, model=args.system_model)
        opus_client = LLMClient(provider=LLMProvider.ANTHROPIC, model=args.judge_anthropic)
        gpt_client = LLMClient(provider=LLMProvider.OPENAI, model=args.judge_openai)
        from backend.src.recommendation.rag.initializer import RAGInitializer
        rag = RAGInitializer()
        if not rag.is_initialized():
            logger.error("RAG index is empty. Run build_rag_index.py first.")
            return 1

    generator = RecommendationGenerator(llm_client=sys_client) if sys_client is not None else None
    judges = [("opus_anthropic", opus_client), ("gpt4o_openai", gpt_client)]

    plan = build_session_plan(args, personas)
    logger.info("Sessions to run: %d, systems: %s", len(plan), SYSTEMS)

    # ---- Generation phase ----
    for sess in plan:
        system_recs: dict[str, list[Recommendation]] = {}
        for sys_label in SYSTEMS:
            sess_path = run_dir / "sessions" / f"{sess['session_id']}__{sys_label}.json"
            if sess_path.exists():
                stored = json.loads(sess_path.read_text(encoding="utf-8"))
                system_recs[sys_label] = [Recommendation.model_validate(r) for r in stored["recommendations"]]
                continue

            logger.info("Generating: %s [%s]", sess["session_id"], sys_label)
            try:
                recs, rag_diag = generate_for_system(
                    system_label=sys_label,
                    persona=sess["persona"],
                    query=sess["query"],
                    generator=generator,
                    rag_initializer=rag,
                    max_recs=args.max_recs,
                    dry_run=args.dry_run,
                )
            except Exception as e:
                logger.error("Generation failed for %s: %s", sys_label, e)
                recs = []
                rag_diag = {}
            system_recs[sys_label] = recs

            sess_record = {
                "session_id": sess["session_id"],
                "persona_id": sess["persona"]["persona_id"],
                "query_type": sess["query_type"],
                "query": sess["query"],
                "seed": sess["seed"],
                "system": sys_label,
                "rag_diagnostics": rag_diag,
                "rec_count": len(recs),
                "recommendations": [r.model_dump(mode="json") for r in recs],
            }
            sess_path.write_text(json.dumps(sess_record, indent=2, default=str), encoding="utf-8")

        if args.skip_judging:
            continue

        # ---- Judging phase (per session) ----
        for sys_a, sys_b in PAIRS:
            recs_a = system_recs.get(sys_a, [])
            recs_b = system_recs.get(sys_b, [])
            for order in (1, 2):
                if order == 1:
                    recs_left, recs_right = recs_a, recs_b
                else:
                    recs_left, recs_right = recs_b, recs_a

                for judge_label, judge_client in judges:
                    out_path = (
                        run_dir / "pairwise"
                        / f"{sess['session_id']}__{sys_a}__VS__{sys_b}__order{order}__{judge_label}.json"
                    )
                    if out_path.exists():
                        continue

                    judgment = call_pairwise_judge(
                        judge_client=judge_client,
                        judge_label=judge_label,
                        persona=sess["persona"],
                        query=sess["query"],
                        recs_A=recs_left,
                        recs_B=recs_right,
                        rubric_text=rubric_text,
                        dry_run=args.dry_run,
                    )

                    out_record = {
                        "session_id": sess["session_id"],
                        "persona_id": sess["persona"]["persona_id"],
                        "query_type": sess["query_type"],
                        "seed": sess["seed"],
                        "sys_a": sys_a,
                        "sys_b": sys_b,
                        "order": order,
                        **judgment,
                    }
                    out_path.write_text(json.dumps(out_record, indent=2), encoding="utf-8")

    # ---- Post-run aggregation ----
    gating_stats = aggregate_gating_stats(run_dir)
    (run_dir / "gating_stats.json").write_text(json.dumps(gating_stats, indent=2), encoding="utf-8")

    if not args.skip_judging:
        pw_summary = aggregate_pairwise_summary(run_dir)
        (run_dir / "pairwise_summary.json").write_text(json.dumps(pw_summary, indent=2), encoding="utf-8")
        logger.info("Confirmed wins per pair:")
        for pair, wins in pw_summary["confirmed_wins_per_pair"].items():
            logger.info("  %s : %s", pair, wins)

    logger.info("Gating stats:")
    for sys_label, e in gating_stats.items():
        logger.info("  %s : gated %d/%d (%.1f%%), mean top=%.3f",
                    sys_label, e["n_gated"], e["n_sessions"],
                    100 * e["gated_frac"], e["top_score_mean"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
