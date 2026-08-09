"""Phase B — LLM-as-judge evaluation of recommendation text quality.

Pipeline per session:
  1. Build AggregatedContext from a persona JSON record (no DB / Plaid required).
  2. Optionally enrich with real ChromaDB RAG retrieval.
  3. Call RecommendationGenerator with the system-under-test LLM (default Sonnet 4.5).
  4. Score each returned Recommendation against the 6-criterion rubric using two
     independent judges (Anthropic Opus + OpenAI GPT-4o by default).
  5. Persist raw recommendations and per-judge judgments under
     tests/eval/llm_quality/results/runs/<timestamp>_<label>/.

Run modes:
  --pilot              : 3 personas x 1 query x 1 seed = 12-output sanity run (~24 judge calls)
  --full               : 6 personas x 3 queries x 2 seeds = 36 sessions (~864 judge calls at 3 replicates)
  --dry-run            : skip every LLM call; validate wiring, paths, and prompt assembly only

Resume: pass --out <existing_run_dir> to skip sessions/judgments whose files already exist.

Usage:
    python tests/eval/llm_quality/run_recommendation_judge.py --pilot --dry-run
    python tests/eval/llm_quality/run_recommendation_judge.py --pilot
    python tests/eval/llm_quality/run_recommendation_judge.py --full --judge-replicates 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path bootstrap — must precede backend.* imports
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
from backend.src.recommendation.engine.schemas import (
    AggregatedContext,
    ConfidenceLevel,
    Recommendation,
    RecommendationCategory,
    RecommendationPriority,
    RiskLevel,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PERSONAS_PATH = SCRIPT_DIR / "recommendation_personas.json"
RUBRIC_PATH = SCRIPT_DIR / "judge_rubric_rec.md"
RESULTS_ROOT = SCRIPT_DIR / "results" / "runs"

CRITERIA = [
    "C1_personalization",
    "C2_risk_alignment",
    "C3_factual_grounding",
    "C4_actionability",
    "C5_diversification",
    "C6_safety_compliance",
]

QUERY_TYPES = ["open_ended", "rebalance", "sector_specific"]

# Default model identifiers
# System-under-test defaults to the production deployed model so the eval is faithful
# to what users actually see. Override via --system-model if the deployment moves.
DEFAULT_SYSTEM_MODEL = "claude-sonnet-4-5"
DEFAULT_JUDGE_ANTHROPIC = "claude-opus-4-7"
DEFAULT_JUDGE_OPENAI = "gpt-4o"

# Pilot subset: representative across risk axis
PILOT_PERSONAS = ["P1_conservative_retiree", "P3_moderate_mid_career", "P5_aggressive_young_professional"]
PILOT_QUERIES = ["open_ended"]
PILOT_SEEDS = [1]

# Generator settings (match production defaults in generator.py)
GEN_TEMPERATURE = 0.4
GEN_MAX_TOKENS = 4096

# Judge settings — temperature 0 for max determinism by default
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 1024

logger = logging.getLogger("recommendation_judge")


# ---------------------------------------------------------------------------
# Persona -> AggregatedContext
# ---------------------------------------------------------------------------

def _format_money(value: float | int | Decimal) -> str:
    return f"${float(value):,.0f}"


def build_portfolio_context(persona: dict) -> str:
    """Render the persona's portfolio_summary as the markdown block the LLM expects."""
    p = persona["portfolio_summary"]
    lines = [
        "## Portfolio Snapshot",
        f"- Total Assets: {_format_money(p['total_assets'])}",
        f"- Total Liabilities: {_format_money(p['total_liabilities'])}",
        f"- Net Worth: {_format_money(p['total_assets'] - p['total_liabilities'])}",
        "",
        "### Allocation by Asset Type",
    ]
    for asset_type, pct in p["allocation_by_asset_type"].items():
        lines.append(f"- {asset_type}: {pct:.1f}%")
    lines.append("")
    lines.append("### Allocation by Sector")
    for sector, pct in p["allocation_by_sector"].items():
        lines.append(f"- {sector}: {pct:.1f}%")
    lines.append("")
    lines.append("### Holdings (tickers)")
    lines.append(", ".join(p["portfolio_tickers"]))
    return "\n".join(lines)


def build_profile_context(persona: dict) -> str:
    """Render the persona's investment_profile as markdown."""
    prof = persona["investment_profile"]
    user = persona["user"]
    lines = [
        "## Investment Profile",
        f"- Age: {user['age']}",
        f"- Occupation: {user['occupation']}",
        f"- Annual Income: {_format_money(user['annual_income'])}",
        f"- Monthly Expenses: {_format_money(user['monthly_expenses'])}",
        f"- Risk Tolerance: {prof['risk_tolerance']}",
        f"- Loss Comfort (1-10): {prof['loss_comfort']}",
        f"- Investment Horizon: {prof['investment_horizon']}",
        f"- Liquidity Needs: {prof['liquidity_needs']}",
        f"- Income Stability: {prof['income_stability']}",
        f"- Has Emergency Fund: {prof['has_emergency_fund']}",
        f"- Debt Level: {prof['debt_level']}",
        f"- Investment Experience: {prof['investment_experience']}",
    ]
    if prof.get("esg_preference") is not None:
        lines.append(f"- ESG Preference: {prof['esg_preference']}")
    if prof.get("goals"):
        lines.append("- Goals:")
        for g in prof["goals"]:
            lines.append(
                f"  - {g['goal_type']} ({g['priority']} priority, "
                f"target {_format_money(g['target_amount'])}): {g.get('description', '')}"
            )
    return "\n".join(lines)


def build_aggregated_context(
    persona: dict,
    query: str,
    rag_initializer=None,
    similarity_threshold: float | None = None,
    rag_diagnostics: dict | None = None,
) -> AggregatedContext:
    """Assemble an AggregatedContext from a persona record, optionally enriched by RAG.

    Args:
        similarity_threshold: If set and the RAG top-hit similarity is below this
            value, the RAG context is dropped (equivalent to RAG-blind for this
            query). Used by the A_gated system in Experiment 4a to test whether
            a similarity gate mitigates context overriding.
        rag_diagnostics: If a mutable dict is passed, this function populates it
            with retrieval telemetry: {"top_score": float, "num_hits": int,
            "gated": bool, "threshold": float | None}. Callers can persist these
            fields onto the session record for later analysis.
    """
    portfolio_block = build_portfolio_context(persona)
    profile_block = build_profile_context(persona)

    rag_text = ""
    sources_used = ["portfolio", "investment_profile"]
    top_score = 0.0
    num_hits = 0
    gated = False
    if rag_initializer is not None:
        try:
            results = rag_initializer.search(query, top_k=5)
            hits = results.get("results", []) or []
            num_hits = len(hits)
            if hits:
                top_score = float(hits[0].get("score", 0.0))
            if similarity_threshold is not None and top_score < similarity_threshold:
                # Gate: retrieval quality is below the threshold; treat as RAG-blind
                # for this query to test whether low-similarity contexts drive the
                # context-overriding effect.
                rag_text = ""
                gated = True
            else:
                rag_text = results.get("context", "") or ""
                if hits:
                    sources_used.append("rag_knowledge_base")
        except Exception as e:
            logger.warning("RAG retrieval failed (%s); continuing without RAG context", e)

    if rag_diagnostics is not None:
        rag_diagnostics.update({
            "top_score": top_score,
            "num_hits": num_hits,
            "gated": gated,
            "threshold": similarity_threshold,
        })

    return AggregatedContext(
        portfolio_context=portfolio_block,
        investment_profile_context=profile_block,
        rag_context=rag_text,
        live_market_context="",  # Skipped for reproducibility (see manuscript note)
        sentiment_context="",
        user_risk_tolerance=persona["investment_profile"]["risk_tolerance"],
        user_investment_horizon=persona["investment_profile"]["investment_horizon"],
        user_experience_level=persona["investment_profile"]["investment_experience"],
        current_sector_allocation=persona["portfolio_summary"]["allocation_by_sector"],
        current_asset_allocation=persona["portfolio_summary"]["allocation_by_asset_type"],
        portfolio_tickers=persona["portfolio_summary"]["portfolio_tickers"],
        excluded_sectors=[],
        data_sources_used=sources_used,
    )


# ---------------------------------------------------------------------------
# Judge prompt + parser
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert evaluator of automated financial-advisory output.
Rate a single recommendation against six criteria using the rubric below.
Be strict and consistent. Do not reward verbosity, fluency, or formatting beyond what the rubric specifies.
Return only valid JSON in the exact format requested at the end of the user message.

== RUBRIC ==
{rubric_text}
== END RUBRIC ==
"""

JUDGE_USER_PROMPT_TEMPLATE = """\
## User profile
{profile_block}

## Portfolio snapshot
{portfolio_block}

## User query
{query}

## Recommendation under review (system identity hidden)
{recommendation_block}

## Output format (JSON only, no prose)
{{
  "C1_personalization": <int 1-5>,
  "C2_risk_alignment": <int 1-5>,
  "C3_factual_grounding": <int 1-5>,
  "C4_actionability": <int 1-5>,
  "C5_diversification": <int 1-5>,
  "C6_safety_compliance": <int 1-5>,
  "rationale": {{
    "C1": "<one-sentence justification>",
    "C2": "<one-sentence justification>",
    "C3": "<one-sentence justification>",
    "C4": "<one-sentence justification>",
    "C5": "<one-sentence justification>",
    "C6": "<one-sentence justification>"
  }},
  "critical_failure": <true if C2==1 or C6==1, else false>
}}
"""


def render_recommendation_for_judge(rec: Recommendation) -> str:
    """Render a Recommendation as a flat block for the judge — no internal scores leaked."""
    parts = [
        f"category: {rec.category.value}",
        f"title: {rec.title}",
        f"summary: {rec.summary}",
        f"detailed_rationale: {rec.detailed_rationale}",
        f"tickers: {rec.tickers}",
        f"suggested_action: {rec.suggested_action}",
        f"suggested_allocation_pct: {rec.suggested_allocation_pct}",
        f"risk_level: {rec.risk_level.value}",
        f"expected_return_range: {rec.expected_return_range}",
        f"time_horizon: {rec.time_horizon}",
        f"confidence: {rec.confidence.value}",
        f"priority: {rec.priority.value}",
    ]
    return "\n".join(parts)


def parse_judge_response(text: str) -> dict | None:
    """Extract the judge JSON. Returns None if unparseable."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n", "", cleaned)
        cleaned = re.sub(r"\n```\s*$", "", cleaned)
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        return None
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    for crit in CRITERIA:
        v = parsed.get(crit)
        if not isinstance(v, int) or not (1 <= v <= 5):
            return None
    return parsed


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------

def session_id(persona_id: str, query_type: str, seed: int) -> str:
    return f"{persona_id}__{query_type}__seed{seed}"


def build_session_plan(args, personas: list[dict]) -> list[dict]:
    """Materialize the (persona, query_type, seed) sessions to run."""
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


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def init_run_dir(args) -> Path:
    label = "pilot" if args.pilot else "full"
    if args.out:
        run_dir = RESULTS_ROOT / args.out
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = RESULTS_ROOT / f"{ts}_{label}"
    (run_dir / "sessions").mkdir(parents=True, exist_ok=True)
    (run_dir / "judgments").mkdir(parents=True, exist_ok=True)

    config = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": label,
        "system_model": args.system_model,
        "judge_anthropic": args.judge_anthropic,
        "judge_openai": args.judge_openai,
        "seeds": args.seeds,
        "max_recs": args.max_recs,
        "judge_replicates": args.judge_replicates,
        "rag_disabled": args.rag_disabled,
        "dry_run": args.dry_run,
        "personas_sha": hash_file(PERSONAS_PATH),
        "rubric_sha": hash_file(RUBRIC_PATH),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return run_dir


# ---------------------------------------------------------------------------
# Recommendation generation per session
# ---------------------------------------------------------------------------

def run_session(
    sess: dict,
    generator: RecommendationGenerator | None,
    rag_initializer,
    max_recs: int,
    dry_run: bool,
) -> dict:
    """Build context, call generator, return session record (input + recs)."""
    context = build_aggregated_context(
        persona=sess["persona"],
        query=sess["query"],
        rag_initializer=rag_initializer,
    )

    if dry_run or generator is None:
        # Stub one Recommendation so downstream prompt assembly is exercised end-to-end.
        recs = [
            Recommendation(
                id=f"dryrun_{sess['session_id']}_0",
                category=RecommendationCategory.HOLD,
                title="[dry-run stub recommendation]",
                summary="Stub summary used to validate judge-prompt assembly without LLM calls.",
                detailed_rationale="Stub rationale.",
                tickers=[],
                suggested_action="Stub action.",
                risk_level=RiskLevel.MODERATE,
                confidence=ConfidenceLevel.MEDIUM,
                priority=RecommendationPriority.MEDIUM,
            )
        ]
    else:
        recs = generator.generate(context=context, query=sess["query"], max_recommendations=max_recs)

    return {
        "session_id": sess["session_id"],
        "persona_id": sess["persona"]["persona_id"],
        "query_type": sess["query_type"],
        "query": sess["query"],
        "seed": sess["seed"],
        "context": {
            "portfolio_context": context.portfolio_context,
            "investment_profile_context": context.investment_profile_context,
            "rag_context": context.rag_context,
            "data_sources_used": context.data_sources_used,
        },
        "recommendations": [r.model_dump(mode="json") for r in recs],
        "rec_count": len(recs),
    }


# ---------------------------------------------------------------------------
# Judging per recommendation
# ---------------------------------------------------------------------------

def call_judge(
    judge_client: LLMClient,
    judge_label: str,
    persona: dict,
    query: str,
    rec: Recommendation,
    rubric_text: str,
    replicate: int,
    dry_run: bool,
) -> dict:
    system_prompt = JUDGE_SYSTEM_PROMPT_TEMPLATE.format(rubric_text=rubric_text)
    user_prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
        profile_block=build_profile_context(persona),
        portfolio_block=build_portfolio_context(persona),
        query=query,
        recommendation_block=render_recommendation_for_judge(rec),
    )

    if dry_run:
        return {
            "judge_label": judge_label,
            "replicate": replicate,
            "status": "dry_run",
            "raw_output": None,
            "parsed": None,
        }

    try:
        resp = judge_client.chat(
            user_message=user_prompt,
            system_prompt=system_prompt,
            max_tokens=JUDGE_MAX_TOKENS,
            temperature=JUDGE_TEMPERATURE,
        )
        parsed = parse_judge_response(resp.content)
        return {
            "judge_label": judge_label,
            "judge_provider": resp.provider,
            "judge_model": resp.model,
            "replicate": replicate,
            "status": "ok" if parsed is not None else "parse_failed",
            "raw_output": resp.content,
            "parsed": parsed,
            "usage": resp.usage,
        }
    except Exception as e:
        logger.error("Judge call failed (%s, replicate=%d): %s", judge_label, replicate, e)
        return {
            "judge_label": judge_label,
            "replicate": replicate,
            "status": "error",
            "error": str(e),
            "raw_output": None,
            "parsed": None,
        }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_summary(run_dir: Path) -> dict:
    """Compute per-criterion means across all valid judgments."""
    judgments_dir = run_dir / "judgments"
    sessions_dir = run_dir / "sessions"

    by_criterion: dict[str, list[int]] = {c: [] for c in CRITERIA}
    by_judge_criterion: dict[str, dict[str, list[int]]] = {}
    critical_failures = 0
    total_judgments = 0
    parse_failures = 0
    errors = 0

    for jpath in judgments_dir.glob("*.json"):
        data = json.loads(jpath.read_text(encoding="utf-8"))
        total_judgments += 1
        if data.get("status") == "parse_failed":
            parse_failures += 1
            continue
        if data.get("status") == "error":
            errors += 1
            continue
        if data.get("status") != "ok":
            continue
        parsed = data.get("parsed") or {}
        judge = data["judge_label"]
        by_judge_criterion.setdefault(judge, {c: [] for c in CRITERIA})
        for c in CRITERIA:
            v = parsed.get(c)
            if isinstance(v, int):
                by_criterion[c].append(v)
                by_judge_criterion[judge][c].append(v)
        if parsed.get("critical_failure"):
            critical_failures += 1

    def _mean(xs: list[int]) -> float | None:
        return round(sum(xs) / len(xs), 3) if xs else None

    return {
        "n_sessions": sum(1 for _ in sessions_dir.glob("*.json")),
        "n_judgments_total": total_judgments,
        "n_parse_failures": parse_failures,
        "n_errors": errors,
        "n_critical_failures": critical_failures,
        "mean_per_criterion_overall": {c: _mean(by_criterion[c]) for c in CRITERIA},
        "mean_per_criterion_per_judge": {
            judge: {c: _mean(scores[c]) for c in CRITERIA}
            for judge, scores in by_judge_criterion.items()
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true", help="3 personas x 1 query x 1 seed = 12-output sanity run")
    mode.add_argument("--full", action="store_true", help="6 personas x 3 queries x N seeds full matrix")

    p.add_argument("--seeds", type=int, default=2, help="Seeds per (persona, query) for --full")
    p.add_argument("--max-recs", type=int, default=4, help="Max recommendations per session")
    p.add_argument("--judge-replicates", type=int, default=1,
                   help="Times each judge re-judges the same rec (for self-consistency variance)")
    p.add_argument("--system-model", default=DEFAULT_SYSTEM_MODEL)
    p.add_argument("--judge-anthropic", default=DEFAULT_JUDGE_ANTHROPIC)
    p.add_argument("--judge-openai", default=DEFAULT_JUDGE_OPENAI)
    p.add_argument("--rag-disabled", action="store_true", help="Skip ChromaDB retrieval (offline mode)")
    p.add_argument("--dry-run", action="store_true", help="Skip every LLM call; validate wiring only")
    p.add_argument("--out", default=None, help="Existing run subdir to resume into")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    personas = json.loads(PERSONAS_PATH.read_text(encoding="utf-8"))["personas"]
    rubric_text = RUBRIC_PATH.read_text(encoding="utf-8")

    run_dir = init_run_dir(args)
    logger.info("Run directory: %s", run_dir)

    # ----- Init clients -----
    if args.dry_run:
        sys_client = None
        opus_client = None
        gpt_client = None
        rag = None
    else:
        sys_client = LLMClient(provider=LLMProvider.ANTHROPIC, model=args.system_model)
        opus_client = LLMClient(provider=LLMProvider.ANTHROPIC, model=args.judge_anthropic)
        gpt_client = LLMClient(provider=LLMProvider.OPENAI, model=args.judge_openai)

        if args.rag_disabled:
            rag = None
        else:
            try:
                from backend.src.recommendation.rag.initializer import RAGInitializer
                rag = RAGInitializer()
                if not rag.is_initialized():
                    logger.warning("ChromaDB has no documents indexed; RAG context will be empty.")
            except Exception as e:
                logger.warning("Could not init RAG (%s); continuing without RAG context", e)
                rag = None

    generator = RecommendationGenerator(llm_client=sys_client) if sys_client is not None else None
    judges = [
        ("opus_anthropic", opus_client),
        ("gpt4o_openai", gpt_client),
    ] if not args.dry_run else [("opus_anthropic", None), ("gpt4o_openai", None)]

    # ----- Plan -----
    plan = build_session_plan(args, personas)
    logger.info("Sessions to run: %d", len(plan))

    # ----- Loop -----
    for sess in plan:
        sess_path = run_dir / "sessions" / f"{sess['session_id']}.json"
        if sess_path.exists():
            logger.info("Resume: session exists, loading: %s", sess_path.name)
            sess_record = json.loads(sess_path.read_text(encoding="utf-8"))
        else:
            logger.info("Generating: %s", sess["session_id"])
            sess_record = run_session(
                sess=sess,
                generator=generator,
                rag_initializer=rag,
                max_recs=args.max_recs,
                dry_run=args.dry_run,
            )
            sess_path.write_text(json.dumps(sess_record, indent=2, default=str), encoding="utf-8")

        # Re-hydrate Recommendations from stored dicts to feed the judges
        rec_dicts = sess_record.get("recommendations", [])
        recs: list[Recommendation] = []
        for rd in rec_dicts:
            try:
                recs.append(Recommendation.model_validate(rd))
            except Exception as e:
                logger.warning("Could not rehydrate rec for judging: %s", e)

        for rec_idx, rec in enumerate(recs):
            for judge_label, judge_client in judges:
                for replicate in range(1, args.judge_replicates + 1):
                    out_path = (
                        run_dir / "judgments"
                        / f"{sess['session_id']}__rec{rec_idx}__{judge_label}__rep{replicate}.json"
                    )
                    if out_path.exists():
                        continue
                    if not args.dry_run and judge_client is None:
                        continue

                    judgment = call_judge(
                        judge_client=judge_client,
                        judge_label=judge_label,
                        persona=sess["persona"],
                        query=sess["query"],
                        rec=rec,
                        rubric_text=rubric_text,
                        replicate=replicate,
                        dry_run=args.dry_run,
                    )
                    judgment["session_id"] = sess["session_id"]
                    judgment["rec_index"] = rec_idx
                    judgment["rec_id"] = rec.id
                    judgment["rubric_version"] = "1.1"
                    out_path.write_text(json.dumps(judgment, indent=2, default=str), encoding="utf-8")
                    if not args.dry_run:
                        # OpenAI tier-1 TPM (~30K/min) is the binding constraint.
                        # 12s for absolute scoring (smaller prompts) keeps us under TPM.
                        time.sleep(12.0 if judge_label == "gpt4o_openai" else 0.5)

    summary = aggregate_summary(run_dir)
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Summary written: %s", summary_path)

    # Pretty stdout summary
    print("\n=== RUN SUMMARY ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
