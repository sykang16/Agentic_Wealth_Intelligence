"""Phase B (ablation half) — pairwise judge of WealthNexus vs three baselines.

Systems compared:
  A — WealthNexus           : full pipeline (profile + portfolio + RAG)
  B1 — Profile-blind         : same generator + system prompt, but the AggregatedContext
                              has the investment_profile_context blanked out
  B2 — RAG-blind             : same generator + system prompt, but rag_context is empty
  B3 — Generic LLM           : zero-shot Sonnet with a neutral advisor prompt; no
                              WealthNexus system prompt and no RAG

For each session (persona x query x seed) we generate 4 recommendation lists, then run
list-level pairwise judgments for the three pairs (A vs B1, A vs B2, A vs B3).

Position bias is controlled by judging each pair in both orders (A,B) and (B,A).
A pair is marked a CONFIRMED win for system X only when both orders independently name X
as overall_winner — disagreements are logged as ties.

Run modes:
  --pilot              : 3 personas x 1 query x 1 seed = 3 sessions, ~36 judge calls
  --full               : 6 personas x 3 queries x 2 seeds = 36 sessions, ~432 judge calls
  --dry-run            : skip every LLM call; validate wiring only

Resume: pass --out <existing_run_dir> to skip files that already exist.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap (mirrors run_recommendation_judge.py)
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

# Reuse shared helpers from the absolute-scoring script
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
    render_recommendation_for_judge,
    session_id,
)

SYSTEMS = ["A_wealthnexus", "B1_profile_blind", "B2_rag_blind", "B3_generic_llm"]
PAIRS = [("A_wealthnexus", "B1_profile_blind"),
         ("A_wealthnexus", "B2_rag_blind"),
         ("A_wealthnexus", "B3_generic_llm")]

GEN_TEMPERATURE = 0.4
GEN_MAX_TOKENS = 4096
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 1024

logger = logging.getLogger("pairwise_baseline")


# ---------------------------------------------------------------------------
# B3 generic-LLM path (no WealthNexus system prompt)
# ---------------------------------------------------------------------------

B3_SYSTEM_PROMPT = """You are an experienced financial advisor providing investment recommendations.

Output your recommendations as a valid JSON array. Each element MUST include the keys:
category, title, summary, detailed_rationale, tickers, suggested_action,
suggested_allocation_pct, risk_level, expected_return_range, time_horizon,
confidence, priority.

Allowed values:
  category: one of buy, sell, hold, rebalance, diversify, risk_reduction, tax_optimization, income_generation
  risk_level: one of low, moderate, high
  confidence: one of low, medium, high
  priority: one of low, medium, high

Return between 1 and {max_recommendations} recommendations as a JSON array. No prose."""

B3_USER_PROMPT = """## User profile
{profile_block}

## Portfolio
{portfolio_block}

## Query
{query}

Provide the JSON array of recommendations."""


def _parse_recommendations_json(text: str) -> list[Recommendation]:
    """Parse a JSON array of recommendation dicts. Mirrors generator._parse_recommendations
    but inlined to avoid coupling to a private method."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n", "", cleaned)
        cleaned = re.sub(r"\n```\s*$", "", cleaned)
    try:
        raw = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", cleaned)
        if not match:
            return []
        try:
            raw = json.loads(match.group())
        except json.JSONDecodeError:
            return []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []

    recs: list[Recommendation] = []
    for i, item in enumerate(raw):
        try:
            recs.append(Recommendation(
                id=f"b3_{i}",
                category=RecommendationCategory(item.get("category", "hold")),
                title=item.get("title", "Untitled"),
                summary=item.get("summary", ""),
                detailed_rationale=item.get("detailed_rationale", ""),
                tickers=item.get("tickers", []) or [],
                suggested_action=item.get("suggested_action", ""),
                suggested_allocation_pct=item.get("suggested_allocation_pct"),
                risk_level=RiskLevel(item.get("risk_level", "moderate")),
                expected_return_range=item.get("expected_return_range"),
                time_horizon=item.get("time_horizon"),
                confidence=ConfidenceLevel(item.get("confidence", "medium")),
                priority=RecommendationPriority(item.get("priority", "medium")),
            ))
        except Exception as e:
            logger.warning("Failed to parse B3 rec %d: %s", i, e)
    return recs


def generate_b3(
    persona: dict,
    query: str,
    sys_client: LLMClient,
    max_recommendations: int,
) -> list[Recommendation]:
    """Generate recommendations using a neutral advisor prompt (no WealthNexus prompt)."""
    system_prompt = B3_SYSTEM_PROMPT.format(max_recommendations=max_recommendations)
    user_prompt = B3_USER_PROMPT.format(
        profile_block=build_profile_context(persona),
        portfolio_block=build_portfolio_context(persona),
        query=query,
    )
    resp = sys_client.chat(
        user_message=user_prompt,
        system_prompt=system_prompt,
        max_tokens=GEN_MAX_TOKENS,
        temperature=GEN_TEMPERATURE,
    )
    return _parse_recommendations_json(resp.content)[:max_recommendations]


# ---------------------------------------------------------------------------
# System dispatcher
# ---------------------------------------------------------------------------

def _stub_rec(label: str, idx: int) -> Recommendation:
    return Recommendation(
        id=f"dryrun_{label}_{idx}",
        category=RecommendationCategory.HOLD,
        title=f"[{label} dry-run stub #{idx}]",
        summary="Stub summary.",
        detailed_rationale="Stub rationale.",
        tickers=[],
        suggested_action="Stub.",
        risk_level=RiskLevel.MODERATE,
        confidence=ConfidenceLevel.MEDIUM,
        priority=RecommendationPriority.MEDIUM,
    )


def generate_for_system(
    system_label: str,
    persona: dict,
    query: str,
    generator: RecommendationGenerator | None,
    sys_client: LLMClient | None,
    rag_initializer,
    max_recs: int,
    dry_run: bool,
) -> list[Recommendation]:
    if dry_run or generator is None or sys_client is None:
        return [_stub_rec(system_label, 0)]

    if system_label == "A_wealthnexus":
        ctx = build_aggregated_context(persona, query, rag_initializer)
        return generator.generate(context=ctx, query=query, max_recommendations=max_recs)

    if system_label == "B1_profile_blind":
        ctx = build_aggregated_context(persona, query, rag_initializer)
        ctx.investment_profile_context = ""
        ctx.user_risk_tolerance = None
        ctx.user_investment_horizon = None
        ctx.user_experience_level = None
        return generator.generate(context=ctx, query=query, max_recommendations=max_recs)

    if system_label == "B2_rag_blind":
        ctx = build_aggregated_context(persona, query, rag_initializer=None)
        return generator.generate(context=ctx, query=query, max_recommendations=max_recs)

    if system_label == "B3_generic_llm":
        return generate_b3(persona, query, sys_client, max_recommendations=max_recs)

    raise ValueError(f"Unknown system label: {system_label}")


# ---------------------------------------------------------------------------
# Pairwise judge prompt
# ---------------------------------------------------------------------------

PAIRWISE_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert evaluator comparing two automated financial-advisory recommendation lists
generated for the same user query. Use the rubric below. Do not reward verbosity, fluency,
or formatting beyond what the rubric specifies. Both lists are produced by automated systems
whose identities are hidden — judge purely on rubric criteria.
Return only valid JSON.

== RUBRIC ==
{rubric_text}
== END RUBRIC ==
"""

PAIRWISE_USER_PROMPT_TEMPLATE = """\
## User profile
{profile_block}

## Portfolio snapshot
{portfolio_block}

## User query
{query}

## System A recommendations (system identity hidden)
{rec_A_block}

## System B recommendations (system identity hidden)
{rec_B_block}

## Output format (JSON only, no prose)
{{
  "winner_per_criterion": {{
    "C1_personalization":  "A" | "B" | "tie",
    "C2_risk_alignment":   "A" | "B" | "tie",
    "C3_factual_grounding":"A" | "B" | "tie",
    "C4_actionability":    "A" | "B" | "tie",
    "C5_diversification":  "A" | "B" | "tie",
    "C6_safety_compliance":"A" | "B" | "tie"
  }},
  "overall_winner": "A" | "B" | "tie",
  "rationale_overall": "<two-sentence justification>"
}}
"""


def render_rec_list(recs: list[Recommendation]) -> str:
    if not recs:
        return "(empty list — system returned no recommendations)"
    lines = []
    for i, r in enumerate(recs, start=1):
        lines.append(f"### Rec #{i}")
        lines.append(render_recommendation_for_judge(r))
        lines.append("")
    return "\n".join(lines)


def parse_pairwise_response(text: str) -> dict | None:
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
    if not isinstance(parsed.get("winner_per_criterion"), dict):
        return None
    if parsed.get("overall_winner") not in ("A", "B", "tie"):
        return None
    return parsed


def call_pairwise_judge(
    judge_client: LLMClient | None,
    judge_label: str,
    persona: dict,
    query: str,
    recs_A: list[Recommendation],
    recs_B: list[Recommendation],
    rubric_text: str,
    dry_run: bool,
) -> dict:
    system_prompt = PAIRWISE_SYSTEM_PROMPT_TEMPLATE.format(rubric_text=rubric_text)
    user_prompt = PAIRWISE_USER_PROMPT_TEMPLATE.format(
        profile_block=build_profile_context(persona),
        portfolio_block=build_portfolio_context(persona),
        query=query,
        rec_A_block=render_rec_list(recs_A),
        rec_B_block=render_rec_list(recs_B),
    )

    if dry_run or judge_client is None:
        return {
            "judge_label": judge_label,
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
        parsed = parse_pairwise_response(resp.content)
        return {
            "judge_label": judge_label,
            "judge_provider": resp.provider,
            "judge_model": resp.model,
            "status": "ok" if parsed is not None else "parse_failed",
            "raw_output": resp.content,
            "parsed": parsed,
            "usage": resp.usage,
        }
    except Exception as e:
        logger.error("Pairwise judge call failed (%s): %s", judge_label, e)
        return {
            "judge_label": judge_label,
            "status": "error",
            "error": str(e),
            "raw_output": None,
            "parsed": None,
        }


# ---------------------------------------------------------------------------
# Aggregation: confirmed wins + criterion win-rates
# ---------------------------------------------------------------------------

def _resolve_pair_winner(order1_parsed: dict | None, order2_parsed: dict | None,
                        sys_a: str, sys_b: str) -> str:
    """Confirmed-win logic with order randomization.

    order1: A=sys_a, B=sys_b
    order2: A=sys_b, B=sys_a
    """
    if not order1_parsed or not order2_parsed:
        return "tie"
    o1 = order1_parsed["overall_winner"]
    o2 = order2_parsed["overall_winner"]
    # Decode both back to system identifiers
    o1_sys = sys_a if o1 == "A" else (sys_b if o1 == "B" else "tie")
    o2_sys = sys_b if o2 == "A" else (sys_a if o2 == "B" else "tie")
    if o1_sys == o2_sys and o1_sys != "tie":
        return o1_sys
    return "tie"


def aggregate_pairwise_summary(run_dir: Path) -> dict:
    """Compute per-pair confirmed wins, ties, and per-criterion win-rates."""
    pw_dir = run_dir / "pairwise"
    files = list(pw_dir.glob("*.json"))

    by_session_pair_judge: dict[tuple[str, str, str, str], dict[int, dict | None]] = {}
    for fp in files:
        data = json.loads(fp.read_text(encoding="utf-8"))
        key = (data["session_id"], data["sys_a"], data["sys_b"], data["judge_label"])
        order = data["order"]  # 1 or 2
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
    p.add_argument("--rag-disabled", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def init_run_dir(args) -> Path:
    label = "pilot_pw" if args.pilot else "full_pw"
    if args.out:
        run_dir = RESULTS_ROOT / args.out
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = RESULTS_ROOT / f"{ts}_{label}"
    (run_dir / "sessions").mkdir(parents=True, exist_ok=True)
    (run_dir / "pairwise").mkdir(parents=True, exist_ok=True)

    config = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": label,
        "system_model": args.system_model,
        "judge_anthropic": args.judge_anthropic,
        "judge_openai": args.judge_openai,
        "seeds": args.seeds,
        "max_recs": args.max_recs,
        "rag_disabled": args.rag_disabled,
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
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

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
    judges = [("opus_anthropic", opus_client), ("gpt4o_openai", gpt_client)]

    plan = build_session_plan(args, personas)
    logger.info("Sessions to run: %d", len(plan))

    for sess in plan:
        # ----- Generate recommendations for all 4 systems -----
        system_recs: dict[str, list[Recommendation]] = {}
        for sys_label in SYSTEMS:
            sess_path = run_dir / "sessions" / f"{sess['session_id']}__{sys_label}.json"
            if sess_path.exists():
                logger.info("Resume: loading %s", sess_path.name)
                stored = json.loads(sess_path.read_text(encoding="utf-8"))
                system_recs[sys_label] = [Recommendation.model_validate(r) for r in stored["recommendations"]]
                continue

            logger.info("Generating: %s [%s]", sess["session_id"], sys_label)
            try:
                recs = generate_for_system(
                    system_label=sys_label,
                    persona=sess["persona"],
                    query=sess["query"],
                    generator=generator,
                    sys_client=sys_client,
                    rag_initializer=rag,
                    max_recs=args.max_recs,
                    dry_run=args.dry_run,
                )
            except Exception as e:
                logger.error("Generation failed for %s: %s", sys_label, e)
                recs = []
            system_recs[sys_label] = recs

            sess_record = {
                "session_id": sess["session_id"],
                "persona_id": sess["persona"]["persona_id"],
                "query_type": sess["query_type"],
                "query": sess["query"],
                "seed": sess["seed"],
                "system": sys_label,
                "rec_count": len(recs),
                "recommendations": [r.model_dump(mode="json") for r in recs],
            }
            sess_path.write_text(json.dumps(sess_record, indent=2, default=str), encoding="utf-8")

        # ----- Pairwise judging -----
        for sys_a, sys_b in PAIRS:
            recs_a = system_recs.get(sys_a, [])
            recs_b = system_recs.get(sys_b, [])
            for order in (1, 2):
                # order=1 : A=sys_a, B=sys_b
                # order=2 : A=sys_b, B=sys_a
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
                    judgment.update({
                        "session_id": sess["session_id"],
                        "sys_a": sys_a,
                        "sys_b": sys_b,
                        "order": order,
                        "rubric_version": "1.1",
                    })
                    out_path.write_text(json.dumps(judgment, indent=2, default=str), encoding="utf-8")
                    if not args.dry_run:
                        # OpenAI tier-1 TPM (~30K/min) is the binding constraint for pairwise
                        # judge calls (~8K input + 1K output = ~9K tokens/call).
                        # 18s -> ~3 calls/min on the OpenAI side, leaves headroom.
                        time.sleep(18.0 if judge_label == "gpt4o_openai" else 0.5)

    summary = aggregate_pairwise_summary(run_dir)
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Summary written: %s", summary_path)

    print("\n=== PAIRWISE RUN SUMMARY ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
