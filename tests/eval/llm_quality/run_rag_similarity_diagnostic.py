"""Phase 1 diagnostic for Experiment 4a (RAG similarity-threshold gating).

Runs the same RAG retrieval that build_aggregated_context() would run for each
(persona x query_type) combination in the LLM-layer evaluation, and records the
top-hit similarity score. No LLM calls; ChromaDB retrieval only.

Purpose: expose the distribution of top-hit similarities so we can pick a
principled gating threshold before running the full A_gated experiment.

Usage:
    python tests/eval/llm_quality/run_rag_similarity_diagnostic.py \\
        --out-dir tests/eval/llm_quality/results/diagnostics/rag_similarity

Outputs (in --out-dir):
    per_query_scores.json   Raw per-(persona, query) top scores + top-5 scores.
    summary.md              Human-readable summary + candidate thresholds.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from statistics import mean, median

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend" / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from tests.eval.llm_quality.run_recommendation_judge import (  # noqa: E402
    PERSONAS_PATH,
    QUERY_TYPES,
)

logger = logging.getLogger("rag_similarity_diagnostic")


def load_personas() -> list[dict]:
    raw = json.loads(PERSONAS_PATH.read_text(encoding="utf-8"))
    return raw["personas"]


def retrieve_top_scores(rag, query: str, top_k: int = 5) -> dict:
    """Call the RAG initializer's search and return top-hit scores."""
    resp = rag.search(query, top_k=top_k)
    hits = resp.get("results", []) or []
    scores = [float(h.get("score", 0.0)) for h in hits]
    sources = [h.get("source", "?") for h in hits]
    return {
        "num_hits": len(hits),
        "top_score": scores[0] if scores else 0.0,
        "all_scores": scores,
        "top_source": sources[0] if sources else None,
        "top_sources": sources[:3],
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True, help="Directory to write diagnostic outputs")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    from backend.src.recommendation.rag.initializer import RAGInitializer
    rag = RAGInitializer()
    stats = rag.get_stats()
    logger.info("RAG index: %d chunks across %d unique documents",
                stats["total_chunks"], stats["unique_documents"])
    if stats["total_chunks"] == 0:
        logger.error("RAG index is empty. Run build_rag_index.py first.")
        return 1

    personas = load_personas()
    logger.info("Loaded %d personas", len(personas))

    rows: list[dict] = []
    for persona in personas:
        pid = persona["persona_id"]
        queries = persona.get("queries", {})
        for qt in QUERY_TYPES:
            query_text = queries.get(qt)
            if not query_text:
                logger.warning("persona=%s missing query_type=%s; skipping", pid, qt)
                continue
            r = retrieve_top_scores(rag, query_text, top_k=args.top_k)
            rows.append({
                "persona_id": pid,
                "query_type": qt,
                "query_text": query_text,
                **r,
            })
            logger.info("[%s/%s] top=%.3f (%d hits, top_source=%s)",
                        pid, qt, r["top_score"], r["num_hits"], r["top_source"])

    (out_dir / "per_query_scores.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    # Compute distribution stats over top scores.
    top_scores = [r["top_score"] for r in rows]
    top_scores_sorted = sorted(top_scores)
    n = len(top_scores)
    stats_summary = {
        "n": n,
        "min": min(top_scores) if top_scores else None,
        "q1": top_scores_sorted[max(0, n // 4 - 1)] if n else None,
        "median": median(top_scores) if top_scores else None,
        "q3": top_scores_sorted[min(n - 1, (3 * n) // 4)] if n else None,
        "max": max(top_scores) if top_scores else None,
        "mean": mean(top_scores) if top_scores else None,
    }

    # Simulate gating rates at candidate thresholds.
    candidates = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    gating = {}
    for t in candidates:
        gated = [r for r in rows if r["top_score"] < t]
        gating[f"{t:.2f}"] = {
            "gated_count": len(gated),
            "gated_frac": len(gated) / n if n else 0.0,
            "gated_ids": [f"{g['persona_id']}/{g['query_type']}" for g in gated],
        }

    md_lines = [
        "# Phase 1 diagnostic: RAG top-hit similarity distribution",
        "",
        f"n = {n} (persona x query_type) pairs; RAG top-k = {args.top_k}",
        "",
        "## Distribution over top-hit similarity",
        "",
        "| statistic | value |",
        "|---|---|",
        f"| min    | {stats_summary['min']:.3f} |",
        f"| Q1     | {stats_summary['q1']:.3f} |",
        f"| median | {stats_summary['median']:.3f} |",
        f"| Q3     | {stats_summary['q3']:.3f} |",
        f"| max    | {stats_summary['max']:.3f} |",
        f"| mean   | {stats_summary['mean']:.3f} |",
        "",
        "## Gating rate at candidate thresholds",
        "",
        "For each candidate threshold t, we count how many (persona, query) pairs",
        "would have `top_score < t` (i.e., RAG context would be gated out).",
        "",
        "| threshold | gated (n) | gated (fraction) |",
        "|---|---|---|",
    ]
    for t_str, info in gating.items():
        md_lines.append(f"| {t_str} | {info['gated_count']}/{n} | {info['gated_frac']:.1%} |")

    md_lines += ["", "## Per-query top scores", "",
                 "| persona | query_type | top_score | top_source |",
                 "|---|---|---|---|"]
    for r in rows:
        md_lines.append(
            f"| {r['persona_id']} | {r['query_type']} | {r['top_score']:.3f} | {r['top_source']} |"
        )

    md_lines += ["", "## Interpretation guide", "",
                 "- A threshold that gates 0% of queries is too permissive to move the needle.",
                 "- A threshold that gates 100% is equivalent to disabling RAG (already tested as B2).",
                 "- Aim for a threshold in the 20-60% gating band so the experiment is informative.",
                 "- Prior report flagged P1/open_ended (top_score 0.43) as the failure case; a",
                 "  useful threshold should gate this case at a minimum.",
                 ""]

    (out_dir / "summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    logger.info("Wrote %s", out_dir / "per_query_scores.json")
    logger.info("Wrote %s", out_dir / "summary.md")

    print("\n=== SUMMARY ===")
    print("\n".join(md_lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
