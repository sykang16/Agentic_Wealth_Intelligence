"""
Baseline comparison for IntentRouter.

Methods evaluated:
  1. WealthNexus Lexical Router (keyword-based, ours)
  2. BL-1: Zero-shot Claude Haiku
  3. BL-3: 3-shot Claude Haiku (3 examples per class)
  4. BL-2: Sentence-BERT + Logistic Regression (few-shot, n=100, LOO-CV)

Usage:
    python tests/eval/run_baseline_comparison.py

Results are printed to stdout and saved as JSON for HTML report generation.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend" / "src"))

from dotenv import load_dotenv
load_dotenv()

TESTSET_PATH = project_root / "tests" / "eval" / "fixtures" / "router_testset.json"
RESULTS_PATH = project_root / "tests" / "eval" / "fixtures" / "baseline_results.json"

CLASSES = ["portfolio_query", "profiling", "recommendation", "general"]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_testset() -> list[dict]:
    with open(TESTSET_PATH, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    total = len(results)
    correct = sum(1 for r in results if r["true"] == r["pred"])

    hard = [r for r in results if r.get("difficulty") == "hard"]
    hard_correct = sum(1 for r in hard if r["true"] == r["pred"])

    # Per-class confusion counts
    confusion = {c: {c2: 0 for c2 in CLASSES} for c in CLASSES}
    for r in results:
        confusion[r["true"]][r["pred"]] += 1

    per_class = {}
    for c in CLASSES:
        tp = confusion[c][c]
        fp = sum(confusion[other][c] for other in CLASSES if other != c)
        fn = sum(confusion[c][other] for other in CLASSES if other != c)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1,
                        "tp": tp, "fp": fp, "fn": fn}

    return {
        "overall_acc":    correct / total,
        "hard_acc":       hard_correct / len(hard) if hard else 0.0,
        "n_total":        total,
        "n_correct":      correct,
        "n_hard":         len(hard),
        "n_hard_correct": hard_correct,
        "per_class":      per_class,
        "confusion":      confusion,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Method 1: WealthNexus Lexical Router
# ─────────────────────────────────────────────────────────────────────────────

def run_keyword_router(cases: list[dict]) -> tuple[list[dict], list[float]]:
    from multi_agent.routing import IntentRouter
    from common.llm_client import LLMClient

    # Build a router with a dummy LLM client (we only use _fallback_classification)
    class _DummyLLM:
        pass
    router = IntentRouter.__new__(IntentRouter)
    router.llm = _DummyLLM()

    results, latencies = [], []
    for case in cases:
        t0 = time.perf_counter()
        intent = router._fallback_classification(case["message"])
        lat = (time.perf_counter() - t0) * 1000  # ms
        latencies.append(lat)
        results.append({
            "text":       case["message"],
            "true":       case["label"],
            "pred":       intent.value,
            "difficulty": case.get("difficulty", "easy"),
            "latency_ms": lat,
        })
    return results, latencies


# ─────────────────────────────────────────────────────────────────────────────
# Method 2: Zero-shot Claude Haiku
# ─────────────────────────────────────────────────────────────────────────────

ZERO_SHOT_SYSTEM = """You are an intent classifier for a personal financial assistant.

Classify each user query into EXACTLY ONE of these four intents:

- portfolio_query: Questions about the user's existing portfolio, current holdings, asset allocation, net worth, account balances, financial position, performance, gains, or losses.
- profiling: Requests to create, update, or discuss the user's investment profile, risk tolerance, financial goals, investment strategy, time horizon, or financial situation assessment.
- recommendation: Requests for specific investment advice, buy/sell recommendations, portfolio rebalancing suggestions, diversification guidance, or asset class suggestions.
- general: Greetings, thanks, general financial education questions, or queries unrelated to the user's personal financial data.

Respond with ONLY the intent label (portfolio_query / profiling / recommendation / general). No explanation."""

VALID_LABELS = set(CLASSES)


def _normalize(raw: str) -> str:
    raw = raw.strip().lower().replace("-", "_")
    if raw in VALID_LABELS:
        return raw
    for label in VALID_LABELS:
        if label in raw:
            return label
    return "general"


def run_zero_shot_claude(cases: list[dict]) -> tuple[list[dict], list[float]]:
    import anthropic
    client = anthropic.Anthropic()

    results, latencies = [], []
    for i, case in enumerate(cases):
        t0 = time.perf_counter()
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=15,
            system=ZERO_SHOT_SYSTEM,
            messages=[{"role": "user", "content": case["message"]}],
        )
        lat = (time.perf_counter() - t0) * 1000
        pred = _normalize(resp.content[0].text)
        latencies.append(lat)
        results.append({
            "text":       case["message"],
            "true":       case["label"],
            "pred":       pred,
            "difficulty": case.get("difficulty", "easy"),
            "latency_ms": lat,
        })
        if (i + 1) % 10 == 0:
            print(f"  zero-shot: {i+1}/100 done")
    return results, latencies


# ─────────────────────────────────────────────────────────────────────────────
# Method 3: 3-shot Claude Haiku
# ─────────────────────────────────────────────────────────────────────────────

FEW_SHOT_EXAMPLES = """
Examples (3 per class):

Query: "What's my net worth?"
Intent: portfolio_query

Query: "How is my asset allocation split?"
Intent: portfolio_query

Query: "Show my investment gains this year"
Intent: portfolio_query

Query: "Build my investment profile"
Intent: profiling

Query: "I want to do a risk assessment"
Intent: profiling

Query: "What's my risk tolerance?"
Intent: profiling

Query: "What should I invest in?"
Intent: recommendation

Query: "Should I buy more AAPL shares?"
Intent: recommendation

Query: "How should I diversify my portfolio?"
Intent: recommendation

Query: "Hello!"
Intent: general

Query: "What is inflation?"
Intent: general

Query: "Thanks for your help"
Intent: general
"""

FEW_SHOT_SYSTEM = f"""You are an intent classifier for a personal financial assistant.

{FEW_SHOT_EXAMPLES}
Classify each user query into EXACTLY ONE of:
- portfolio_query: Questions about existing portfolio, holdings, balances, performance
- profiling: Building/updating investment profile, risk tolerance, goals
- recommendation: Investment advice, buy/sell suggestions, diversification
- general: Greetings, education, off-topic

Respond with ONLY the intent label. No explanation."""


def run_few_shot_claude(cases: list[dict]) -> tuple[list[dict], list[float]]:
    import anthropic
    client = anthropic.Anthropic()

    results, latencies = [], []
    for i, case in enumerate(cases):
        t0 = time.perf_counter()
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=15,
            system=FEW_SHOT_SYSTEM,
            messages=[{"role": "user", "content": case["message"]}],
        )
        lat = (time.perf_counter() - t0) * 1000
        pred = _normalize(resp.content[0].text)
        latencies.append(lat)
        results.append({
            "text":       case["message"],
            "true":       case["label"],
            "pred":       pred,
            "difficulty": case.get("difficulty", "easy"),
            "latency_ms": lat,
        })
        if (i + 1) % 10 == 0:
            print(f"  3-shot: {i+1}/100 done")
    return results, latencies


# ─────────────────────────────────────────────────────────────────────────────
# Method 4: Sentence-BERT + Logistic Regression (LOO-CV)
# ─────────────────────────────────────────────────────────────────────────────

def run_sbert_lr(cases: list[dict]) -> tuple[list[dict], list[float]]:
    """
    Sentence-BERT embeddings (all-MiniLM-L6-v2) + Logistic Regression.
    Leave-one-out cross-validation on 100 examples to simulate fine-tuning
    with limited labelled data (n=100), analogous to few-shot FinBERT.
    Latency measured as inference-only time (encoding + prediction).
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder

    print("  SBERT: loading model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts  = [c["message"] for c in cases]
    labels = [c["label"]   for c in cases]

    print("  SBERT: encoding 100 texts...")
    t_enc_start = time.perf_counter()
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)
    t_enc_total = (time.perf_counter() - t_enc_start) * 1000  # total ms

    le = LabelEncoder()
    y  = le.fit_transform(labels)

    results, latencies = [], []

    for i in range(len(cases)):
        # Train on all except i
        idx_train = [j for j in range(len(cases)) if j != i]
        X_train = embeddings[idx_train]
        y_train = y[idx_train]

        clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        clf.fit(X_train, y_train)

        # Inference latency: just the predict step (embedding pre-computed)
        t0 = time.perf_counter()
        pred_idx = clf.predict(embeddings[i : i + 1])[0]
        lat = (time.perf_counter() - t0) * 1000
        # Add per-example encoding share
        per_enc = t_enc_total / len(cases)
        lat += per_enc

        pred = le.inverse_transform([pred_idx])[0]
        latencies.append(lat)
        results.append({
            "text":       cases[i]["message"],
            "true":       cases[i]["label"],
            "pred":       pred,
            "difficulty": cases[i].get("difficulty", "easy"),
            "latency_ms": lat,
        })

    return results, latencies


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def avg(lst: list[float]) -> float:
    return sum(lst) / len(lst) if lst else 0.0


def print_summary(name: str, metrics: dict, lat: float) -> None:
    print(
        f"  {name:<35} "
        f"overall={metrics['overall_acc']:.1%}  "
        f"hard={metrics['hard_acc']:.1%}  "
        f"lat={lat:.1f}ms"
    )


def main() -> None:
    print("Loading test set...")
    cases = load_testset()
    print(f"  {len(cases)} cases loaded ({sum(1 for c in cases if c['difficulty']=='hard')} hard)")

    all_results = {}

    # ── WealthNexus keyword router ──
    print("\n[1/4] WealthNexus Lexical Router...")
    kw_res, kw_lat = run_keyword_router(cases)
    kw_m = compute_metrics(kw_res)
    print_summary("WealthNexus (Ours)", kw_m, avg(kw_lat))
    all_results["keyword"] = {"metrics": kw_m, "avg_latency_ms": avg(kw_lat),
                               "p95_latency_ms": sorted(kw_lat)[94],
                               "cost_per_query": 0.0, "results": kw_res}

    # ── Zero-shot Claude ──
    print("\n[2/4] Zero-shot Claude (claude-haiku)...")
    zs_res, zs_lat = run_zero_shot_claude(cases)
    zs_m = compute_metrics(zs_res)
    print_summary("BL-1: Zero-shot Claude Haiku", zs_m, avg(zs_lat))
    # Haiku: input $0.80/M, output $4/M; ~300 in + 5 out tokens per call
    zs_cost = (300 * 0.80 + 5 * 4.0) / 1_000_000
    all_results["zero_shot"] = {"metrics": zs_m, "avg_latency_ms": avg(zs_lat),
                                 "p95_latency_ms": sorted(zs_lat)[94],
                                 "cost_per_query": zs_cost, "results": zs_res}

    # ── 3-shot Claude ──
    print("\n[3/4] 3-shot Claude (claude-haiku)...")
    fs_res, fs_lat = run_few_shot_claude(cases)
    fs_m = compute_metrics(fs_res)
    print_summary("BL-3: 3-shot Claude Haiku", fs_m, avg(fs_lat))
    # Longer system prompt: ~600 in + 5 out tokens per call
    fs_cost = (600 * 0.80 + 5 * 4.0) / 1_000_000
    all_results["few_shot"] = {"metrics": fs_m, "avg_latency_ms": avg(fs_lat),
                                "p95_latency_ms": sorted(fs_lat)[94],
                                "cost_per_query": fs_cost, "results": fs_res}

    # ── SBERT + LR ──
    print("\n[4/4] Sentence-BERT + Logistic Regression (LOO-CV)...")
    sb_res, sb_lat = run_sbert_lr(cases)
    sb_m = compute_metrics(sb_res)
    print_summary("BL-2: SBERT+LR (FinBERT-style)", sb_m, avg(sb_lat))
    all_results["sbert_lr"] = {"metrics": sb_m, "avg_latency_ms": avg(sb_lat),
                                "p95_latency_ms": sorted(sb_lat)[94],
                                "cost_per_query": 0.0, "results": sb_res}

    # ── Save ──
    save_data = {}
    for key, val in all_results.items():
        save_data[key] = {
            "metrics":          val["metrics"],
            "avg_latency_ms":   val["avg_latency_ms"],
            "p95_latency_ms":   val["p95_latency_ms"],
            "cost_per_query":   val["cost_per_query"],
        }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    print("\n=== FINAL SUMMARY ===")
    print(f"{'Method':<35} {'Overall':>8} {'Hard':>8} {'Latency':>10} {'Cost/q':>10}")
    print("-" * 72)
    rows = [
        ("WealthNexus (Ours)",         "keyword"),
        ("BL-1: Zero-shot Claude",     "zero_shot"),
        ("BL-3: 3-shot Claude",        "few_shot"),
        ("BL-2: SBERT+LR",             "sbert_lr"),
    ]
    for name, key in rows:
        d  = all_results[key]
        m  = d["metrics"]
        print(
            f"{name:<35} {m['overall_acc']:>7.1%} {m['hard_acc']:>8.1%}"
            f" {d['avg_latency_ms']:>8.1f}ms {d['cost_per_query']:>9.4f}$"
        )


if __name__ == "__main__":
    main()
