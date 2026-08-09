"""Generate an expanded IntentRouter test set (target n=500) via multi-model
generation with cross-model label verification.

Design:
  * 100 original utterances preserved as-is (real-authored, in router_testset.json).
  * 400 new utterances generated in balanced fashion: 4 models x 100 utterances
    (25 per intent class) each. Models are chosen to sit outside the classifier
    baselines being evaluated (Haiku BL-1/BL-3 excluded).
  * Every new utterance is cross-labelled by two other models; disagreements are
    written to a review file for manual reconciliation.

Output files (in tests/eval/fixtures/):
  * router_testset_v2_raw.json          -- all generations + per-model labels
  * router_testset_v2.json              -- final labelled test set (n=500)
  * router_testset_v2_disagreements.csv -- items flagged for manual review

Usage:
    python tests/eval/generate_router_testset_v2.py --step generate
    python tests/eval/generate_router_testset_v2.py --step verify
    python tests/eval/generate_router_testset_v2.py --step finalize
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend" / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger("gen_router_v2")

CLASSES = ["portfolio_query", "profiling", "recommendation", "general"]

# 4 generator models, chosen to sit outside the classifier baselines (Haiku
# BL-1/BL-3 excluded to prevent test-set contamination).
GENERATOR_MODELS = [
    ("claude_opus", "anthropic", "claude-opus-4-7"),
    ("claude_sonnet", "anthropic", "claude-sonnet-4-5"),
    ("gpt_4o", "openai", "gpt-4o"),
    ("gemini_flash", "gemini", "gemini-flash-latest"),
]

# Cross-model verifiers: for each utterance, verify with the two OTHER strongest
# non-generator models (rotates so that no verifier is also the generator).
VERIFIER_ROTATION = {
    "claude_opus": [("gpt_4o", "openai", "gpt-4o"),
                    ("gemini_flash", "gemini", "gemini-flash-latest")],
    "claude_sonnet": [("gpt_4o", "openai", "gpt-4o"),
                      ("gemini_flash", "gemini", "gemini-flash-latest")],
    "gpt_4o": [("claude_opus", "anthropic", "claude-opus-4-7"),
               ("gemini_flash", "gemini", "gemini-flash-latest")],
    "gemini_flash": [("claude_opus", "anthropic", "claude-opus-4-7"),
                     ("gpt_4o", "openai", "gpt-4o")],
}

N_PER_CLASS_PER_MODEL = 25   # 25 * 4 classes = 100 per model, * 4 models = 400 new

ORIGINAL_TESTSET_PATH = SCRIPT_DIR / "fixtures" / "router_testset.json"
RAW_OUT = SCRIPT_DIR / "fixtures" / "router_testset_v2_raw.json"
FINAL_OUT = SCRIPT_DIR / "fixtures" / "router_testset_v2.json"
DISAGREE_CSV = SCRIPT_DIR / "fixtures" / "router_testset_v2_disagreements.csv"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CLASS_DEFINITIONS = {
    "portfolio_query": (
        "Questions about the user's existing portfolio: current holdings, asset "
        "allocation, net worth, account balances, performance, gains/losses."
    ),
    "profiling": (
        "Requests to build/update the user's investment profile: risk tolerance, "
        "goals, time horizon, personal financial situation, strategy assessment."
    ),
    "recommendation": (
        "Requests for specific investment advice: buy/sell suggestions, "
        "rebalancing, diversification guidance, asset class recommendations."
    ),
    "general": (
        "Greetings, thanks, general financial education (not personal), off-topic, "
        "or clarifying questions unrelated to the user's own portfolio."
    ),
}

GENERATION_PROMPT = """You are generating a labelled test set for a financial-assistant intent
classifier. Generate exactly {n} distinct user utterances that clearly belong to the
intent class {target_class}.

Class definitions:
- portfolio_query: {portfolio_query}
- profiling: {profiling}
- recommendation: {recommendation}
- general: {general}

Style diversity requirements:
- Mix formal and informal register (some polite/formal, some casual/short).
- Vary length: some 3-6 words, some 15-25 words.
- Include colloquial phrasings and non-native-speaker style occasionally.
- Include {n_hard} "hard" cases: ambiguous, borderline, or expressed via
  indirect phrasing that still clearly belongs to {target_class}.

Reference examples of {target_class} (do NOT copy verbatim):
{reference_examples}

Return a JSON array of exactly {n} strings. Each string is one utterance. Nothing else.
Do not include labels; do not include commentary. Just the JSON array."""

VERIFY_PROMPT = """You are a strict intent classifier for a personal financial assistant.
Classify the following user utterance into EXACTLY ONE of these intents:

- portfolio_query: {portfolio_query}
- profiling: {profiling}
- recommendation: {recommendation}
- general: {general}

Utterance: {utterance}

Respond with ONLY the intent label (portfolio_query / profiling / recommendation / general).
No explanation."""


# ---------------------------------------------------------------------------
# LLM wrappers
# ---------------------------------------------------------------------------

def call_anthropic(model: str, system_prompt: str, user_prompt: str,
                   max_tokens: int = 4096, temperature: float = 0.7) -> str:
    import anthropic
    client = anthropic.Anthropic()
    # Opus 4.7 deprecated the temperature parameter; only pass it for older models.
    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    if "opus-4-7" not in model:
        kwargs["temperature"] = temperature
    resp = client.messages.create(**kwargs)
    return resp.content[0].text


def call_openai(model: str, system_prompt: str, user_prompt: str,
                max_tokens: int = 4096, temperature: float = 0.7) -> str:
    import openai
    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content


def call_gemini(model: str, system_prompt: str, user_prompt: str,
                max_tokens: int = 4096, temperature: float = 0.7) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    resp = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=max_tokens,
            temperature=temperature,
            # gemini-*-latest are thinking models: without disabling thinking,
            # small max_output_tokens are fully consumed by internal reasoning
            # and resp.text comes back None.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return resp.text


CALLERS = {
    "anthropic": call_anthropic,
    "openai": call_openai,
    "gemini": call_gemini,
}


def call_llm(provider: str, model: str, system_prompt: str, user_prompt: str,
             **kwargs) -> str:
    for attempt in range(3):
        try:
            return CALLERS[provider](model, system_prompt, user_prompt, **kwargs)
        except Exception as e:
            logger.warning("Call failed (%s/%s attempt %d): %s", provider, model, attempt + 1, e)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed after 3 retries: {provider}/{model}")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _class_examples(original: list[dict], cls: str, k: int = 5,
                    seed: int = 42) -> list[str]:
    rng = random.Random(seed + hash(cls) % 10000)
    same = [c["message"] for c in original if c["label"] == cls]
    rng.shuffle(same)
    return same[:k]


def _parse_json_array(text: str) -> list[str]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```\s*$", "", text)
    try:
        arr = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\[[\s\S]*\]", text)
        if not m:
            return []
        try:
            arr = json.loads(m.group())
        except json.JSONDecodeError:
            return []
    if not isinstance(arr, list):
        return []
    return [str(x).strip() for x in arr if isinstance(x, str) and str(x).strip()]


def step_generate(args: argparse.Namespace) -> int:
    original = json.loads(ORIGINAL_TESTSET_PATH.read_text(encoding="utf-8"))
    logger.info("Loaded %d original utterances", len(original))

    n_hard = max(3, N_PER_CLASS_PER_MODEL // 5)
    system_prompt = ("You are a careful data-generation assistant. You output only "
                     "valid JSON when a JSON array is requested.")

    # Resume-safe: load existing items and skip (generator, class) pairs already done.
    all_items: list[dict] = []
    done_pairs: set[tuple[str, str]] = set()
    if RAW_OUT.exists():
        all_items = json.loads(RAW_OUT.read_text(encoding="utf-8"))
        for it in all_items:
            done_pairs.add((it["generator"], it["generator_label"]))
        logger.info("Resuming: %d existing items; %d (gen, class) pairs done",
                    len(all_items), len(done_pairs))

    for gen_key, provider, model in GENERATOR_MODELS:
        logger.info("Generator: %s (%s / %s)", gen_key, provider, model)
        for cls in CLASSES:
            if (gen_key, cls) in done_pairs:
                logger.info("  [%s/%s] skip (already done)", gen_key, cls)
                continue
            refs = _class_examples(original, cls, k=5)
            prompt = GENERATION_PROMPT.format(
                n=N_PER_CLASS_PER_MODEL,
                n_hard=n_hard,
                target_class=cls,
                portfolio_query=CLASS_DEFINITIONS["portfolio_query"],
                profiling=CLASS_DEFINITIONS["profiling"],
                recommendation=CLASS_DEFINITIONS["recommendation"],
                general=CLASS_DEFINITIONS["general"],
                reference_examples="\n".join(f"  - {r}" for r in refs),
            )

            try:
                raw = call_llm(provider, model, system_prompt, prompt,
                               max_tokens=4096, temperature=0.85)
            except Exception as e:
                logger.error("Generation failed (%s/%s): %s", gen_key, cls, e)
                continue

            utterances = _parse_json_array(raw)
            logger.info("  [%s/%s]: parsed %d utterances", gen_key, cls, len(utterances))

            for utt in utterances[:N_PER_CLASS_PER_MODEL]:
                all_items.append({
                    "message": utt,
                    "generator": gen_key,
                    "generator_label": cls,
                    "difficulty": "unknown",
                })

    RAW_OUT.parent.mkdir(parents=True, exist_ok=True)
    RAW_OUT.write_text(json.dumps(all_items, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %d generated items to %s", len(all_items), RAW_OUT)
    return 0


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

VALID_LABELS = set(CLASSES)


def _normalize(raw: str) -> str:
    raw = raw.strip().lower().replace("-", "_").replace(".", "")
    for label in VALID_LABELS:
        if label in raw:
            return label
    return "general"


def step_verify(args: argparse.Namespace) -> int:
    items = json.loads(RAW_OUT.read_text(encoding="utf-8"))
    logger.info("Verifying %d items", len(items))

    system_prompt = ("You are a strict, careful intent classifier. Respond with only "
                     "the label word, nothing else.")

    for i, item in enumerate(items):
        if "verifier_labels" in item:
            continue  # resume-safe
        gen_key = item["generator"]
        verifiers = VERIFIER_ROTATION[gen_key]
        labels: dict[str, str] = {}
        for v_key, v_prov, v_model in verifiers:
            prompt = VERIFY_PROMPT.format(
                utterance=item["message"],
                portfolio_query=CLASS_DEFINITIONS["portfolio_query"],
                profiling=CLASS_DEFINITIONS["profiling"],
                recommendation=CLASS_DEFINITIONS["recommendation"],
                general=CLASS_DEFINITIONS["general"],
            )
            try:
                raw = call_llm(v_prov, v_model, system_prompt, prompt,
                               max_tokens=20, temperature=0.0)
                labels[v_key] = _normalize(raw)
            except Exception as e:
                logger.error("Verify failed (%s): %s", v_key, e)
                labels[v_key] = None
        item["verifier_labels"] = labels

        if (i + 1) % 20 == 0:
            RAW_OUT.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info("  verified %d/%d", i + 1, len(items))

    RAW_OUT.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Verification complete; state saved to %s", RAW_OUT)
    return 0


# ---------------------------------------------------------------------------
# Finalize
# ---------------------------------------------------------------------------

def step_finalize(args: argparse.Namespace) -> int:
    original = json.loads(ORIGINAL_TESTSET_PATH.read_text(encoding="utf-8"))
    generated = json.loads(RAW_OUT.read_text(encoding="utf-8"))
    logger.info("Original: %d, generated: %d", len(original), len(generated))

    kept: list[dict] = []
    disagreements: list[dict] = []

    for item in generated:
        gen_label = item["generator_label"]
        v_labels = item.get("verifier_labels", {})
        v_values = [v for v in v_labels.values() if v]

        # 3/3 agreement: generator + both verifiers all agree
        if len(v_values) >= 2 and all(v == gen_label for v in v_values):
            kept.append({
                "message": item["message"],
                "label": gen_label,
                "difficulty": "easy",
                "source": item["generator"],
                "consensus": "3/3",
            })
        # 2/3 agreement: one verifier agrees
        elif len(v_values) >= 2 and sum(1 for v in v_values if v == gen_label) == 1:
            kept.append({
                "message": item["message"],
                "label": gen_label,
                "difficulty": "hard",
                "source": item["generator"],
                "consensus": "2/3",
            })
            disagreements.append({**item, "resolution": "kept_as_hard"})
        # 0/3 verifier agreement (both disagree with generator)
        else:
            # If both verifiers agree with each other on a different label, use that.
            if len(v_values) >= 2 and v_values[0] == v_values[1] and v_values[0] != gen_label:
                kept.append({
                    "message": item["message"],
                    "label": v_values[0],
                    "difficulty": "hard",
                    "source": item["generator"],
                    "consensus": "2/3-relabel",
                    "generator_label": gen_label,
                })
                disagreements.append({**item, "resolution": f"relabel_to_{v_values[0]}"})
            else:
                # Truly ambiguous: skip, log
                disagreements.append({**item, "resolution": "skipped_ambiguous"})

    # Original 100 with source label
    original_annotated = [
        {**c, "source": "original", "consensus": "gold"} for c in original
    ]

    final = original_annotated + kept
    # Balance check
    from collections import Counter
    dist = Counter(c["label"] for c in final)
    logger.info("Final class distribution: %s", dict(dist))
    logger.info("Final total: %d (kept from generation: %d, disagreements: %d)",
                len(final), len(kept), len(disagreements))

    FINAL_OUT.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")

    if disagreements:
        with DISAGREE_CSV.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "message", "generator", "generator_label",
                "verifier_labels", "resolution",
            ])
            w.writeheader()
            for d in disagreements:
                w.writerow({
                    "message": d["message"],
                    "generator": d["generator"],
                    "generator_label": d["generator_label"],
                    "verifier_labels": json.dumps(d.get("verifier_labels", {})),
                    "resolution": d["resolution"],
                })
        logger.info("Wrote %d disagreements to %s", len(disagreements), DISAGREE_CSV)

    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step", required=True, choices=["generate", "verify", "finalize"])
    args = parser.parse_args()

    if args.step == "generate":
        return step_generate(args)
    elif args.step == "verify":
        return step_verify(args)
    elif args.step == "finalize":
        return step_finalize(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
