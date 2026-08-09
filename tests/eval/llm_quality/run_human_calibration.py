"""Phase C — Human calibration round.

Two subcommands:

    export    Sample N recommendations from an absolute-scoring run and write a CSV
              template for annotators (PI + co-author + student volunteer). Annotators
              fill in C1..C6 columns plus a comment field. Sample is stratified across
              persona x query so coverage is balanced. The output also writes a
              _decode.json file with the rec_id <-> source mapping; KEEP THIS PRIVATE
              and do NOT share with annotators.

    import    Read the filled-in CSVs (one per annotator), join with the LLM-judge
              judgments via the _decode mapping, and report:
                - Krippendorff alpha across the 3 annotators (interval level), per criterion
                - Pearson r between human mean and judge mean (per criterion + composite)
                - Bland-Altman style mean-difference for systematic bias detection
              Output: agreement_report.json + agreement_report.md alongside the
              annotator CSVs.

Reporting choice (Option B in judge_rubric_rec.md): we report all measured agreement
statistics verbatim with no pre-committed pass thresholds.

Usage:

    # 1) Export
    python tests/eval/llm_quality/run_human_calibration.py export \\
        --run-dir tests/eval/llm_quality/results/runs/20260507_xxxxxx_full \\
        --n 50 --annotators pi coauthor student \\
        --out-dir tests/eval/llm_quality/results/calibration/round1

    # Distribute the three CSVs in --out-dir to the three annotators.
    # KEEP _decode.json and the run-dir private from annotators.

    # 2) Import (after annotators return filled CSVs)
    python tests/eval/llm_quality/run_human_calibration.py import \\
        --calib-dir tests/eval/llm_quality/results/calibration/round1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend" / "src"))

from tests.eval.llm_quality.run_recommendation_judge import (  # noqa: E402
    CRITERIA,
    PERSONAS_PATH,
    build_portfolio_context,
    build_profile_context,
)

logger = logging.getLogger("human_calibration")

CSV_INPUT_COLUMNS = [
    "rec_id",
    "persona_id",
    "persona_label",
    "query",
    "user_profile",
    "portfolio_snapshot",
    "rec_title",
    "rec_category",
    "rec_summary",
    "rec_detailed_rationale",
    "rec_tickers",
    "rec_suggested_action",
    "rec_suggested_allocation_pct",
    "rec_risk_level",
    "rec_expected_return_range",
    "rec_time_horizon",
    "rec_confidence",
    "rec_priority",
]
CSV_SCORE_COLUMNS = [
    "C1_personalization",
    "C2_risk_alignment",
    "C3_factual_grounding",
    "C4_actionability",
    "C5_diversification",
    "C6_safety_compliance",
    "comment",
]


# ---------------------------------------------------------------------------
# Stratified sampler
# ---------------------------------------------------------------------------

def collect_recs(run_dir: Path) -> list[dict]:
    """Walk a run dir's sessions/ and emit one row per recommendation with full metadata."""
    sessions_dir = run_dir / "sessions"
    if not sessions_dir.is_dir():
        raise FileNotFoundError(f"No sessions/ subdir under {run_dir}")
    rows: list[dict] = []
    for sp in sorted(sessions_dir.glob("*.json")):
        sess = json.loads(sp.read_text(encoding="utf-8"))
        # Some pairwise runs include a "system" key; absolute runs don't.
        system_label = sess.get("system", "A_wealthnexus")
        for rec_idx, rec in enumerate(sess.get("recommendations", [])):
            rows.append({
                "rec_id": rec.get("id", f"{sess['session_id']}_rec{rec_idx}"),
                "session_id": sess["session_id"],
                "persona_id": sess["persona_id"],
                "query_type": sess["query_type"],
                "query": sess["query"],
                "seed": sess["seed"],
                "system_label": system_label,
                "rec_index": rec_idx,
                "rec": rec,
            })
    return rows


def stratified_sample(rows: list[dict], n: int, seed: int = 1) -> list[dict]:
    """Stratified sample by (persona_id, query_type) so coverage is balanced."""
    rng = random.Random(seed)
    by_strata: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_strata[(r["persona_id"], r["query_type"])].append(r)

    strata = list(by_strata.values())
    rng.shuffle(strata)
    for s in strata:
        rng.shuffle(s)

    selected: list[dict] = []
    while strata and len(selected) < n:
        new_strata = []
        for s in strata:
            if not s:
                continue
            selected.append(s.pop(0))
            if len(selected) >= n:
                break
            new_strata.append(s)
        strata = new_strata
    return selected


# ---------------------------------------------------------------------------
# Persona block lookup
# ---------------------------------------------------------------------------

def load_personas() -> dict[str, dict]:
    raw = json.loads(PERSONAS_PATH.read_text(encoding="utf-8"))
    return {p["persona_id"]: p for p in raw["personas"]}


def render_input_row(rec_row: dict, personas_by_id: dict[str, dict]) -> dict:
    persona = personas_by_id[rec_row["persona_id"]]
    rec = rec_row["rec"]
    return {
        "rec_id": rec_row["rec_id"],
        "persona_id": rec_row["persona_id"],
        "persona_label": persona["label"],
        "query": rec_row["query"],
        "user_profile": build_profile_context(persona),
        "portfolio_snapshot": build_portfolio_context(persona),
        "rec_title": rec.get("title", ""),
        "rec_category": rec.get("category", ""),
        "rec_summary": rec.get("summary", ""),
        "rec_detailed_rationale": rec.get("detailed_rationale", ""),
        "rec_tickers": ", ".join(rec.get("tickers", []) or []),
        "rec_suggested_action": rec.get("suggested_action", ""),
        "rec_suggested_allocation_pct": rec.get("suggested_allocation_pct", ""),
        "rec_risk_level": rec.get("risk_level", ""),
        "rec_expected_return_range": rec.get("expected_return_range", "") or "",
        "rec_time_horizon": rec.get("time_horizon", "") or "",
        "rec_confidence": rec.get("confidence", ""),
        "rec_priority": rec.get("priority", ""),
    }


# ---------------------------------------------------------------------------
# Cohen's kappa (linear-weighted, for ordinal data with N=2 annotators)
# ---------------------------------------------------------------------------

def cohen_kappa_linear_weighted(rater_a: list[int | None], rater_b: list[int | None],
                                  k: int = 5) -> float:
    """Linear-weighted Cohen's kappa for two raters on a 1..k ordinal scale.

    Linear weighting credits partial agreement: scores 1 vs 2 disagree less than 1 vs 5.
    Formula: kappa = 1 - (sum w_ij * o_ij) / (sum w_ij * e_ij)
    where w_ij = |i - j| / (k - 1).

    rater_a, rater_b: parallel lists; entries None are dropped pairwise.
    Returns kappa in [-1, 1]; nan if undetermined.
    """
    pairs = [(a, b) for a, b in zip(rater_a, rater_b) if a is not None and b is not None]
    n = len(pairs)
    if n < 2:
        return float("nan")

    # Marginal frequencies
    a_counts = [0] * (k + 1)
    b_counts = [0] * (k + 1)
    for a, b in pairs:
        a_counts[a] += 1
        b_counts[b] += 1

    obs = 0.0
    exp = 0.0
    denom_w = 0.0
    for i in range(1, k + 1):
        for j in range(1, k + 1):
            w = abs(i - j) / (k - 1)
            o_ij = sum(1 for a, b in pairs if a == i and b == j) / n
            e_ij = (a_counts[i] / n) * (b_counts[j] / n)
            obs += w * o_ij
            exp += w * e_ij
            denom_w += w
    if exp == 0:
        return 1.0 if obs == 0 else float("nan")
    return 1.0 - obs / exp


# ---------------------------------------------------------------------------
# Krippendorff's alpha (interval level), implemented from the standard formula
# ---------------------------------------------------------------------------

def krippendorff_alpha_interval(matrix: list[list[float | None]]) -> float:
    """Compute Krippendorff's alpha for interval-level data.

    matrix: rows = units (recs), cols = annotators. Use None for missing values.
    Returns alpha in roughly [-1, 1]; nan if undetermined.
    """
    sum_obs_within = 0.0
    n_pairs_obs = 0
    all_values: list[float] = []

    for row in matrix:
        valid = [v for v in row if v is not None]
        m = len(valid)
        if m < 2:
            continue
        for i in range(m):
            for j in range(m):
                if i != j:
                    sum_obs_within += (valid[i] - valid[j]) ** 2
                    n_pairs_obs += 1
        all_values.extend(valid)

    if n_pairs_obs == 0 or len(all_values) < 2:
        return float("nan")

    Do = sum_obs_within / n_pairs_obs

    n = len(all_values)
    sum_v = sum(all_values)
    sum_v_sq = sum(v * v for v in all_values)
    # For interval data: sum over all ordered pairs (a,b) with a != b of (a-b)^2
    # = 2 * (n * sum_v_sq - sum_v^2)
    sum_pair_sq = 2.0 * (n * sum_v_sq - sum_v * sum_v)
    n_pairs_exp = n * (n - 1)
    De = sum_pair_sq / n_pairs_exp

    if De == 0:
        return 1.0  # zero variance, full agreement by definition
    return 1.0 - Do / De


def _self_test_alpha() -> None:
    """Quick sanity checks for the alpha and Spearman implementations."""
    # Perfect agreement -> 1.0
    assert abs(krippendorff_alpha_interval([[5, 5], [4, 4], [3, 3]]) - 1.0) < 1e-9
    # Perfect disagreement (alternating) -> negative
    a = krippendorff_alpha_interval([[1, 5], [5, 1], [1, 5]])
    assert a < 0, f"expected negative, got {a}"
    # Single-annotator coverage on every unit -> nan
    assert math.isnan(krippendorff_alpha_interval([[5, None], [4, None]]))
    # Spearman: monotonic non-linear pair -> rho = 1.0 even if Pearson < 1.
    rho = spearman_rho([1, 2, 3, 4, 5], [1, 4, 9, 16, 25])
    assert abs(rho - 1.0) < 1e-9, f"expected rho=1, got {rho}"
    # Ties produce fractional ranks: identical inputs -> rho = 1.0 (degenerate)
    rho_eq = spearman_rho([1, 2, 2, 3], [1, 2, 2, 3])
    assert abs(rho_eq - 1.0) < 1e-9, f"expected rho=1 with ties, got {rho_eq}"


# ---------------------------------------------------------------------------
# Pearson r and Spearman rho
# ---------------------------------------------------------------------------

def pearson_r(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def _average_ranks(values: list[float]) -> list[float]:
    """Fractional (average) ranks. Ties get the mean of the ranks they span."""
    indexed = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and values[indexed[j + 1]] == values[indexed[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-indexed
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_rho(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation with tie-aware (fractional) ranks.

    Appropriate for ordinal Likert data where Pearson assumes interval scale.
    """
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    rx = _average_ranks(list(xs))
    ry = _average_ranks(list(ys))
    return pearson_r(rx, ry)


# ---------------------------------------------------------------------------
# Export subcommand
# ---------------------------------------------------------------------------

def cmd_export(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_recs(run_dir)
    if not rows:
        logger.error("No recommendations found in %s", run_dir)
        return 1
    sampled = stratified_sample(rows, n=args.n, seed=args.sample_seed)
    logger.info("Sampled %d recs (target %d) from %d available", len(sampled), args.n, len(rows))

    personas_by_id = load_personas()
    rendered = [render_input_row(r, personas_by_id) for r in sampled]

    # Decode mapping (PRIVATE — not shared with annotators)
    decode = {
        "run_dir": str(run_dir),
        "sample_seed": args.sample_seed,
        "n_sampled": len(sampled),
        "rec_to_source": {
            r["rec_id"]: {
                "session_id": r["session_id"],
                "persona_id": r["persona_id"],
                "query_type": r["query_type"],
                "seed": r["seed"],
                "system_label": r["system_label"],
                "rec_index": r["rec_index"],
            }
            for r in sampled
        },
    }
    (out_dir / "_decode.json").write_text(json.dumps(decode, indent=2), encoding="utf-8")

    # One CSV per annotator. Each annotator gets the SAME rec set; row order randomized
    # per annotator with their own RNG so any time-of-day fatigue effect averages out.
    for annotator in args.annotators:
        rng = random.Random(hashlib.md5(annotator.encode()).hexdigest())
        ann_rows = list(rendered)
        rng.shuffle(ann_rows)
        csv_path = out_dir / f"annotations_{annotator}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_INPUT_COLUMNS + CSV_SCORE_COLUMNS)
            writer.writeheader()
            for row in ann_rows:
                out_row = dict(row)
                for col in CSV_SCORE_COLUMNS:
                    out_row[col] = ""
                writer.writerow(out_row)
        logger.info("Wrote %s (%d rows)", csv_path, len(ann_rows))

    instructions_path = out_dir / "INSTRUCTIONS.md"
    instructions_path.write_text(
        _instructions_template(annotators=args.annotators, n=len(sampled)),
        encoding="utf-8",
    )
    logger.info("Wrote %s", instructions_path)
    logger.info("DO NOT share %s with annotators.", out_dir / "_decode.json")
    return 0


def _instructions_template(annotators: list[str], n: int) -> str:
    return f"""# Phase C — Human Calibration Round 1 (Annotator Instructions)

Thank you for taking part. There are {len(annotators)} of you ({', '.join(annotators)})
and you are independently rating the same {n} recommendations. Please do not discuss
ratings with each other before all CSVs are returned.

## What you will do

You have been given a CSV (`annotations_<your_name>.csv`) with one row per recommendation.
For each row, fill in the seven blank columns at the right:

  - C1_personalization      (1-5 integer)
  - C2_risk_alignment       (1-5 integer)
  - C3_factual_grounding    (1-5 integer)
  - C4_actionability        (1-5 integer)
  - C5_diversification      (1-5 integer)
  - C6_safety_compliance    (1-5 integer)
  - comment                 (free text, optional, useful when score <= 2)

Use the rubric in `judge_rubric_rec.md` (Section 2). The exact same rubric is being used
by the LLM judges; this calibration round measures how the LLM judges agree with humans.

## Important

- You are blind to which system produced each recommendation. Score by the rubric only.
- Do NOT look at LLM-judge scores before finishing your annotations.
- Score independently. Do not consult the other annotators until all CSVs are returned.
- If you find a row where the rec is malformed (e.g., empty fields), score what is
  present per the rubric and add a comment explaining.

When you are done, return the filled CSV to the calibration coordinator.
"""


# ---------------------------------------------------------------------------
# Import subcommand
# ---------------------------------------------------------------------------

def _read_annotator_csv(path: Path) -> dict[str, dict[str, int | None]]:
    """Read an annotator CSV, trying common encodings (Excel on Windows defaults to cp1252)."""
    raw = path.read_bytes()
    text: str | None = None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "cp949", "latin-1"):
        try:
            text = raw.decode(enc)
            if enc != "utf-8":
                logger.info("Decoded %s as %s (not UTF-8)", path.name, enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise UnicodeDecodeError("utf-8", raw, 0, 1, f"Could not decode {path} with any tried encoding")

    out: dict[str, dict[str, int | None]] = {}
    import io
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
            scores: dict[str, int | None] = {}
            for crit in CRITERIA:
                v = (row.get(crit) or "").strip()
                if v == "":
                    scores[crit] = None
                else:
                    try:
                        iv = int(v)
                    except ValueError:
                        scores[crit] = None
                        continue
                    if not (1 <= iv <= 5):
                        scores[crit] = None
                        continue
                    scores[crit] = iv
            out[row["rec_id"]] = scores
    return out


def _load_judge_means(run_dir: Path) -> dict[str, dict[str, float]]:
    """For each rec_id, mean judge score across both judges + replicates."""
    judgments_dir = run_dir / "judgments"
    by_rec: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for jp in judgments_dir.glob("*.json"):
        d = json.loads(jp.read_text(encoding="utf-8"))
        if d.get("status") != "ok":
            continue
        rec_id = d.get("rec_id")
        if not rec_id:
            continue
        parsed = d.get("parsed") or {}
        for crit in CRITERIA:
            v = parsed.get(crit)
            if isinstance(v, int):
                by_rec[rec_id][crit].append(v)
    return {
        rec_id: {crit: (sum(scores) / len(scores)) if scores else float("nan")
                 for crit, scores in crit_dict.items()}
        for rec_id, crit_dict in by_rec.items()
    }


def cmd_import(args: argparse.Namespace) -> int:
    calib_dir = Path(args.calib_dir).resolve()
    decode_path = calib_dir / "_decode.json"
    if not decode_path.exists():
        logger.error("Missing %s", decode_path)
        return 1
    decode = json.loads(decode_path.read_text(encoding="utf-8"))
    run_dir = Path(decode["run_dir"]).resolve()

    annotator_files = sorted(calib_dir.glob("annotations_*.csv"))
    if not annotator_files:
        logger.error("No annotator CSVs found in %s", calib_dir)
        return 1
    logger.info("Annotators: %s", [p.stem.replace("annotations_", "") for p in annotator_files])

    annotator_data: dict[str, dict[str, dict[str, int | None]]] = {}
    for fp in annotator_files:
        ann = fp.stem.replace("annotations_", "")
        annotator_data[ann] = _read_annotator_csv(fp)

    judge_means = _load_judge_means(run_dir)

    rec_ids = sorted(decode["rec_to_source"].keys())

    # Separate primary annotators from retest CSVs — retest only feeds test-retest stats.
    annotator_names = list(annotator_data.keys())
    primary_only = [a for a in annotator_names if not a.endswith("_retest")]

    # Krippendorff alpha per criterion across primary annotators (works for N>=2)
    alphas: dict[str, float] = {}
    for crit in CRITERIA:
        matrix = []
        for rid in rec_ids:
            row = []
            for ann in primary_only:
                row.append(annotator_data[ann].get(rid, {}).get(crit))
            matrix.append(row)
        alphas[crit] = krippendorff_alpha_interval(matrix)

    # Raw agreement rate (exact + within-1) per criterion, per pair.
    # Important when marginals are skewed: Cohen's kappa underestimates agreement
    # in this regime (the "kappa paradox" / Feinstein-Cicchetti 1990).
    raw_agreement_per_pair: dict[str, dict[str, dict[str, float]]] = {}
    for i, ann_a in enumerate(primary_only):
        for ann_b in primary_only[i + 1:]:
            pk = f"{ann_a}__vs__{ann_b}"
            raw_agreement_per_pair[pk] = {}
            for crit in CRITERIA:
                a_vals = [annotator_data[ann_a].get(rid, {}).get(crit) for rid in rec_ids]
                b_vals = [annotator_data[ann_b].get(rid, {}).get(crit) for rid in rec_ids]
                paired = [(a, b) for a, b in zip(a_vals, b_vals) if a is not None and b is not None]
                n = len(paired)
                exact = sum(1 for a, b in paired if a == b)
                within1 = sum(1 for a, b in paired if abs(a - b) <= 1)
                raw_agreement_per_pair[pk][crit] = {
                    "n": n,
                    "exact_rate": exact / n if n else float("nan"),
                    "within1_rate": within1 / n if n else float("nan"),
                }

    # Cohen's kappa (linear-weighted), Pearson r, and Spearman rho per criterion for each
    # primary annotator pair. Spearman is the appropriate correlation for ordinal Likert
    # data; Pearson is retained as a secondary reference.
    kappa_per_pair: dict[str, dict[str, float]] = {}
    inter_rater_r_per_pair: dict[str, dict[str, float]] = {}
    inter_rater_rho_per_pair: dict[str, dict[str, float]] = {}
    for i, ann_a in enumerate(primary_only):
        for ann_b in primary_only[i + 1:]:
            pair_key = f"{ann_a}__vs__{ann_b}"
            kappa_per_pair[pair_key] = {}
            inter_rater_r_per_pair[pair_key] = {}
            inter_rater_rho_per_pair[pair_key] = {}
            for crit in CRITERIA:
                a_vals = [annotator_data[ann_a].get(rid, {}).get(crit) for rid in rec_ids]
                b_vals = [annotator_data[ann_b].get(rid, {}).get(crit) for rid in rec_ids]
                kappa_per_pair[pair_key][crit] = cohen_kappa_linear_weighted(a_vals, b_vals)
                paired = [(a, b) for a, b in zip(a_vals, b_vals) if a is not None and b is not None]
                if paired:
                    xs, ys = zip(*paired)
                    inter_rater_r_per_pair[pair_key][crit] = pearson_r(list(xs), list(ys))
                    inter_rater_rho_per_pair[pair_key][crit] = spearman_rho(list(xs), list(ys))
                else:
                    inter_rater_r_per_pair[pair_key][crit] = float("nan")
                    inter_rater_rho_per_pair[pair_key][crit] = float("nan")

    # Test-retest reliability: if a "<name>_retest" CSV is present, compare to "<name>" CSV
    # on the subset of overlapping rec_ids.
    test_retest: dict[str, dict[str, float]] = {}
    for ann in list(primary_only):
        retest_name = f"{ann}_retest"
        if retest_name not in annotator_data:
            continue
        original = annotator_data[ann]
        retest = annotator_data[retest_name]
        overlap_ids = [rid for rid in rec_ids if rid in original and rid in retest]
        if not overlap_ids:
            continue
        per_crit: dict[str, float] = {"n": float(len(overlap_ids))}
        for crit in CRITERIA:
            a_vals = [original[rid].get(crit) for rid in overlap_ids]
            b_vals = [retest[rid].get(crit) for rid in overlap_ids]
            per_crit[f"{crit}_kappa"] = cohen_kappa_linear_weighted(a_vals, b_vals)
            paired = [(a, b) for a, b in zip(a_vals, b_vals) if a is not None and b is not None]
            if paired:
                xs, ys = zip(*paired)
                per_crit[f"{crit}_pearson_r"] = pearson_r(list(xs), list(ys))
                per_crit[f"{crit}_spearman_rho"] = spearman_rho(list(xs), list(ys))
            else:
                per_crit[f"{crit}_pearson_r"] = float("nan")
                per_crit[f"{crit}_spearman_rho"] = float("nan")
        test_retest[ann] = per_crit

    # Spearman rho (primary, ordinal-appropriate) and Pearson r (secondary) between
    # human-mean and judge-mean per criterion.
    correlations: dict[str, dict] = {}
    for crit in CRITERIA:
        humans: list[float] = []
        judges: list[float] = []
        for rid in rec_ids:
            human_vals = [annotator_data[ann].get(rid, {}).get(crit) for ann in annotator_data]
            human_vals = [v for v in human_vals if v is not None]
            if not human_vals:
                continue
            jm = judge_means.get(rid, {}).get(crit)
            if jm is None or math.isnan(jm):
                continue
            humans.append(sum(human_vals) / len(human_vals))
            judges.append(jm)
        r = pearson_r(humans, judges)
        rho = spearman_rho(humans, judges)
        # Bland-Altman style: mean of (human - judge), 1.96 * sd
        diffs = [h - j for h, j in zip(humans, judges)]
        mean_diff = sum(diffs) / len(diffs) if diffs else float("nan")
        if len(diffs) > 1:
            sd = math.sqrt(sum((d - mean_diff) ** 2 for d in diffs) / (len(diffs) - 1))
        else:
            sd = float("nan")
        correlations[crit] = {
            "n": len(humans),
            "spearman_rho": rho,
            "pearson_r": r,
            "mean_diff_human_minus_judge": mean_diff,
            "loa_low": mean_diff - 1.96 * sd if not math.isnan(sd) else float("nan"),
            "loa_high": mean_diff + 1.96 * sd if not math.isnan(sd) else float("nan"),
        }

    # Composite (mean over criteria, then correlate)
    comp_humans: list[float] = []
    comp_judges: list[float] = []
    for rid in rec_ids:
        per_crit_human = []
        per_crit_judge = []
        for crit in CRITERIA:
            human_vals = [annotator_data[ann].get(rid, {}).get(crit) for ann in annotator_data]
            human_vals = [v for v in human_vals if v is not None]
            if not human_vals:
                per_crit_human.append(None)
            else:
                per_crit_human.append(sum(human_vals) / len(human_vals))
            jm = judge_means.get(rid, {}).get(crit)
            per_crit_judge.append(jm if (jm is not None and not math.isnan(jm)) else None)
        if any(v is None for v in per_crit_human) or any(v is None for v in per_crit_judge):
            continue
        comp_humans.append(sum(per_crit_human) / len(per_crit_human))
        comp_judges.append(sum(per_crit_judge) / len(per_crit_judge))
    composite_r = pearson_r(comp_humans, comp_judges)
    composite_rho = spearman_rho(comp_humans, comp_judges)

    report = {
        "n_recs": len(rec_ids),
        "n_annotators": len(primary_only),
        "annotators": primary_only,
        "raw_agreement_per_pair": raw_agreement_per_pair,
        "krippendorff_alpha_interval_per_criterion": alphas,
        "cohen_kappa_linear_weighted_per_pair": kappa_per_pair,
        "inter_rater_pearson_r_per_pair": inter_rater_r_per_pair,
        "inter_rater_spearman_rho_per_pair": inter_rater_rho_per_pair,
        "judge_human_correlation_per_criterion": correlations,
        "composite_pearson_r": composite_r,
        "composite_spearman_rho": composite_rho,
        "test_retest": test_retest,
    }

    (calib_dir / "agreement_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = _render_report_md(report)
    (calib_dir / "agreement_report.md").write_text(md, encoding="utf-8")
    logger.info("Wrote %s and %s", calib_dir / "agreement_report.json", calib_dir / "agreement_report.md")

    print("\n=== AGREEMENT REPORT ===")
    print(md)
    return 0


def _render_report_md(report: dict) -> str:
    lines = [
        "# Phase C - Human Calibration Agreement Report",
        "",
        f"- Recommendations annotated: **{report['n_recs']}**",
        f"- Annotators: **{', '.join(report['annotators'])}** ({report['n_annotators']})",
        "",
        "## Inter-annotator agreement",
        "",
        "We report raw agreement rate, Cohen's kappa (linear-weighted, ordinal), Spearman "
        "rho (primary for ordinal Likert data), Pearson r (secondary), and Krippendorff "
        "alpha. Raw agreement and Spearman rho are the most interpretable statistics when "
        "marginal distributions are skewed (the kappa paradox: high agreement, low kappa).",
        "",
        "### Raw agreement rate per pair",
        "",
    ]
    for pair, crits in report["raw_agreement_per_pair"].items():
        lines.append(f"**{pair}**")
        lines.append("")
        lines.append("| Criterion | n | Exact | Within-1 |")
        lines.append("|---|---|---|---|")
        for crit, stats in crits.items():
            lines.append(f"| {crit} | {int(stats['n'])} | {stats['exact_rate']:.1%} | {stats['within1_rate']:.1%} |")
        lines.append("")
    lines.append("### Cohen's kappa (linear-weighted) per pair")
    lines.append("")
    for pair, kappas in report["cohen_kappa_linear_weighted_per_pair"].items():
        lines.append(f"**{pair}**")
        lines.append("")
        lines.append("| Criterion | kappa | Spearman rho | Pearson r |")
        lines.append("|---|---|---|---|")
        rs = report["inter_rater_pearson_r_per_pair"].get(pair, {})
        rhos = report.get("inter_rater_spearman_rho_per_pair", {}).get(pair, {})
        for crit, kv in kappas.items():
            rv = rs.get(crit, float("nan"))
            rhov = rhos.get(crit, float("nan"))
            lines.append(f"| {crit} | {kv:.3f} | {rhov:.3f} | {rv:.3f} |")
        lines.append("")
    lines.append("### Krippendorff alpha (interval, all annotators)")
    lines.append("")
    lines.append("| Criterion | alpha |")
    lines.append("|---|---|")
    for crit, a in report["krippendorff_alpha_interval_per_criterion"].items():
        lines.append(f"| {crit} | {a:.3f} |")
    lines.append("")
    lines.append("## Judge-vs-Human correlation")
    lines.append("")
    lines.append("Spearman rho is the primary statistic (ordinal Likert); Pearson r is reported for reference.")
    lines.append("")
    lines.append("| Criterion | n | Spearman rho | Pearson r | mean(H-J) | LoA low | LoA high |")
    lines.append("|---|---|---|---|---|---|---|")
    for crit, c in report["judge_human_correlation_per_criterion"].items():
        rho = c.get("spearman_rho", float("nan"))
        lines.append(
            f"| {crit} | {c['n']} | {rho:.3f} | {c['pearson_r']:.3f} | "
            f"{c['mean_diff_human_minus_judge']:.3f} | "
            f"{c['loa_low']:.3f} | {c['loa_high']:.3f} |"
        )
    lines.append("")
    comp_rho = report.get("composite_spearman_rho", float("nan"))
    lines.append(
        f"## Composite agreement: Spearman rho = **{comp_rho:.3f}** "
        f"(Pearson r = {report['composite_pearson_r']:.3f})"
    )
    if report.get("test_retest"):
        lines.append("")
        lines.append("## Test-retest reliability (intra-rater)")
        lines.append("")
        for ann, stats in report["test_retest"].items():
            lines.append(f"### {ann} (n = {int(stats['n'])} re-rated items)")
            lines.append("")
            lines.append("| Criterion | kappa | Spearman rho | Pearson r |")
            lines.append("|---|---|---|---|")
            for crit in CRITERIA:
                k = stats.get(f"{crit}_kappa", float("nan"))
                rho = stats.get(f"{crit}_spearman_rho", float("nan"))
                r = stats.get(f"{crit}_pearson_r", float("nan"))
                lines.append(f"| {crit} | {k:.3f} | {rho:.3f} | {r:.3f} |")
            lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Test-retest export
# ---------------------------------------------------------------------------

def cmd_retest_export(args: argparse.Namespace) -> int:
    """Generate a test-retest CSV for one annotator.

    Reads the annotator's existing CSV, randomly subsamples N rows, and writes a new
    'annotations_<name>_retest.csv' with all score columns blanked. Wait ~7 days
    before having the annotator re-rate, to capture intra-rater consistency.
    """
    calib_dir = Path(args.calib_dir).resolve()
    src = calib_dir / f"annotations_{args.annotator}.csv"
    if not src.exists():
        logger.error("Source CSV not found: %s", src)
        return 1

    rng = random.Random(args.seed)
    with src.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if len(rows) < args.n:
        logger.error("Only %d rows available, cannot sample %d", len(rows), args.n)
        return 1

    sampled = rng.sample(rows, args.n)
    rng.shuffle(sampled)

    out_path = calib_dir / f"annotations_{args.annotator}_retest.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_INPUT_COLUMNS + CSV_SCORE_COLUMNS)
        writer.writeheader()
        for row in sampled:
            out_row = {c: row.get(c, "") for c in CSV_INPUT_COLUMNS}
            for col in CSV_SCORE_COLUMNS:
                out_row[col] = ""  # Blank — annotator re-rates fresh
            writer.writerow(out_row)
    logger.info("Wrote %s (%d rows for test-retest)", out_path, args.n)
    logger.info("After ~7 days delay, %s should re-score these rows blind to original ratings.", args.annotator)
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_export = sub.add_parser("export", help="Sample recs and produce annotator CSVs")
    p_export.add_argument("--run-dir", required=True, help="Path to a results/runs/<dir> from run_recommendation_judge.py")
    p_export.add_argument("--n", type=int, default=50, help="Target sample size (default 50)")
    p_export.add_argument("--annotators", nargs="+", required=True, help="Annotator names (one CSV each)")
    p_export.add_argument("--out-dir", required=True, help="Where to write annotator CSVs and _decode.json")
    p_export.add_argument("--sample-seed", type=int, default=1)

    p_import = sub.add_parser("import", help="Ingest filled CSVs and compute agreement stats")
    p_import.add_argument("--calib-dir", required=True, help="Calibration round dir (must contain _decode.json)")

    p_retest = sub.add_parser("retest-export",
        help="Generate a test-retest CSV for one annotator (subsample of original CSV)")
    p_retest.add_argument("--calib-dir", required=True, help="Existing calibration round dir")
    p_retest.add_argument("--annotator", required=True, help="Annotator whose subset is re-rated (e.g. 'pi')")
    p_retest.add_argument("--n", type=int, default=10, help="Number of recs to re-rate (default 10)")
    p_retest.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    _self_test_alpha()
    if args.cmd == "export":
        return cmd_export(args)
    elif args.cmd == "import":
        return cmd_import(args)
    elif args.cmd == "retest-export":
        return cmd_retest_export(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
