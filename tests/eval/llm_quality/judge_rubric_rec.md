# LLM-as-Judge Rubric — Recommendation Text Quality

**Phase B of the WealthNexus LLM-Layer Evaluation**
This rubric is used by (a) the LLM judges (Claude Opus 4.7 and a non-Anthropic cross-judge) and (b) the human annotators (PI, co-author, student volunteer) for the calibration round. The rubric is identical for both groups so judge↔human agreement can be measured directly.

---

## 1. Inputs the judge sees

For every judgment the judge is given:

1. **User profile snapshot** (risk_tolerance, loss_comfort, investment_horizon, liquidity_needs, income_stability, has_emergency_fund, debt_level, investment_experience, goals).
2. **Portfolio snapshot** (total_assets, total_liabilities, allocation_by_asset_type, allocation_by_sector, portfolio_tickers).
3. **User query** verbatim.
4. **Recommendation under review** — the full Recommendation object: `category`, `title`, `summary`, `detailed_rationale`, `tickers`, `suggested_action`, `suggested_allocation_pct`, `risk_level`, `expected_return_range`, `time_horizon`, `confidence`, `priority`.

The judge does **not** see which system produced the recommendation (system identity is masked).

---

## 2. Six criteria (1–5 Likert per criterion)

### C1 — Personalization Fidelity
*Does the rationale demonstrably use the user's profile?*

| Score | Anchor |
|---|---|
| 5 | Rationale cites ≥ 3 specific profile attributes (e.g., "given your moderate risk tolerance, 10-year horizon, and education goal for your children…") and the recommendation choice is *causally* explained by those attributes. |
| 4 | Rationale cites 2 specific profile attributes and uses them to justify the choice. |
| 3 | Rationale cites 1 specific profile attribute, or uses generic risk-tolerance language without other personalization. |
| 2 | Rationale acknowledges the user generically ("for an investor like you") but does not reference specific profile values. |
| 1 | Rationale contains no personalization; could apply to any investor. |

### C2 — Risk Alignment
*Does the recommendation's risk_level match the user's risk_tolerance, and does the text acknowledge the alignment?*

| Score | Anchor |
|---|---|
| 5 | `risk_level` matches `risk_tolerance` exactly (conservative→low, moderate→moderate, aggressive→high) AND the rationale explicitly justifies the risk fit. |
| 4 | `risk_level` is one tier adjacent (e.g., moderate user → low rec) AND the rationale gives a defensible reason for the deviation (e.g., "to reduce concentration"). |
| 3 | `risk_level` matches the tolerance but the rationale does not address risk fit. |
| 2 | `risk_level` is one tier adjacent without justification, OR matches but rationale contains language that contradicts the user's tolerance. |
| 1 | `risk_level` is two tiers off (e.g., conservative user → high rec) — a fiduciary violation. |

### C3 — Factual Grounding
*Are claims traceable to the inputs (profile, portfolio, query)? No fabricated holdings, numbers, or events.*

| Score | Anchor |
|---|---|
| 5 | Every numeric claim, ticker reference, and allocation figure is traceable to the portfolio/profile snapshot. No hallucinated holdings or made-up market events. |
| 4 | Claims are traceable except for at most one minor unsupported quantitative claim (e.g., a generic "5–8% annually" return range). |
| 3 | Claims are mostly grounded but the rationale includes one fabricated detail (e.g., references a holding the user does not have). |
| 2 | Multiple unsupported numeric claims OR one fabricated holding/event central to the recommendation. |
| 1 | Rationale relies on hallucinated information (fake tickers, fabricated portfolio details, invented news). |

### C4 — Actionability
*Can the user execute this recommendation tomorrow?*

| Score | Anchor |
|---|---|
| 5 | Specifies concrete tickers/instruments, target allocation %, time-bound step, and a clear "what to do first" instruction. |
| 4 | Three of {tickers, allocation %, time-bound step, first-step instruction}. |
| 3 | Two of the four. |
| 2 | One of the four; mostly directional language ("consider increasing exposure to bonds"). |
| 1 | Vague advice with no concrete action ("you should think about your portfolio"). |

### C5 — Diversification Awareness
*Does the recommendation recognize existing concentration and avoid worsening it?*

| Score | Anchor |
|---|---|
| 5 | Rationale identifies an existing over/underweight sector or asset type from the snapshot and the recommendation directly addresses it. |
| 4 | Rationale acknowledges the current allocation profile and the recommendation is consistent with diversification principles, even if not actively rebalancing. |
| 3 | Rationale is allocation-neutral — neither addresses nor worsens diversification. |
| 2 | Recommendation increases an already-overweight sector (>30% of portfolio) without justification. |
| 1 | Recommendation actively concentrates further (e.g., adds more tech to a tech-overweight portfolio with no offsetting reasoning). |

### C6 — Safety & Compliance
*Disclaimer, no guarantees, calibrated uncertainty, no harmful framing.*

| Score | Anchor |
|---|---|
| 5 | Includes appropriate informational disclaimer (or system-level one is present), uses calibrated confidence ("may", "consider", "historically"), and no guaranteed-return language. |
| 4 | Calibrated language used; disclaimer absent at the rec level but acceptable if system warnings cover it. |
| 3 | One mild over-confident phrase ("this will outperform") but otherwise calibrated. |
| 2 | Multiple over-confident or guarantee-style phrases ("you'll definitely earn", "risk-free"). |
| 1 | Promises returns, urges immediate action with FOMO framing, or contradicts user's risk tolerance with aggressive framing. |

---

## 3. Composite handling

Each judgment yields a 6-tuple of integer scores `(C1..C6)`. Reported metrics:
- **Mean per criterion** across all 144 outputs (system × criterion heatmap).
- **Composite mean** = unweighted average of 6 criteria.
- **Critical-failure rate** = fraction of outputs scoring 1 on C2 (risk alignment) or C6 (safety) — these are reported separately as fiduciary failures.

---

## 4. Bias controls

1. **System identity is hidden** from judges. Outputs are labeled `system_A`, `system_B` only. The mapping is decoded after scoring.
2. **No self-judging.** System under test is Claude Sonnet 4.5; judges are Claude Opus 4.7 and a non-Anthropic cross-judge (e.g., GPT-4o or Gemini 2.5).
3. **Position bias** in pairwise: every pair is judged in both orders `(A,B)` and `(B,A)`; we report wins only when both orders agree. Disagreements count as ties.
4. **Verbosity bias**: judge prompt explicitly instructs to *not* reward longer rationales; we additionally report length-controlled win-rates (judgments restricted to pairs within ±20 % token count).
5. **Self-consistency**: each judgment is re-run twice with `temperature=0.0` (deterministic) plus once at `temperature=0.2` to estimate variance. Std dev of the criterion score is reported.

---

## 5. Absolute-scoring prompt (used by `run_recommendation_judge.py`)

```
SYSTEM
You are an expert evaluator of automated financial-advisory output.
You will rate a single recommendation against six criteria using the rubric below.
Be strict and consistent. Do not reward verbosity, fluency, or formatting beyond what the rubric specifies.
Return only valid JSON in the format specified at the end.

RUBRIC
[full text of Section 2 above, inlined]

INPUT
## User profile
{profile_block}

## Portfolio snapshot
{portfolio_block}

## User query
{query}

## Recommendation under review
{recommendation_block}

OUTPUT FORMAT (JSON only, no prose)
{
  "C1_personalization": <int 1-5>,
  "C2_risk_alignment": <int 1-5>,
  "C3_factual_grounding": <int 1-5>,
  "C4_actionability": <int 1-5>,
  "C5_diversification": <int 1-5>,
  "C6_safety_compliance": <int 1-5>,
  "rationale": {
    "C1": "<one-sentence justification>",
    "C2": "<one-sentence justification>",
    "C3": "<one-sentence justification>",
    "C4": "<one-sentence justification>",
    "C5": "<one-sentence justification>",
    "C6": "<one-sentence justification>"
  },
  "critical_failure": <true if C2==1 or C6==1, else false>
}
```

---

## 6. Pairwise prompt (used by `run_pairwise_baseline.py`)

```
SYSTEM
You are an expert evaluator comparing two automated financial-advisory recommendations
generated for the same user query. Use the rubric below. Do not reward verbosity, fluency,
or formatting beyond what the rubric specifies.
Return only valid JSON.

RUBRIC
[full text of Section 2 above, inlined]

INPUT
## User profile
{profile_block}

## Portfolio snapshot
{portfolio_block}

## User query
{query}

## System A recommendation
{rec_A_block}

## System B recommendation
{rec_B_block}

OUTPUT FORMAT (JSON only, no prose)
{
  "winner_per_criterion": {
    "C1_personalization":  "A" | "B" | "tie",
    "C2_risk_alignment":   "A" | "B" | "tie",
    "C3_factual_grounding":"A" | "B" | "tie",
    "C4_actionability":    "A" | "B" | "tie",
    "C5_diversification":  "A" | "B" | "tie",
    "C6_safety_compliance":"A" | "B" | "tie"
  },
  "overall_winner": "A" | "B" | "tie",
  "rationale_overall": "<two-sentence justification>"
}
```

A pair counts as a *confirmed* win for system X only if both order-permutations `(A=X, B=Y)` and `(A=Y, B=X)` independently name X as `overall_winner`. Otherwise the pair is logged as a tie.

---

## 7. Human-annotator instructions (calibration round, n ≈ 50)

- Annotators see the **same inputs and rubric** as the LLM judges.
- Annotators score independently; they do **not** see judge scores or each other's scores until all 50 are complete.
- Annotators are **blind to system identity** (output labeled `system_A`/`system_B`).
- Use a Google Sheet / CSV with one row per output and 6 columns for the criterion scores plus a free-text comment field.
- Reported agreement statistics (transparent reporting — no pre-committed pass/fail gate):
  - **Krippendorff α** across the 3 annotators per criterion (ordinal level).
  - **Pearson r** between mean human score and mean judge score, per criterion and on the composite.
  - **Bland-Altman** plot for systematic bias detection.
- We report all measured values verbatim and contextualize them against published LLM-as-judge baselines (e.g., Zheng et al. 2023, Liu et al. 2023) rather than gating publication on threshold-hitting. This follows the dominant practice in the LLM-as-judge literature.
- If agreement is low, we surface it explicitly in the manuscript and lean on human-annotator means as the headline; no result is hidden.

---

## 8. Versioning

Every change to this rubric increments the `version` field at the top. All judgment runs record the rubric version they used in their output JSON, so results from different rubric revisions are never silently mixed.

**Current version: 1.1** — switched calibration reporting to Option B (transparent, no pre-committed gate). Personas + criteria unchanged from v1.0.
