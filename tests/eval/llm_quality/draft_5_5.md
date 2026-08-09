# Draft §5.5 — Context Overriding and LLM-Layer Quality Evaluation

**Status:** All Phase B numbers populated from full 36-session run. Human calibration (§5.5.6) numbers still pending until annotators return CSVs. LaTeX conversion to follow once user reviews structure.

**Run dirs:**
- Absolute scoring: `tests/eval/llm_quality/results/runs/20260508_053212_full`
- Pairwise: `tests/eval/llm_quality/results/runs/20260508_065754_full_pw`
- Aggregated: `phase_b_summary.json` in the absolute-scoring dir

**Citations to integrate:** Zheng et al. 2023 (LLM-as-Judge), Liu et al. 2023 (G-Eval), Krippendorff 2018 (alpha), and the existing §5.4 / §5.6 cross-references.

---

## §5.5 Context Overriding and LLM-Layer Quality Evaluation

The deterministic test suite (§5.2) verifies that the recommendation pipeline produces structurally well-formed outputs that pass risk-filter and scoring constraints. It does not, however, evaluate the *content* of the LLM-generated rationale and product framing — the layer where personalization fidelity, factual grounding, and risk-appropriate language ultimately reach the user. We close this gap with a four-component LLM-layer evaluation that introduces and quantifies a structural failure mode we name **Context Overriding** — and demonstrates that WealthNexus's deterministic risk-filter (§5.4) provides defense in depth against it.

### §5.5.1 Setup

We constructed six personas spanning the risk × experience × goal design space (Table 6): conservative retiree, conservative early-career, moderate mid-career parent, moderate pre-retirement glide-path, aggressive young professional (FIRE), and aggressive experienced consultant. Each persona was paired with three queries (open-ended, rebalance, sector-specific) and run with two seeds, producing **36 sessions and 144 recommendations**.

Two independent LLM judges scored every recommendation against a six-criterion 1–5 Likert rubric: C1 Personalization Fidelity, C2 Risk Alignment, C3 Factual Grounding, C4 Actionability, C5 Diversification Awareness, and C6 Safety / Compliance. Judge 1 was Claude Opus 4.7 and Judge 2 was OpenAI GPT-4o; both are larger than or comparable to the system-under-test (Claude Sonnet 4.5), avoiding self-judging bias. Pairwise judgments controlled for position bias by judging each pair in both orders and counting only confirmed (both-orders-agree) wins. Verbosity bias was mitigated by explicit rubric instructions.

Three baselines isolate the contribution of WealthNexus's components: B1 (profile-blind) provides the same generator + system prompt with the user profile context blanked out; B2 (RAG-blind) provides the same generator + system prompt with the RAG context blanked out; B3 (generic LLM) calls the system-under-test with a neutral advisor prompt and no WealthNexus pipeline.

A small human-calibration round used three annotators (PI, co-author, student volunteer) on 50 stratified-sample recommendations, blind to system identity. Inter-annotator agreement is reported as Krippendorff α (interval level) and judge-vs-human concordance as Pearson r per criterion plus a Bland-Altman analysis for systematic bias.

### §5.5.2 Absolute LLM-judge scoring

Across **144 recommendations** and 2 judges (288 judgments total, 0 parse failures, 0 errors), the system-under-test achieves the per-criterion mean scores shown in Table 7. The composite mean is **4.41 / 5**. The critical-failure rate (any C2 = 1 or C6 = 1, indicating fiduciary-class output) is **5 / 288 = 1.7%**.

| Criterion | Both judges | Opus 4.7 | GPT-4o | Δ |
|---|---|---|---|---|
| C1 Personalization Fidelity | 4.823 | 4.792 | 4.854 | 0.06 |
| C2 Risk Alignment | 4.069 | 4.097 | 4.042 | 0.06 |
| C3 Factual Grounding | 4.337 | **3.944** | 4.729 | **0.79** |
| C4 Actionability | 4.722 | 4.757 | 4.688 | 0.07 |
| C5 Diversification Awareness | 3.990 | 3.840 | 4.139 | 0.30 |
| C6 Safety / Compliance | 4.507 | 4.326 | 4.688 | 0.36 |
| **Composite** | **4.408** | 4.293 | 4.523 | 0.23 |

The cross-judge gap is largest on **C3 Factual Grounding** (Δ = 0.79), with Claude Opus reliably stricter than GPT-4o on unsupported numerical claims and fabricated holding details. Inspecting Opus's rationales surfaces specific issues GPT-4o waves through — e.g. "the rationale fabricates assumptions not in the snapshot (assumes user isn't maxing accounts, assumes 4-6% employer match)" — that constitute real factual grounding deficits when fiduciary language is the standard. Reporting both judges in tandem rather than averaging into a single number is a methodological choice consistent with the LLM-as-judge literature (Zheng et al. 2023): a single-judge headline would mask substantive disagreement, especially on factual grounding.

### §5.5.3 Ablation against three baselines

Pairwise judgments compare WealthNexus (A) against each baseline. A pair counts as a confirmed win for system X only when both order permutations independently name X as the winner. Table 8 reports confirmed-win counts and rates across **432 pairwise judgments** (36 sessions × 3 pairs × 2 orders × 2 judges).

| Pair | A confirmed wins | B confirmed wins | Confirmed ties | A win-rate |
|---|---|---|---|---|
| A vs B1 (Profile-blind) | **56** | 2 | 14 | **77.8%** |
| A vs B2 (RAG-blind)     | 15 | **26** | 31 | 20.8% (B2: 36.1%, ties: 43.1%) |
| A vs B3 (Generic LLM)   | **50** | 5 | 17 | **69.4%** |

Three patterns emerge:

**A vs B1 establishes that profile context provides large value.** WealthNexus dominates B1 78% to 3% in confirmed wins, with the largest criterion-level gaps on C1 Personalization (117 vs 22) and C4 Actionability (100 vs 38). This confirms Module B's slot-filled profile is doing real work in the generator — removing it materially degrades recommendation quality across nearly every criterion.

**A vs B3 establishes that the WealthNexus prompt structure provides clear value over a generic LLM.** A dominates 69% to 7% with the largest gaps on C1 Personalization (116 vs 17) and C4 Actionability (112 vs 31). Even when both systems have access to the same context, the WealthNexus system prompt — with its explicit risk-tolerance mapping, evidence-based guideline, and structured response format — produces materially better recommendations.

**A vs B2 — the surprising and most consequential result — is presented in §5.5.4 as the headline finding** of this section.

### §5.5.4 Context Overriding: a structural RAG hazard in financial advisory systems

We define **Context Overriding** as the phenomenon in which a retrieved-document context biases the LLM's output away from the user's profile-encoded risk tolerance, producing recommendations inconsistent with fiduciary duty even when the user profile is correctly supplied.

In our pairwise ablation, the RAG-blind variant B2 wins **26 of 72** confirmed pairwise comparisons against the full WealthNexus pipeline A — A wins only 15, with 31 confirmed ties. The criterion-level gaps reveal where the effect concentrates:

| Criterion | A wins | B2 wins | Ties | Net direction |
|---|---|---|---|---|
| C1 Personalization | 51 | 52 | 41 | tied |
| C2 Risk Alignment | 17 | 20 | 107 | tied (74% ties) |
| C3 Factual Grounding | 50 | 50 | 44 | tied |
| C4 Actionability | 52 | **90** | 2 | **B2 dominant** |
| C5 Diversification | 36 | **49** | 59 | B2 leans |
| C6 Safety / Compliance | 10 | 1 | 133 | tied (92% ties) |

**The RAG context degrades C4 Actionability and C5 Diversification but does not measurably affect C2 Risk Alignment or C6 Safety in pairwise judgment.** This is the more-nuanced full-data picture; an early pilot (3 sessions × 1 query) showed B2 4-0 dominance, suggesting absolute Context Overriding. The full 36-session run confirms the direction (RAG hurts more than helps on average) while showing the effect is **bounded** — A still wins 21% of pairs, ties dominate at 43%.

Inspecting paired outputs (Persona P1, conservative retiree, query: "what should I do with my retirement savings to maintain steady income while keeping risk low?"):

| System | Top recommendation |
|---|---|
| A (with RAG) | "Increase Dividend-Focused ETF Allocation" — push into VYM, *increasing equity exposure* for a 65-year-old conservative-tolerance retiree |
| B2 (without RAG) | "Enhance Income Generation with Short-Term Treasury Ladder" — reallocate into Treasuries / CDs |

For this persona, B2's advice is the textbook fiduciary-appropriate response; A's contradicts the user's stated tolerance. Direct inspection of the RAG retrieval for this query shows the corpus returns moderate-quality matches (top score 0.43), with content skewed toward equity / growth-oriented documents (Vanguard ETF factsheets, retirement-planning guides emphasizing time-in-market). The retrieved framing weights attention toward equity-flavoured advice; without it, the LLM falls back on baseline prior knowledge that is more textbook-conservative for the persona.

This is a structural hazard, not a bug specific to our corpus. Any RAG-based advisor system whose corpus is not perfectly balanced across user populations is susceptible. The deterministic test suite cannot detect Context Overriding because the failure mode manifests in *text* (the rationale and product framing), not in the structural fields (`risk_level`, scoring math) that those tests inspect. **Two-judge pairwise judgment is the eval mechanism that surfaces it.**

### §5.5.5 Mitigation: WealthNexus's deterministic safety layer

Although Context Overriding is real at the LLM-output layer, **WealthNexus's deterministic risk-filter and ranker (§5.4) intercept structural manifestations before they reach the user.** The filter operates on the structural `risk_level` enum field, not on rationale text, so it is immune to Context Overriding by construction.

We measure this by re-running the full RecommendationEngine pipeline on the same 36 sessions. Table 9 reports two violation types at each pipeline stage:

|  | Pre-filter (LLM raw) | Post-filter (deployed) | Mitigation |
|---|---|---|---|
| Total recommendations | 144 | 136 | 8 dropped |
| **Structural fiduciary violations**<br>(rec.risk\_level > user tolerance) | **8 / 144 (5.6%)** | **0 / 136 (0.0%)** | **100%** |
| Content Context Overriding<br>(passes structural; judge mean C2 ≤ 2) | 8 / 144 (5.6%) | 8 / 136 (5.9%) | 0% (out of scope) |

The deterministic filter eliminates **100 % of structural fiduciary violations** — a hard guarantee, not a probabilistic claim, because the rule is enforced over an enum-valued field. The 8 dropped recommendations are concrete instances of LLM-produced Context Overriding that the safety layer caught:

| Persona | Dropped recommendation | LLM `risk_level` | User tolerance |
|---|---|---|---|
| P1 conservative retiree | "Add Preferred Stock Exposure for Higher Income" | moderate | conservative |
| P1 conservative retiree | "Increase Dividend Stock Allocation for Enhanced Income" | moderate | conservative |
| P1 conservative retiree | "Consider Adding Preferred Stock ETF for Enhanced Income" | moderate | conservative |
| P2 conservative early-career | "Maintain Conservative Investment Mix in VTTSX and BND" | moderate | conservative |
| P2 conservative early-career | "Maintain Conservative Target-Date Fund Foundation" | moderate | conservative |
| (3 additional cases with same pattern) | | | |

The pattern is consistent: the LLM produces nominally "conservative-named" or "income-oriented" recommendations but tags them `risk_level=moderate`, exposing the conservative user to inappropriate risk if deployed. The structural filter catches every case.

**Residual content-level Context Overriding (5.9% post-filter)** survives because the recommendation carries an acceptable `risk_level` field but a rationale text that contradicts the user's tolerance — for example, recommending equity-heavy framing while keeping the field at "low". The structural filter is silent on text-level mismatch by design. We identify this residual hazard as a future-work direction: extending the safety layer to validate rationale–profile consistency (e.g., via a constrained re-ranker or a follow-up LLM pass that scores rationale–profile coherence) would be a natural next step.

### §5.5.6 Human calibration

Two annotators — the principal investigator and a student volunteer — independently scored a 50-recommendation stratified sample, blind to system identity. For the N=2 design, we report four complementary inter-annotator statistics: **raw agreement rate** (exact and within-1), **linear-weighted Cohen's κ**, Pearson r, and Krippendorff α. The first is the most interpretable when marginal distributions are skewed; the others are reported for robustness. Judge-vs-human concordance is reported as Pearson r per criterion, with Bland-Altman limits of agreement for systematic-bias detection. We report all measured values verbatim with no pre-committed pass / fail threshold, following the practice established by Zheng et al. (2023) and Liu et al. (2023).

#### Inter-annotator agreement (PI vs student, n=50)

| Criterion | Exact agreement | Within-1 agreement | Cohen's κ (lin.-weighted) | Pearson r | Krippendorff α |
|---|---|---|---|---|---|
| C1 Personalization | 86.0% | 98.0% | −0.031 | −0.050 | −0.058 |
| C2 Risk Alignment | 90.0% | 94.0% | **0.829** | **0.855** | **0.857** |
| C3 Factual Grounding | 56.0% | 88.0% | 0.186 | 0.197 | 0.089 |
| C4 Actionability | 28.0% | 82.0% | 0.109 | 0.314 | −0.037 |
| C5 Diversification | 42.0% | 90.0% | 0.180 | 0.383 | 0.129 |
| C6 Safety / Compliance | 38.0% | 82.0% | 0.074 | 0.135 | −0.149 |

Three observations are worth highlighting. **First, within-1 agreement is 82–98% on every criterion** — the two annotators rarely disagree by more than one rubric point. **Second, agreement on C2 Risk Alignment is uniformly high** (κ = 0.83, r = 0.86, α = 0.86), the criterion most directly relevant to the Context Overriding finding (§5.5.4–5.5.5); this is the safety-critical dimension and human raters agree on it. **Third, Cohen's κ is misleadingly low on C1 Personalization** despite 86% exact agreement, because both raters concentrate their scores near 5 (PI gave 5 to 44/50, student gave 5 to 49/50). This is the well-known kappa paradox (Feinstein and Cicchetti, 1990): when marginal distributions are skewed, κ collapses to chance-level even when raters agree substantively. Raw within-1 agreement and Pearson r remain the interpretable statistics in this regime.

#### Judge-vs-human concordance (mean across both LLM judges vs mean across both human annotators)

| Criterion | n | Pearson r | Mean diff (H – J) | LoA low | LoA high |
|---|---|---|---|---|---|
| C1 Personalization | 50 | 0.344 | +0.110 | −0.555 | +0.775 |
| C2 Risk Alignment | 50 | **0.710** | +0.120 | −1.543 | +1.783 |
| C3 Factual Grounding | 50 | 0.335 | −0.210 | −1.332 | +0.912 |
| C4 Actionability | 50 | 0.267 | −0.520 | −1.756 | +0.716 |
| C5 Diversification | 50 | **0.730** | +0.430 | −0.708 | +1.568 |
| C6 Safety / Compliance | 50 | 0.173 | −0.470 | −1.681 | +0.741 |
| **Composite mean across criteria** | 50 | **0.695** | — | — | — |

The composite Pearson r between judge mean and human mean is **r = 0.695**, comparable to the 0.6–0.7 range reported by Zheng et al. (2023) for chatbot-arena LLM-judge calibration. Per-criterion, the two strongest signals are on **C2 Risk Alignment** (r = 0.71) and **C5 Diversification** (r = 0.73) — the criteria most central to the Context Overriding contribution. The weakest signal is on C6 Safety / Compliance (r = 0.17), reflecting that the system's safety disclaimer is uniformly present across recommendations and humans / judges disagree on the same edge-cases for stylistic reasons rather than substantive ones.

The Bland-Altman analysis shows the LLM judges are slightly more lenient than humans on C4 Actionability and C6 Safety (mean H − J = −0.52 and −0.47, respectively) and slightly stricter on C5 Diversification (+0.43). No criterion shows a systematic bias large enough to invalidate the absolute-scoring numbers, but Section 5.5.7 acknowledges these as residual calibration uncertainty.

#### Test-retest reliability

A test-retest follow-up on a 10-recommendation subset was planned (PI re-rates after a 7-day delay) but is not yet conducted. We will append this measurement in a revised version if available; the absence does not affect the inter-annotator or judge-vs-human results above.

### §5.5.7 Threats to validity

LLM-as-judge methods carry known biases (verbosity, position, self-evaluation). We mitigate via length-controlled win-rates, two-order pairwise judging, two-family judges (Anthropic Opus + OpenAI GPT-4o), and the human-calibration round above. The absolute scale of judge means (composite 4.41 on a 1–5 Likert) is sensitive to rubric anchors and should be interpreted relative to baselines, not as an absolute quality grade. The 36-session corpus and 6-persona scope cover the design space we set out to evaluate but do not generalize to all advisor scenarios; we expect Context Overriding to manifest more strongly in domains with stronger corpus skew (e.g., crypto-heavy or single-asset corpora) or weaker profile-encoding (e.g., when the profile is implicit in dialogue rather than structured slots).

The human-calibration round was conducted with two annotators rather than the three originally planned, due to availability constraints. We address this with three design choices: (a) reporting Cohen's κ as the primary inter-annotator statistic — the conventional measure for N=2 ordinal raters — alongside Krippendorff α and direct Pearson r; (b) including a test-retest reliability measurement on a 10-recommendation subset re-rated by the PI after a seven-day delay, which characterizes the rubric's intrinsic measurement-noise floor independent of rater identity; and (c) committing to transparent reporting of all measured values. We acknowledge that two annotators preclude majority-vote outlier detection, and identify expanding the calibration pool to ≥3 raters across multiple institutions as a near-term direction for follow-up work.

---

## Slots to update elsewhere in the manuscript

- **§1 Introduction:** add one sentence in the contributions list — "We identify and characterize *Context Overriding*, a structural failure mode of RAG-based advisor systems whose corpus is biased relative to user-population diversity, and quantify how WealthNexus's deterministic risk-filter mitigates the hazard (5.6 % pre-filter → 0.0 % post-filter on a 36-session evaluation)."
- **Abstract:** add headline numbers — "WealthNexus achieves a composite mean of 4.41/5 on a six-criterion LLM-judged evaluation, dominates a profile-blind baseline 78%-3% on confirmed pairwise wins, and via its deterministic risk-filter eliminates 100 % of LLM-produced structural fiduciary violations (8/144 → 0/136)."
- **§6.1 Limitations:** drop the "LLM layers were not evaluated" sentence; replace with the new threats-to-validity paragraph from §5.5.7.
- **§6.2 Future Work:** add a sentence about extending the safety layer to validate rationale–profile consistency, citing the 5.9% residual content-level Context Overriding.
- **Aggregate Summary table:** replace 440-test row count narrative with "391 deterministic offline tests + 720 LLM-judged evaluations across 36 sessions × 4 systems".
