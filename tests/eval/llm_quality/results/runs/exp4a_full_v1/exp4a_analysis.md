# Experiment 4a: RAG similarity-threshold gating -- analysis

## Run summary

- Run dir: `.\tests\eval\llm_quality\results\runs\exp4a_full_v1`
- Baseline (A vs B2 original): not provided

## Gating rates per system

| system | sessions | gated (n) | gated (%) | mean top-hit sim. |
|---|---|---|---|---|
| A_gated_040 | 36 | 10 | 27.8% | 0.297 |
| A_gated_045 | 36 | 22 | 61.1% | 0.297 |
| A_wealthnexus | 36 | 0 | 0.0% | 0.308 |
| B2_rag_blind | 36 | 0 | 0.0% | 0.000 |

## Pairwise confirmed-win results

Rates and Wilson 95% CIs are computed over available verdicts (i.e., verdicts where both order permutations were successfully judged). ``Unavail.`` counts verdicts dropped because at least one of the two order-permutation judgments failed to parse. `p` is the two-sided exact binomial p-value testing H0: p(A wins) = p(B wins) over decided verdicts (ties excluded).

| Pair (A vs B) | A/B/Tie | Unavail. | A rate (of avail.) | 95% CI | p (decided) |
|---|---|---|---|---|---|
| A_gated_040 vs A_wealthnexus | 9/16/47 | 0 | 12.5% | [6.7%, 22.1%] | 0.230 |
| A_gated_040 vs B2_rag_blind | 8/22/42 | 0 | 11.1% | [5.7%, 20.4%] | 0.016 |
| A_gated_045 vs A_wealthnexus | 13/14/45 | 0 | 18.1% | [10.9%, 28.5%] | 1.000 |
| A_gated_045 vs B2_rag_blind | 5/14/53 | 0 | 6.9% | [3.0%, 15.2%] | 0.064 |
| A_wealthnexus vs B2_rag_blind | 7/16/49 | 0 | 9.7% | [4.8%, 18.7%] | 0.093 |

## Per-criterion winner distribution

### A_gated_040 vs A_wealthnexus

| Criterion | A wins | B wins | Tie |
|---|---|---|---|
| C1_personalization | 36 | 47 | 61 |
| C2_risk_alignment | 5 | 28 | 111 |
| C3_factual_grounding | 41 | 38 | 65 |
| C4_actionability | 53 | 50 | 41 |
| C5_diversification | 24 | 39 | 81 |
| C6_safety_compliance | 2 | 15 | 127 |

### A_gated_040 vs B2_rag_blind

| Criterion | A wins | B wins | Tie |
|---|---|---|---|
| C1_personalization | 32 | 46 | 66 |
| C2_risk_alignment | 17 | 26 | 101 |
| C3_factual_grounding | 29 | 53 | 62 |
| C4_actionability | 32 | 71 | 41 |
| C5_diversification | 25 | 46 | 73 |
| C6_safety_compliance | 5 | 8 | 131 |

### A_gated_045 vs A_wealthnexus

| Criterion | A wins | B wins | Tie |
|---|---|---|---|
| C1_personalization | 35 | 42 | 67 |
| C2_risk_alignment | 7 | 16 | 121 |
| C3_factual_grounding | 43 | 39 | 62 |
| C4_actionability | 52 | 55 | 37 |
| C5_diversification | 38 | 26 | 80 |
| C6_safety_compliance | 3 | 10 | 131 |

### A_gated_045 vs B2_rag_blind

| Criterion | A wins | B wins | Tie |
|---|---|---|---|
| C1_personalization | 36 | 34 | 74 |
| C2_risk_alignment | 12 | 11 | 121 |
| C3_factual_grounding | 35 | 35 | 74 |
| C4_actionability | 36 | 64 | 44 |
| C5_diversification | 30 | 33 | 81 |
| C6_safety_compliance | 7 | 2 | 135 |

### A_wealthnexus vs B2_rag_blind

| Criterion | A wins | B wins | Tie |
|---|---|---|---|
| C1_personalization | 43 | 45 | 56 |
| C2_risk_alignment | 25 | 8 | 111 |
| C3_factual_grounding | 34 | 47 | 63 |
| C4_actionability | 45 | 62 | 37 |
| C5_diversification | 31 | 39 | 74 |
| C6_safety_compliance | 11 | 3 | 130 |

## Interpretation guide

For each candidate manuscript story:

- **Gate closes the A-B2 gap**: A_gated_XX vs B2 should be closer to 50/50 or A-favouring, compared to the baseline A vs B2 (36-session pre-experiment).
- **Gate does not hurt vs full RAG**: A_gated_XX vs A_wealthnexus should be close to 50/50 (many ties) or A_gated-favouring; a strong A_wealthnexus victory would suggest gating discards useful context.
- **Dose-response**: as the threshold rises 0.40 -> 0.45, gating rate rises and, if the mechanism is retrieval-quality-driven, the A_gated_XX vs B2 rate should shift toward B2 (equivalent to disabling RAG).
