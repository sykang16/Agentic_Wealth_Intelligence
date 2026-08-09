# Phase 1 diagnostic: RAG top-hit similarity distribution

n = 18 (persona x query_type) pairs; RAG top-k = 5

## Distribution over top-hit similarity

| statistic | value |
|---|---|
| min    | 0.000 |
| Q1     | 0.346 |
| median | 0.395 |
| Q3     | 0.425 |
| max    | 0.494 |
| mean   | 0.360 |

## Gating rate at candidate thresholds

For each candidate threshold t, we count how many (persona, query) pairs
would have `top_score < t` (i.e., RAG context would be gated out).

| threshold | gated (n) | gated (fraction) |
|---|---|---|
| 0.40 | 10/18 | 55.6% |
| 0.45 | 16/18 | 88.9% |
| 0.50 | 18/18 | 100.0% |
| 0.55 | 18/18 | 100.0% |
| 0.60 | 18/18 | 100.0% |
| 0.65 | 18/18 | 100.0% |
| 0.70 | 18/18 | 100.0% |

## Per-query top scores

| persona | query_type | top_score | top_source |
|---|---|---|---|
| P1_conservative_retiree | open_ended | 0.494 | Wealth Intelligence Research |
| P1_conservative_retiree | rebalance | 0.435 | Wealth Intelligence Research |
| P1_conservative_retiree | sector_specific | 0.304 | Wealth Intelligence Research |
| P2_conservative_early_career | open_ended | 0.418 | Wealth Intelligence Research |
| P2_conservative_early_career | rebalance | 0.389 | Wealth Intelligence Research |
| P2_conservative_early_career | sector_specific | 0.387 | Wealth Intelligence Research |
| P3_moderate_mid_career | open_ended | 0.420 | Wealth Intelligence Research |
| P3_moderate_mid_career | rebalance | 0.429 | Vanguard |
| P3_moderate_mid_career | sector_specific | 0.416 | Vanguard |
| P4_moderate_pre_retirement | open_ended | 0.425 | Wealth Intelligence Research |
| P4_moderate_pre_retirement | rebalance | 0.363 | Wealth Intelligence Research |
| P4_moderate_pre_retirement | sector_specific | 0.477 | Wealth Intelligence Research |
| P5_aggressive_young_professional | open_ended | 0.395 | Wealth Intelligence Research |
| P5_aggressive_young_professional | rebalance | 0.394 | Wealth Intelligence Research |
| P5_aggressive_young_professional | sector_specific | 0.000 | None |
| P6_aggressive_experienced_consultant | open_ended | 0.385 | Wealth Intelligence Research |
| P6_aggressive_experienced_consultant | rebalance | 0.346 | Wealth Intelligence Research |
| P6_aggressive_experienced_consultant | sector_specific | 0.000 | None |

## Interpretation guide

- A threshold that gates 0% of queries is too permissive to move the needle.
- A threshold that gates 100% is equivalent to disabling RAG (already tested as B2).
- Aim for a threshold in the 20-60% gating band so the experiment is informative.
- Prior report flagged P1/open_ended (top_score 0.43) as the failure case; a
  useful threshold should gate this case at a minimum.
