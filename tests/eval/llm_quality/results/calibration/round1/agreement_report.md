# Phase C - Human Calibration Agreement Report

- Recommendations annotated: **50**
- Annotators: **pi, student** (2)

## Inter-annotator agreement

We report raw agreement rate, Cohen's kappa (linear-weighted, ordinal), Spearman rho (primary for ordinal Likert data), Pearson r (secondary), and Krippendorff alpha. Raw agreement and Spearman rho are the most interpretable statistics when marginal distributions are skewed (the kappa paradox: high agreement, low kappa).

### Raw agreement rate per pair

**pi__vs__student**

| Criterion | n | Exact | Within-1 |
|---|---|---|---|
| C1_personalization | 50 | 86.0% | 98.0% |
| C2_risk_alignment | 50 | 90.0% | 94.0% |
| C3_factual_grounding | 50 | 56.0% | 88.0% |
| C4_actionability | 50 | 28.0% | 82.0% |
| C5_diversification | 50 | 42.0% | 90.0% |
| C6_safety_compliance | 50 | 38.0% | 82.0% |

### Cohen's kappa (linear-weighted) per pair

**pi__vs__student**

| Criterion | kappa | Spearman rho | Pearson r |
|---|---|---|---|
| C1_personalization | -0.031 | -0.053 | -0.050 |
| C2_risk_alignment | 0.829 | 0.916 | 0.855 |
| C3_factual_grounding | 0.186 | 0.181 | 0.197 |
| C4_actionability | 0.109 | 0.357 | 0.314 |
| C5_diversification | 0.180 | 0.393 | 0.383 |
| C6_safety_compliance | 0.074 | 0.060 | 0.135 |

### Krippendorff alpha (interval, all annotators)

| Criterion | alpha |
|---|---|
| C1_personalization | -0.058 |
| C2_risk_alignment | 0.857 |
| C3_factual_grounding | 0.089 |
| C4_actionability | -0.037 |
| C5_diversification | 0.129 |
| C6_safety_compliance | -0.149 |

## Judge-vs-Human correlation

Spearman rho is the primary statistic (ordinal Likert); Pearson r is reported for reference.

| Criterion | n | Spearman rho | Pearson r | mean(H-J) | LoA low | LoA high |
|---|---|---|---|---|---|---|
| C1_personalization | 50 | 0.298 | 0.344 | 0.110 | -0.555 | 0.775 |
| C2_risk_alignment | 50 | 0.860 | 0.710 | 0.120 | -1.543 | 1.783 |
| C3_factual_grounding | 50 | 0.378 | 0.335 | -0.210 | -1.332 | 0.912 |
| C4_actionability | 50 | 0.299 | 0.267 | -0.520 | -1.756 | 0.716 |
| C5_diversification | 50 | 0.759 | 0.730 | 0.430 | -0.708 | 1.568 |
| C6_safety_compliance | 50 | 0.200 | 0.173 | -0.470 | -1.681 | 0.741 |

## Composite agreement: Spearman rho = **0.612** (Pearson r = 0.695)