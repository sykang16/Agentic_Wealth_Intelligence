# Draft Table 5 — Compressed §5.2–§5.6 unit-test summary

**Goal.** Replace four sub-sections of the current manuscript (Asset Agent §5.2, Profiling §5.3, Recommendation Engine Scoring + Quality §5.4, Orchestrator §5.5, Adversarial Safety §5.6) with one summary table plus 1–2 paragraphs. This frees ~1.5 ACM 2-col pages for the new §5.5 Context Overriding section without exceeding the 8-page limit.

The detailed test breakdowns and methodology stay available in the supplementary appendix / public repo, linked from the table caption.

---

## Proposed Table 5 (replaces former §§5.2–5.6 Tables)

```latex
\begin{table}[h]
\centering
\caption{Deterministic offline test suite summary. All tests run without LLM calls;
detailed methodology and per-test results available at \url{<<repo_url>>}.}
\label{tab:deterministic_tests}
\begin{tabular}{lcc l}
\toprule
\textbf{Module} & \textbf{Tests} & \textbf{Pass} & \textbf{Headline finding} \\
\midrule
Asset Agent (Module A)              &  68 & 100\%  & Net worth verified to \$705{,}764.50 within 0.01\% relative tolerance; 9 query types correctly dispatched \\
Profiling Module (Module B)         &  73 & 100\%  & 14-slot extraction parser handles JSON / markdown / null / list / boolean cases; state-machine completion is monotonically non-decreasing \\
Recommendation Scoring (Ranker)     &  53 & 100\%  & Composite formula $0.40 R + 0.35 A + 0.25 D$ verified algebraically; 9 risk-tolerance combinations correct \\
Recommendation Quality (Engine)     &  60 & 100\%  & Risk-filter, sector exclusion, threshold suppression, and 4-warning generation all correct under 8 schema-level scenarios \\
Orchestrator (Routing + Multi-turn) &  37 & 100\%  & Intent dispatch, 4-node error isolation, and multi-turn profile state threading verified \\
Adversarial Safety                  & 100 & 100\%  & 0\% violation rate under 300-product risk-injection attack; data-absence returns empty list, not nearest-match fallback \\
\midrule
\textbf{Total}                      & \textbf{391} & \textbf{100\%} & \\
\bottomrule
\end{tabular}
\end{table}
```

(Note: total is 391, not 440 as in the current manuscript. The current 440 includes the 49 IntentRouter tests counted under §5.1 + the comparative experiments. Verify the count when porting.)

## Proposed paragraph (replaces §5.2–§5.6 narrative; ~150 words)

```
The deterministic offline test suite (Table~\ref{tab:deterministic_tests}) verifies
the structural correctness of every module that does not depend on LLM
generation: portfolio aggregation, profile slot-filling and the conversational
state machine, recommendation scoring and filtering, orchestrator routing
across four agents, and adversarial-safety guarantees. All 391 tests pass
in under one second of total wall time, with no external API calls, network
dependencies, or non-determinism. Methodology and per-test breakdowns are
published with the source code at \url{<<repo_url>>}. The remainder of this
section focuses on the two evaluations that admit non-deterministic behaviour
and therefore require dedicated treatment: the IntentRouter baseline
comparison (§5.\ref{sec:baseline}) and the LLM-layer quality evaluation
(§5.\ref{sec:llm_quality}). Together these establish that WealthNexus is
both structurally correct and qualitatively competitive on dimensions that
the deterministic suite cannot reach.
```

## What's preserved vs deferred

**Preserved in compressed Table 5:**
- Total test counts and pass rates per module (the most paper-grade evidence).
- One-line headline finding per module.
- Reference to the public repo for full methodology.

**Deferred to supplementary / repo:**
- Per-test category breakdowns (e.g. "9 query types in AssetQueryInterface").
- Specific numeric tolerances (\$705,764.50 stays in the table because it is the most concrete claim; other tolerances move out).
- The per-fold variance discussion for FinBERT cross-validation (already lives in §5.7 baseline comparison and stays there).

## Page-budget impact

Current §5.2–§5.6 in the preprint occupy approximately 2.3 ACM 2-col pages including 5 tables. Compressing to one table + one paragraph reclaims approximately 1.7 pages. The new §5.5 Context Overriding section is sized for ~1.5 ACM 2-col pages (1 setup paragraph, 4 tables, 1 paired-output exhibit, 4 short result paragraphs). Net change: roughly break-even on length, with reviewer-impact concentrated on the Context Overriding finding.

## Cross-references that need updating

After the renumber, all references to the old §5.2–§5.6 must be updated:

- §5.7 Baseline Comparison stays where it is; its explicit cross-reference "Section~5.7" in Literature Review §2 stays valid.
- §5.8 Scoring Ablation stays where it is; the same cross-reference rule applies.
- The Aggregate Summary table at the bottom of §5 must update its row labels and totals.
- The Conclusion paragraph that lists "440 offline tests" must be re-stated either as "391 deterministic offline tests + the new LLM-layer evaluation in §5.5" or rolled into a new aggregate phrase.
