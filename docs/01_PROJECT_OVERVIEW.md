# Agentic Wealth Intelligence System

## Project Overview

## What We're Building
This project implements a **Multi-Agent Wealth Intelligence System** where specialized AI agents collaborate to provide comprehensive financial advisory services.
Agentic Wealth Intelligence system with 3 modules:
1. Asset Management (holistic view + LLM queries)
2. Investment Profiling (conversational slot-filling)
3. Real-time Recommendations (RAG + MCP)

### Problem Statement
- **Current Challenge**: Financial data is fragmented across multiple platforms
- **Gap**: Simple data visualization without actionable insights
- **Solution**: Evolve from passive dashboards to proactive AI-driven financial advisor

## Core Business Goals
1. **Holistic Asset Management**: Unified view of user's complete financial status and provides LLM-powered natural language interface
- Example
```
User: "What's my current liquidity ratio?"
System: "Your liquidity ratio is 0.35. You have $15,000 in liquid assets 
        (cash + checking) against $43,000 in total current assets. 
        This is below the recommended 0.5 threshold."
```

2. **Intelligent Profiling**: Natural conversation-based investment tendency assessment and advise based on profiling result and real asset allocation
- Example
```
Agent: "Tell me about your investment goals."
User: "I want to save for retirement but I'm worried about market crashes."
Agent: "I understand you're risk-averse. How far away is your retirement?"
User: "About 30 years."
Agent: "That's helpful! With a long time horizon, you can afford more 
       volatility. On a scale of 1-10, how would you feel seeing your 
       portfolio drop 20% in a year?"
```

3. **Real-Time Advisory**: Hybrid recommendations combining static knowledge with live market data
- Example
```
User: "Should I invest in tech stocks?"
System: 
- Considers: Your conservative risk profile + 30-year horizon +
             current 60% tech allocation
- Retrieves: Latest FOMC minutes, sector analysis from Vector DB
- Fetches: Current tech sector performance via yfinance (live quotes + news)
- Recommends: "Given your already high tech exposure (60%), consider
              diversifying. Recent Fed signals suggest rising rates,
              which historically pressure tech valuations..."
```

## System Architecture

**LangGraph Hybrid Supervisor Routing** — the orchestrator uses a two-tier Router (keyword fast-path + LLM fallback) that feeds into a flat outer graph. Recommendation queries are handled by an inner Supervisor subgraph that iteratively gathers portfolio and profile context before calling the engine.

### Outer Orchestrator Graph

```
                 ┌──────────────────────────────────────────────────┐
     User ──────►│                    Router                        │
                 │  Tier 1: Keyword match  (fast, no LLM call)      │
                 │  Tier 2: LLM fallback   (T=0, max_tokens=20)     │
                 └─────────────────────┬────────────────────────────┘
                                       │ conditional routing
       ┌───────────────────────────────┼────────────────┬──────────────────┐
       │ portfolio_query               │ profiling      │ recommendation   │ general
       ▼                              ▼                ▼                  ▼
┌─────────────┐           ┌──────────────┐   ┌──────────────────┐  ┌──────────┐
│  Portfolio  │           │  Profiling   │   │    Recommend     │  │ General  │
│  Module A   │           │   Module B   │   │  [Subgraph]      │  │  (LLM)   │
│  AssetAgent │           │ slot-filling │   │  Supervisor loop │  │ fallback │
│  net worth  │           │  13 slots    │   │  Portfolio+Prof  │  │          │
│  allocation │           │              │   │  +RAG+live data  │  │          │
└──────┬──────┘           └──────┬───────┘   └────────┬─────────┘  └────┬─────┘
       └─────────────────────────┴───────────────────┴─────────────────┘
                                                       │
                                              ┌────────▼────────┐
                                              │     Respond     │
                                              │ append history  │──► END
                                              └─────────────────┘
```

### Recommendation Supervisor Subgraph

Embedded as the `recommend` node in the outer graph. The Supervisor iteratively decides which context to gather before calling the engine.

```
  outer state ──► Supervisor (LLM decision · guard: steps <= 5)
                       │                  │                 │
               portfolio_fetch    profiling_fetch    recommend synthesis
               AssetAgent         ProfilingAgent     enriched query
               .process()         .get_profile_      + RAG (ChromaDB)
                                  summary()          + live data (MCP
                                                       -> yfinance)
                       └──────────────────┴─────────────────┘
                                   loops back to Supervisor
                                               │
                                          finish (steps >= 5
                                          or context complete)
                                               │
                                        outer Respond node
```