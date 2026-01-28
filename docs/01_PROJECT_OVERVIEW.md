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
- Fetches: Current tech sector performance via MCP
- Recommends: "Given your already high tech exposure (60%), consider 
              diversifying. Recent Fed signals suggest rising rates, 
              which historically pressure tech valuations..."
```

## System Architecture
Multi-Agent Design - This system uses 4 specialized AI agents coordinated by an orchestrator:

```
┌─────────────────────────────────────────────────────┐
│          Orchestrator Agent (Main)                  │
│  - Routes user queries to appropriate agents        │
│  - Manages conversation flow                        │
│  - Aggregates results from multiple agents          │
└─────────────────────────────────────────────────────┘
                          ║
        ╔═════════════════╬═════════════════╗
        ║                 ║                 ║
┌───────▼──────┐  ┌──────▼───────┐  ┌─────▼────────┐
│ Asset Agent  │  │Profile Agent │  │ Advisor Agent│
│  (Module A)  │  │  (Module B)  │  │  (Module C)  │
└──────────────┘  └──────────────┘  └──────────────┘
```