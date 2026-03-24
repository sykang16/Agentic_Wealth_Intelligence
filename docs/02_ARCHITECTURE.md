# System Architecture

## Tech Stack
- **Language**: Python 3.12+
- **Agent Framework**: LangGraph 0.2+
- **LLM**: Anthropic Claude (Sonnet 4.5/4.6)
- **Data Models**: Pydantic v2
- **Vector DB**: ChromaDB
- **MCP Layer**: FastMCP + custom servers (abstraction over external data sources)
- **Market Data**: yfinance via `MarketMCPServer` (live quotes, ETF data, news headlines — no API key)
- **Testing**: pytest

## Module Interactions
User Query → Asset Management (Module A)
     ↓
Investment Profiling (Module B) → InvestmentProfile
     ↓
Recommendation Engine (Module C)
  ├─ Vector DB / RAG (static knowledge: ETF factsheets, FOMC minutes, guides)
  └─ MCP Layer → MarketMCPServer → yfinance (live quotes + news headlines)
     ↓
Personalized Recommendation

## Project Structure

```
agentic-wealth-intelligence/
├── .claude/
│   └── context.md
├── docs/                             # Documentation
├── backend/
│   └── src/
│       ├── agents/
│       │   └── orchestrator.py      # Main coordinator (LangGraph)
│       ├── multi_agent/
│       │   ├── graph.py             # LangGraph workflow
│       │   ├── state.py             # State definitions
│       │   └── routing.py           # Keyword intent routing
│       ├── api/                      # FastAPI endpoints
│       ├── asset_management/
│       │   ├── aggregator.py        # Portfolio aggregation
│       │   ├── calculator.py        # Financial metrics
│       │   ├── query_handler.py     # NL query handler
│       │   └── price_updater.py     # yfinance price fetching + history
│       ├── profiling/
│       │   └── agent.py             # Conversational slot-filling
│       ├── mcp/
│       │   ├── servers/
│       │   │   ├── market_server.py # MarketMCPServer → yfinance (quotes, news)
│       │   │   ├── news_server.py   # NewsMCPServer (Alpha Vantage news)
│       │   │   └── portfolio_server.py
│       │   ├── integration/
│       │   │   ├── hybrid_data_provider.py  # Sync+async MCP facade
│       │   │   └── data_normalizer.py
│       │   └── schemas.py           # StockQuote, ETFProfile, HybridDataResponse
│       ├── recommendation/
│       │   ├── engine/
│       │   │   ├── engine.py        # Main recommendation logic
│       │   │   ├── context_builder.py  # Aggregates portfolio+RAG+MCP live data
│       │   │   └── ranker.py        # Composite scoring
│       │   ├── rag/
│       │   │   ├── vector_store.py  # ChromaDB interface
│       │   │   ├── retriever.py     # Search & ranking
│       │   │   ├── freshness.py     # Document expiration policy
│       │   │   └── initializer.py   # RAG setup with seed documents
│       │   └── collectors/
│       │       ├── alpha_vantage.py # News sentiment (API key required)
│       │       ├── sec_edgar.py     # SEC filings
│       │       ├── news_api.py      # NewsAPI articles
│       │       └── manager.py       # Collector orchestration
│       └── common/
│           ├── models.py            # Pydantic schemas
│           └── llm_client.py
├── ui/
│   ├── streamlit_app.py             # 7-tab Streamlit app
│   └── styles.py                    # CSS
├── tests/
│   └── eval/                        # Offline evaluation suite (448 tests)
├── data/
│   ├── synthetic/
│   │   ├── portfolios.json          # Synthetic user portfolios
│   │   └── price_history.json       # Price history (yfinance, rolling 365d)
│   └── documents/                   # Seed RAG documents (FOMC, ETF guides)
└── requirements.txt
```