# Module C: Real-Time Product Recommendation

## Objective

Generate personalized investment recommendations by combining:
- User portfolio and risk profile
- Static financial knowledge (RAG / Vector DB)
- Live market data via MCP layer → yfinance (quotes + news headlines)

---

## Architecture Components

### 1. RAG Knowledge Base (Static)

Stored in ChromaDB (`./data/chroma`). Documents are chunked (512–1024 tokens) and embedded with `sentence-transformers`.

Seed documents loaded from `data/documents/`:
- ETF fact sheets and prospectuses
- FOMC meeting minutes
- Investment strategy guides
- Historical market analysis

#### Document Expiration Policy

| Document Type | Retention | Auto-extend on access |
|---|---|---|
| News Article | 7 days | No |
| Market Analysis | 30 days | +7 days per access |
| FOMC Minutes | 90 days | +30 days per access |
| Research Report | 90 days | +30 days per access |
| ETF Factsheet | 365 days | No |
| Investment Guide | 365 days | No |

Expired documents are excluded from search results automatically. Run **Cleanup Expired Documents** in the UI to physically remove them from the index.

### 2. Data Collectors (Knowledge Ingestion)

Collectors populate the RAG index with fresh external data. Managed by `DataCollectionManager`.

| Collector | Source | API Key | Data |
|---|---|---|---|
| `AlphaVantageCollector` | Alpha Vantage | Required | News sentiment (NEWS_SENTIMENT endpoint) |
| `AlphaVantageCollector` (ETF) | Yahoo Finance (yfinance) | None | ETF overviews, financials |
| `SECEdgarCollector` | SEC EDGAR | None | 10-K, 10-Q, 8-K filings |
| `NewsAPICollector` | NewsAPI.org | Required | Financial news articles |

### 3. Live Market Data — MCP Layer

No API key required. yfinance is the underlying data source, accessed through the MCP abstraction layer.

**Call chain at recommendation time:**

```
ContextBuilder._add_live_context()
  → HybridDataProvider.get_enriched_context_sync()   [sync wrapper; runs in clean thread]
    → MarketMCPServer.get_stock_quote(symbol)         [yfinance: price + change%]
    → MarketMCPServer.get_ticker_news(symbol)         [yfinance: recent headlines]
```

- Up to 10 portfolio tickers fetched for quotes; up to 5 for news (2 headlines each)
- Results injected as `live_market_context` into the LLM prompt
- 5-minute cache inside `MarketMCPServer` prevents redundant yfinance calls

**`MarketMCPServer` tools:**

| Tool | Description |
|---|---|
| `get_stock_quote(symbol)` | Current price, change%, volume via `yf.Ticker.history(period="2d")` |
| `get_etf_profile(symbol)` | Name, sector, P/E, dividend yield, 52-week range via `yf.Ticker.info` |
| `get_ticker_news(symbol, limit)` | Recent headlines via `yf.Ticker.news` |
| `get_multiple_quotes(symbols)` | Batch quotes, up to 10 symbols |

**`render_live_data_section()`** (UI — Knowledge Search tab):
- Calls yfinance directly (no MCP) for the interactive "Refresh Live Quotes" widget
- Displays current price, change%, volume for portfolio holdings

### 4. Recommendation Engine

`backend/src/recommendation/engine/engine.py`

```python
class RecommendationEngine:
    def generate_recommendations(
        self,
        user_id: str,
        query: str = "",
        max_recommendations: int = 5,
        include_live_data: bool = True,
    ) -> RecommendationResponse:
        # 1. Build aggregated context
        context = self.context_builder.build_context(
            user_id, query=query, include_live_data=include_live_data
        )
        # 2. Generate via LLM (Claude)
        # 3. Parse + rank recommendations
        # 4. Apply risk filter + composite scoring
        # 5. Return RecommendationResponse
```

#### Composite Scoring

```
composite = 0.40 * relevance_score
          + 0.35 * risk_alignment_score
          + 0.25 * diversification_score
```

Default minimum threshold: `0.55`. Recommendations below this are suppressed.

#### Risk Filtering

| User Risk Tolerance | Allowed Risk Levels |
|---|---|
| Conservative | LOW only |
| Moderate | LOW, MODERATE |
| Aggressive | LOW, MODERATE, HIGH |
| No profile | All levels (conservative defaults applied) |

---

## Implementation Files

```
backend/src/
├── mcp/
│   ├── servers/
│   │   ├── market_server.py       # MarketMCPServer: quotes, ETF profiles, news (yfinance)
│   │   ├── news_server.py         # NewsMCPServer: market news (Alpha Vantage)
│   │   ├── portfolio_server.py    # PortfolioMCPServer
│   │   └── base.py                # BaseMCPServer (FastMCP wrapper)
│   ├── integration/
│   │   ├── hybrid_data_provider.py  # Combines RAG + MCP live data; sync wrapper
│   │   └── data_normalizer.py       # Cross-source field normalization
│   └── schemas.py                   # StockQuote, ETFProfile, HybridDataResponse, MCPDataSource
└── recommendation/
    ├── engine/
    │   ├── engine.py              # Main orchestration + LLM call
    │   ├── context_builder.py     # Aggregates portfolio + RAG + MCP live data
    │   ├── ranker.py              # Composite score computation
    │   └── schemas.py             # RecommendationRequest/Response models
    ├── rag/
    │   ├── vector_store.py        # ChromaDB CRUD + expiry filtering
    │   ├── retriever.py           # Semantic search + re-ranking
    │   ├── freshness.py           # Per-type retention policy
    │   ├── initializer.py         # RAG setup, seed document loading
    │   └── schemas.py             # Document, DocumentMetadata, DocumentType
    └── collectors/
        ├── base.py                # BaseCollector, CollectorResult
        ├── alpha_vantage.py       # News (Alpha Vantage API) + ETF (yfinance)
        ├── sec_edgar.py           # SEC EDGAR filings
        ├── news_api.py            # NewsAPI articles
        └── manager.py             # DataCollectionManager
```

---

## Evaluation Results

All evaluation tests are offline (no LLM calls). See `tests/eval/` for details.

| Test File | Tests | Key Coverage |
|---|---|---|
| `test_recommendation_ranker.py` | 53 | Composite score, risk matrix, relevance grid |
| `test_recommendation_quality.py` | 60 | Risk filter, sector filter, warning generation, schema |
| `test_adversarial_safety.py` | 29 | Poisoned datasets (0% violation), relevance inflation guard, data-absence silence |
