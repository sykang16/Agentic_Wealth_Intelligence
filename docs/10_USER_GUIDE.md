# User Guide

## Getting Started

After starting the Streamlit app (`python -m streamlit run ui/streamlit_app.py`), you'll see a multi-tab interface with a sidebar for user selection.

### Sidebar

Select a user portfolio from the dropdown. The sidebar shows:
- User info (name, age, occupation, income)
- Quick stats (net worth, liquidity ratio)
- Example queries you can click to use

---

## Tabs

### 1. Dashboard

Overview of the selected user's financial position:
- **Metrics cards**: Net worth, total assets, liabilities, savings rate
- **Asset allocation pie chart**: How assets are distributed
- **Net worth breakdown**: Visual breakdown of assets vs liabilities
- **Top holdings bar chart**: Largest investment positions

### 2. AI Advisor

Unified chat interface powered by the multi-agent orchestrator. Type any message and the system automatically routes it to the right module:

- **"What's my net worth?"** — Routes to Portfolio module
- **"Help me build my profile"** — Routes to Profiling module
- **"What should I invest in?"** — Routes to Recommendation module
- **"Hello!"** — General conversation

Each response shows which module handled the message. Conversation history persists across turns.

### 3. Ask Questions

Direct portfolio query interface. Ask natural language questions about your portfolio:
- Net worth and asset breakdown
- Asset and sector allocation
- Holdings and gains/losses
- Liquidity analysis
- Real estate and account details

Responses include relevant visualizations (charts, tables).

### 4. Investment Profile

Build your personalized investment profile through a guided conversation:

1. Click **Start New Profile** or **Continue Profile**
2. Answer the advisor's questions about:
   - Risk tolerance
   - Investment horizon
   - Experience level
   - Financial goals
   - Preferences (ESG, sectors)
3. Progress bar shows completion percentage
4. Completed profiles improve recommendation quality

### 5. Recommendations

Generate AI-powered investment recommendations:

1. Optionally enter a specific query
2. Select AI model and number of recommendations
3. Click **Generate Recommendations**
4. Review prioritized recommendations with:
   - Category (buy, sell, rebalance, diversify, etc.)
   - Risk level and expected returns
   - Specific tickers and suggested actions
   - Confidence and relevance scores

### 6. Knowledge Search

Search the financial knowledge base (RAG-powered):
- ETF fact sheets and profiles
- FOMC meeting summaries
- Investment guides
- Collected news articles

Also includes live market data (fetched via Yahoo Finance, no API key required).

#### Price Data

Update and track historical prices for all portfolio holdings:

- **Update Prices** — fetches current prices from Yahoo Finance for every holding across all users and saves them to the portfolio file. All asset types are supported (stocks, ETFs, bonds, mutual funds, crypto).
- **Price History** — select any ticker to view a line chart and table of daily closing prices. History is retained for up to 365 days per symbol.

#### Document Expiration Policy

Every document in the knowledge base has an expiration date based on its **publish date** and type. Expired documents are automatically excluded from search results and recommendations even before cleanup.

| Document Type | Retention | Auto-extend on access |
|---|---|---|
| News Article | **7 days** | No |
| Market Analysis | **30 days** | +7 days per access |
| FOMC Minutes | **90 days** | +30 days per access |
| Research Report | **90 days** | +30 days per access |
| ETF Factsheet | **365 days** | No |
| Investment Guide | **365 days** | No |

**Why expiration matters:** Financial data becomes stale quickly. News older than a week is typically no longer relevant to current market conditions. Research and regulatory documents (FOMC, 10-K) remain useful longer. ETF factsheets and investment guides are considered reference material and kept for a full year.

**Auto-extend on access:** For document types marked with auto-extend, each time a document is retrieved in a search or recommendation, its expiry date is pushed forward. This keeps frequently-used research alive longer without manual intervention.

**Cleanup Expired Documents:** Removes physically expired chunks from the ChromaDB index. This is optional — expired documents are already hidden from search — but running it periodically keeps the index lean and improves search performance.

### 7. Holdings

Detailed view of all investment holdings:
- Holdings by sector (bar chart)
- Unrealized gains and losses (chart)
- Complete holdings table with all metrics

---

## Tips

- Complete your investment profile before generating recommendations for more personalized results
- The AI Advisor tab is the easiest way to interact — just type naturally
- Use the Knowledge Search to explore financial concepts before making decisions
- Configure optional API keys in `.env` for live market data
