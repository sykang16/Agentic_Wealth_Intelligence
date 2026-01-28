# Module C: Real-Time Product Recommendation

#### 🟡 Priority: MEDIUM

## Objective
Hybrid recommendations (static + live data)

## Architecture Components
1. Vector Database (Static Knowledge)**
- [ ] Store financial documents:
  - ETF prospectuses and fact sheets
  - FOMC meeting minutes
  - Investment strategy guides
  - Historical market analysis
- [ ] Semantic search capability
- [ ] Document chunking strategy (512-1024 tokens)
- [ ] Embedding model: OpenAI text-embedding-3 or similar
2. MCP Server (live)
- [ ] Connect to real-time data sources:
  - Market news APIs (Bloomberg, Reuters)
  - Stock/ETF price feeds
  - Economic indicators (Fed data, unemployment, CPI)
  - Sentiment analysis from financial news
- [ ] Implement MCP protocol for standardized data access
3. Recommendation Engine
- [ ] Freshness management:
  - Time-based retention policy for news (e.g., 30 days)
  - Periodic re-embedding of updated documents
- [ ] Hybrid search:
  - Semantic search for conceptual matching
  - Keyword search for specific products
- [ ] Re-ranking strategy to prioritize recent + relevant results

## Recommendation Logic
```python
class RecommendationEngine:
    async def generate_recommendation(
        self,
        user_profile: InvestmentProfile,
        user_portfolio: AssetPortfolio,
        query: str
    ) -> Recommendation:
        # 1. Retrieve relevant static documents from Vector DB
        static_context = await self.vector_db.search(query, k=5)
        
        # 2. Fetch live market data via MCP
        live_data = await self.mcp_client.get_market_data()
        
        # 3. Combine contexts and generate recommendation
        recommendation = await self.llm.generate(
            profile=user_profile,
            portfolio=user_portfolio,
            static_knowledge=static_context,
            market_data=live_data
        )
        
        return recommendation
```

## RAG Implementation
[Details on vector DB, retrieval, freshness]

## MCP Integration
[Details on servers, data sources]

## Implementation Files
```
src/
├── recommendation/
│   ├── __init__.py
│   ├── engine.py             # Main recommendation logic
│   ├── rag/
│   │   ├── vector_store.py   # Vector DB interface
│   │   ├── embeddings.py     # Document embedding
│   │   ├── retrieval.py      # Search & ranking
│   │   └── freshness.py      # Retention policy manager
│   ├── mcp/
│   │   ├── client.py         # MCP client
│   │   ├── servers/
│   │   │   ├── news.py       # News feed MCP server
│   │   │   ├── market.py     # Market data MCP server
│   │   │   └── sentiment.py  # Sentiment analysis
│   │   └── schemas.py
│   └── prompts/
│       └── recommendation_prompt.py
```

