### `.claude/context.md` (Most Important for Claude Code)

# Project Context for Claude Code

## What We're Building
Agentic Wealth Intelligence system with 3 modules:
1. Asset Management (holistic view + LLM queries)
2. Investment Profiling (conversational slot-filling)
3. Real-time Recommendations (RAG + MCP)

## Tech Stack
- Python 3.11+, LangGraph, Anthropic Claude
- Pydantic v2, Chroma/Pinecone
- MCP for real-time data

## Code Style
- Type hints everywhere
- Pydantic for all data models
- Async/await for I/O
- pytest for testing
- Black formatting, 100 char lines

## File Structure
- [Architecture](../docs/02_ARCHITECTURE.md)

## Important Notes
- Use synthetic data (no real APIs)
- Follow slot-filling pattern for profiling
- Implement freshness policies for RAG

## Quick Links
- Architecture: [docs/02_ARCHITECTURE.md](../docs/02_ARCHITECTURE.md)
- Current module: docs/0X_MODULE_Y.md
- Phase plan: [docs/06_PHASES.md](../docs/06_PHASES.md)


## Current Phase
Phase 7: Integration & Testing (Complete)

## Completed
- [x] Phase 1: Foundation (project structure, data models, synthetic data)
- [x] Phase 2: Asset Management Module
  - [x] LLM Client (Anthropic Claude integration)
  - [x] Portfolio Aggregator (data loading and metrics)
  - [x] Query Interface (natural language processing)
  - [x] Visualization Engine (Plotly charts)
  - [x] Asset Agent (LangGraph-based)
  - [x] Streamlit UI
- [x] Phase 3: Investment Profiling Agent
  - [x] Profiling schemas (ConversationState, ProfileSlots, SlotStatus)
  - [x] LLM prompts (extractor, validator, question generator)
  - [x] Agent nodes (extractor, validator, question_gen, checker)
  - [x] LangGraph workflow (ProfilingAgent)
  - [x] Streamlit UI (Investment Profile tab)
  - [x] Unit and integration tests
- [x] Phase 4: RAG System Setup
  - [x] RAG schemas (Document, DocumentChunk, DocumentMetadata, SearchQuery, RetrievalResult)
  - [x] Synthetic financial documents (ETF fact sheets, FOMC summaries, investment guides)
  - [x] Embeddings module (SentenceTransformerEmbeddings, DocumentChunker, EmbeddingPipeline)
  - [x] ChromaDB vector store (document storage, search, filtering)
  - [x] Hybrid retrieval system (semantic + keyword search, re-ranking)
  - [x] Freshness management (retention policies, expiry handling)
  - [x] Auto-initialization on app startup
  - [x] Real data collectors (Alpha Vantage, SEC EDGAR, News API)
  - [x] Incremental document addition
  - [x] Data collection UI with buttons to collect real data
  - [x] Unit and integration tests
- [x] Phase 5: MCP Integration
  - [x] MCP server infrastructure (base server, FastMCP)
  - [x] Market data server (quotes, ETF profiles, market overview)
  - [x] News server (market news, ticker sentiment)
  - [x] Portfolio server (analysis, holdings, RAG search)
  - [x] MCP client with rate limiting and retry logic
  - [x] Hybrid data provider (RAG + MCP combined context)
  - [x] Data normalizer (cross-source normalization)
  - [x] Unit and integration tests
- [x] Phase 6: Recommendation Engine
  - [x] Engine schemas (Recommendation, RecommendationRequest, RecommendationResponse, AggregatedContext)
  - [x] Context builder (aggregates portfolio, profile, RAG, MCP data)
  - [x] LLM-based recommendation generator with structured JSON output
  - [x] Prompt templates (recommendation generation, explanation enhancement)
  - [x] Recommendation ranker (risk filtering, sector exclusion, diversification scoring)
  - [x] Main engine orchestrator (pipeline: context -> generate -> rank -> response)
  - [x] Streamlit UI (Recommendations tab with query input, provider selection, result cards)
  - [x] Unit tests (66 tests: schemas, ranker, generator, context builder, engine)
  - [x] Integration tests (4 tests: full pipeline, risk filtering, category filtering)

- [x] Phase 7: Integration & Testing
  - [x] LangGraph multi-agent orchestrator (state, routing, nodes, graph)
  - [x] Intent classification (LLM-based with keyword fallback)
  - [x] WealthOrchestrator high-level API
  - [x] FastAPI REST endpoints (chat, portfolio, profiling, recommendations)
  - [x] API schemas and dependency injection
  - [x] Streamlit "AI Advisor" tab (unified orchestrator chat)
  - [x] Unit tests (LLM client, collectors, MCP server, orchestrator routing/nodes/graph)
  - [x] Integration tests (orchestrator full graph, API endpoints)
  - [x] End-to-end tests (full pipeline: portfolio → profiling → recommendations)
  - [x] Documentation (README, API reference, user guide, orchestrator design)
