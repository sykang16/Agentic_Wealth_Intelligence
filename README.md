---
title: Agentic Wealth Intelligence
emoji: 📈
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.55.0
app_file: ui/streamlit_app.py
pinned: false
---

# Agentic Wealth Intelligence

AI-powered wealth management system with multi-agent orchestration, conversational profiling, RAG-enhanced recommendations, and real-time market data integration.

## Architecture

**LangGraph Hybrid Supervisor Routing** — two-tier intent classification feeds into a flat outer graph; the Recommend node embeds a Supervisor subgraph for multi-step context gathering.

### Outer Orchestrator Graph

```
                 ┌──────────────────────────────────────┐
     User ──────►│             Router                   │
                 │  Tier 1: Keyword match  (fast path)  │
                 │  Tier 2: LLM fallback   (T=0)        │
                 └──────────────┬───────────────────────┘
                                │ conditional routing
       ┌────────────────────────┼──────────────────┬──────────────┐
       ▼                        ▼                  ▼              ▼
┌────────────┐       ┌──────────────┐   ┌──────────────┐  ┌──────────┐
│ Portfolio  │       │  Profiling   │   │  Recommend   │  │ General  │
│  Module A  │       │   Module B   │   │  [Subgraph]  │  │  (LLM)   │
│ AssetAgent │       │ slot-filling │   │  Supervisor  │  │          │
└──────┬─────┘       └──────┬───────┘   └──────┬───────┘  └────┬─────┘
       └────────────────────┴──────────────────┴────────────────┘
                                                │
                                         ┌──────▼──────┐
                                         │   Respond   │──► END
                                         └─────────────┘
```

### Recommendation Supervisor Subgraph

```
┌───────────────────────────────────────────────────────────────┐
│             Supervisor  (LLM decision · guard: steps <= 5)    │
│        ┌────────────────────┬──────────────────┐              │
│        ▼                    ▼                  ▼              │
│ ┌─────────────┐   ┌──────────────┐   ┌──────────────┐        │
│ │portfolio_   │   │profiling_    │   │  recommend   │        │
│ │fetch        │   │fetch         │   │  synthesis   │        │
│ │AssetAgent   │   │get_profile_  │   │  RAG + MCP   │        │
│ │.process()   │   │  summary()   │   │              │        │
│ └──────┬──────┘   └──────┬───────┘   └──────┬───────┘        │
│        └─────────────────┴──────────────────┘                 │
│                    loops back to Supervisor                    │
│                                          finish ─────► Respond │
└───────────────────────────────────────────────────────────────┘
```

**Modules:**
- **Portfolio Analysis** — Natural language queries about your holdings, net worth, allocation, gains/losses
- **Investment Profiling** — Conversational slot-filling to build your risk/preference profile
- **Recommendations** — AI-generated investment advice using portfolio data, RAG knowledge base, and live market data
- **AI Advisor** — Unified chat interface that automatically routes to the right module

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and set your API keys:

```bash
cp .env.example .env
```

Required (at least one LLM provider):
- `OPENAI_API_KEY` — OpenAI GPT-4o
- `ANTHROPIC_API_KEY` — Anthropic Claude
- `GEMINI_API_KEY` — Google Gemini

Optional (for live data):
- `ALPHA_VANTAGE_API_KEY` — Market quotes and financial news
- `NEWS_API_KEY` — Business news
- `SEC_USER_AGENT` — SEC EDGAR filings

### 3. Generate Synthetic Data

```python
from backend.src.data_generation import generate_sample_data
generate_sample_data()
```

### 4. Run the Streamlit UI

```bash
python -m streamlit run ui/streamlit_app.py
```

### 5. Run the API Server

```bash
uvicorn backend.src.api.app:app --reload
```

API documentation available at `http://localhost:8000/docs`.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/chat` | Unified orchestrator chat |
| `GET` | `/api/v1/portfolio/{user_id}` | Portfolio summary |
| `POST` | `/api/v1/portfolio/{user_id}/query` | Natural language portfolio query |
| `POST` | `/api/v1/profiling/{user_id}/start` | Start profiling session |
| `POST` | `/api/v1/profiling/{user_id}/respond` | Continue profiling conversation |
| `POST` | `/api/v1/recommendations/generate` | Generate recommendations |
| `GET` | `/health` | Health check |

## Project Structure

```
backend/
├── src/
│   ├── agents/              # Agent implementations
│   │   ├── asset_agent.py   # Portfolio query agent
│   │   └── orchestrator.py  # Multi-agent orchestrator
│   ├── api/                 # FastAPI application
│   │   ├── app.py           # App factory
│   │   ├── dependencies.py  # Dependency injection
│   │   ├── schemas.py       # Request/response models
│   │   └── routes/          # API route handlers
│   ├── asset_management/    # Module A: Portfolio data
│   ├── common/              # Shared models, LLM client
│   ├── multi_agent/         # LangGraph orchestrator
│   │   ├── state.py         # State definitions
│   │   ├── routing.py       # Intent classification
│   │   ├── nodes.py         # Graph node functions
│   │   └── graph.py         # Graph construction
│   ├── profiling/           # Module B: Slot-filling agent
│   ├── recommendation/      # Module C: RAG + recommendations
│   └── mcp/                 # MCP servers for live data
ui/
├── streamlit_app.py         # Main Streamlit application
tests/
├── unit/                    # Unit tests
├── integration/             # Integration tests
└── e2e/                     # End-to-end tests
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test categories
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/e2e/ -v
```

## Tech Stack

- **Python 3.11+** — Core language
- **LangGraph** — Multi-agent orchestration and profiling workflows
- **FastAPI** — REST API layer
- **Streamlit** — Interactive UI
- **ChromaDB** — Vector store for RAG
- **Sentence Transformers** — Document embeddings
- **MCP (Model Context Protocol)** — Live market data integration
- **Pydantic v2** — Data validation

## Documentation

See [docs/00_INDEX.md](docs/00_INDEX.md) for the full documentation index.
