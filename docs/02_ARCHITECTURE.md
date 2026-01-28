# System Architecture

## Tech Stack
- **Language**: Python 3.11+
- **Agent Framework**: LangGraph 0.2+
- **LLM**: Anthropic Claude (Sonnet 4.5)
- **Data Models**: Pydantic v2
- **Vector DB**: ChromaDB
- **Testing**: pytest

## Module Interactions
User Query → Asset Management (Module A)
     ↓
Investment Profiling (Module B) → InvestmentProfile
     ↓
Recommendation Engine (Module C)
  ├─ Vector DB (static knowledge)
  └─ MCP Server (live data)
     ↓
Personalized Recommendation

## Project Structure

```
agentic-wealth-intelligence/
├── .claude/
│   └── context.md                    # This file
├── docs/
│   ├── 00_INDEX.md                   # Documentation navigation
│   ├── 01_PROJECT_OVERVIEW.md
│   ├── 02_ARCHITECTURE.md
│   ├── 03_MODULE_A_ASSET.md
│   ├── 04_MODULE_B_PROFILING.md
│   ├── 05_MODULE_C_RECOMMENDATION.md
│   ├── 06_PHASES.md
│   ├── 07_TECHNICAL_SPECS.md
│   └── 08_UI_DESIGN.md
├── backend/
│   └── src/
│       ├── agents/
│       │   ├── orchestrator.py      # Main coordinator
│       │   ├── asset_agent.py       # Module A
│       │   ├── profiling_agent.py   # Module B
│       │   └── advisory_agent.py    # Module C
│       ├── multi_agent/
│       │   ├── graph.py             # LangGraph workflow
│       │   ├── state.py             # State definitions
│       │   └── routing.py           # Routing logic
│       ├── api/                      # FastAPI endpoints (Phase 3+)
│       ├── asset_management/         # Tools for Asset Agent
│       ├── profiling/                # Tools for Profiling Agent
│       ├── recommendation/           # Tools for Advisory Agent
│       ├── data_generation/          # Synthetic data
│       └── common/
│           ├── models.py             # Pydantic schemas
│           ├── llm_client.py
│           └── utils.py
├── ui/                               # Streamlit app (Phase 2+)
│   ├── streamlit_app.py
│   ├── pages/
│   ├── components/
│   ├── utils/
│   └── config.py                     # UI configuration
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── data/
│   ├── synthetic/                    # Generated test data
│   └── documents/                    # Static financial docs
└── README.md
```