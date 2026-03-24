# Technical Specifications

## Dependencies
```txt
# Core
python>=3.12
anthropic>=0.40.0
langgraph>=0.2.0
langchain>=0.3.0
langchain-anthropic>=0.3.0
pydantic>=2.0.0
pydantic-settings>=2.0.0

# Data & Vector DB
chromadb>=0.5.0
sentence-transformers>=2.0.0

# MCP (Model Context Protocol)
mcp>=1.0.0
fastmcp>=0.1.0

# Market Data (no API key required — accessed via MarketMCPServer)
yfinance>=0.2.40

# HTTP
httpx>=0.27.0

# Visualization / UI
streamlit>=1.40.0
plotly>=5.24.0
altair>=5.4.0
pandas>=2.0.0

# API
fastapi>=0.110.0
uvicorn>=0.27.0

# Testing
pytest>=8.0.0
pytest-asyncio>=0.24.0
pytest-cov>=5.0.0

# Development
python-dotenv>=1.0.0
faker>=30.0.0
black>=24.0.0
ruff>=0.6.0
```

## Environment Variables
```bash
# .env.example
ANTHROPIC_API_KEY=your_api_key_here        # Required
CHROMA_PATH=./data/chroma                  # Optional (default: ./data/chroma)
LOG_LEVEL=INFO

# Optional data collectors (features work without these via yfinance)
ALPHA_VANTAGE_API_KEY=your_key_here        # Enables "Collect News" in Knowledge Search
NEWS_API_KEY=your_key_here                 # Enables NewsAPI news collection
```

## Development Setup
```bash
# Commands to set up project
```

## Code Style Guidelines
- Type hints: required
- Line length: 100
- Formatter: black
- Linter: ruff
- Docstrings: Google style

## Testing Requirements
- Coverage: >80%
- Unit + integration tests
- Mock external APIs

## Async/Await
- LLM calls use async/await (LangGraph nodes)
- yfinance and ChromaDB operations are synchronous — do not wrap in asyncio.run() inside Streamlit

## Error Handling
- Comprehensive try/catch blocks
- Log at appropriate levels (DEBUG, INFO, WARNING, ERROR)
- Graceful degradation

## Key Implementation Patterns
1. Pydantic for All Data Models
```
from pydantic import BaseModel, Field

class InvestmentProfile(BaseModel):
    risk_tolerance: Literal["conservative", "moderate", "aggressive"]
    time_horizon: Literal["short", "medium", "long"]
    # ... all other fields
```

2. LangGraph State Management
```
from typing import TypedDict
from langgraph.graph import StateGraph

class SystemState(TypedDict):
    messages: List[Message]
    active_agent: str
    user_profile: Optional[InvestmentProfile]
    # ... other state
```

3. Agent Base Class
```
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    async def process(self, state: SystemState) -> SystemState:
        """Process agent logic"""
        pass
```

4. Synthetic Data Generation
```
class SyntheticAPIGenerator:
    def generate_bank_account(self) -> Dict:
        return {
            "account_id": fake.uuid4(),
            "balance": fake.pydecimal(left_digits=6, right_digits=2),
            "account_type": fake.random_element(["checking", "savings"])
        }
```

## Success Metrics

### Module A: Asset Management
- [ ] Query response time < 2 seconds
- [ ] Support 10+ query types
- [ ] Visualization generation success rate > 95%

### Module B: Profiling
- [ ] Complete profile extraction in < 10 conversation turns
- [ ] Slot extraction accuracy > 90%
- [ ] Natural conversation flow (user satisfaction)

### Module C: Recommendations
- [ ] Retrieval precision > 80%
- [ ] Response includes both static + live data
- [ ] Recommendation generation time < 5 seconds

---

## Risk Management

### Technical Risks
- **LLM API Rate Limits**: Implement caching and request batching
- **Vector DB Performance**: Monitor query latency, optimize indices
- **yfinance Availability**: yfinance may throttle or return empty data for some symbols
- **Data Quality**: Validate synthetic data realism

### Mitigation Strategies
- Comprehensive error handling with per-symbol try/except in price fetching
- Retry logic with exponential backoff for LLM calls
- Graceful degradation: recommendations work without live data (`include_live_data=False`)
- Extensive testing with edge cases (448 offline eval tests)
