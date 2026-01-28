# Technical Specifications

## Dependencies
```txt
# Core
python>=3.11
anthropic>=0.40.0
langgraph>=0.2.0
langchain>=0.3.0
pydantic>=2.0.0

# Data & Vector DB
chromadb>=0.5.0  # or pinecone-client
sentence-transformers>=2.0.0

# Visualization
plotly>=5.0.0
pandas>=2.0.0

# MCP (if available)
mcp-sdk  # Check for official package

# Testing
pytest>=8.0.0
pytest-asyncio>=0.23.0
pytest-cov>=4.0.0

# Development
python-dotenv>=1.0.0
black>=24.0.0
ruff>=0.0.292
```

## Environment Variables
```bash
# .env.example
ANTHROPIC_API_KEY=your_api_key_here
VECTOR_DB_TYPE=chroma  # or pinecone
CHROMA_PATH=./data/chroma
LOG_LEVEL=INFO
ENABLE_SYNTHETIC_DATA=true
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
- Use async/await for all I/O operations
- API calls, database queries, LLM calls must be async

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
- **MCP Server Downtime**: Implement fallback mechanisms
- **Data Quality**: Validate synthetic data realism

### Mitigation Strategies
- Comprehensive error handling
- Retry logic with exponential backoff
- Graceful degradation (work without MCP if needed)
- Extensive testing with edge cases
