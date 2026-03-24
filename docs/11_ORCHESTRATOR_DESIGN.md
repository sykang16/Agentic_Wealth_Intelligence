# Orchestrator Design

## Overview

The multi-agent orchestrator uses a LangGraph `StateGraph` to route user messages to the appropriate module. It provides a unified conversational interface over the three core modules (Portfolio, Profiling, Recommendations) plus a general-purpose LLM fallback.

## Architecture

```
                    ┌─────────────┐
        START ────► │   router    │
                    └──────┬──────┘
                           │ (conditional edge based on intent)
              ┌────────────┼────────────┬──────────────┐
              ▼            ▼            ▼              ▼
        ┌───────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
        │  portfolio │ │ profiling│ │ recommend │ │  general  │
        │  (Mod A)   │ │ (Mod B)  │ │ (Mod C)  │ │  (LLM)   │
        └─────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
              │            │            │             │
              └────────────┴────────────┴─────────────┘
                                 │
                                 ▼
                              respond ──► END
```

## State

The orchestrator uses a `TypedDict` state (LangGraph convention):

```python
class OrchestratorState(TypedDict, total=False):
    user_id: str              # Current user
    messages: list[dict]      # Conversation history
    current_message: str      # Message being processed
    intent: str               # Classified intent
    response: str             # Generated response
    module_source: str        # Which module handled it
    profiling_state: Any      # Persisted profiling state
    error: str | None         # Error if any
```

## Intent Classification

The `IntentRouter` classifies messages using a two-tier approach:

1. **LLM classification** (primary) — A structured prompt asks the LLM to return one of four intent labels
2. **Keyword fallback** — If the LLM fails or returns an unknown label, keyword matching is used

### Intent Values

| Intent | Routes To | Example Messages |
|--------|-----------|-----------------|
| `portfolio_query` | Portfolio node | "What's my net worth?", "Show holdings" |
| `profiling` | Profiling node | "Build my profile", "Risk assessment" |
| `recommendation` | Recommend node | "What should I invest in?", "Diversify" |
| `general` | General node | "Hello", "Thanks", "What can you do?" |

## Node Functions

Each node is a pure function `(state) -> dict` returning a partial state update. Module dependencies are injected via closure at graph construction time.

- **`router_node`** — Classifies intent, sets `state.intent`
- **`portfolio_node`** — Calls `AssetAgent.process()`, populates `state.response`
- **`profiling_node`** — Manages `ProfilingAgent` lifecycle (start/continue), persists `state.profiling_state`
- **`recommend_node`** — Calls `RecommendationEngine.generate_recommendations()`, formats response text
- **`general_node`** — Direct LLM response for greetings, help, off-topic
- **`respond_node`** — Appends the exchange to message history

## Error Handling

Each module node wraps its logic in try/except. On failure:
- `state.error` is populated with the error message
- `state.response` contains a user-friendly error message
- The `respond_node` still appends the exchange to history

## `WealthOrchestrator` Class

High-level API wrapping the graph:

```python
orchestrator = WealthOrchestrator(aggregator, llm_client, rag)
state = orchestrator.create_initial_state("user_001")
state = orchestrator.process_message("What's my net worth?", state)
response = orchestrator.get_response(state)  # OrchestratorResponse
```

## Extension Points

- **New modules**: Add a new `UserIntent` value, create a node function, register it in `build_orchestrator_graph()`
- **Better routing**: Replace the LLM prompt in `routing.py` or add few-shot examples
- **State persistence**: Replace the in-memory `profiling_state` with database storage
- **Streaming**: LangGraph supports streaming; adapt `process_message()` to yield intermediate states

## Files

| File | Purpose |
|------|---------|
| `backend/src/multi_agent/state.py` | State types and response model |
| `backend/src/multi_agent/routing.py` | Intent classification |
| `backend/src/multi_agent/nodes.py` | Node function factories |
| `backend/src/multi_agent/graph.py` | Graph construction |
| `backend/src/agents/orchestrator.py` | High-level orchestrator API |
