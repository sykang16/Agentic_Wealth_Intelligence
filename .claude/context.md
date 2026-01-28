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
- Phase plan: [docs/06_PHASES.md](../ocs/06_PHASES.md)


## Current Phase
Phase [X]: [Brief description]

## Active Tasks
- [ ] Task 1
- [ ] Task 2
