"""FastAPI dependency providers."""

import logging
from functools import lru_cache
from pathlib import Path

from backend.src.agents.orchestrator import WealthOrchestrator
from backend.src.asset_management.aggregator import PortfolioAggregator
from backend.src.common.llm_client import LLMClient
from backend.src.logging_db import LogRepository, get_engine, get_session_factory
from backend.src.recommendation.rag.initializer import RAGInitializer

logger = logging.getLogger(__name__)

_project_root = Path(__file__).parent.parent.parent.parent


@lru_cache()
def get_aggregator() -> PortfolioAggregator:
    """Get or create the cached PortfolioAggregator."""
    data_path = _project_root / "data" / "synthetic" / "synthetic_portfolios.json"
    aggregator = PortfolioAggregator(data_path)
    aggregator.load_data()
    return aggregator


@lru_cache()
def get_llm_client() -> LLMClient:
    """Get or create the cached LLMClient."""
    return LLMClient()


@lru_cache()
def get_rag() -> RAGInitializer:
    """Get or create the cached RAGInitializer."""
    persist_dir = str(_project_root / "data" / "chroma")
    rag = RAGInitializer(persist_directory=persist_dir)
    return rag


@lru_cache()
def get_orchestrator() -> WealthOrchestrator:
    """Get or create the cached WealthOrchestrator."""
    aggregator = get_aggregator()
    llm_client = get_llm_client()
    rag = get_rag()
    return WealthOrchestrator(aggregator, llm_client, rag)


@lru_cache()
def get_log_repository() -> LogRepository:
    """Get or create the cached LogRepository (SQLite-backed)."""
    engine = get_engine()
    factory = get_session_factory(engine)
    return LogRepository(factory)
