"""FastAPI dependency providers."""

import logging
import os
from functools import lru_cache
from pathlib import Path

from backend.src.agents.orchestrator import WealthOrchestrator
from backend.src.asset_management.aggregator import PortfolioAggregator
from backend.src.common.llm_client import LLMClient
from backend.src.logging_db import LogRepository, get_engine, get_session_factory
from backend.src.recommendation.rag.initializer import RAGInitializer

# Import PlaidToken model so it registers on Base.metadata before create_all()
import backend.src.plaid.token_store  # noqa: F401

logger = logging.getLogger(__name__)

_project_root = Path(__file__).parent.parent.parent.parent


@lru_cache()
def get_aggregator() -> PortfolioAggregator:
    """Get or create the cached PortfolioAggregator.

    Registers a PlaidPortfolioSource if PLAID_CLIENT_ID and PLAID_SECRET
    are present in the environment.
    """
    data_path = _project_root / "data" / "synthetic" / "synthetic_portfolios.json"
    aggregator = PortfolioAggregator(data_path)
    aggregator.load_data()

    # Wire Plaid source only when credentials are configured
    if os.environ.get("PLAID_CLIENT_ID") and os.environ.get("PLAID_SECRET"):
        try:
            aggregator.register_plaid_source(get_plaid_source())
        except Exception as exc:
            logger.warning("Failed to register Plaid source: %s", exc)

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


# ------------------------------------------------------------------
# Plaid dependencies
# ------------------------------------------------------------------


@lru_cache()
def get_plaid_token_repo() -> "backend.src.plaid.token_store.PlaidTokenRepository":
    """Get or create the cached PlaidTokenRepository."""
    from backend.src.plaid.token_store import PlaidTokenRepository

    engine = get_engine()
    factory = get_session_factory(engine)
    return PlaidTokenRepository(factory)


@lru_cache()
def get_plaid_client() -> "backend.src.plaid.client.PlaidClient | None":
    """Get or create the cached PlaidClient, or None if credentials are absent."""
    from backend.src.plaid.client import PlaidClient

    client_id = os.environ.get("PLAID_CLIENT_ID")
    secret = os.environ.get("PLAID_SECRET")
    env = os.environ.get("PLAID_ENV", "sandbox")

    if not client_id or not secret:
        return None
    return PlaidClient(client_id=client_id, secret=secret, env=env)


@lru_cache()
def get_plaid_source() -> "backend.src.plaid.source.PlaidPortfolioSource | None":
    """Get or create the cached PlaidPortfolioSource, or None if Plaid is unconfigured."""
    from backend.src.plaid.adapter import PlaidPortfolioAdapter
    from backend.src.plaid.source import PlaidPortfolioSource

    client = get_plaid_client()
    if client is None:
        return None

    env = os.environ.get("PLAID_ENV", "sandbox")
    return PlaidPortfolioSource(
        plaid_client=client,
        adapter=PlaidPortfolioAdapter(),
        token_repo=get_plaid_token_repo(),
        current_env=env,
    )
