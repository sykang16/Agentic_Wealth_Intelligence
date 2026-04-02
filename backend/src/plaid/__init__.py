"""Plaid integration package for real account data ingestion."""

from backend.src.plaid.adapter import PlaidPortfolioAdapter
from backend.src.plaid.client import PlaidClient
from backend.src.plaid.source import PlaidPortfolioSource
from backend.src.plaid.token_store import PlaidTokenRepository

__all__ = [
    "PlaidClient",
    "PlaidPortfolioAdapter",
    "PlaidPortfolioSource",
    "PlaidTokenRepository",
]
