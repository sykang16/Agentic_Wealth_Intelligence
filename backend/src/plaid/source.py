"""PlaidPortfolioSource: live Plaid data provider for PortfolioAggregator.

PortfolioAggregator calls register_plaid_source(source) once at startup.
Subsequent calls to get_user_ids() and get_portfolio() transparently include
Plaid-connected users alongside synthetic ones.
"""

import logging
from datetime import datetime, timedelta, timezone

from backend.src.common.models import UserPortfolio
from backend.src.plaid.adapter import PlaidPortfolioAdapter
from backend.src.plaid.client import PlaidClient, PlaidIntegrationError
from backend.src.plaid.token_store import PlaidTokenRepository

logger = logging.getLogger(__name__)

# Cache TTL: how long to reuse a fetched portfolio before calling Plaid again
_CACHE_TTL_MINUTES = 5


class PlaidPortfolioSource:
    """Provides UserPortfolio objects built from live Plaid API data.

    Supports multiple environments simultaneously (e.g. sandbox + production).
    Pass a ``clients`` dict keyed by env name, or a single ``plaid_client``
    with ``current_env`` for backward compatibility.

    Data is cached per user_id with a 5-minute TTL to avoid excessive
    Plaid API calls during a single session.
    """

    def __init__(
        self,
        adapter: PlaidPortfolioAdapter,
        token_repo: PlaidTokenRepository,
        plaid_client: PlaidClient | None = None,
        current_env: str = "sandbox",
        clients: dict[str, PlaidClient] | None = None,
    ) -> None:
        if clients is not None:
            self._clients = clients
        elif plaid_client is not None:
            self._clients = {current_env: plaid_client}
        else:
            raise ValueError("Provide either 'clients' dict or 'plaid_client'.")
        self._adapter = adapter
        self._repo = token_repo
        # TTL cache: user_id -> (UserPortfolio, fetched_at)
        self._cache: dict[str, tuple[UserPortfolio, datetime]] = {}

    # ------------------------------------------------------------------
    # Public interface (mirrors PortfolioAggregator's internal API)
    # ------------------------------------------------------------------

    def get_user_ids(self) -> list[str]:
        """Return all active Plaid user_ids whose env has a configured client."""
        records = self._repo.get_all_active()
        return [r.user_id for r in records if r.env in self._clients]

    def has_user(self, user_id: str) -> bool:
        """Return True if user_id has an active Plaid token with a configured client."""
        record = self._repo.get_by_user_id(user_id)
        return record is not None and record.env in self._clients

    def get_portfolio(self, user_id: str) -> UserPortfolio | None:
        """Return a UserPortfolio for the given Plaid user_id.

        Uses the 5-minute TTL cache to reduce Plaid API calls.
        Returns None if the user is not found or the API call fails.
        """
        # Check cache first
        cached = self._cache.get(user_id)
        if cached:
            portfolio, fetched_at = cached
            age = datetime.now(timezone.utc) - fetched_at
            if age < timedelta(minutes=_CACHE_TTL_MINUTES):
                return portfolio

        # Cache miss or expired — fetch from Plaid
        record = self._repo.get_by_user_id(user_id)
        if not record or not record.is_active:
            return None
        if record.env != self._current_env:
            logger.warning(
                "Token env '%s' does not match current env '%s' for user %s",
                record.env,
                self._current_env,
                user_id,
            )
            return None

        try:
            portfolio = self._fetch_portfolio(
                user_id, record.access_token, record.products, record.env,
                age=record.age,
                annual_income=float(record.annual_income) if record.annual_income is not None else None,
                occupation=record.occupation,
            )
            self._cache[user_id] = (portfolio, datetime.now(timezone.utc))
            return portfolio
        except PlaidIntegrationError as exc:
            logger.error("Failed to fetch Plaid portfolio for %s: %s", user_id, exc)
            # Return stale cache if available rather than failing completely
            if cached:
                logger.info("Returning stale cache for %s", user_id)
                return cached[0]
            return None

    def invalidate_cache(self, user_id: str) -> None:
        """Force the next get_portfolio() call to re-fetch from Plaid."""
        self._cache.pop(user_id, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_portfolio(
        self,
        user_id: str,
        access_token: str,
        products_str: str,
        env: str,
        age: int | None = None,
        annual_income: float | None = None,
        occupation: str | None = None,
    ) -> UserPortfolio:
        """Call Plaid APIs and build a UserPortfolio."""
        client = self._clients.get(env)
        if client is None:
            raise PlaidIntegrationError(
                f"No Plaid client configured for env '{env}'."
            )
        products = [p.strip() for p in products_str.split(",")]

        accounts = client.get_accounts(access_token)
        institution_name = client.get_institution_name(access_token)

        # Fetch holdings only if the Investments product was requested
        holdings, securities = [], []
        if "investments" in products:
            holdings, securities = client.get_investment_holdings(access_token)

        # Fetch identity only if the Identity product was requested
        identity = None
        if "identity" in products:
            identity = client.get_identity(access_token)

        return self._adapter.build_portfolio(
            user_id=user_id,
            accounts=accounts,
            institution_name=institution_name,
            holdings=holdings,
            securities=securities,
            identity=identity,
            age=age,
            annual_income=annual_income,
            occupation=occupation,
        )
