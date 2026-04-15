"""Plaid API client wrapper.

Wraps the plaid-python SDK to provide a clean interface and isolate SDK
exception types behind PlaidIntegrationError.
"""

import logging

logger = logging.getLogger(__name__)

# Map environment name strings to Plaid API hosts.
# Note: Plaid deprecated the "development" environment — Limited Production now
# uses the "production" host with restricted access. "development" is aliased to
# "production" so existing configs continue to work without a DNS failure.
_ENV_MAP = {
    "sandbox": "https://sandbox.plaid.com",
    "development": "https://production.plaid.com",  # deprecated; aliased to production
    "production": "https://production.plaid.com",
}


class PlaidIntegrationError(Exception):
    """Raised when a Plaid API call fails."""


class PlaidClient:
    """Thin wrapper around the Plaid Python SDK.

    Instantiate once (e.g. via lru_cache in dependencies.py) and reuse.
    All SDK exceptions are caught and re-raised as PlaidIntegrationError.
    """

    def __init__(self, client_id: str, secret: str, env: str = "sandbox") -> None:
        """Initialize the Plaid client.

        Args:
            client_id: Plaid client_id from the dashboard.
            secret:    Plaid secret for the target environment.
            env:       One of "sandbox", "development", "production".
        """
        try:
            import plaid
            from plaid.api import plaid_api
        except ImportError as exc:
            raise ImportError(
                "plaid-python is not installed. "
                "Add 'plaid-python>=28.0.0' to requirements.txt."
            ) from exc

        host = _ENV_MAP.get(env, _ENV_MAP["sandbox"])
        configuration = plaid.Configuration(
            host=host,
            api_key={"clientId": client_id, "secret": secret},
        )
        api_client = plaid.ApiClient(configuration)
        self._client = plaid_api.PlaidApi(api_client)
        self._env = env

    # ------------------------------------------------------------------
    # Link token
    # ------------------------------------------------------------------

    def create_link_token(self, user_id: str, products: list[str]) -> str:
        """Create a Plaid Link token for the frontend flow.

        Args:
            user_id:  A stable identifier for this user (used as Plaid client_user_id).
            products: List of Plaid product names, e.g. ["transactions", "investments"].

        Returns:
            The link_token string to pass to Plaid Link JS.
        """
        try:
            from plaid.model.link_token_create_request import LinkTokenCreateRequest
            from plaid.model.link_token_create_request_user import (
                LinkTokenCreateRequestUser,
            )
            from plaid.model.country_code import CountryCode
            from plaid.model.products import Products

            product_enums = [Products(p) for p in products]
            request = LinkTokenCreateRequest(
                user=LinkTokenCreateRequestUser(client_user_id=user_id),
                client_name="Wealth Intelligence",
                products=product_enums,
                country_codes=[CountryCode("US")],
                language="en",
            )
            response = self._client.link_token_create(request)
            return response["link_token"]
        except Exception as exc:
            logger.error("create_link_token failed: %s", exc)
            raise PlaidIntegrationError(f"Failed to create link token: {exc}") from exc

    # ------------------------------------------------------------------
    # Token exchange
    # ------------------------------------------------------------------

    def exchange_public_token(self, public_token: str) -> tuple[str, str]:
        """Exchange a public_token for a persistent access_token.

        Returns:
            Tuple of (access_token, item_id).
        """
        try:
            from plaid.model.item_public_token_exchange_request import (
                ItemPublicTokenExchangeRequest,
            )

            request = ItemPublicTokenExchangeRequest(public_token=public_token)
            response = self._client.item_public_token_exchange(request)
            return response["access_token"], response["item_id"]
        except Exception as exc:
            logger.error("exchange_public_token failed: %s", exc)
            raise PlaidIntegrationError(f"Token exchange failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Account data
    # ------------------------------------------------------------------

    def get_accounts(self, access_token: str) -> list[dict]:
        """Fetch all accounts for the item.

        Returns:
            List of raw account dicts from the Plaid API response.
        """
        try:
            from plaid.model.accounts_get_request import AccountsGetRequest

            response = self._client.accounts_get(
                AccountsGetRequest(access_token=access_token)
            )
            return [acc.to_dict() for acc in response["accounts"]]
        except Exception as exc:
            logger.error("get_accounts failed: %s", exc)
            raise PlaidIntegrationError(f"Failed to fetch accounts: {exc}") from exc

    def get_identity(self, access_token: str) -> dict | None:
        """Fetch identity data for the item (requires Identity product).

        Returns:
            Raw identity dict, or None if the product is not available.
        """
        try:
            from plaid.model.identity_get_request import IdentityGetRequest

            response = self._client.identity_get(
                IdentityGetRequest(access_token=access_token)
            )
            accounts = response.get("accounts", [])
            if accounts:
                return accounts[0].to_dict()
            return None
        except Exception as exc:
            logger.warning("get_identity failed (non-fatal): %s", exc)
            return None

    def get_investment_holdings(
        self, access_token: str
    ) -> tuple[list[dict], list[dict]]:
        """Fetch investment holdings and their securities.

        Returns:
            Tuple of (holdings_list, securities_list) as raw dicts.
            Returns ([], []) if the Investments product is not enabled.
        """
        try:
            from plaid.model.investments_holdings_get_request import (
                InvestmentsHoldingsGetRequest,
            )

            response = self._client.investments_holdings_get(
                InvestmentsHoldingsGetRequest(access_token=access_token)
            )
            holdings = [h.to_dict() for h in response.get("holdings", [])]
            securities = [s.to_dict() for s in response.get("securities", [])]
            return holdings, securities
        except Exception as exc:
            logger.warning("get_investment_holdings failed (non-fatal): %s", exc)
            return [], []

    # ------------------------------------------------------------------
    # Institution metadata
    # ------------------------------------------------------------------

    def get_institution_name(self, access_token: str) -> str:
        """Resolve the institution name for an item.

        Returns:
            Institution name string, or "Unknown Institution" on failure.
        """
        try:
            from plaid.model.item_get_request import ItemGetRequest
            from plaid.model.institutions_get_by_id_request import (
                InstitutionsGetByIdRequest,
            )
            from plaid.model.country_code import CountryCode

            item_resp = self._client.item_get(
                ItemGetRequest(access_token=access_token)
            )
            institution_id = item_resp["item"]["institution_id"]
            if not institution_id:
                return "Unknown Institution"

            inst_resp = self._client.institutions_get_by_id(
                InstitutionsGetByIdRequest(
                    institution_id=institution_id,
                    country_codes=[CountryCode("US")],
                )
            )
            return inst_resp["institution"]["name"]
        except Exception as exc:
            logger.warning("get_institution_name failed (non-fatal): %s", exc)
            return "Unknown Institution"
