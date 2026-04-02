"""Maps raw Plaid API responses to the internal UserPortfolio schema.

Downstream agents and MCP servers receive a UserPortfolio object regardless
of whether it was built from synthetic JSON or live Plaid data.
"""

from datetime import datetime, timezone
from decimal import Decimal

from backend.src.common.models import (
    AccountStatus,
    AssetType,
    BankAccount,
    BankAccountType,
    Holding,
    InvestmentAccount,
    InvestmentAccountType,
    PortfolioSummary,
    User,
    UserPortfolio,
)


# ------------------------------------------------------------------
# Enum mapping helpers
# ------------------------------------------------------------------

_BANK_SUBTYPE_MAP: dict[str, BankAccountType] = {
    "checking": BankAccountType.CHECKING,
    "savings": BankAccountType.SAVINGS,
    "money market": BankAccountType.SAVINGS,
    "money_market": BankAccountType.SAVINGS,
    "cd": BankAccountType.SAVINGS,
    "paypal": BankAccountType.CHECKING,
    "prepaid": BankAccountType.CHECKING,
}

_INVESTMENT_SUBTYPE_MAP: dict[str, InvestmentAccountType] = {
    "ira": InvestmentAccountType.IRA,
    "roth": InvestmentAccountType.ROTH_IRA,
    "roth ira": InvestmentAccountType.ROTH_IRA,
    "401k": InvestmentAccountType.FOUR01K,
    "401(k)": InvestmentAccountType.FOUR01K,
    "403b": InvestmentAccountType.FOUR01K,
    "403(b)": InvestmentAccountType.FOUR01K,
    "brokerage": InvestmentAccountType.BROKERAGE,
}

_ASSET_TYPE_MAP: dict[str, AssetType] = {
    "equity": AssetType.STOCK,
    "etf": AssetType.ETF,
    "mutual fund": AssetType.MUTUAL_FUND,
    "fixed income": AssetType.BOND,
    "bond": AssetType.BOND,
    "cryptocurrency": AssetType.CRYPTO,
    "crypto": AssetType.CRYPTO,
    "cash": AssetType.STOCK,  # treat cash-like as stock for schema compatibility
}


def _map_bank_subtype(subtype: str | None) -> BankAccountType:
    if subtype:
        return _BANK_SUBTYPE_MAP.get(subtype.lower(), BankAccountType.CHECKING)
    return BankAccountType.CHECKING


def _map_investment_subtype(subtype: str | None) -> InvestmentAccountType:
    if subtype:
        return _INVESTMENT_SUBTYPE_MAP.get(subtype.lower(), InvestmentAccountType.BROKERAGE)
    return InvestmentAccountType.BROKERAGE


def _map_security_type(sec_type: str | None) -> AssetType:
    if sec_type:
        return _ASSET_TYPE_MAP.get(sec_type.lower(), AssetType.STOCK)
    return AssetType.STOCK


def _to_decimal(value) -> Decimal:
    """Safely convert a value to Decimal, returning 0 on failure."""
    try:
        return Decimal(str(value)) if value is not None else Decimal("0")
    except Exception:
        return Decimal("0")


# ------------------------------------------------------------------
# Adapter
# ------------------------------------------------------------------


class PlaidPortfolioAdapter:
    """Converts raw Plaid API response dicts into a UserPortfolio object."""

    def build_portfolio(
        self,
        user_id: str,
        accounts: list[dict],
        institution_name: str,
        holdings: list[dict],
        securities: list[dict],
        identity: dict | None = None,
        age: int | None = None,
        annual_income: float | None = None,
        occupation: str | None = None,
    ) -> UserPortfolio:
        """Build a UserPortfolio from Plaid API data.

        Args:
            user_id:          Pre-generated user ID (e.g. "plaid_abc12345").
            accounts:         Raw accounts from /accounts/get.
            institution_name: Name of the financial institution.
            holdings:         Raw holdings from /investments/holdings/get.
            securities:       Security master list from the same holdings call.
            identity:         Optional raw identity dict from /identity/get.

        Returns:
            A fully populated UserPortfolio ready for use by agents.
        """
        user = self._build_user(user_id, identity, age=age, annual_income=annual_income, occupation=occupation)
        bank_accounts = self._build_bank_accounts(user_id, accounts, institution_name)
        investment_accounts = self._build_investment_accounts(user_id, accounts, institution_name)
        holding_objects = self._build_holdings(holdings, securities)
        summary = self._build_summary(user_id, bank_accounts, investment_accounts, holding_objects)

        return UserPortfolio(
            user=user,
            bank_accounts=bank_accounts,
            investment_accounts=investment_accounts,
            holdings=holding_objects,
            real_estate=[],        # Plaid has no real estate product
            investment_profile=None,  # filled later via profiling agent
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Sub-builders
    # ------------------------------------------------------------------

    def _build_user(
        self,
        user_id: str,
        identity: dict | None,
        age: int | None = None,
        annual_income: float | None = None,
        occupation: str | None = None,
    ) -> User:
        """Extract name/email from Plaid identity; fall back to stubs."""
        name = "Plaid User"
        email = "unknown@plaid.user"

        if identity:
            owners = identity.get("owners", [])
            if owners:
                names = owners[0].get("names", [])
                if names:
                    name = names[0]
                emails = owners[0].get("emails", [])
                if emails:
                    email = emails[0].get("data", email)

        return User(
            user_id=user_id,
            name=name,
            email=email,
            age=age if age is not None else 30,
            occupation=occupation if occupation else "Unknown",
            annual_income=Decimal(str(annual_income)) if annual_income is not None else Decimal("0"),
            monthly_expenses=Decimal("0"),
        )

    def _build_bank_accounts(
        self,
        user_id: str,
        accounts: list[dict],
        institution_name: str,
    ) -> list[BankAccount]:
        """Build BankAccount objects from Plaid depository accounts."""
        result = []
        for acc in accounts:
            if acc.get("type") != "depository":
                continue
            balances = acc.get("balances", {})
            result.append(
                BankAccount(
                    account_id=acc["account_id"],
                    user_id=user_id,
                    account_type=_map_bank_subtype(acc.get("subtype")),
                    bank_name=institution_name,
                    balance=_to_decimal(balances.get("current")),
                    currency=balances.get("iso_currency_code") or "USD",
                    status=AccountStatus.ACTIVE,
                )
            )
        return result

    def _build_investment_accounts(
        self,
        user_id: str,
        accounts: list[dict],
        institution_name: str,
    ) -> list[InvestmentAccount]:
        """Build InvestmentAccount objects from Plaid investment/brokerage accounts."""
        result = []
        for acc in accounts:
            if acc.get("type") not in ("investment", "brokerage"):
                continue
            balances = acc.get("balances", {})
            total = _to_decimal(balances.get("current"))
            available = _to_decimal(balances.get("available"))
            result.append(
                InvestmentAccount(
                    account_id=acc["account_id"],
                    user_id=user_id,
                    account_type=_map_investment_subtype(acc.get("subtype")),
                    institution=institution_name,
                    total_value=total,
                    cash_balance=available,
                    status=AccountStatus.ACTIVE,
                )
            )
        return result

    def _build_holdings(
        self,
        holdings: list[dict],
        securities: list[dict],
    ) -> list[Holding]:
        """Build Holding objects from Plaid holdings and securities master data."""
        # Index securities by security_id for O(1) lookup
        sec_map: dict[str, dict] = {s["security_id"]: s for s in securities}

        result = []
        for h in holdings:
            sec_id = h.get("security_id", "")
            sec = sec_map.get(sec_id, {})

            quantity = _to_decimal(h.get("quantity"))
            # Skip zero/negative quantity holdings to satisfy Holding.quantity gt=0 constraint
            if quantity <= 0:
                continue

            cost_basis_total = _to_decimal(h.get("cost_basis"))
            institution_price = _to_decimal(h.get("institution_price"))

            # Derive average cost per share from total cost basis
            avg_cost = cost_basis_total / quantity if cost_basis_total > 0 else Decimal("0")

            symbol = sec.get("ticker_symbol") or sec.get("name", "UNKNOWN")[:10]
            name = sec.get("name") or symbol

            result.append(
                Holding(
                    holding_id=f"{h.get('account_id', '')}_{sec_id}",
                    account_id=h.get("account_id", ""),
                    asset_type=_map_security_type(sec.get("type")),
                    symbol=symbol,
                    name=name,
                    quantity=quantity,
                    average_cost=avg_cost,
                    current_price=institution_price,
                    sector=None,  # Plaid does not provide sector info
                )
            )
        return result

    def _build_summary(
        self,
        user_id: str,
        bank_accounts: list[BankAccount],
        investment_accounts: list[InvestmentAccount],
        holdings: list[Holding],
    ) -> PortfolioSummary:
        """Compute a PortfolioSummary from Plaid account data."""
        total_bank = sum(
            acc.balance for acc in bank_accounts if acc.status == AccountStatus.ACTIVE
        )
        total_invest = sum(
            acc.total_value
            for acc in investment_accounts
            if acc.status == AccountStatus.ACTIVE
        )
        total_assets = total_bank + total_invest
        # Liabilities: Plaid has a Liabilities product but it is not included
        # in the default products set; default to zero.
        total_liabilities = Decimal("0")
        total_net_worth = total_assets - total_liabilities

        # Asset-type allocation: prefer holdings-based (when investments product is active),
        # otherwise fall back to account-balance-based allocation so the pie chart is never empty.
        alloc_by_type: dict[str, float] = {}
        total_holding_value = sum(h.market_value for h in holdings)
        if total_holding_value > 0:
            for h in holdings:
                key = h.asset_type.value
                alloc_by_type[key] = (
                    alloc_by_type.get(key, 0.0)
                    + float(h.market_value / total_holding_value * 100)
                )
        elif total_assets > 0:
            # No holdings data — compute allocation from account balances.
            # Group bank accounts by subtype (checking / savings).
            for acc in bank_accounts:
                if acc.status != AccountStatus.ACTIVE or acc.balance <= 0:
                    continue
                key = acc.account_type.value  # "checking" or "savings"
                alloc_by_type[key] = (
                    alloc_by_type.get(key, 0.0)
                    + float(acc.balance / total_assets * 100)
                )
            # Group investment accounts by subtype (ira / roth_ira / 401k / brokerage).
            for acc in investment_accounts:
                if acc.status != AccountStatus.ACTIVE or acc.total_value <= 0:
                    continue
                key = acc.account_type.value  # "ira", "roth_ira", "401k", "brokerage"
                alloc_by_type[key] = (
                    alloc_by_type.get(key, 0.0)
                    + float(acc.total_value / total_assets * 100)
                )

        # Sector allocation: Plaid does not provide sector info
        alloc_by_sector: dict[str, float] = {}

        liquidity_ratio = (
            float(total_bank / total_assets) if total_assets > 0 else 0.0
        )
        debt_ratio = (
            float(total_liabilities / total_assets) if total_assets > 0 else 0.0
        )

        return PortfolioSummary(
            user_id=user_id,
            total_assets=total_assets,
            total_liabilities=total_liabilities,
            total_net_worth=total_net_worth,
            cash_and_equivalents=total_bank,
            stocks=Decimal("0"),
            bonds=Decimal("0"),
            etfs=Decimal("0"),
            mutual_funds=Decimal("0"),
            crypto=Decimal("0"),
            real_estate_equity=Decimal("0"),
            allocation_by_asset_type=alloc_by_type,
            allocation_by_sector=alloc_by_sector,
            liquidity_ratio=liquidity_ratio,
            debt_to_asset_ratio=debt_ratio,
            monthly_savings_rate=None,
            last_updated=datetime.now(timezone.utc),
        )
