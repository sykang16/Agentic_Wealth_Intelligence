"""Data aggregator for multi-source financial data integration."""

import json
from decimal import Decimal
from pathlib import Path

from backend.src.common.models import (
    AccountStatus,
    AssetType,
    BankAccount,
    Holding,
    InvestmentAccount,
    InvestmentProfile,
    PortfolioSummary,
    PropertyStatus,
    RealEstate,
    User,
    UserPortfolio,
)


class PortfolioAggregator:
    """Aggregates financial data from multiple sources."""

    def __init__(self, data_path: str | Path | None = None):
        """Initialize aggregator.

        Args:
            data_path: Path to synthetic data JSON file.
        """
        self.data_path = Path(data_path) if data_path else None
        self._portfolios: dict[str, UserPortfolio] = {}
        self._loaded = False
        # Optional live Plaid data source; set via register_plaid_source()
        self._plaid_source = None

    def load_data(self, data_path: str | Path | None = None) -> None:
        """Load portfolio data from JSON file.

        Args:
            data_path: Path to JSON file. Uses instance path if not provided.
        """
        path = Path(data_path) if data_path else self.data_path
        if not path or not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        with open(path) as f:
            data = json.load(f)

        for portfolio_data in data:
            portfolio = self._parse_portfolio(portfolio_data)
            self._portfolios[portfolio.user.user_id] = portfolio

        self._loaded = True

    def reload(self) -> None:
        """Reload portfolio data from disk (useful after adding new synthetic users)."""
        self._portfolios.clear()
        self._loaded = False
        self.load_data()

    def register_plaid_source(self, source) -> None:
        """Attach a live Plaid data source.

        When registered, get_user_ids() and get_portfolio() will transparently
        include Plaid-connected users alongside synthetic ones.

        Args:
            source: A PlaidPortfolioSource instance.
        """
        self._plaid_source = source

    def _parse_portfolio(self, data: dict) -> UserPortfolio:
        """Parse portfolio data from JSON."""
        user = User(**data["user"])

        bank_accounts = [BankAccount(**acc) for acc in data.get("bank_accounts", [])]
        investment_accounts = [
            InvestmentAccount(**acc) for acc in data.get("investment_accounts", [])
        ]
        holdings = [Holding(**h) for h in data.get("holdings", [])]
        real_estate = [RealEstate(**prop) for prop in data.get("real_estate", [])]

        summary = None
        if data.get("summary"):
            summary = PortfolioSummary(**data["summary"])

        investment_profile = None
        if data.get("investment_profile"):
            investment_profile = InvestmentProfile(**data["investment_profile"])

        return UserPortfolio(
            user=user,
            bank_accounts=bank_accounts,
            investment_accounts=investment_accounts,
            holdings=holdings,
            real_estate=real_estate,
            investment_profile=investment_profile,
            summary=summary,
        )

    def update_investment_profile(self, user_id: str, profile: InvestmentProfile) -> None:
        """Replace the investment profile for a user in memory.

        Call save_portfolios() afterwards to persist to disk.

        Args:
            user_id: Target user ID.
            profile: New InvestmentProfile to store.

        Raises:
            RuntimeError: If data has not been loaded.
            ValueError: If the user ID does not exist.
        """
        if not self._loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        portfolio = self._portfolios.get(user_id)
        if portfolio is None:
            raise ValueError(f"No portfolio found for user '{user_id}'")
        portfolio.investment_profile = profile

    def save_portfolios(self) -> None:
        """Persist current in-memory portfolios back to the JSON file."""
        if not self.data_path:
            raise RuntimeError("No data_path configured. Cannot save.")
        data = [portfolio.model_dump(mode="json") for portfolio in self._portfolios.values()]
        with open(self.data_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def get_portfolio(self, user_id: str) -> UserPortfolio | None:
        """Get portfolio for a specific user.

        Checks synthetic portfolios first, then the Plaid source if registered.
        """
        if not self._loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        if user_id in self._portfolios:
            return self._portfolios[user_id]
        if self._plaid_source and self._plaid_source.has_user(user_id):
            return self._plaid_source.get_portfolio(user_id)
        return None

    def get_all_portfolios(self) -> list[UserPortfolio]:
        """Get all loaded portfolios (synthetic only)."""
        if not self._loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        return list(self._portfolios.values())

    def get_user_ids(self) -> list[str]:
        """Get all user IDs (synthetic + Plaid if source is registered)."""
        if not self._loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        ids = list(self._portfolios.keys())
        if self._plaid_source:
            ids += self._plaid_source.get_user_ids()
        return ids

    def get_total_cash(self, user_id: str) -> Decimal:
        """Get total cash and equivalents for a user."""
        portfolio = self.get_portfolio(user_id)
        if not portfolio:
            return Decimal("0")

        bank_total = sum(
            acc.balance
            for acc in portfolio.bank_accounts
            if acc.status == AccountStatus.ACTIVE
        )
        investment_cash = sum(
            acc.cash_balance
            for acc in portfolio.investment_accounts
            if acc.status == AccountStatus.ACTIVE
        )
        return bank_total + investment_cash

    def get_holdings_by_type(self, user_id: str) -> dict[str, list[Holding]]:
        """Get holdings grouped by asset type."""
        portfolio = self.get_portfolio(user_id)
        if not portfolio:
            return {}

        result: dict[str, list[Holding]] = {}
        for holding in portfolio.holdings:
            asset_type = holding.asset_type.value
            if asset_type not in result:
                result[asset_type] = []
            result[asset_type].append(holding)
        return result

    def get_holdings_by_sector(self, user_id: str) -> dict[str, list[Holding]]:
        """Get holdings grouped by sector."""
        portfolio = self.get_portfolio(user_id)
        if not portfolio:
            return {}

        result: dict[str, list[Holding]] = {}
        for holding in portfolio.holdings:
            sector = holding.sector or "Unknown"
            if sector not in result:
                result[sector] = []
            result[sector].append(holding)
        return result

    def get_asset_allocation(
        self, user_id: str, exclude: list[str] | None = None
    ) -> dict[str, float]:
        """Get asset allocation percentages.

        Args:
            user_id: User ID.
            exclude: List of asset types to exclude (e.g., ["real_estate"]).

        Returns:
            Dict of asset type to percentage.
        """
        portfolio = self.get_portfolio(user_id)
        if not portfolio or not portfolio.summary:
            return {}

        allocation = portfolio.summary.allocation_by_asset_type.copy()

        if exclude:
            # Normalize exclude list (handle variations like "real estate", "real_estate")
            exclude_normalized = [e.lower().replace(" ", "_") for e in exclude]

            # Remove excluded asset types
            for key in list(allocation.keys()):
                key_normalized = key.lower().replace(" ", "_")
                if key_normalized in exclude_normalized:
                    del allocation[key]

            # Recalculate percentages to sum to 100%
            if allocation:
                total = sum(allocation.values())
                if total > 0:
                    allocation = {k: (v / total) * 100 for k, v in allocation.items()}

        return allocation

    def get_sector_allocation(self, user_id: str) -> dict[str, float]:
        """Get sector allocation percentages."""
        portfolio = self.get_portfolio(user_id)
        if not portfolio or not portfolio.summary:
            return {}
        return portfolio.summary.allocation_by_sector

    def get_net_worth(self, user_id: str) -> Decimal:
        """Get user's net worth."""
        portfolio = self.get_portfolio(user_id)
        if not portfolio or not portfolio.summary:
            return Decimal("0")
        return portfolio.summary.total_net_worth

    def get_liquidity_ratio(self, user_id: str) -> float:
        """Get user's liquidity ratio."""
        portfolio = self.get_portfolio(user_id)
        if not portfolio or not portfolio.summary:
            return 0.0
        return portfolio.summary.liquidity_ratio

    def get_real_estate_summary(self, user_id: str) -> dict:
        """Get real estate summary for a user."""
        portfolio = self.get_portfolio(user_id)
        if not portfolio:
            return {}

        owned_properties = [
            p for p in portfolio.real_estate if p.status == PropertyStatus.OWNED
        ]

        total_value = sum(p.current_value for p in owned_properties)
        total_equity = sum(p.equity for p in owned_properties)
        total_mortgage = sum(p.mortgage_balance or Decimal("0") for p in owned_properties)
        total_rental_income = sum(
            p.monthly_rental_income or Decimal("0") for p in owned_properties
        )

        return {
            "count": len(owned_properties),
            "total_value": total_value,
            "total_equity": total_equity,
            "total_mortgage": total_mortgage,
            "monthly_rental_income": total_rental_income,
            "properties": [
                {
                    "type": p.property_type.value,
                    "address": p.address,
                    "value": p.current_value,
                    "equity": p.equity,
                    "appreciation_percent": p.appreciation_percent,
                }
                for p in owned_properties
            ],
        }

    def get_top_holdings(self, user_id: str, n: int = 5) -> list[Holding]:
        """Get top N holdings by market value."""
        portfolio = self.get_portfolio(user_id)
        if not portfolio:
            return []

        sorted_holdings = sorted(
            portfolio.holdings, key=lambda h: h.market_value, reverse=True
        )
        return sorted_holdings[:n]

    def get_portfolio_metrics(self, user_id: str) -> dict:
        """Get comprehensive portfolio metrics."""
        portfolio = self.get_portfolio(user_id)
        if not portfolio:
            return {}

        summary = portfolio.summary
        if not summary:
            return {}

        # Calculate additional metrics
        total_unrealized_gain = sum(h.unrealized_gain_loss for h in portfolio.holdings)
        total_cost_basis = sum(h.cost_basis for h in portfolio.holdings)

        return {
            "net_worth": summary.total_net_worth,
            "total_assets": summary.total_assets,
            "total_liabilities": summary.total_liabilities,
            "cash_and_equivalents": summary.cash_and_equivalents,
            "liquidity_ratio": summary.liquidity_ratio,
            "debt_to_asset_ratio": summary.debt_to_asset_ratio,
            "monthly_savings_rate": summary.monthly_savings_rate,
            "total_unrealized_gain": total_unrealized_gain,
            "total_cost_basis": total_cost_basis,
            "unrealized_gain_percent": (
                float(total_unrealized_gain / total_cost_basis * 100)
                if total_cost_basis > 0
                else 0.0
            ),
            "num_bank_accounts": len(
                [a for a in portfolio.bank_accounts if a.status == AccountStatus.ACTIVE]
            ),
            "num_investment_accounts": len(
                [a for a in portfolio.investment_accounts if a.status == AccountStatus.ACTIVE]
            ),
            "num_holdings": len(portfolio.holdings),
            "num_properties": len(
                [p for p in portfolio.real_estate if p.status == PropertyStatus.OWNED]
            ),
        }

    def format_portfolio_context(self, user_id: str) -> str:
        """Format portfolio data as context for LLM."""
        portfolio = self.get_portfolio(user_id)
        if not portfolio:
            return "No portfolio data found."

        user = portfolio.user
        summary = portfolio.summary
        metrics = self.get_portfolio_metrics(user_id)

        context_parts = [
            f"## User Profile",
            f"- Name: {user.name}",
            f"- Age: {user.age}",
            f"- Occupation: {user.occupation}",
            f"- Annual Income: ${user.annual_income:,.2f}",
            f"- Monthly Expenses: ${user.monthly_expenses:,.2f}",
            f"- Monthly Savings: ${user.monthly_savings:,.2f}",
            "",
            f"## Portfolio Summary",
            f"- Net Worth: ${summary.total_net_worth:,.2f}" if summary else "",
            f"- Total Assets: ${summary.total_assets:,.2f}" if summary else "",
            f"- Total Liabilities: ${summary.total_liabilities:,.2f}" if summary else "",
            f"- Liquidity Ratio: {metrics.get('liquidity_ratio', 0):.2%}",
            f"- Debt-to-Asset Ratio: {metrics.get('debt_to_asset_ratio', 0):.2%}",
            f"- Monthly Savings Rate: {metrics.get('monthly_savings_rate', 0):.2%}" if metrics.get('monthly_savings_rate') else "",
            "",
            f"## Asset Allocation",
        ]

        if summary:
            for asset_type, pct in summary.allocation_by_asset_type.items():
                context_parts.append(f"- {asset_type.title()}: {pct:.1f}%")

        context_parts.extend(["", "## Bank Accounts"])
        for acc in portfolio.bank_accounts:
            if acc.status == AccountStatus.ACTIVE:
                context_parts.append(
                    f"- {acc.bank_name} ({acc.account_type.value}): ${acc.balance:,.2f}"
                )

        context_parts.extend(["", "## Investment Accounts"])
        for acc in portfolio.investment_accounts:
            if acc.status == AccountStatus.ACTIVE:
                context_parts.append(
                    f"- {acc.institution} ({acc.account_type.value}): ${acc.total_value:,.2f}"
                )

        context_parts.extend(["", "## Top Holdings"])
        for holding in self.get_top_holdings(user_id, 5):
            gain_pct = holding.unrealized_gain_loss_percent
            gain_sign = "+" if gain_pct >= 0 else ""
            context_parts.append(
                f"- {holding.symbol} ({holding.name}): ${holding.market_value:,.2f} ({gain_sign}{gain_pct:.1f}%)"
            )

        re_summary = self.get_real_estate_summary(user_id)
        if re_summary.get("count", 0) > 0:
            context_parts.extend(["", "## Real Estate"])
            for prop in re_summary["properties"]:
                context_parts.append(
                    f"- {prop['type'].replace('_', ' ').title()}: ${prop['value']:,.2f} (Equity: ${prop['equity']:,.2f})"
                )

        return "\n".join(context_parts)
