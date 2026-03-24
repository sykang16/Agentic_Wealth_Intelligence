"""Price updater service — uses yfinance (no API key, no quota).

Price history is appended to a JSON sidecar file next to the portfolio data:
    data/synthetic/price_history.json

History entry per symbol:
    {"date": "2024-01-15", "price": 185.23, "updated_at": "<iso>"}

One entry per calendar day; re-running on the same day overwrites that day's entry.
History is capped at HISTORY_KEEP_DAYS entries per symbol.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Callable

import yfinance as yf

from backend.src.common.models import (
    AccountStatus,
    AssetType,
    PortfolioSummary,
    PropertyStatus,
    UserPortfolio,
)

from .aggregator import PortfolioAggregator

logger = logging.getLogger(__name__)

HISTORY_KEEP_DAYS = 365  # rolling window kept per symbol


class PriceUpdater:
    """Fetches live prices via yfinance and updates stored portfolio data.

    No API key required. All asset types (stocks, ETFs, bonds, mutual funds,
    crypto) are supported. Prices are fetched in a single batched request.
    """

    def __init__(
        self,
        aggregator: PortfolioAggregator,
        history_path: str | Path | None = None,
    ):
        self._aggregator = aggregator
        if history_path:
            self._history_path = Path(history_path)
        elif aggregator.data_path:
            self._history_path = aggregator.data_path.parent / "price_history.json"
        else:
            self._history_path = None

    # ------------------------------------------------------------------
    # History I/O
    # ------------------------------------------------------------------

    def _load_history(self) -> dict:
        if self._history_path and self._history_path.exists():
            with open(self._history_path) as f:
                return json.load(f)
        return {}

    def _save_history(self, history: dict) -> None:
        if not self._history_path:
            return
        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._history_path, "w") as f:
            json.dump(history, f, indent=2)

    def get_price_history(self, symbol: str) -> list[dict]:
        """Return stored price history for *symbol*, oldest first."""
        return self._load_history().get(symbol.upper(), [])

    def _append_to_history(self, prices: dict[str, Decimal], timestamp: datetime) -> None:
        """Append today's prices to history; one entry per symbol per day."""
        history = self._load_history()
        date_str = timestamp.strftime("%Y-%m-%d")

        for symbol, price in prices.items():
            entries: list[dict] = history.setdefault(symbol, [])
            entry = {
                "date": date_str,
                "price": float(price),
                "updated_at": timestamp.isoformat(),
            }
            # Overwrite today's entry if it already exists, else append
            if entries and entries[-1]["date"] == date_str:
                entries[-1] = entry
            else:
                entries.append(entry)
            # Trim to rolling window
            if len(entries) > HISTORY_KEEP_DAYS:
                history[symbol] = entries[-HISTORY_KEEP_DAYS:]

        self._save_history(history)

    # ------------------------------------------------------------------
    # Staleness detection
    # ------------------------------------------------------------------

    def is_stale(self, max_age_hours: int = 24) -> bool:
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        for portfolio in self._aggregator.get_all_portfolios():
            for holding in portfolio.holdings:
                if holding.last_updated < cutoff:
                    return True
        return False

    def get_oldest_update(self) -> datetime | None:
        oldest: datetime | None = None
        for portfolio in self._aggregator.get_all_portfolios():
            for holding in portfolio.holdings:
                if oldest is None or holding.last_updated < oldest:
                    oldest = holding.last_updated
        return oldest

    # ------------------------------------------------------------------
    # Symbol helpers
    # ------------------------------------------------------------------

    def get_unique_symbols(self) -> list[str]:
        """All unique ticker symbols across all portfolios (original form)."""
        symbols: set[str] = set()
        for portfolio in self._aggregator.get_all_portfolios():
            for holding in portfolio.holdings:
                symbols.add(holding.symbol.upper())
        return sorted(symbols)

    def _build_yf_symbol_map(self) -> dict[str, str]:
        """Return mapping of yfinance ticker → original portfolio symbol.

        Crypto holdings get a '-USD' suffix for Yahoo Finance (e.g. ADA → ADA-USD).
        All other asset types are passed through unchanged.
        """
        mapping: dict[str, str] = {}
        for portfolio in self._aggregator.get_all_portfolios():
            for holding in portfolio.holdings:
                original = holding.symbol.upper()
                if holding.asset_type == AssetType.CRYPTO:
                    yf_sym = f"{original}-USD"
                else:
                    yf_sym = original
                mapping[yf_sym] = original
        return mapping

    # ------------------------------------------------------------------
    # Price fetching
    # ------------------------------------------------------------------

    def update_all_prices(
        self,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> dict:
        """Fetch current prices for all symbols via yfinance, update holdings,
        append to price history, and persist to disk.

        Returns:
            {"updated": [...], "failed": [...], "skipped": []}
        """
        # Build yfinance symbol → original symbol mapping (crypto gets -USD suffix)
        yf_map = self._build_yf_symbol_map()  # {yf_sym: original_sym}
        items = sorted(yf_map.items())         # [(yf_sym, original), ...]
        total = len(items)

        if not items:
            return {"updated": [], "failed": [], "skipped": []}

        updated: list[str] = []
        failed: list[str] = []
        prices: dict[str, Decimal] = {}  # keyed by original symbol

        # Fetch one ticker at a time — avoids batch retry loops
        for idx, (yf_sym, original) in enumerate(items):
            if progress_callback:
                progress_callback(idx, total, original)
            try:
                hist = yf.Ticker(yf_sym).history(period="2d")
                if hist.empty:
                    raise ValueError("no data returned")
                price = Decimal(str(round(float(hist["Close"].iloc[-1]), 4)))
                prices[original] = price
                updated.append(original)
            except Exception as exc:
                logger.warning("Failed to fetch %s (%s): %s", original, yf_sym, exc)
                failed.append(original)

        if progress_callback:
            progress_callback(total, total, "Saving...")

        # Apply new prices to holdings
        now = datetime.now()
        for portfolio in self._aggregator.get_all_portfolios():
            changed = False
            for holding in portfolio.holdings:
                sym = holding.symbol.upper()
                if sym in prices:
                    holding.current_price = prices[sym]
                    holding.last_updated = now
                    changed = True
            if changed:
                portfolio.summary = self._recalculate_summary(portfolio)

        self._aggregator.save_portfolios()

        if prices:
            self._append_to_history(prices, now)

        return {"updated": updated, "failed": failed, "skipped": []}

    # ------------------------------------------------------------------
    # Summary recalculation
    # ------------------------------------------------------------------

    def _recalculate_summary(self, portfolio: UserPortfolio) -> PortfolioSummary:
        """Recalculate portfolio summary from updated holdings."""
        user = portfolio.user
        bank_accounts = portfolio.bank_accounts
        investment_accounts = portfolio.investment_accounts
        holdings = portfolio.holdings
        real_estate = portfolio.real_estate

        cash_and_equivalents = sum(
            (acc.balance for acc in bank_accounts if acc.status == AccountStatus.ACTIVE),
            Decimal("0"),
        ) + sum(
            (acc.cash_balance for acc in investment_accounts if acc.status == AccountStatus.ACTIVE),
            Decimal("0"),
        )

        stocks = bonds = etfs = mutual_funds = crypto = Decimal("0")
        sector_values: dict[str, Decimal] = {}

        for holding in holdings:
            value = holding.market_value
            if holding.asset_type == AssetType.STOCK:
                stocks += value
            elif holding.asset_type == AssetType.BOND:
                bonds += value
            elif holding.asset_type == AssetType.ETF:
                etfs += value
            elif holding.asset_type == AssetType.MUTUAL_FUND:
                mutual_funds += value
            elif holding.asset_type == AssetType.CRYPTO:
                crypto += value
            if holding.sector:
                sector_values[holding.sector] = (
                    sector_values.get(holding.sector, Decimal("0")) + value
                )

        real_estate_equity = sum(
            (prop.equity for prop in real_estate if prop.status == PropertyStatus.OWNED),
            Decimal("0"),
        )
        total_mortgages = sum(
            (prop.mortgage_balance or Decimal("0") for prop in real_estate),
            Decimal("0"),
        )

        total_assets = (
            cash_and_equivalents + stocks + bonds + etfs + mutual_funds + crypto
            + sum((prop.current_value for prop in real_estate), Decimal("0"))
        )
        total_liabilities = total_mortgages
        total_net_worth = total_assets - total_liabilities

        allocation_by_asset_type: dict[str, float] = {}
        if total_assets > 0:
            allocation_by_asset_type = {
                "cash": float(cash_and_equivalents / total_assets * 100),
                "stocks": float(stocks / total_assets * 100),
                "bonds": float(bonds / total_assets * 100),
                "etfs": float(etfs / total_assets * 100),
                "mutual_funds": float(mutual_funds / total_assets * 100),
                "crypto": float(crypto / total_assets * 100),
                "real_estate": float(
                    sum((prop.current_value for prop in real_estate), Decimal("0"))
                    / total_assets * 100
                ),
            }

        allocation_by_sector: dict[str, float] = {}
        total_investment_value = stocks + bonds + etfs + mutual_funds + crypto
        if total_investment_value > 0:
            for sector, value in sector_values.items():
                allocation_by_sector[sector] = float(value / total_investment_value * 100)

        liquidity_ratio = float(cash_and_equivalents / total_assets) if total_assets > 0 else 0.0
        debt_to_asset_ratio = float(total_liabilities / total_assets) if total_assets > 0 else 0.0
        monthly_savings_rate = (
            float(user.monthly_savings / user.monthly_income)
            if user.monthly_income > 0 else None
        )

        return PortfolioSummary(
            user_id=user.user_id,
            total_assets=total_assets.quantize(Decimal("0.01")),
            total_liabilities=total_liabilities.quantize(Decimal("0.01")),
            total_net_worth=total_net_worth.quantize(Decimal("0.01")),
            cash_and_equivalents=cash_and_equivalents.quantize(Decimal("0.01")),
            stocks=stocks.quantize(Decimal("0.01")),
            bonds=bonds.quantize(Decimal("0.01")),
            etfs=etfs.quantize(Decimal("0.01")),
            mutual_funds=mutual_funds.quantize(Decimal("0.01")),
            crypto=crypto.quantize(Decimal("0.01")),
            real_estate_equity=real_estate_equity.quantize(Decimal("0.01")),
            allocation_by_asset_type=allocation_by_asset_type,
            allocation_by_sector=allocation_by_sector,
            liquidity_ratio=round(liquidity_ratio, 4),
            debt_to_asset_ratio=round(debt_to_asset_ratio, 4),
            monthly_savings_rate=round(monthly_savings_rate, 4) if monthly_savings_rate else None,
            last_updated=datetime.now(),
        )
