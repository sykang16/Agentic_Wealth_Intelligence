"""Unit tests for PriceUpdater."""

import asyncio
import json
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.src.asset_management.aggregator import PortfolioAggregator
from backend.src.asset_management.price_updater import PriceUpdater
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


# ── Fixtures ──────────────────────────────────────────────────────────


def _make_holding(
    symbol: str,
    price: Decimal,
    qty: Decimal = Decimal("10"),
    last_updated: datetime | None = None,
    asset_type: AssetType = AssetType.STOCK,
) -> Holding:
    return Holding(
        holding_id=f"h_{symbol}",
        account_id="acc_001",
        asset_type=asset_type,
        symbol=symbol,
        name=f"{symbol} Corp",
        quantity=qty,
        average_cost=Decimal("100"),
        current_price=price,
        sector="Technology",
        last_updated=last_updated or datetime.now(),
    )


def _make_portfolio(
    user_id: str = "user_001",
    holdings: list[Holding] | None = None,
) -> UserPortfolio:
    user = User(
        user_id=user_id,
        name="Test User",
        email="test@example.com",
        age=35,
        occupation="Engineer",
        annual_income=Decimal("120000"),
        monthly_expenses=Decimal("5000"),
    )
    bank_accounts = [
        BankAccount(
            account_id="bank_001",
            user_id=user_id,
            account_type=BankAccountType.CHECKING,
            bank_name="Test Bank",
            balance=Decimal("10000"),
        )
    ]
    investment_accounts = [
        InvestmentAccount(
            account_id="acc_001",
            user_id=user_id,
            account_type=InvestmentAccountType.BROKERAGE,
            institution="Test Broker",
            total_value=Decimal("50000"),
            cash_balance=Decimal("5000"),
        )
    ]
    if holdings is None:
        holdings = [
            _make_holding("AAPL", Decimal("150")),
            _make_holding("MSFT", Decimal("300")),
        ]
    summary = PortfolioSummary(
        user_id=user_id,
        total_assets=Decimal("100000"),
        total_liabilities=Decimal("0"),
        total_net_worth=Decimal("100000"),
    )
    return UserPortfolio(
        user=user,
        bank_accounts=bank_accounts,
        investment_accounts=investment_accounts,
        holdings=holdings,
        summary=summary,
    )


@pytest.fixture()
def aggregator(tmp_path: Path) -> PortfolioAggregator:
    """Create a PortfolioAggregator with two users and write data to a temp file."""
    portfolio1 = _make_portfolio(
        "user_001",
        [
            _make_holding("AAPL", Decimal("150")),
            _make_holding("MSFT", Decimal("300")),
        ],
    )
    portfolio2 = _make_portfolio(
        "user_002",
        [
            _make_holding("MSFT", Decimal("300")),
            _make_holding("GOOGL", Decimal("2800")),
        ],
    )

    data_file = tmp_path / "portfolios.json"
    data = [
        portfolio1.model_dump(mode="json"),
        portfolio2.model_dump(mode="json"),
    ]
    data_file.write_text(json.dumps(data, default=str))

    agg = PortfolioAggregator(data_path=data_file)
    agg.load_data()
    return agg


# ── Tests ──────────────────────────────────────────────────────────────


class TestIsSale:
    """Staleness detection."""

    def test_is_stale_true(self, aggregator: PortfolioAggregator):
        """Holdings with last_updated > 24h ago should be considered stale."""
        # Push all holdings into the past
        old_time = datetime.now() - timedelta(hours=25)
        for p in aggregator.get_all_portfolios():
            for h in p.holdings:
                h.last_updated = old_time

        updater = PriceUpdater(aggregator)
        assert updater.is_stale() is True

    def test_is_stale_false(self, aggregator: PortfolioAggregator):
        """Holdings updated within the last 24h should NOT be stale."""
        recent = datetime.now() - timedelta(hours=1)
        for p in aggregator.get_all_portfolios():
            for h in p.holdings:
                h.last_updated = recent

        updater = PriceUpdater(aggregator)
        assert updater.is_stale() is False

    def test_is_stale_custom_threshold(self, aggregator: PortfolioAggregator):
        """Custom max_age_hours threshold should be respected."""
        old_time = datetime.now() - timedelta(hours=5)
        for p in aggregator.get_all_portfolios():
            for h in p.holdings:
                h.last_updated = old_time

        updater = PriceUpdater(aggregator)
        assert updater.is_stale(max_age_hours=4) is True
        assert updater.is_stale(max_age_hours=6) is False


class TestGetUniqueSymbols:
    """Symbol deduplication."""

    def test_get_unique_symbols(self, aggregator: PortfolioAggregator):
        """Symbols should be deduplicated across users and sorted."""
        updater = PriceUpdater(aggregator)
        symbols = updater.get_unique_symbols()
        assert symbols == ["AAPL", "GOOGL", "MSFT"]


class TestUpdateHoldingsPrices:
    """End-to-end update flow (with mocked API)."""

    def test_update_holdings_prices(self, aggregator: PortfolioAggregator):
        """After update, current_price and last_updated should change."""
        updater = PriceUpdater(aggregator, api_key="test-key")

        new_prices = {
            "AAPL": Decimal("160"),
            "MSFT": Decimal("310"),
            "GOOGL": Decimal("2900"),
        }

        async def _mock_fetch(symbol, client=None):
            return new_prices.get(symbol.upper())

        updater.fetch_price = _mock_fetch

        result = asyncio.run(updater.update_all_prices())

        assert set(result["updated"]) == {"AAPL", "GOOGL", "MSFT"}
        assert result["failed"] == []

        # Verify prices updated
        p1 = aggregator.get_portfolio("user_001")
        for h in p1.holdings:
            if h.symbol == "AAPL":
                assert h.current_price == Decimal("160")
            elif h.symbol == "MSFT":
                assert h.current_price == Decimal("310")

        p2 = aggregator.get_portfolio("user_002")
        for h in p2.holdings:
            if h.symbol == "GOOGL":
                assert h.current_price == Decimal("2900")


class TestRecalculateSummary:
    """Summary fields update after price change."""

    def test_recalculate_summary(self, aggregator: PortfolioAggregator):
        """Summary total_assets should reflect new market values."""
        updater = PriceUpdater(aggregator, api_key="test-key")

        portfolio = aggregator.get_portfolio("user_001")

        # Recalculate the baseline summary from actual holdings
        original_summary = updater._recalculate_summary(portfolio)

        # Double the price of all holdings
        for h in portfolio.holdings:
            h.current_price = h.current_price * 2

        new_summary = updater._recalculate_summary(portfolio)

        # Holdings value should have increased
        assert new_summary.stocks > original_summary.stocks
        assert new_summary.total_assets > original_summary.total_assets
        assert new_summary.total_net_worth > original_summary.total_net_worth


class TestPartialFailure:
    """Failed symbols should not block others."""

    def test_partial_failure(self, aggregator: PortfolioAggregator):
        """If one symbol fails, others should still update."""
        updater = PriceUpdater(aggregator, api_key="test-key")

        async def _mock_fetch(symbol, client=None):
            if symbol.upper() == "MSFT":
                return None  # Simulate failure
            return Decimal("999")

        updater.fetch_price = _mock_fetch

        result = asyncio.run(updater.update_all_prices())

        assert "MSFT" in result["failed"]
        assert "AAPL" in result["updated"]
        assert "GOOGL" in result["updated"]

        # AAPL should be updated
        p1 = aggregator.get_portfolio("user_001")
        aapl = [h for h in p1.holdings if h.symbol == "AAPL"][0]
        assert aapl.current_price == Decimal("999")

        # MSFT should retain its original price
        msft = [h for h in p1.holdings if h.symbol == "MSFT"][0]
        assert msft.current_price == Decimal("300")


class TestSkipUnconfigured:
    """No API key → graceful error."""

    def test_skip_unconfigured(self, aggregator: PortfolioAggregator):
        """Without an API key, update_all_prices returns an error dict."""
        updater = PriceUpdater(aggregator, api_key=None)

        # Also clear the env var
        with patch.dict("os.environ", {}, clear=True):
            updater._api_key = None
            result = asyncio.run(updater.update_all_prices())

        assert "error" in result
        assert result["updated"] == []

    def test_fetch_price_returns_none_without_key(self, aggregator: PortfolioAggregator):
        """fetch_price should return None if no API key."""
        updater = PriceUpdater(aggregator, api_key=None)
        updater._api_key = None

        result = asyncio.run(updater.fetch_price("AAPL"))
        assert result is None


class TestSavePortfolios:
    """Test that aggregator.save_portfolios() persists data."""

    def test_save_and_reload(self, aggregator: PortfolioAggregator):
        """Saved data should be loadable and match."""
        # Modify a price
        p = aggregator.get_portfolio("user_001")
        p.holdings[0].current_price = Decimal("999.99")

        aggregator.save_portfolios()

        # Reload
        agg2 = PortfolioAggregator(data_path=aggregator.data_path)
        agg2.load_data()
        p2 = agg2.get_portfolio("user_001")
        assert p2.holdings[0].current_price == Decimal("999.99")


class TestGetOldestUpdate:
    """Test get_oldest_update helper."""

    def test_get_oldest_update(self, aggregator: PortfolioAggregator):
        """Should return the earliest last_updated across all holdings."""
        updater = PriceUpdater(aggregator)

        old_time = datetime(2020, 1, 1)
        aggregator.get_portfolio("user_002").holdings[0].last_updated = old_time

        assert updater.get_oldest_update() == old_time

    def test_get_oldest_update_empty(self):
        """Should return None when there are no portfolios."""
        agg = PortfolioAggregator()
        agg._portfolios = {}
        agg._loaded = True
        updater = PriceUpdater(agg)
        assert updater.get_oldest_update() is None


class TestSkipUnsupportedAssetTypes:
    """Crypto and mutual fund symbols should be excluded from fetching."""

    def test_crypto_and_mutual_fund_excluded(self, tmp_path: Path):
        """get_unique_symbols should not include crypto or mutual fund symbols."""
        portfolio = _make_portfolio(
            "user_001",
            [
                _make_holding("AAPL", Decimal("150"), asset_type=AssetType.STOCK),
                _make_holding("ETH", Decimal("3000"), asset_type=AssetType.CRYPTO),
                _make_holding("VFIAX", Decimal("400"), asset_type=AssetType.MUTUAL_FUND),
                _make_holding("VOO", Decimal("450"), asset_type=AssetType.ETF),
            ],
        )

        data_file = tmp_path / "portfolios.json"
        data_file.write_text(json.dumps([portfolio.model_dump(mode="json")], default=str))
        agg = PortfolioAggregator(data_path=data_file)
        agg.load_data()

        updater = PriceUpdater(agg)
        assert updater.get_unique_symbols() == ["AAPL", "VOO"]
        assert updater.get_skipped_symbols() == ["ETH", "VFIAX"]

    def test_skipped_symbols_not_fetched(self, tmp_path: Path):
        """update_all_prices should not attempt to fetch skipped symbols."""
        portfolio = _make_portfolio(
            "user_001",
            [
                _make_holding("AAPL", Decimal("150"), asset_type=AssetType.STOCK),
                _make_holding("BTC", Decimal("60000"), asset_type=AssetType.CRYPTO),
            ],
        )

        data_file = tmp_path / "portfolios.json"
        data_file.write_text(json.dumps([portfolio.model_dump(mode="json")], default=str))
        agg = PortfolioAggregator(data_path=data_file)
        agg.load_data()

        updater = PriceUpdater(agg, api_key="test-key")
        fetched_symbols = []

        async def _mock_fetch(symbol, client=None):
            fetched_symbols.append(symbol)
            return Decimal("999")

        updater.fetch_price = _mock_fetch
        result = asyncio.run(updater.update_all_prices())

        # Only AAPL should have been fetched, BTC should be in skipped
        assert fetched_symbols == ["AAPL"]
        assert "AAPL" in result["updated"]
        assert "BTC" in result["skipped"]


class TestRateLimitEarlyStop:
    """When daily quota is hit, remaining symbols should be skipped."""

    def test_rate_limit_stops_early(self, aggregator: PortfolioAggregator):
        """After a rate-limit error, remaining symbols should be skipped."""
        from backend.src.asset_management.price_updater import _RateLimitExceeded

        updater = PriceUpdater(aggregator, api_key="test-key")
        call_count = 0

        async def _mock_fetch(symbol, client=None):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise _RateLimitExceeded("daily limit")
            return Decimal("999")

        updater.fetch_price = _mock_fetch
        result = asyncio.run(updater.update_all_prices())

        # First symbol succeeded, second hit rate limit, third was skipped
        assert len(result["updated"]) == 1
        assert len(result["failed"]) == 1  # the one that hit the limit
        assert len(result["skipped"]) == 1  # remaining symbols
        assert result.get("rate_limited") is True
        # Should not have attempted the 3rd symbol
        assert call_count == 2
