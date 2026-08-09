"""Unit tests for Asset Agent and related modules."""

from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.src.asset_management.aggregator import PortfolioAggregator
from backend.src.asset_management.query_interface import (
    AssetQueryInterface,
    QueryIntent,
    QueryType,
)
from backend.src.asset_management.visualization import VisualizationEngine
from backend.src.common.models import (
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


@pytest.fixture
def sample_portfolio() -> UserPortfolio:
    """Create a sample portfolio for testing."""
    user = User(
        user_id="test_user",
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
            user_id="test_user",
            account_type=BankAccountType.CHECKING,
            bank_name="Chase",
            balance=Decimal("10000"),
        ),
        BankAccount(
            account_id="bank_002",
            user_id="test_user",
            account_type=BankAccountType.SAVINGS,
            bank_name="Ally",
            balance=Decimal("25000"),
        ),
    ]

    investment_accounts = [
        InvestmentAccount(
            account_id="inv_001",
            user_id="test_user",
            account_type=InvestmentAccountType.BROKERAGE,
            institution="Fidelity",
            total_value=Decimal("100000"),
            cash_balance=Decimal("5000"),
        ),
    ]

    holdings = [
        Holding(
            holding_id="hold_001",
            account_id="inv_001",
            asset_type=AssetType.STOCK,
            symbol="AAPL",
            name="Apple Inc.",
            quantity=Decimal("50"),
            average_cost=Decimal("150"),
            current_price=Decimal("180"),
            sector="Technology",
        ),
        Holding(
            holding_id="hold_002",
            account_id="inv_001",
            asset_type=AssetType.ETF,
            symbol="VOO",
            name="Vanguard S&P 500",
            quantity=Decimal("100"),
            average_cost=Decimal("400"),
            current_price=Decimal("450"),
            sector="Broad Market",
        ),
    ]

    summary = PortfolioSummary(
        user_id="test_user",
        total_assets=Decimal("135000"),
        total_liabilities=Decimal("0"),
        total_net_worth=Decimal("135000"),
        cash_and_equivalents=Decimal("40000"),
        stocks=Decimal("9000"),
        etfs=Decimal("45000"),
        allocation_by_asset_type={
            "cash": 29.6,
            "stocks": 6.7,
            "etfs": 33.3,
        },
        allocation_by_sector={
            "Technology": 16.7,
            "Broad Market": 83.3,
        },
        liquidity_ratio=0.30,
        debt_to_asset_ratio=0.0,
        monthly_savings_rate=0.50,
    )

    return UserPortfolio(
        user=user,
        bank_accounts=bank_accounts,
        investment_accounts=investment_accounts,
        holdings=holdings,
        summary=summary,
    )


@pytest.fixture
def aggregator_with_data(sample_portfolio: UserPortfolio, tmp_path: Path):
    """Create aggregator with sample data."""
    import json

    data_file = tmp_path / "portfolios.json"
    data = [sample_portfolio.model_dump(mode="json")]
    with open(data_file, "w") as f:
        json.dump(data, f, default=str)

    aggregator = PortfolioAggregator(data_file)
    aggregator.load_data()
    return aggregator


class TestPortfolioAggregator:
    """Tests for PortfolioAggregator."""

    def test_load_data(self, aggregator_with_data: PortfolioAggregator):
        """Test loading portfolio data."""
        assert len(aggregator_with_data.get_user_ids()) == 1
        assert "test_user" in aggregator_with_data.get_user_ids()

    def test_get_portfolio(self, aggregator_with_data: PortfolioAggregator):
        """Test getting a specific portfolio."""
        portfolio = aggregator_with_data.get_portfolio("test_user")
        assert portfolio is not None
        assert portfolio.user.name == "Test User"

    def test_get_total_cash(self, aggregator_with_data: PortfolioAggregator):
        """Test calculating total cash."""
        cash = aggregator_with_data.get_total_cash("test_user")
        # Bank accounts (10000 + 25000) + investment cash (5000)
        assert cash == Decimal("40000")

    def test_get_holdings_by_type(self, aggregator_with_data: PortfolioAggregator):
        """Test grouping holdings by type."""
        by_type = aggregator_with_data.get_holdings_by_type("test_user")
        assert "stock" in by_type
        assert "etf" in by_type
        assert len(by_type["stock"]) == 1
        assert len(by_type["etf"]) == 1

    def test_get_net_worth(self, aggregator_with_data: PortfolioAggregator):
        """Test getting net worth."""
        net_worth = aggregator_with_data.get_net_worth("test_user")
        assert net_worth == Decimal("135000")

    def test_get_top_holdings(self, aggregator_with_data: PortfolioAggregator):
        """Test getting top holdings."""
        top = aggregator_with_data.get_top_holdings("test_user", 5)
        assert len(top) == 2
        # VOO should be first (45000 > 9000)
        assert top[0].symbol == "VOO"

    def test_format_portfolio_context(self, aggregator_with_data: PortfolioAggregator):
        """Test formatting portfolio as context."""
        context = aggregator_with_data.format_portfolio_context("test_user")
        assert "Test User" in context
        assert "Net Worth" in context
        assert "AAPL" in context or "VOO" in context


class TestQueryInterface:
    """Tests for AssetQueryInterface."""

    def test_fallback_intent_detection_net_worth(self):
        """Test fallback intent detection for net worth."""
        aggregator = MagicMock()
        interface = AssetQueryInterface(aggregator, llm_client=MagicMock())

        intent = interface._fallback_intent_detection("What's my net worth?")
        assert intent.query_type == QueryType.NET_WORTH

    def test_fallback_intent_detection_allocation(self):
        """Test fallback intent detection for allocation."""
        aggregator = MagicMock()
        interface = AssetQueryInterface(aggregator, llm_client=MagicMock())

        intent = interface._fallback_intent_detection("Show my asset allocation")
        assert intent.query_type == QueryType.ASSET_ALLOCATION
        assert intent.requires_visualization

    def test_fallback_intent_detection_liquidity(self):
        """Test fallback intent detection for liquidity."""
        aggregator = MagicMock()
        interface = AssetQueryInterface(aggregator, llm_client=MagicMock())

        intent = interface._fallback_intent_detection("What's my liquidity ratio?")
        assert intent.query_type == QueryType.LIQUIDITY

    def test_fallback_intent_detection_holdings(self):
        """Test fallback intent detection for holdings."""
        aggregator = MagicMock()
        interface = AssetQueryInterface(aggregator, llm_client=MagicMock())

        intent = interface._fallback_intent_detection("Show my top 5 holdings")
        assert intent.query_type == QueryType.TOP_HOLDINGS


class TestVisualizationEngine:
    """Tests for VisualizationEngine."""

    def test_create_asset_allocation_pie(self):
        """Test creating asset allocation pie chart."""
        viz = VisualizationEngine()
        allocation = {"stocks": 50.0, "bonds": 30.0, "cash": 20.0}

        fig = viz.create_asset_allocation_pie(allocation)
        assert fig is not None
        assert len(fig.data) > 0

    def test_create_holdings_bar(self, sample_portfolio: UserPortfolio):
        """Test creating holdings bar chart."""
        viz = VisualizationEngine()

        fig = viz.create_holdings_bar(sample_portfolio.holdings)
        assert fig is not None
        assert len(fig.data) > 0

    def test_create_net_worth_breakdown(self, sample_portfolio: UserPortfolio):
        """Test creating net worth waterfall chart."""
        viz = VisualizationEngine()

        fig = viz.create_net_worth_breakdown(sample_portfolio)
        assert fig is not None

    def test_create_metrics_dashboard(self):
        """Test creating metrics dashboard."""
        viz = VisualizationEngine()
        metrics = {
            "net_worth": 135000,
            "liquidity_ratio": 0.30,
            "monthly_savings_rate": 0.50,
            "total_assets": 135000,
            "total_unrealized_gain": 5000,
            "debt_to_asset_ratio": 0.0,
        }

        fig = viz.create_metrics_dashboard(metrics)
        assert fig is not None

    def test_empty_figure(self):
        """Test creating empty figure with message."""
        viz = VisualizationEngine()
        fig = viz._empty_figure("No data available")
        assert fig is not None
