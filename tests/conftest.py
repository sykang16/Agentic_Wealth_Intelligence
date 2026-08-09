"""Pytest configuration and shared fixtures."""

from decimal import Decimal
from datetime import datetime, date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from backend.src.common.llm_client import LLMResponse
from backend.src.common.models import (
    AccountStatus,
    AssetType,
    BankAccount,
    BankAccountType,
    Holding,
    InvestmentAccount,
    InvestmentAccountType,
    PortfolioSummary,
    RealEstate,
    PropertyType,
    PropertyStatus,
    User,
    UserPortfolio,
)


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def data_dir(project_root: Path) -> Path:
    """Return the data directory."""
    return project_root / "data"


@pytest.fixture
def synthetic_data_dir(data_dir: Path) -> Path:
    """Return the synthetic data directory."""
    return data_dir / "synthetic"


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client that returns configurable responses."""
    mock = MagicMock()
    mock.chat.return_value = LLMResponse(
        content="Mock response",
        model="test-model",
        provider="test",
        usage={"input_tokens": 10, "output_tokens": 5},
    )
    mock.chat_with_history.return_value = LLMResponse(
        content="Mock response",
        model="test-model",
        provider="test",
        usage={"input_tokens": 10, "output_tokens": 5},
    )
    return mock


@pytest.fixture
def sample_user() -> User:
    """Create a sample user."""
    return User(
        user_id="user_001",
        name="Test User",
        email="test@example.com",
        age=35,
        occupation="Software Engineer",
        annual_income=Decimal("120000"),
        monthly_expenses=Decimal("5000"),
        created_at=datetime.now(),
    )


@pytest.fixture
def sample_holdings() -> list[Holding]:
    """Create sample holdings."""
    return [
        Holding(
            holding_id="h1",
            account_id="acc1",
            asset_type=AssetType.STOCK,
            symbol="AAPL",
            name="Apple Inc.",
            quantity=Decimal("50"),
            average_cost=Decimal("150.00"),
            current_price=Decimal("175.00"),
            sector="Technology",
            last_updated=datetime.now(),
        ),
        Holding(
            holding_id="h2",
            account_id="acc1",
            asset_type=AssetType.ETF,
            symbol="VOO",
            name="Vanguard S&P 500 ETF",
            quantity=Decimal("100"),
            average_cost=Decimal("380.00"),
            current_price=Decimal("420.00"),
            sector="Broad Market",
            last_updated=datetime.now(),
        ),
        Holding(
            holding_id="h3",
            account_id="acc1",
            asset_type=AssetType.STOCK,
            symbol="MSFT",
            name="Microsoft Corporation",
            quantity=Decimal("30"),
            average_cost=Decimal("300.00"),
            current_price=Decimal("350.00"),
            sector="Technology",
            last_updated=datetime.now(),
        ),
    ]


@pytest.fixture
def sample_portfolio(sample_user, sample_holdings) -> UserPortfolio:
    """Create a sample portfolio for testing."""
    bank_accounts = [
        BankAccount(
            account_id="ba1",
            user_id="user_001",
            account_type=BankAccountType.CHECKING,
            bank_name="Test Bank",
            balance=Decimal("15000"),
            currency="USD",
            interest_rate=Decimal("0.01"),
            status=AccountStatus.ACTIVE,
            last_updated=datetime.now(),
        ),
        BankAccount(
            account_id="ba2",
            user_id="user_001",
            account_type=BankAccountType.SAVINGS,
            bank_name="Test Bank",
            balance=Decimal("50000"),
            currency="USD",
            interest_rate=Decimal("0.04"),
            status=AccountStatus.ACTIVE,
            last_updated=datetime.now(),
        ),
    ]

    investment_accounts = [
        InvestmentAccount(
            account_id="acc1",
            user_id="user_001",
            account_type=InvestmentAccountType.BROKERAGE,
            institution="Test Brokerage",
            total_value=Decimal("62250"),
            cash_balance=Decimal("5000"),
            status=AccountStatus.ACTIVE,
            last_updated=datetime.now(),
        ),
    ]

    real_estate = [
        RealEstate(
            property_id="re1",
            user_id="user_001",
            property_type=PropertyType.PRIMARY_RESIDENCE,
            address="123 Main St",
            purchase_price=Decimal("400000"),
            current_value=Decimal("500000"),
            purchase_date=date(2020, 1, 15),
            mortgage_balance=Decimal("300000"),
            status=PropertyStatus.OWNED,
            last_updated=datetime.now(),
        ),
    ]

    summary = PortfolioSummary(
        user_id="user_001",
        total_assets=Decimal("627250"),
        total_liabilities=Decimal("300000"),
        total_net_worth=Decimal("327250"),
        cash_and_equivalents=Decimal("65000"),
        stocks=Decimal("20250"),
        etfs=Decimal("42000"),
        real_estate_equity=Decimal("200000"),
        liquidity_ratio=0.1036,
        debt_to_asset_ratio=0.4785,
        monthly_savings_rate=0.5833,
        allocation_by_asset_type={
            "cash": 10.36,
            "stocks": 5.59,
            "etf": 6.70,
            "real_estate": 79.72,
        },
        allocation_by_sector={
            "Technology": 65.0,
            "Broad Market": 35.0,
        },
    )

    return UserPortfolio(
        user=sample_user,
        bank_accounts=bank_accounts,
        investment_accounts=investment_accounts,
        holdings=sample_holdings,
        real_estate=real_estate,
        summary=summary,
    )


@pytest.fixture
def sample_aggregator_with_data(sample_portfolio):
    """Create a mock aggregator pre-loaded with sample data."""
    aggregator = MagicMock()
    aggregator.get_portfolio.return_value = sample_portfolio
    aggregator.get_user_ids.return_value = ["user_001"]
    aggregator.get_net_worth.return_value = Decimal("327250")
    aggregator.get_total_cash.return_value = Decimal("65000")
    aggregator.get_top_holdings.return_value = sample_portfolio.holdings
    aggregator.get_asset_allocation.return_value = {
        "cash": 10.36,
        "stocks": 5.59,
        "etf": 6.70,
        "real_estate": 79.72,
    }
    aggregator.get_sector_allocation.return_value = {
        "Technology": 65.0,
        "Broad Market": 35.0,
    }
    aggregator.get_portfolio_metrics.return_value = {
        "net_worth": 327250,
        "total_assets": 627250,
        "total_liabilities": 300000,
        "liquidity_ratio": 0.1036,
    }
    aggregator.format_portfolio_context.return_value = "Portfolio context for user_001"
    return aggregator
