"""Unit tests for Pydantic data models."""

from datetime import date, datetime
from decimal import Decimal

import pytest

from backend.src.common.models import (
    AccountStatus,
    AssetType,
    BankAccount,
    BankAccountType,
    ExperienceLevel,
    GoalType,
    Holding,
    InvestmentAccount,
    InvestmentAccountType,
    InvestmentGoal,
    InvestmentHorizon,
    InvestmentProfile,
    PortfolioSummary,
    Priority,
    PropertyStatus,
    PropertyType,
    RealEstate,
    RiskTolerance,
    User,
    UserPortfolio,
)


class TestUser:
    """Tests for User model."""

    def test_user_creation(self):
        """Test basic user creation."""
        user = User(
            user_id="user_001",
            name="John Doe",
            email="john@example.com",
            age=35,
            occupation="Software Engineer",
            annual_income=Decimal("120000"),
            monthly_expenses=Decimal("5000"),
        )

        assert user.user_id == "user_001"
        assert user.name == "John Doe"
        assert user.monthly_income == Decimal("10000")
        assert user.monthly_savings == Decimal("5000")

    def test_user_monthly_calculations(self):
        """Test computed monthly income and savings."""
        user = User(
            user_id="user_002",
            name="Jane Doe",
            email="jane@example.com",
            age=40,
            occupation="Manager",
            annual_income=Decimal("240000"),
            monthly_expenses=Decimal("15000"),
        )

        assert user.monthly_income == Decimal("20000")
        assert user.monthly_savings == Decimal("5000")


class TestBankAccount:
    """Tests for BankAccount model."""

    def test_bank_account_creation(self):
        """Test basic bank account creation."""
        account = BankAccount(
            account_id="bank_001",
            user_id="user_001",
            account_type=BankAccountType.CHECKING,
            bank_name="Chase",
            balance=Decimal("5000.50"),
            status=AccountStatus.ACTIVE,
        )

        assert account.account_id == "bank_001"
        assert account.account_type == BankAccountType.CHECKING
        assert account.balance == Decimal("5000.50")
        assert account.status == AccountStatus.ACTIVE

    def test_bank_account_default_values(self):
        """Test default values."""
        account = BankAccount(
            account_id="bank_002",
            user_id="user_001",
            account_type=BankAccountType.SAVINGS,
            bank_name="Bank of America",
            balance=Decimal("10000"),
        )

        assert account.currency == "USD"
        assert account.interest_rate == Decimal("0")
        assert account.status == AccountStatus.ACTIVE


class TestHolding:
    """Tests for Holding model."""

    def test_holding_computed_fields(self):
        """Test computed market value and gain/loss."""
        holding = Holding(
            holding_id="hold_001",
            account_id="inv_001",
            asset_type=AssetType.STOCK,
            symbol="AAPL",
            name="Apple Inc.",
            quantity=Decimal("100"),
            average_cost=Decimal("150"),
            current_price=Decimal("180"),
            sector="Technology",
        )

        assert holding.market_value == Decimal("18000")
        assert holding.cost_basis == Decimal("15000")
        assert holding.unrealized_gain_loss == Decimal("3000")
        assert holding.unrealized_gain_loss_percent == 20.0

    def test_holding_negative_gain(self):
        """Test negative unrealized gain (loss)."""
        holding = Holding(
            holding_id="hold_002",
            account_id="inv_001",
            asset_type=AssetType.STOCK,
            symbol="META",
            name="Meta Platforms",
            quantity=Decimal("50"),
            average_cost=Decimal("300"),
            current_price=Decimal("250"),
        )

        assert holding.unrealized_gain_loss == Decimal("-2500")
        assert holding.unrealized_gain_loss_percent < 0


class TestRealEstate:
    """Tests for RealEstate model."""

    def test_real_estate_equity(self):
        """Test equity calculation with mortgage."""
        property = RealEstate(
            property_id="prop_001",
            user_id="user_001",
            property_type=PropertyType.PRIMARY_RESIDENCE,
            address="123 Main St",
            purchase_price=Decimal("500000"),
            current_value=Decimal("600000"),
            purchase_date=date(2020, 1, 1),
            mortgage_balance=Decimal("300000"),
        )

        assert property.equity == Decimal("300000")
        assert property.appreciation == Decimal("100000")
        assert property.appreciation_percent == 20.0

    def test_real_estate_no_mortgage(self):
        """Test equity when property is paid off."""
        property = RealEstate(
            property_id="prop_002",
            user_id="user_001",
            property_type=PropertyType.RENTAL,
            address="456 Oak Ave",
            purchase_price=Decimal("300000"),
            current_value=Decimal("400000"),
            purchase_date=date(2015, 6, 15),
            mortgage_balance=None,
        )

        assert property.equity == Decimal("400000")


class TestInvestmentProfile:
    """Tests for InvestmentProfile model."""

    def test_profile_completeness_full(self):
        """Test profile completeness when all fields filled."""
        profile = InvestmentProfile(
            user_id="user_001",
            monthly_income=Decimal("10000"),
            monthly_expenses=Decimal("5000"),
            risk_tolerance=RiskTolerance.MODERATE,
            investment_horizon=InvestmentHorizon.LONG,
            goals=[
                InvestmentGoal(
                    goal_type=GoalType.RETIREMENT,
                    target_amount=Decimal("1000000"),
                    priority=Priority.HIGH,
                )
            ],
            investment_experience=ExperienceLevel.INTERMEDIATE,
            liquidity_needs=Priority.LOW,
            loss_comfort="7",
            income_stability="stable",
            has_emergency_fund=True,
            debt_level="low",
        )

        assert profile.profile_completeness == 1.0

    def test_profile_completeness_partial(self):
        """Test profile completeness when partially filled."""
        profile = InvestmentProfile(
            user_id="user_001",
            risk_tolerance=RiskTolerance.AGGRESSIVE,
            investment_horizon=InvestmentHorizon.LONG,
        )

        # 2 out of 9 fields filled (risk_tolerance, investment_horizon)
        assert profile.profile_completeness < 1.0
        assert profile.profile_completeness > 0.0

    def test_profile_completeness_empty(self):
        """Test profile completeness when empty."""
        profile = InvestmentProfile(user_id="user_001")

        assert profile.profile_completeness == 0.0


class TestUserPortfolio:
    """Tests for UserPortfolio aggregate model."""

    def test_portfolio_creation(self):
        """Test creating a complete portfolio."""
        user = User(
            user_id="user_001",
            name="Test User",
            email="test@example.com",
            age=30,
            occupation="Engineer",
            annual_income=Decimal("100000"),
            monthly_expenses=Decimal("4000"),
        )

        bank_account = BankAccount(
            account_id="bank_001",
            user_id="user_001",
            account_type=BankAccountType.CHECKING,
            bank_name="Chase",
            balance=Decimal("10000"),
        )

        portfolio = UserPortfolio(
            user=user,
            bank_accounts=[bank_account],
        )

        assert portfolio.user.user_id == "user_001"
        assert len(portfolio.bank_accounts) == 1
        assert portfolio.investment_accounts == []
        assert portfolio.holdings == []
        assert portfolio.real_estate == []
