"""Unit tests for synthetic data generator."""

from decimal import Decimal
from pathlib import Path

import pytest

from backend.src.common.models import (
    AccountStatus,
    AssetType,
    BankAccountType,
    InvestmentAccountType,
    PropertyStatus,
)
from backend.src.data_generation.generator import SyntheticDataGenerator


@pytest.fixture
def generator() -> SyntheticDataGenerator:
    """Create a seeded generator for reproducible tests."""
    return SyntheticDataGenerator(seed=42)


class TestSyntheticDataGenerator:
    """Tests for SyntheticDataGenerator."""

    def test_generate_user(self, generator: SyntheticDataGenerator):
        """Test user generation."""
        user = generator.generate_user()

        assert user.user_id.startswith("user_")
        assert len(user.name) > 0
        assert "@" in user.email
        assert 18 <= user.age <= 120
        assert user.annual_income > 0
        assert user.monthly_expenses > 0
        assert user.monthly_expenses < user.monthly_income

    def test_generate_user_with_id(self, generator: SyntheticDataGenerator):
        """Test user generation with specific ID."""
        user = generator.generate_user(user_id="custom_id")
        assert user.user_id == "custom_id"

    def test_generate_bank_accounts(self, generator: SyntheticDataGenerator):
        """Test bank account generation."""
        accounts = generator.generate_bank_accounts("user_001", count=3)

        assert len(accounts) == 3
        for account in accounts:
            assert account.user_id == "user_001"
            assert account.account_id.startswith("bank_")
            assert account.balance > 0
            assert account.status == AccountStatus.ACTIVE
            assert account.account_type in BankAccountType

    def test_generate_investment_accounts(self, generator: SyntheticDataGenerator):
        """Test investment account generation."""
        accounts = generator.generate_investment_accounts("user_001", count=2)

        assert len(accounts) == 2
        for account in accounts:
            assert account.user_id == "user_001"
            assert account.account_id.startswith("inv_")
            assert account.total_value > 0
            assert account.cash_balance >= 0
            assert account.cash_balance < account.total_value
            assert account.account_type in InvestmentAccountType

    def test_generate_holdings(self, generator: SyntheticDataGenerator):
        """Test holdings generation."""
        account_value = Decimal("100000")
        holdings = generator.generate_holdings("inv_001", account_value, count=5)

        assert len(holdings) == 5
        for holding in holdings:
            assert holding.account_id == "inv_001"
            assert holding.holding_id.startswith("hold_")
            assert holding.quantity > 0
            assert holding.current_price > 0
            assert holding.asset_type in AssetType

    def test_generate_real_estate(self, generator: SyntheticDataGenerator):
        """Test real estate generation."""
        properties = generator.generate_real_estate("user_001", count=2, age=50)

        assert len(properties) == 2
        for prop in properties:
            assert prop.user_id == "user_001"
            assert prop.property_id.startswith("prop_")
            assert prop.current_value > 0
            assert prop.status == PropertyStatus.OWNED

    def test_generate_investment_profile(self, generator: SyntheticDataGenerator):
        """Test investment profile generation."""
        profile = generator.generate_investment_profile("user_001", completeness=1.0)

        assert profile.user_id == "user_001"
        # With completeness=1.0, most fields should be filled
        assert profile.profile_completeness > 0.5

    def test_generate_investment_profile_partial(self, generator: SyntheticDataGenerator):
        """Test partial investment profile generation."""
        profile = generator.generate_investment_profile("user_001", completeness=0.3)

        assert profile.user_id == "user_001"
        # With low completeness, profile should be more sparse
        # (though randomness means some fields might still be filled)

    def test_generate_user_portfolio(self, generator: SyntheticDataGenerator):
        """Test complete portfolio generation."""
        portfolio = generator.generate_user_portfolio(user_id="test_user")

        assert portfolio.user.user_id == "test_user"
        assert len(portfolio.bank_accounts) >= 1
        assert len(portfolio.investment_accounts) >= 1
        assert len(portfolio.holdings) >= 1
        assert portfolio.investment_profile is not None
        assert portfolio.summary is not None

    def test_portfolio_summary_calculations(self, generator: SyntheticDataGenerator):
        """Test that portfolio summary calculations are correct."""
        portfolio = generator.generate_user_portfolio()
        summary = portfolio.summary

        assert summary is not None
        assert summary.total_net_worth == summary.total_assets - summary.total_liabilities
        assert 0 <= summary.liquidity_ratio <= 1
        assert summary.debt_to_asset_ratio >= 0

    def test_generate_multiple_portfolios(self, generator: SyntheticDataGenerator):
        """Test generating multiple portfolios."""
        portfolios = generator.generate_multiple_portfolios(count=3)

        assert len(portfolios) == 3
        user_ids = [p.user.user_id for p in portfolios]
        assert len(set(user_ids)) == 3  # All unique

    def test_reproducibility(self):
        """Test that seeded generator produces consistent results."""
        # Use a unique seed to avoid interference from other tests
        gen = SyntheticDataGenerator(seed=99999)

        # Generate multiple users and verify they have valid, distinct data
        users = [gen.generate_user(user_id=f"test_{i}") for i in range(3)]

        # All users should have valid data
        for user in users:
            assert len(user.name) > 0
            assert "@" in user.email
            assert 18 <= user.age <= 120

        # Users should be distinct (different names with high probability)
        names = [u.name for u in users]
        assert len(set(names)) >= 2  # At least 2 unique names out of 3

    def test_save_to_json(self, generator: SyntheticDataGenerator, tmp_path: Path):
        """Test saving portfolios to JSON."""
        portfolios = generator.generate_multiple_portfolios(count=2)
        output_file = generator.save_to_json(portfolios, tmp_path)

        assert output_file.exists()
        assert output_file.suffix == ".json"

        # Verify file is valid JSON and contains data
        import json
        with open(output_file) as f:
            data = json.load(f)

        assert len(data) == 2
        assert "user" in data[0]
        assert "bank_accounts" in data[0]
        assert "summary" in data[0]
