"""Synthetic data generator for the Agentic Wealth Intelligence system."""

import json
import random
import uuid
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from faker import Faker

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
    InvestmentProfile,
    PortfolioSummary,
    Priority,
    PropertyStatus,
    PropertyType,
    RealEstate,
    RiskTolerance,
    InvestmentHorizon,
    User,
    UserPortfolio,
)
from backend.src.data_generation.constants import (
    BANKS,
    BONDS,
    CITIES,
    CRYPTO,
    ETFS,
    INSTITUTIONS,
    INTEREST_RATES,
    MUTUAL_FUNDS,
    OCCUPATIONS,
    SAVINGS_RATES,
    STOCKS,
)


fake = Faker()
Faker.seed(42)
random.seed(42)


class SyntheticDataGenerator:
    """Generate synthetic financial data for testing and development."""

    def __init__(self, seed: int | None = 42):
        """Initialize generator with optional seed for reproducibility."""
        if seed is not None:
            Faker.seed(seed)
            random.seed(seed)
        self.fake = Faker()

    def _generate_id(self, prefix: str) -> str:
        """Generate a unique ID with prefix."""
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    def _random_decimal(self, min_val: float, max_val: float, decimals: int = 2) -> Decimal:
        """Generate a random decimal value."""
        value = random.uniform(min_val, max_val)
        return Decimal(str(round(value, decimals)))

    def _get_savings_rate_range(self, age: int) -> tuple[float, float]:
        """Get savings rate range based on age."""
        if age < 30:
            return SAVINGS_RATES["20-30"]
        elif age < 40:
            return SAVINGS_RATES["30-40"]
        elif age < 50:
            return SAVINGS_RATES["40-50"]
        elif age < 60:
            return SAVINGS_RATES["50-60"]
        else:
            return SAVINGS_RATES["60+"]

    def generate_user(self, user_id: str | None = None) -> User:
        """Generate a synthetic user."""
        if user_id is None:
            user_id = self._generate_id("user")

        occupation_data = random.choice(OCCUPATIONS)
        annual_income = self._random_decimal(*occupation_data["income_range"])

        age = random.randint(25, 70)
        savings_rate_range = self._get_savings_rate_range(age)
        savings_rate = random.uniform(*savings_rate_range)
        monthly_income = annual_income / 12
        monthly_expenses = monthly_income * Decimal(str(1 - savings_rate))

        return User(
            user_id=user_id,
            name=self.fake.name(),
            email=self.fake.email(),
            age=age,
            occupation=occupation_data["title"],
            annual_income=annual_income,
            monthly_expenses=monthly_expenses.quantize(Decimal("0.01")),
            created_at=self.fake.date_time_between(start_date="-5y", end_date="now"),
        )

    def generate_bank_accounts(
        self, user_id: str, count: int | None = None
    ) -> list[BankAccount]:
        """Generate synthetic bank accounts for a user."""
        if count is None:
            count = random.randint(1, 3)

        accounts = []
        account_types = list(BankAccountType)

        for i in range(count):
            account_type = account_types[i % len(account_types)]
            interest_rate_range = INTEREST_RATES.get(
                account_type.value, (0.0, 0.01)
            )

            # Balance ranges by account type
            if account_type == BankAccountType.CHECKING:
                balance = self._random_decimal(1000, 25000)
            elif account_type == BankAccountType.SAVINGS:
                balance = self._random_decimal(5000, 100000)
            else:  # money_market
                balance = self._random_decimal(10000, 200000)

            accounts.append(
                BankAccount(
                    account_id=self._generate_id("bank"),
                    user_id=user_id,
                    account_type=account_type,
                    bank_name=random.choice(BANKS),
                    balance=balance,
                    currency="USD",
                    interest_rate=self._random_decimal(*interest_rate_range, decimals=4),
                    status=AccountStatus.ACTIVE,
                    last_updated=datetime.now(),
                )
            )

        return accounts

    def generate_investment_accounts(
        self, user_id: str, count: int | None = None
    ) -> list[InvestmentAccount]:
        """Generate synthetic investment accounts for a user."""
        if count is None:
            count = random.randint(1, 3)

        accounts = []
        account_types = [
            InvestmentAccountType.BROKERAGE,
            InvestmentAccountType.FOUR01K,
            InvestmentAccountType.ROTH_IRA,
            InvestmentAccountType.IRA,
        ]

        selected_types = random.sample(account_types, min(count, len(account_types)))

        for account_type in selected_types:
            # Value ranges by account type
            if account_type == InvestmentAccountType.FOUR01K:
                total_value = self._random_decimal(20000, 500000)
            elif account_type in [InvestmentAccountType.IRA, InvestmentAccountType.ROTH_IRA]:
                total_value = self._random_decimal(10000, 200000)
            else:  # brokerage
                total_value = self._random_decimal(5000, 300000)

            cash_balance = total_value * self._random_decimal(0.02, 0.10)

            accounts.append(
                InvestmentAccount(
                    account_id=self._generate_id("inv"),
                    user_id=user_id,
                    account_type=account_type,
                    institution=random.choice(INSTITUTIONS),
                    total_value=total_value,
                    cash_balance=cash_balance.quantize(Decimal("0.01")),
                    status=AccountStatus.ACTIVE,
                    last_updated=datetime.now(),
                )
            )

        return accounts

    def generate_holdings(
        self, account_id: str, account_value: Decimal, count: int | None = None
    ) -> list[Holding]:
        """Generate synthetic holdings for an investment account."""
        if count is None:
            count = random.randint(3, 10)

        holdings = []
        remaining_value = account_value * Decimal("0.95")  # Leave some as cash

        # Determine asset mix
        asset_pools = [
            (STOCKS, AssetType.STOCK, 0.5),
            (ETFS, AssetType.ETF, 0.3),
            (BONDS, AssetType.BOND, 0.1),
            (MUTUAL_FUNDS, AssetType.MUTUAL_FUND, 0.05),
            (CRYPTO, AssetType.CRYPTO, 0.05),
        ]

        selected_assets = []
        for pool, asset_type, weight in asset_pools:
            num_from_pool = max(1, int(count * weight))
            selected = random.sample(pool, min(num_from_pool, len(pool)))
            for asset in selected:
                selected_assets.append((asset, asset_type))

        # Shuffle and trim to count
        random.shuffle(selected_assets)
        selected_assets = selected_assets[:count]

        # Distribute value among holdings
        weights = [random.random() for _ in selected_assets]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        for (asset, asset_type), weight in zip(selected_assets, weights):
            target_value = remaining_value * Decimal(str(weight))
            price_range = asset.get("price_range", (10, 100))
            current_price = self._random_decimal(*price_range)

            # Calculate quantity
            quantity = (target_value / current_price).quantize(Decimal("0.0001"))
            if quantity <= 0:
                quantity = Decimal("0.0001")

            # Average cost is typically close to current price (random gain/loss)
            gain_loss_pct = random.uniform(-0.30, 0.50)
            average_cost = current_price / Decimal(str(1 + gain_loss_pct))
            average_cost = average_cost.quantize(Decimal("0.01"))

            holdings.append(
                Holding(
                    holding_id=self._generate_id("hold"),
                    account_id=account_id,
                    asset_type=asset_type,
                    symbol=asset["symbol"],
                    name=asset["name"],
                    quantity=quantity,
                    average_cost=average_cost,
                    current_price=current_price,
                    sector=asset.get("sector"),
                    last_updated=datetime.now(),
                )
            )

        return holdings

    def generate_real_estate(
        self, user_id: str, count: int | None = None, age: int = 40
    ) -> list[RealEstate]:
        """Generate synthetic real estate for a user."""
        if count is None:
            # Probability of owning property increases with age
            if age < 30:
                count = random.choices([0, 1], weights=[0.7, 0.3])[0]
            elif age < 40:
                count = random.choices([0, 1], weights=[0.4, 0.6])[0]
            elif age < 50:
                count = random.choices([0, 1, 2], weights=[0.2, 0.6, 0.2])[0]
            else:
                count = random.choices([0, 1, 2], weights=[0.1, 0.5, 0.4])[0]

        if count == 0:
            return []

        properties = []
        city, state = random.choice(CITIES)

        # First property is usually primary residence
        property_types = [PropertyType.PRIMARY_RESIDENCE]
        if count > 1:
            property_types.extend(
                random.choices(
                    [PropertyType.RENTAL, PropertyType.VACATION, PropertyType.LAND],
                    weights=[0.6, 0.3, 0.1],
                    k=count - 1,
                )
            )

        for prop_type in property_types:
            # Value ranges by property type
            if prop_type == PropertyType.PRIMARY_RESIDENCE:
                purchase_price = self._random_decimal(200000, 800000)
            elif prop_type == PropertyType.RENTAL:
                purchase_price = self._random_decimal(150000, 500000)
            elif prop_type == PropertyType.VACATION:
                purchase_price = self._random_decimal(100000, 400000)
            else:  # land
                purchase_price = self._random_decimal(50000, 200000)

            # Appreciation over time
            years_owned = random.randint(1, 15)
            appreciation_rate = random.uniform(0.02, 0.08)  # 2-8% annual
            current_value = purchase_price * Decimal(
                str((1 + appreciation_rate) ** years_owned)
            )
            current_value = current_value.quantize(Decimal("0.01"))

            # Mortgage
            has_mortgage = random.random() < 0.7
            if has_mortgage:
                # Remaining mortgage is typically 50-90% of original
                original_ltv = random.uniform(0.7, 0.95)
                paydown_pct = random.uniform(0.1, 0.5)
                mortgage_balance = purchase_price * Decimal(
                    str(original_ltv * (1 - paydown_pct))
                )
                mortgage_balance = mortgage_balance.quantize(Decimal("0.01"))
            else:
                mortgage_balance = None

            # Rental income for rental properties
            if prop_type == PropertyType.RENTAL:
                monthly_rental_income = self._random_decimal(1500, 4000)
            else:
                monthly_rental_income = None

            purchase_date = date.today() - timedelta(days=years_owned * 365)

            properties.append(
                RealEstate(
                    property_id=self._generate_id("prop"),
                    user_id=user_id,
                    property_type=prop_type,
                    address=f"{self.fake.street_address()}, {city}, {state}",
                    purchase_price=purchase_price,
                    current_value=current_value,
                    purchase_date=purchase_date,
                    mortgage_balance=mortgage_balance,
                    monthly_rental_income=monthly_rental_income,
                    status=PropertyStatus.OWNED,
                    last_updated=datetime.now(),
                )
            )

        return properties

    def generate_investment_profile(
        self, user_id: str, user: User | None = None, completeness: float = 1.0
    ) -> InvestmentProfile:
        """Generate a synthetic investment profile."""
        # Determine which fields to fill based on completeness
        fill_field = lambda: random.random() < completeness

        # Risk tolerance correlates with age (younger = more aggressive)
        age = user.age if user else random.randint(25, 70)
        if age < 35:
            risk_weights = [0.1, 0.3, 0.6]  # conservative, moderate, aggressive
        elif age < 50:
            risk_weights = [0.2, 0.5, 0.3]
        else:
            risk_weights = [0.4, 0.4, 0.2]

        risk_tolerance = (
            random.choices(list(RiskTolerance), weights=risk_weights)[0]
            if fill_field()
            else None
        )

        # Investment horizon correlates with age
        if age < 40:
            horizon_weights = [0.1, 0.3, 0.6]
        elif age < 55:
            horizon_weights = [0.2, 0.5, 0.3]
        else:
            horizon_weights = [0.4, 0.4, 0.2]

        investment_horizon = (
            random.choices(list(InvestmentHorizon), weights=horizon_weights)[0]
            if fill_field()
            else None
        )

        # Goals
        goals = []
        if fill_field():
            num_goals = random.randint(1, 3)
            goal_types = random.sample(list(GoalType), num_goals)
            for goal_type in goal_types:
                target_amount = self._random_decimal(10000, 1000000)
                years_to_goal = random.randint(1, 20)
                goals.append(
                    InvestmentGoal(
                        goal_type=goal_type,
                        target_amount=target_amount,
                        target_date=date.today() + timedelta(days=years_to_goal * 365),
                        priority=random.choice(list(Priority)),
                    )
                )

        # Experience correlates slightly with age
        experience = (
            random.choices(
                list(ExperienceLevel),
                weights=[0.4, 0.4, 0.2] if age < 35 else [0.2, 0.4, 0.4],
            )[0]
            if fill_field()
            else None
        )

        # Preferences
        preferred_sectors = (
            random.sample(
                ["Technology", "Healthcare", "Financial Services", "Energy", "Consumer"],
                random.randint(1, 3),
            )
            if fill_field()
            else []
        )

        excluded_sectors = (
            random.sample(
                ["Tobacco", "Gambling", "Weapons", "Fossil Fuels"],
                random.randint(0, 2),
            )
            if fill_field()
            else []
        )

        # Income/expenses from user if available
        monthly_income = user.monthly_income if user and fill_field() else None
        monthly_expenses = user.monthly_expenses if user and fill_field() else None
        monthly_savings_target = (
            (monthly_income - monthly_expenses) * Decimal("0.8")
            if monthly_income and monthly_expenses and fill_field()
            else None
        )
        if monthly_savings_target:
            monthly_savings_target = monthly_savings_target.quantize(Decimal("0.01"))

        return InvestmentProfile(
            user_id=user_id,
            monthly_income=monthly_income,
            monthly_expenses=monthly_expenses,
            monthly_savings_target=monthly_savings_target,
            risk_tolerance=risk_tolerance,
            investment_horizon=investment_horizon,
            goals=goals,
            investment_experience=experience,
            preferred_sectors=preferred_sectors,
            excluded_sectors=excluded_sectors,
            esg_preference=random.choice([True, False, None]) if fill_field() else None,
            liquidity_needs=random.choice(list(Priority)) if fill_field() else None,
            tax_sensitivity=random.choice(list(Priority)) if fill_field() else None,
            last_updated=datetime.now(),
        )

    def calculate_portfolio_summary(
        self,
        user_id: str,
        user: User,
        bank_accounts: list[BankAccount],
        investment_accounts: list[InvestmentAccount],
        holdings: list[Holding],
        real_estate: list[RealEstate],
    ) -> PortfolioSummary:
        """Calculate portfolio summary from all assets."""
        # Cash and equivalents (bank accounts + investment cash)
        cash_and_equivalents = sum(
            (acc.balance for acc in bank_accounts if acc.status == AccountStatus.ACTIVE),
            Decimal("0"),
        ) + sum(
            (acc.cash_balance for acc in investment_accounts if acc.status == AccountStatus.ACTIVE),
            Decimal("0"),
        )

        # Holdings by type
        stocks = Decimal("0")
        bonds = Decimal("0")
        etfs = Decimal("0")
        mutual_funds = Decimal("0")
        crypto = Decimal("0")

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
                sector_values[holding.sector] = sector_values.get(
                    holding.sector, Decimal("0")
                ) + value

        # Real estate
        real_estate_equity = sum(
            (prop.equity for prop in real_estate if prop.status == PropertyStatus.OWNED),
            Decimal("0"),
        )
        total_mortgages = sum(
            (prop.mortgage_balance or Decimal("0") for prop in real_estate),
            Decimal("0"),
        )

        # Totals
        total_assets = (
            cash_and_equivalents
            + stocks
            + bonds
            + etfs
            + mutual_funds
            + crypto
            + sum((prop.current_value for prop in real_estate), Decimal("0"))
        )
        total_liabilities = total_mortgages
        total_net_worth = total_assets - total_liabilities

        # Allocation by asset type
        allocation_by_asset_type = {}
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
                    / total_assets
                    * 100
                ),
            }

        # Allocation by sector
        allocation_by_sector = {}
        total_investment_value = stocks + bonds + etfs + mutual_funds + crypto
        if total_investment_value > 0:
            for sector, value in sector_values.items():
                allocation_by_sector[sector] = float(
                    value / total_investment_value * 100
                )

        # Metrics
        liquidity_ratio = (
            float(cash_and_equivalents / total_assets) if total_assets > 0 else 0.0
        )
        debt_to_asset_ratio = (
            float(total_liabilities / total_assets) if total_assets > 0 else 0.0
        )
        monthly_savings_rate = (
            float(user.monthly_savings / user.monthly_income)
            if user.monthly_income > 0
            else None
        )

        return PortfolioSummary(
            user_id=user_id,
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

    def generate_user_portfolio(
        self,
        user_id: str | None = None,
        profile_completeness: float = 1.0,
    ) -> UserPortfolio:
        """Generate a complete portfolio for a user."""
        # Generate user
        user = self.generate_user(user_id)

        # Generate accounts
        bank_accounts = self.generate_bank_accounts(user.user_id)
        investment_accounts = self.generate_investment_accounts(user.user_id)

        # Generate holdings for each investment account
        holdings = []
        for account in investment_accounts:
            account_holdings = self.generate_holdings(
                account.account_id, account.total_value
            )
            holdings.extend(account_holdings)

        # Generate real estate
        real_estate = self.generate_real_estate(user.user_id, age=user.age)

        # Generate investment profile
        investment_profile = self.generate_investment_profile(
            user.user_id, user, completeness=profile_completeness
        )

        # Calculate summary
        summary = self.calculate_portfolio_summary(
            user.user_id,
            user,
            bank_accounts,
            investment_accounts,
            holdings,
            real_estate,
        )

        return UserPortfolio(
            user=user,
            bank_accounts=bank_accounts,
            investment_accounts=investment_accounts,
            holdings=holdings,
            real_estate=real_estate,
            investment_profile=investment_profile,
            summary=summary,
        )

    def generate_multiple_portfolios(self, count: int = 5) -> list[UserPortfolio]:
        """Generate multiple user portfolios."""
        portfolios = []
        for i in range(count):
            # Vary profile completeness
            completeness = random.choice([0.3, 0.5, 0.7, 1.0])
            portfolio = self.generate_user_portfolio(
                user_id=f"user_{i+1:03d}",
                profile_completeness=completeness,
            )
            portfolios.append(portfolio)
        return portfolios

    def save_to_json(
        self, portfolios: list[UserPortfolio], output_dir: str | Path
    ) -> Path:
        """Save portfolios to JSON file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "synthetic_portfolios.json"

        # Convert to JSON-serializable format
        data = [portfolio.model_dump(mode="json") for portfolio in portfolios]

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

        return output_file


def generate_sample_data(output_dir: str | Path = "data/synthetic", count: int = 5) -> Path:
    """Convenience function to generate sample data."""
    generator = SyntheticDataGenerator(seed=42)
    portfolios = generator.generate_multiple_portfolios(count=count)
    return generator.save_to_json(portfolios, output_dir)


if __name__ == "__main__":
    output_path = generate_sample_data()
    print(f"Generated synthetic data at: {output_path}")
