"""Pydantic data models for the Agentic Wealth Intelligence system."""

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, computed_field


# ============================================================
# ENUMS
# ============================================================


class AccountStatus(str, Enum):
    """Status of financial accounts."""

    ACTIVE = "active"
    CLOSED = "closed"
    FROZEN = "frozen"


class BankAccountType(str, Enum):
    """Types of bank accounts."""

    CHECKING = "checking"
    SAVINGS = "savings"
    MONEY_MARKET = "money_market"


class InvestmentAccountType(str, Enum):
    """Types of investment accounts."""

    BROKERAGE = "brokerage"
    IRA = "ira"
    ROTH_IRA = "roth_ira"
    FOUR01K = "401k"


class AssetType(str, Enum):
    """Types of investment assets."""

    STOCK = "stock"
    ETF = "etf"
    BOND = "bond"
    MUTUAL_FUND = "mutual_fund"
    CRYPTO = "crypto"


class PropertyType(str, Enum):
    """Types of real estate properties."""

    PRIMARY_RESIDENCE = "primary_residence"
    RENTAL = "rental"
    VACATION = "vacation"
    LAND = "land"
    COMMERCIAL = "commercial"


class PropertyStatus(str, Enum):
    """Status of real estate properties."""

    OWNED = "owned"
    SOLD = "sold"
    PENDING = "pending"


class RiskTolerance(str, Enum):
    """Investment risk tolerance levels."""

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class InvestmentHorizon(str, Enum):
    """Investment time horizon."""

    SHORT = "short"  # < 3 years
    MEDIUM = "medium"  # 3-10 years
    LONG = "long"  # > 10 years


class ExperienceLevel(str, Enum):
    """Investment experience levels."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class Priority(str, Enum):
    """Priority levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class GoalType(str, Enum):
    """Types of investment goals."""

    RETIREMENT = "retirement"
    HOUSE = "house"
    EDUCATION = "education"
    EMERGENCY = "emergency"
    WEALTH = "wealth"
    OTHER = "other"


# ============================================================
# CORE MODELS
# ============================================================


class User(BaseModel):
    """User/Customer information."""

    user_id: str = Field(..., description="Unique user identifier")
    name: str = Field(..., description="Full name")
    email: str = Field(..., description="Email address")
    age: int = Field(..., ge=18, le=120, description="Age in years")
    occupation: str = Field(..., description="Current occupation")
    annual_income: Decimal = Field(..., ge=0, description="Annual income in USD")
    monthly_expenses: Decimal = Field(..., ge=0, description="Average monthly expenses in USD")
    created_at: datetime = Field(default_factory=datetime.now)

    @computed_field
    @property
    def monthly_income(self) -> Decimal:
        """Calculate monthly income from annual."""
        return self.annual_income / 12

    @computed_field
    @property
    def monthly_savings(self) -> Decimal:
        """Calculate monthly savings potential."""
        return self.monthly_income - self.monthly_expenses


class BankAccount(BaseModel):
    """Bank account (checking, savings, money market)."""

    account_id: str = Field(..., description="Unique account identifier")
    user_id: str = Field(..., description="Owner's user ID")
    account_type: BankAccountType = Field(..., description="Type of bank account")
    bank_name: str = Field(..., description="Name of the bank")
    balance: Decimal = Field(..., ge=0, description="Current balance in USD")
    currency: str = Field(default="USD", description="Account currency")
    interest_rate: Decimal = Field(default=Decimal("0"), ge=0, description="Annual interest rate")
    status: AccountStatus = Field(default=AccountStatus.ACTIVE, description="Account status")
    last_updated: datetime = Field(default_factory=datetime.now)


class InvestmentAccount(BaseModel):
    """Investment account (brokerage, IRA, 401k)."""

    account_id: str = Field(..., description="Unique account identifier")
    user_id: str = Field(..., description="Owner's user ID")
    account_type: InvestmentAccountType = Field(..., description="Type of investment account")
    institution: str = Field(..., description="Financial institution name")
    total_value: Decimal = Field(..., ge=0, description="Total account value in USD")
    cash_balance: Decimal = Field(..., ge=0, description="Cash balance in USD")
    status: AccountStatus = Field(default=AccountStatus.ACTIVE, description="Account status")
    last_updated: datetime = Field(default_factory=datetime.now)


class Holding(BaseModel):
    """Individual asset holding within an investment account."""

    holding_id: str = Field(..., description="Unique holding identifier")
    account_id: str = Field(..., description="Parent investment account ID")
    asset_type: AssetType = Field(..., description="Type of asset")
    symbol: str = Field(..., description="Ticker symbol")
    name: str = Field(..., description="Asset name")
    quantity: Decimal = Field(..., gt=0, description="Number of units held")
    average_cost: Decimal = Field(..., ge=0, description="Average cost per unit")
    current_price: Decimal = Field(..., ge=0, description="Current market price per unit")
    sector: str | None = Field(default=None, description="Market sector")
    last_updated: datetime = Field(default_factory=datetime.now)

    @computed_field
    @property
    def market_value(self) -> Decimal:
        """Calculate current market value."""
        return self.quantity * self.current_price

    @computed_field
    @property
    def cost_basis(self) -> Decimal:
        """Calculate total cost basis."""
        return self.quantity * self.average_cost

    @computed_field
    @property
    def unrealized_gain_loss(self) -> Decimal:
        """Calculate unrealized gain/loss."""
        return self.market_value - self.cost_basis

    @computed_field
    @property
    def unrealized_gain_loss_percent(self) -> float:
        """Calculate unrealized gain/loss percentage."""
        if self.cost_basis == 0:
            return 0.0
        return float((self.unrealized_gain_loss / self.cost_basis) * 100)


class RealEstate(BaseModel):
    """Real estate property."""

    property_id: str = Field(..., description="Unique property identifier")
    user_id: str = Field(..., description="Owner's user ID")
    property_type: PropertyType = Field(..., description="Type of property")
    address: str = Field(..., description="Property address")
    purchase_price: Decimal = Field(..., ge=0, description="Original purchase price")
    current_value: Decimal = Field(..., ge=0, description="Current estimated value")
    purchase_date: date = Field(..., description="Date of purchase")
    mortgage_balance: Decimal | None = Field(
        default=None, ge=0, description="Outstanding mortgage balance"
    )
    monthly_rental_income: Decimal | None = Field(
        default=None, ge=0, description="Monthly rental income if applicable"
    )
    status: PropertyStatus = Field(default=PropertyStatus.OWNED, description="Property status")
    last_updated: datetime = Field(default_factory=datetime.now)

    @computed_field
    @property
    def equity(self) -> Decimal:
        """Calculate property equity (value - mortgage)."""
        if self.mortgage_balance is None:
            return self.current_value
        return self.current_value - self.mortgage_balance

    @computed_field
    @property
    def appreciation(self) -> Decimal:
        """Calculate total appreciation since purchase."""
        return self.current_value - self.purchase_price

    @computed_field
    @property
    def appreciation_percent(self) -> float:
        """Calculate appreciation percentage."""
        if self.purchase_price == 0:
            return 0.0
        return float((self.appreciation / self.purchase_price) * 100)


# ============================================================
# INVESTMENT PROFILE (Module B)
# ============================================================


class InvestmentGoal(BaseModel):
    """Individual investment goal."""

    goal_type: GoalType = Field(..., description="Type of financial goal")
    target_amount: Decimal = Field(..., ge=0, description="Target amount in USD")
    target_date: date | None = Field(default=None, description="Target date to achieve goal")
    priority: Priority = Field(default=Priority.MEDIUM, description="Goal priority")
    description: str | None = Field(default=None, description="Additional details")


class InvestmentProfile(BaseModel):
    """User's investment profile for personalized recommendations."""

    user_id: str = Field(..., description="User ID")

    # Income & Expenses
    monthly_income: Decimal | None = Field(default=None, ge=0, description="Monthly income")
    monthly_expenses: Decimal | None = Field(default=None, ge=0, description="Monthly expenses")
    monthly_savings_target: Decimal | None = Field(
        default=None, ge=0, description="Monthly savings target"
    )

    # Risk Assessment
    risk_tolerance: RiskTolerance | None = Field(default=None, description="Risk tolerance level")
    investment_horizon: InvestmentHorizon | None = Field(
        default=None, description="Investment time horizon"
    )

    # Goals
    goals: list[InvestmentGoal] = Field(default_factory=list, description="Investment goals")

    # Experience
    investment_experience: ExperienceLevel | None = Field(
        default=None, description="Investment experience level"
    )

    # Preferences
    preferred_sectors: list[str] = Field(default_factory=list, description="Preferred sectors")
    excluded_sectors: list[str] = Field(default_factory=list, description="Excluded sectors")
    esg_preference: bool | None = Field(default=None, description="Prefer ESG investments")

    # Constraints
    liquidity_needs: Priority | None = Field(default=None, description="Liquidity requirements")
    tax_sensitivity: Priority | None = Field(default=None, description="Tax sensitivity level")

    # Profiling-only slots (collected by ProfilingAgent, not derivable from financial data)
    loss_comfort: str | None = Field(default=None, description="Loss comfort level (1-10 scale)")
    income_stability: str | None = Field(default=None, description="Income stability (stable/variable/uncertain)")
    has_emergency_fund: bool | None = Field(default=None, description="Has emergency fund")
    debt_level: str | None = Field(default=None, description="Debt level (none/low/moderate/high)")

    # Metadata
    last_updated: datetime = Field(default_factory=datetime.now)

    @computed_field
    @property
    def profile_completeness(self) -> float:
        """Calculate profile completeness (0.0 - 1.0).

        Counts 11 required fields: the 9 profiling-agent slots plus
        monthly_income and monthly_expenses (required financial context).
        """
        required_fields = [
            self.risk_tolerance,
            self.loss_comfort,
            self.investment_horizon,
            self.liquidity_needs,
            self.income_stability,
            self.has_emergency_fund,
            self.debt_level,
            self.investment_experience,
            self.monthly_income,
            self.monthly_expenses,
        ]
        goals_filled = len(self.goals) > 0

        filled = sum(1 for f in required_fields if f is not None) + (1 if goals_filled else 0)
        total = len(required_fields) + 1  # 11 total

        return round(filled / total, 2)


# ============================================================
# PORTFOLIO SUMMARY (Computed for Module A)
# ============================================================


class PortfolioSummary(BaseModel):
    """Aggregated portfolio summary for a user."""

    user_id: str = Field(..., description="User ID")

    # Totals
    total_assets: Decimal = Field(..., ge=0, description="Total asset value")
    total_liabilities: Decimal = Field(..., ge=0, description="Total liabilities (mortgages)")
    total_net_worth: Decimal = Field(..., description="Net worth (assets - liabilities)")

    # By Category
    cash_and_equivalents: Decimal = Field(default=Decimal("0"), ge=0)
    stocks: Decimal = Field(default=Decimal("0"), ge=0)
    bonds: Decimal = Field(default=Decimal("0"), ge=0)
    etfs: Decimal = Field(default=Decimal("0"), ge=0)
    mutual_funds: Decimal = Field(default=Decimal("0"), ge=0)
    crypto: Decimal = Field(default=Decimal("0"), ge=0)
    real_estate_equity: Decimal = Field(default=Decimal("0"), ge=0)

    # Allocation percentages
    allocation_by_asset_type: dict[str, float] = Field(default_factory=dict)
    allocation_by_sector: dict[str, float] = Field(default_factory=dict)

    # Metrics
    liquidity_ratio: float = Field(
        default=0.0, ge=0, le=1, description="Liquid assets / total assets"
    )
    debt_to_asset_ratio: float = Field(
        default=0.0, ge=0, description="Total liabilities / total assets"
    )
    monthly_savings_rate: float | None = Field(
        default=None, description="(income - expenses) / income"
    )

    last_updated: datetime = Field(default_factory=datetime.now)


# ============================================================
# AGGREGATE CONTAINER
# ============================================================


class UserPortfolio(BaseModel):
    """Complete portfolio data for a user."""

    user: User
    bank_accounts: list[BankAccount] = Field(default_factory=list)
    investment_accounts: list[InvestmentAccount] = Field(default_factory=list)
    holdings: list[Holding] = Field(default_factory=list)
    real_estate: list[RealEstate] = Field(default_factory=list)
    investment_profile: InvestmentProfile | None = None
    summary: PortfolioSummary | None = None
