"""FastAPI routes for user management (add/delete synthetic users, list all users)."""

import logging
import os
import random

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from backend.src.api.dependencies import get_aggregator, get_plaid_token_repo
from backend.src.asset_management.aggregator import PortfolioAggregator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/users", tags=["users"])

# ------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------


class GenerateSyntheticRequest(BaseModel):
    user_id: str | None = None
    age: int | None = None
    annual_income: float | None = None
    risk_tolerance: str | None = None  # conservative | moderate | aggressive
    occupation: str | None = None


class GenerateSyntheticResponse(BaseModel):
    user_id: str
    name: str
    status: str = "created"


class UserListItem(BaseModel):
    user_id: str
    name: str
    source: str       # "synthetic" or "plaid"
    institution: str | None = None
    created_at: str | None = None


class UserListResponse(BaseModel):
    users: list[UserListItem]
    total: int


class DeleteUserResponse(BaseModel):
    deleted: bool
    user_id: str


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------


@router.get("", response_model=UserListResponse)
def list_users(
    aggregator: PortfolioAggregator = Depends(get_aggregator),
    token_repo=Depends(get_plaid_token_repo),
):
    """Return all users with their data source (synthetic or Plaid)."""
    user_ids = aggregator.get_user_ids()
    plaid_records = {r.user_id: r for r in token_repo.get_all_active()}

    users = []
    for uid in user_ids:
        if uid in plaid_records:
            rec = plaid_records[uid]
            users.append(
                UserListItem(
                    user_id=uid,
                    name=rec.institution_name or uid,
                    source="plaid",
                    institution=rec.institution_name,
                    created_at=rec.created_at.isoformat() if rec.created_at else None,
                )
            )
        else:
            portfolio = aggregator.get_portfolio(uid)
            name = portfolio.user.name if portfolio else uid
            users.append(UserListItem(user_id=uid, name=name, source="synthetic"))

    return UserListResponse(users=users, total=len(users))


@router.post("/generate-synthetic", response_model=GenerateSyntheticResponse)
def generate_synthetic_user(
    request: GenerateSyntheticRequest,
    aggregator: PortfolioAggregator = Depends(get_aggregator),
):
    """Generate a new synthetic user with randomized portfolio data.

    The generated portfolio is appended to synthetic_portfolios.json and
    the aggregator is reloaded so the new user is immediately available.
    """
    from backend.src.data_generation.generator import SyntheticDataGenerator

    # Assign an auto-incremented user_id if none was provided
    user_id = request.user_id
    if not user_id:
        existing = set(aggregator.get_user_ids())
        # Find the next unused user_NNN id
        i = 1
        while True:
            candidate = f"user_{i:03d}"
            if candidate not in existing:
                user_id = candidate
                break
            i += 1

    # Reject duplicate user_id
    if aggregator.get_portfolio(user_id) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"User '{user_id}' already exists.",
        )

    # Use a random seed so each generation is unique
    seed = random.randint(0, 999_999)
    gen = SyntheticDataGenerator(seed=seed)

    # Generate the portfolio with optional overrides
    portfolio = gen.generate_user_portfolio(user_id=user_id)

    # Apply overrides
    if request.age is not None:
        portfolio.user.age = request.age
    if request.occupation:
        portfolio.user.occupation = request.occupation
    if request.annual_income is not None:
        from decimal import Decimal
        portfolio.user.annual_income = Decimal(str(request.annual_income))
    if request.risk_tolerance and portfolio.investment_profile:
        from backend.src.common.models import RiskTolerance
        try:
            portfolio.investment_profile.risk_tolerance = RiskTolerance(request.risk_tolerance)
        except ValueError:
            pass  # Ignore invalid enum values; keep generated value

    # Store the new portfolio in memory and persist to JSON
    # Access the internal dict directly to add without triggering reload
    aggregator._portfolios[user_id] = portfolio
    try:
        aggregator.save_portfolios()
    except Exception as exc:
        # Remove from memory if save failed to keep state consistent
        aggregator._portfolios.pop(user_id, None)
        logger.error("Failed to save synthetic portfolio for %s: %s", user_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist portfolio data.",
        )

    logger.info("Generated synthetic user %s (%s)", user_id, portfolio.user.name)
    return GenerateSyntheticResponse(user_id=user_id, name=portfolio.user.name)


@router.delete("/{user_id}", response_model=DeleteUserResponse)
def delete_user(
    user_id: str,
    aggregator: PortfolioAggregator = Depends(get_aggregator),
):
    """Delete a synthetic user.

    Only synthetic users can be deleted this way.
    For Plaid users, use DELETE /api/v1/plaid/disconnect/{user_id}.
    """
    if user_id.startswith("plaid_"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Use DELETE /api/v1/plaid/disconnect/{user_id} to remove Plaid users.",
        )

    if aggregator.get_portfolio(user_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{user_id}' not found.",
        )

    # Remove from memory and persist
    aggregator._portfolios.pop(user_id, None)
    try:
        aggregator.save_portfolios()
    except Exception as exc:
        logger.error("Failed to persist deletion of user %s: %s", user_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User removed from memory but disk persistence failed.",
        )

    return DeleteUserResponse(deleted=True, user_id=user_id)
