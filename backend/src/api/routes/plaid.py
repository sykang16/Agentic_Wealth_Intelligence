"""FastAPI routes for Plaid account connection management."""

import logging
import os

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from backend.src.api.dependencies import (
    get_aggregator,
    get_plaid_client,
    get_plaid_source,
    get_plaid_token_repo,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/plaid", tags=["plaid"])


# ------------------------------------------------------------------
# Request / Response schemas
# ------------------------------------------------------------------


class LinkTokenRequest(BaseModel):
    user_id: str = "new_user"
    products: list[str] = ["transactions", "investments"]


class LinkTokenResponse(BaseModel):
    link_token: str


class ExchangeTokenRequest(BaseModel):
    public_token: str
    products: list[str] = ["transactions", "investments"]


class ExchangeTokenResponse(BaseModel):
    user_id: str
    institution: str
    status: str = "connected"


class PlaidStatusResponse(BaseModel):
    connected: bool
    user_id: str | None = None
    institution: str | None = None
    products: list[str] = []
    env: str | None = None
    created_at: str | None = None


class DisconnectResponse(BaseModel):
    disconnected: bool
    user_id: str


# ------------------------------------------------------------------
# Helper: check whether Plaid is configured
# ------------------------------------------------------------------


def _require_plaid_client(plaid_client):
    """Raise 503 if the Plaid client is not available."""
    if plaid_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Plaid is not configured. Set PLAID_CLIENT_ID and PLAID_SECRET.",
        )


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------


@router.post("/link-token", response_model=LinkTokenResponse)
def create_link_token(
    request: LinkTokenRequest,
    plaid_client=Depends(get_plaid_client),
):
    """Create a Plaid Link token for the frontend OAuth flow."""
    _require_plaid_client(plaid_client)
    try:
        token = plaid_client.create_link_token(
            user_id=request.user_id,
            products=request.products,
        )
        return LinkTokenResponse(link_token=token)
    except Exception as exc:
        logger.error("link-token creation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Plaid error: {exc}",
        )


@router.post("/exchange-token", response_model=ExchangeTokenResponse)
def exchange_token(
    request: ExchangeTokenRequest,
    plaid_client=Depends(get_plaid_client),
    token_repo=Depends(get_plaid_token_repo),
    aggregator=Depends(get_aggregator),
):
    """Exchange a public_token for a persistent access_token and store it.

    After a successful exchange the Plaid user becomes available in the
    aggregator's get_user_ids() result immediately (the PlaidPortfolioSource
    reads from the DB on every call).
    """
    _require_plaid_client(plaid_client)
    try:
        access_token, item_id = plaid_client.exchange_public_token(request.public_token)
        institution_name = plaid_client.get_institution_name(access_token)
        env = os.environ.get("PLAID_ENV", "sandbox")

        user_id = token_repo.save(
            item_id=item_id,
            access_token=access_token,
            institution_name=institution_name,
            products=request.products,
            env=env,
        )

        # Invalidate any stale cache for this user in the live Plaid source
        plaid_source = aggregator._plaid_source
        if plaid_source:
            plaid_source.invalidate_cache(user_id)

        return ExchangeTokenResponse(
            user_id=user_id,
            institution=institution_name,
        )
    except Exception as exc:
        logger.error("exchange-token failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Plaid error: {exc}",
        )


@router.get("/status/{user_id}", response_model=PlaidStatusResponse)
def get_plaid_status(
    user_id: str,
    token_repo=Depends(get_plaid_token_repo),
):
    """Check whether a user has an active Plaid connection.

    Never exposes the access_token.
    """
    record = token_repo.get_by_user_id(user_id)
    if not record:
        return PlaidStatusResponse(connected=False)

    return PlaidStatusResponse(
        connected=True,
        user_id=record.user_id,
        institution=record.institution_name,
        products=record.products.split(","),
        env=record.env,
        created_at=record.created_at.isoformat() if record.created_at else None,
    )


@router.delete("/disconnect/{user_id}", response_model=DisconnectResponse)
def disconnect_plaid(
    user_id: str,
    token_repo=Depends(get_plaid_token_repo),
    aggregator=Depends(get_aggregator),
):
    """Deactivate a Plaid connection and remove the user from the aggregator."""
    disconnected = token_repo.deactivate(user_id)
    if not disconnected:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active Plaid connection found for user '{user_id}'.",
        )

    # Remove from source cache so it stops appearing in get_user_ids()
    plaid_source = aggregator._plaid_source
    if plaid_source:
        plaid_source.invalidate_cache(user_id)

    return DisconnectResponse(disconnected=True, user_id=user_id)


@router.get("/connections")
def list_plaid_connections(token_repo=Depends(get_plaid_token_repo)):
    """List all active Plaid connections (admin use)."""
    records = token_repo.get_all_active()
    return [
        {
            "user_id": r.user_id,
            "institution": r.institution_name,
            "products": r.products.split(","),
            "env": r.env,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in records
    ]
