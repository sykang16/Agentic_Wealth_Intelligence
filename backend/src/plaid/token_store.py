"""SQLAlchemy model and repository for storing Plaid access tokens."""

import os
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Integer, Numeric, String, Text
from sqlalchemy.orm import Session, sessionmaker

# Import the shared Base so PlaidToken is registered in the same metadata
# and gets created by engine.py's Base.metadata.create_all()
from backend.src.logging_db.models import Base


# ------------------------------------------------------------------
# Token encryption helpers (Fernet / AES-128-CBC + HMAC-SHA256)
# ------------------------------------------------------------------

def _get_fernet():
    """Return a Fernet instance if PLAID_ENCRYPTION_KEY is set, else None."""
    key = os.environ.get("PLAID_ENCRYPTION_KEY", "")
    if not key:
        return None
    try:
        from cryptography.fernet import Fernet
        return Fernet(key.encode())
    except Exception:
        return None


def _encrypt_token(plaintext: str) -> str:
    """Encrypt an access token if an encryption key is configured."""
    f = _get_fernet()
    return f.encrypt(plaintext.encode()).decode() if f else plaintext


def _decrypt_token(stored: str) -> str:
    """Decrypt a stored token; falls back to plain text for sandbox/legacy tokens."""
    f = _get_fernet()
    if f is None:
        return stored
    try:
        return f.decrypt(stored.encode()).decode()
    except Exception:
        return stored  # plain-text fallback for unencrypted legacy records


class PlaidToken(Base):
    """Stores Plaid access tokens linked to internal user IDs."""

    __tablename__ = "plaid_tokens"

    # user_id follows the pattern "plaid_{item_id[:8]}"
    user_id = Column(String(64), primary_key=True, index=True, nullable=False)
    item_id = Column(String(128), nullable=False, unique=True)
    # access_token is stored plain in sandbox; encrypt at rest for production
    access_token = Column(Text, nullable=False)
    institution_name = Column(String(256), nullable=True)
    # comma-separated product list, e.g. "transactions,investments"
    products = Column(String(256), nullable=False, default="transactions")
    # "sandbox" | "development" | "production"
    env = Column(String(32), nullable=False, default="sandbox")
    is_active = Column(Boolean, nullable=False, default=True)
    # User profile fields collected at connection time
    age = Column(Integer, nullable=True)
    annual_income = Column(Numeric(precision=18, scale=2), nullable=True)
    occupation = Column(String(256), nullable=True)
    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class PlaidTokenRecord:
    """Plain data object returned from PlaidTokenRepository queries."""

    __slots__ = (
        "user_id",
        "item_id",
        "access_token",
        "institution_name",
        "products",
        "env",
        "is_active",
        "created_at",
        "updated_at",
        "age",
        "annual_income",
        "occupation",
    )

    def __init__(self, row: PlaidToken) -> None:
        self.user_id = row.user_id
        self.item_id = row.item_id
        self.access_token = _decrypt_token(row.access_token)
        self.institution_name = row.institution_name
        self.products = row.products
        self.env = row.env
        self.is_active = row.is_active
        self.created_at = row.created_at
        self.updated_at = row.updated_at
        self.age = row.age
        self.annual_income = row.annual_income
        self.occupation = row.occupation


class PlaidTokenRepository:
    """CRUD operations for Plaid access tokens."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._factory = session_factory

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(
        self,
        item_id: str,
        access_token: str,
        institution_name: str | None,
        products: list[str],
        env: str,
        age: int | None = None,
        annual_income: float | None = None,
        occupation: str | None = None,
        custom_user_id: str | None = None,
    ) -> str:
        """Upsert a Plaid token record and return the user_id used.

        If ``custom_user_id`` is provided and non-empty, it is used as the
        user_id (allowing the caller to pick a friendly name like 'user_101').
        Otherwise the user_id is auto-derived as ``plaid_{item_id[:8]}``.
        """
        user_id = custom_user_id.strip() if custom_user_id and custom_user_id.strip() else f"plaid_{item_id[:8]}"
        products_str = ",".join(products)
        stored_token = _encrypt_token(access_token)

        with self._factory() as session:
            existing = session.get(PlaidToken, user_id)
            if existing:
                existing.access_token = stored_token
                existing.institution_name = institution_name
                existing.products = products_str
                existing.env = env
                existing.is_active = True
                existing.updated_at = datetime.now(timezone.utc)
                if age is not None:
                    existing.age = age
                if annual_income is not None:
                    existing.annual_income = annual_income
                if occupation is not None:
                    existing.occupation = occupation
            else:
                record = PlaidToken(
                    user_id=user_id,
                    item_id=item_id,
                    access_token=stored_token,
                    institution_name=institution_name,
                    products=products_str,
                    env=env,
                    age=age,
                    annual_income=annual_income,
                    occupation=occupation,
                )
                session.add(record)
            session.commit()

        return user_id

    def deactivate(self, user_id: str) -> bool:
        """Mark a token as inactive (soft delete)."""
        with self._factory() as session:
            record = session.get(PlaidToken, user_id)
            if not record:
                return False
            record.is_active = False
            record.updated_at = datetime.now(timezone.utc)
            session.commit()
        return True

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_by_user_id(self, user_id: str) -> PlaidTokenRecord | None:
        """Return the token record for a user_id, or None if not found."""
        with self._factory() as session:
            row = session.get(PlaidToken, user_id)
            if not row or not row.is_active:
                return None
            return PlaidTokenRecord(row)

    def get_all_active(self) -> list[PlaidTokenRecord]:
        """Return all active Plaid token records."""
        with self._factory() as session:
            rows = (
                session.query(PlaidToken)
                .filter(PlaidToken.is_active == True)  # noqa: E712
                .all()
            )
            return [PlaidTokenRecord(r) for r in rows]
