"""Database engine and session factory for interaction logging."""

from pathlib import Path

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.src.logging_db.models import Base

_DEFAULT_DB_DIR = Path(__file__).parent.parent.parent.parent / "data"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "trace.db"


def get_engine(db_url: str | None = None) -> Engine:
    """Create a SQLAlchemy engine.

    Args:
        db_url: SQLAlchemy database URL. Defaults to ``sqlite:///data/trace.db``.
    """
    if db_url is None:
        _DEFAULT_DB_DIR.mkdir(parents=True, exist_ok=True)
        db_url = f"sqlite:///{_DEFAULT_DB_PATH}"

    extra_kwargs = {}
    if ":memory:" in db_url:
        extra_kwargs["connect_args"] = {"check_same_thread": False}
        extra_kwargs["poolclass"] = StaticPool

    engine = create_engine(db_url, echo=False, **extra_kwargs)
    Base.metadata.create_all(engine)
    return engine


def get_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Return a sessionmaker bound to *engine*."""
    return sessionmaker(bind=engine)
