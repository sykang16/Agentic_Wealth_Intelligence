"""SQLAlchemy models for interaction logging."""

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class SessionRecord(Base):
    """A conversation session."""

    __tablename__ = "sessions"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(64), index=True, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    state_json = Column(Text, nullable=True)

    interactions = relationship(
        "InteractionLog", back_populates="session", cascade="all, delete-orphan"
    )


class InteractionLog(Base):
    """A single interaction within a session."""

    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("sessions.id"), index=True, nullable=False)
    node_name = Column(String(64), nullable=False, default="")
    user_message = Column(Text, nullable=False, default="")
    response_content = Column(Text, nullable=False, default="")
    intent = Column(String(64), nullable=False, default="")
    module_source = Column(String(64), nullable=False, default="")
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    session = relationship("SessionRecord", back_populates="interactions")
