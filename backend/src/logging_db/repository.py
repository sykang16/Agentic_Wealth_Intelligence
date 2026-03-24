"""Repository layer for interaction logging CRUD."""

import json
import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import sessionmaker

from backend.src.logging_db.models import InteractionLog, SessionRecord

# Keys that are safe to serialise into state_json.
_SERIALIZABLE_KEYS = (
    "user_id",
    "messages",
    "current_message",
    "intent",
    "response",
    "module_source",
    "error",
)


class LogRepository:
    """CRUD operations for session and interaction logging."""

    def __init__(self, session_factory: sessionmaker):
        self._session_factory = session_factory

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def create_session(self, user_id: str) -> str:
        """Create a new session record and return its id."""
        session_id = str(uuid.uuid4())
        with self._session_factory() as db:
            record = SessionRecord(id=session_id, user_id=user_id)
            db.add(record)
            db.commit()
        return session_id

    def list_sessions(self, user_id: str | None = None, limit: int = 50) -> list[dict]:
        """Return recent sessions, optionally filtered by user_id."""
        with self._session_factory() as db:
            query = db.query(SessionRecord).order_by(SessionRecord.created_at.desc())
            if user_id:
                query = query.filter(SessionRecord.user_id == user_id)
            rows = query.limit(limit).all()
            return [
                {
                    "id": r.id,
                    "user_id": r.user_id,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "updated_at": r.updated_at.isoformat() if r.updated_at else None,
                    "interaction_count": len(r.interactions),
                }
                for r in rows
            ]

    def get_session_with_interactions(self, session_id: str) -> dict | None:
        """Return full session detail including all interaction logs."""
        with self._session_factory() as db:
            record = db.query(SessionRecord).filter_by(id=session_id).first()
            if record is None:
                return None
            return {
                "id": record.id,
                "user_id": record.user_id,
                "created_at": record.created_at.isoformat() if record.created_at else None,
                "updated_at": record.updated_at.isoformat() if record.updated_at else None,
                "state_json": record.state_json,
                "interactions": [
                    {
                        "id": i.id,
                        "node_name": i.node_name,
                        "user_message": i.user_message,
                        "response_content": i.response_content,
                        "intent": i.intent,
                        "module_source": i.module_source,
                        "error": i.error,
                        "created_at": i.created_at.isoformat() if i.created_at else None,
                    }
                    for i in record.interactions
                ],
            }

    # ------------------------------------------------------------------
    # Interactions
    # ------------------------------------------------------------------

    def log_interaction(
        self,
        session_id: str,
        user_message: str,
        response_content: str,
        intent: str = "",
        module_source: str = "",
        node_name: str = "",
        error: str | None = None,
    ) -> int:
        """Append an interaction log entry. Returns the log id."""
        with self._session_factory() as db:
            log = InteractionLog(
                session_id=session_id,
                node_name=node_name,
                user_message=user_message,
                response_content=response_content,
                intent=intent,
                module_source=module_source,
                error=error,
            )
            db.add(log)
            db.commit()
            db.refresh(log)
            return log.id

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def save_session_state(self, session_id: str, state: dict) -> None:
        """Persist JSON-safe fields of orchestrator state."""
        safe: dict = {}
        for key in _SERIALIZABLE_KEYS:
            if key in state:
                safe[key] = state[key]

        # Strip messages to role+content only
        if "messages" in safe:
            safe["messages"] = [
                {"role": m.get("role", ""), "content": m.get("content", "")}
                for m in safe["messages"]
            ]

        with self._session_factory() as db:
            record = db.query(SessionRecord).filter_by(id=session_id).first()
            if record:
                record.state_json = json.dumps(safe)
                record.updated_at = datetime.now(timezone.utc)
                db.commit()

    def load_session_state(self, session_id: str) -> dict | None:
        """Load persisted state for a session. Returns None if not found."""
        with self._session_factory() as db:
            record = db.query(SessionRecord).filter_by(id=session_id).first()
            if record is None or record.state_json is None:
                return None
            return json.loads(record.state_json)
