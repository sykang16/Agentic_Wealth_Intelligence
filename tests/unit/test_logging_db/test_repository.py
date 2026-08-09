"""Tests for LogRepository against in-memory SQLite."""

import pytest

from backend.src.logging_db.engine import get_engine, get_session_factory
from backend.src.logging_db.repository import LogRepository


@pytest.fixture
def repo():
    engine = get_engine("sqlite:///:memory:")
    factory = get_session_factory(engine)
    return LogRepository(factory)


def test_create_session(repo):
    session_id = repo.create_session("user-1")
    assert isinstance(session_id, str)
    assert len(session_id) == 36  # UUID format


def test_log_interaction(repo):
    session_id = repo.create_session("user-1")
    log_id = repo.log_interaction(
        session_id=session_id,
        user_message="What is my net worth?",
        response_content="Your net worth is $1M.",
        intent="portfolio_query",
        module_source="Portfolio",
    )
    assert isinstance(log_id, int)
    assert log_id >= 1


def test_save_and_load_state(repo):
    session_id = repo.create_session("user-1")
    state = {
        "user_id": "user-1",
        "messages": [
            {"role": "user", "content": "Hello", "visualization": "should-be-stripped"},
        ],
        "current_message": "Hello",
        "intent": "general",
        "response": "Hi there!",
        "module_source": "General",
        "error": None,
        "profiling_state": {"should": "be excluded"},
    }
    repo.save_session_state(session_id, state)
    loaded = repo.load_session_state(session_id)

    assert loaded is not None
    assert loaded["user_id"] == "user-1"
    assert loaded["intent"] == "general"
    # Messages should only have role+content
    assert loaded["messages"] == [{"role": "user", "content": "Hello"}]
    # profiling_state should not be persisted
    assert "profiling_state" not in loaded


def test_load_session_state_not_found(repo):
    assert repo.load_session_state("nonexistent-id") is None


def test_list_sessions_by_user(repo):
    repo.create_session("user-1")
    repo.create_session("user-1")
    repo.create_session("user-2")

    all_sessions = repo.list_sessions()
    assert len(all_sessions) == 3

    user1_sessions = repo.list_sessions(user_id="user-1")
    assert len(user1_sessions) == 2
    assert all(s["user_id"] == "user-1" for s in user1_sessions)


def test_get_session_with_interactions(repo):
    session_id = repo.create_session("user-1")
    repo.log_interaction(
        session_id=session_id,
        user_message="Q1",
        response_content="A1",
        intent="general",
    )
    repo.log_interaction(
        session_id=session_id,
        user_message="Q2",
        response_content="A2",
        intent="portfolio_query",
        module_source="Portfolio",
    )

    detail = repo.get_session_with_interactions(session_id)
    assert detail is not None
    assert detail["user_id"] == "user-1"
    assert len(detail["interactions"]) == 2
    assert detail["interactions"][0]["user_message"] == "Q1"
    assert detail["interactions"][1]["intent"] == "portfolio_query"


def test_get_session_not_found(repo):
    assert repo.get_session_with_interactions("nonexistent") is None
