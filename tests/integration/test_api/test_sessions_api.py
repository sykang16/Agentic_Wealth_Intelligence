"""Integration tests for session listing and detail endpoints."""

import pytest


def test_list_sessions_empty(api_client):
    resp = api_client.get("/api/v1/sessions")
    assert resp.status_code == 200
    data = resp.json()
    assert "sessions" in data
    assert isinstance(data["sessions"], list)


def test_session_created_after_chat(api_client):
    # Send a chat message — this should create a session
    chat_resp = api_client.post(
        "/api/v1/chat",
        json={"message": "Hello", "user_id": "user-001"},
    )
    assert chat_resp.status_code == 200
    session_id = chat_resp.json().get("session_id")
    assert session_id is not None

    # List sessions — should include the new one
    list_resp = api_client.get("/api/v1/sessions", params={"user_id": "user-001"})
    assert list_resp.status_code == 200
    sessions = list_resp.json()["sessions"]
    assert any(s["id"] == session_id for s in sessions)


def test_get_session_detail(api_client):
    # Create via chat
    chat_resp = api_client.post(
        "/api/v1/chat",
        json={"message": "What is my net worth?", "user_id": "user-001"},
    )
    session_id = chat_resp.json()["session_id"]

    # Get detail
    detail_resp = api_client.get(f"/api/v1/sessions/{session_id}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["id"] == session_id
    assert len(detail["interactions"]) >= 1
    assert detail["interactions"][0]["user_message"] == "What is my net worth?"


def test_get_session_not_found(api_client):
    resp = api_client.get("/api/v1/sessions/nonexistent-id")
    assert resp.status_code == 404


def test_chat_with_session_id_restores_state(api_client):
    # First message creates a session
    resp1 = api_client.post(
        "/api/v1/chat",
        json={"message": "Hello", "user_id": "user-001"},
    )
    session_id = resp1.json()["session_id"]

    # Second message with same session_id
    resp2 = api_client.post(
        "/api/v1/chat",
        json={"message": "What's my portfolio?", "user_id": "user-001", "session_id": session_id},
    )
    assert resp2.status_code == 200
    assert resp2.json()["session_id"] == session_id

    # Session should now have 2 interactions
    detail = api_client.get(f"/api/v1/sessions/{session_id}").json()
    assert len(detail["interactions"]) == 2
