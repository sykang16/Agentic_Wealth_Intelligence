"""Tests for LangSmith tracing configuration."""

import os
from unittest.mock import patch

from backend.src.common.tracing import configure_tracing, is_tracing_enabled


def test_tracing_disabled_by_default():
    with patch.dict(os.environ, {}, clear=True):
        assert is_tracing_enabled() is False


def test_tracing_enabled_with_env_vars():
    env = {
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_API_KEY": "lsv2_test_key",
    }
    with patch.dict(os.environ, env, clear=True):
        assert is_tracing_enabled() is True


def test_tracing_disabled_without_api_key():
    env = {"LANGCHAIN_TRACING_V2": "true"}
    with patch.dict(os.environ, env, clear=True):
        assert is_tracing_enabled() is False


def test_configure_tracing_runs_without_error():
    # Should not raise regardless of env state
    configure_tracing()
