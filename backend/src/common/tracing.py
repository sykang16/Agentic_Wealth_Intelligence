"""LangSmith tracing configuration.

LangGraph/LangChain SDK auto-traces when these environment variables are set:
- LANGCHAIN_TRACING_V2=true
- LANGCHAIN_API_KEY=<your-langsmith-api-key>
- LANGCHAIN_PROJECT=<project-name>  (optional, defaults to "default")

This module provides helpers to check status and log it at startup.
"""

import logging
import os

logger = logging.getLogger(__name__)


def is_tracing_enabled() -> bool:
    """Return True if LangSmith tracing environment variables are configured."""
    tracing_flag = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
    api_key = os.getenv("LANGCHAIN_API_KEY", "")
    return tracing_flag and bool(api_key)


def configure_tracing() -> None:
    """Log LangSmith tracing status at startup. No-op if disabled."""
    if is_tracing_enabled():
        project = os.getenv("LANGCHAIN_PROJECT", "default")
        logger.info("LangSmith tracing enabled (project=%s)", project)
    else:
        logger.info(
            "LangSmith tracing disabled. Set LANGCHAIN_TRACING_V2=true "
            "and LANGCHAIN_API_KEY to enable."
        )
