"""Base class for data collectors."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..rag.schemas import Document

logger = logging.getLogger(__name__)


class CollectorStatus(str, Enum):
    """Status of a collection operation."""

    SUCCESS = "success"
    PARTIAL = "partial"  # Some documents collected, some failed
    FAILED = "failed"
    NO_API_KEY = "no_api_key"
    RATE_LIMITED = "rate_limited"


@dataclass
class CollectorResult:
    """Result from a data collection operation."""

    collector_name: str
    status: CollectorStatus
    documents: list[Document] = field(default_factory=list)
    documents_collected: int = 0
    errors: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_success(self) -> bool:
        """Check if collection was successful."""
        return self.status in (CollectorStatus.SUCCESS, CollectorStatus.PARTIAL)


class BaseCollector(ABC):
    """Abstract base class for data collectors."""

    name: str = "base"
    description: str = "Base collector"

    def __init__(self):
        """Initialize the collector."""
        self._last_collection: datetime | None = None

    @abstractmethod
    def is_configured(self) -> bool:
        """Check if the collector has required API keys/configuration.

        Returns:
            True if properly configured.
        """
        pass

    @abstractmethod
    def collect(self, **kwargs) -> CollectorResult:
        """Collect documents from the data source.

        Args:
            **kwargs: Collector-specific parameters.

        Returns:
            CollectorResult with collected documents.
        """
        pass

    def get_status(self) -> dict:
        """Get collector status information.

        Returns:
            Dictionary with status info.
        """
        return {
            "name": self.name,
            "description": self.description,
            "configured": self.is_configured(),
            "last_collection": self._last_collection.isoformat() if self._last_collection else None,
        }

    def _create_result(
        self,
        status: CollectorStatus,
        documents: list[Document] | None = None,
        errors: list[str] | None = None,
    ) -> CollectorResult:
        """Helper to create a CollectorResult.

        Args:
            status: Collection status.
            documents: Collected documents.
            errors: Any errors encountered.

        Returns:
            CollectorResult instance.
        """
        docs = documents or []
        self._last_collection = datetime.now()

        return CollectorResult(
            collector_name=self.name,
            status=status,
            documents=docs,
            documents_collected=len(docs),
            errors=errors or [],
        )
