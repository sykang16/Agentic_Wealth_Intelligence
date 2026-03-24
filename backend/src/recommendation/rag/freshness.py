"""Freshness management for RAG documents with time-based retention policies."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable

from .schemas import Document, DocumentMetadata, DocumentType
from .vector_store import ChromaVectorStore


class RetentionPolicy(str, Enum):
    """Pre-defined retention policies."""

    NEWS = "news"  # Short retention for news articles
    MARKET_DATA = "market_data"  # Medium retention for market analysis
    RESEARCH = "research"  # Long retention for research reports
    REFERENCE = "reference"  # Very long retention for reference documents
    PERMANENT = "permanent"  # No expiration


@dataclass
class RetentionConfig:
    """Configuration for document retention."""

    policy: RetentionPolicy
    retention_days: int
    auto_extend_on_access: bool = False
    extension_days: int = 0

    @classmethod
    def from_policy(cls, policy: RetentionPolicy) -> "RetentionConfig":
        """Create RetentionConfig from a predefined policy.

        Args:
            policy: The retention policy to use.

        Returns:
            RetentionConfig instance.
        """
        policy_defaults = {
            RetentionPolicy.NEWS: (7, False, 0),  # 7 days, no extension
            RetentionPolicy.MARKET_DATA: (30, True, 7),  # 30 days, extend by 7 on access
            RetentionPolicy.RESEARCH: (90, True, 30),  # 90 days, extend by 30 on access
            RetentionPolicy.REFERENCE: (365, False, 0),  # 1 year, no extension
            RetentionPolicy.PERMANENT: (36500, False, 0),  # ~100 years (effectively permanent)
        }

        days, auto_extend, ext_days = policy_defaults[policy]
        return cls(
            policy=policy,
            retention_days=days,
            auto_extend_on_access=auto_extend,
            extension_days=ext_days,
        )


@dataclass
class DocumentTypePolicy:
    """Maps document types to retention policies."""

    policies: dict[DocumentType, RetentionConfig] = field(default_factory=dict)

    def __post_init__(self):
        """Set default policies for each document type."""
        if not self.policies:
            self.policies = {
                DocumentType.NEWS_ARTICLE: RetentionConfig.from_policy(RetentionPolicy.NEWS),
                DocumentType.MARKET_ANALYSIS: RetentionConfig.from_policy(RetentionPolicy.MARKET_DATA),
                DocumentType.FOMC_MINUTES: RetentionConfig.from_policy(RetentionPolicy.RESEARCH),
                DocumentType.RESEARCH_REPORT: RetentionConfig.from_policy(RetentionPolicy.RESEARCH),
                DocumentType.ETF_FACTSHEET: RetentionConfig.from_policy(RetentionPolicy.REFERENCE),
                DocumentType.INVESTMENT_GUIDE: RetentionConfig.from_policy(RetentionPolicy.REFERENCE),
            }

    def get_policy(self, doc_type: DocumentType) -> RetentionConfig:
        """Get retention policy for a document type.

        Args:
            doc_type: The document type.

        Returns:
            RetentionConfig for the document type.
        """
        return self.policies.get(
            doc_type,
            RetentionConfig.from_policy(RetentionPolicy.RESEARCH),  # Default
        )

    def set_policy(self, doc_type: DocumentType, config: RetentionConfig) -> None:
        """Set retention policy for a document type.

        Args:
            doc_type: The document type.
            config: The retention configuration.
        """
        self.policies[doc_type] = config


class FreshnessManager:
    """Manages document freshness and retention in the RAG system."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        type_policies: DocumentTypePolicy | None = None,
    ):
        """Initialize the freshness manager.

        Args:
            vector_store: ChromaDB vector store instance.
            type_policies: Document type to retention policy mapping.
        """
        self.vector_store = vector_store
        self.type_policies = type_policies or DocumentTypePolicy()

    def set_expiry_for_document(self, document: Document) -> Document:
        """Set expiry date for a document based on its type.

        Args:
            document: Document to set expiry for.

        Returns:
            Document with updated expiry date.
        """
        policy = self.type_policies.get_policy(document.metadata.document_type)

        if policy.policy == RetentionPolicy.PERMANENT:
            document.metadata.expiry_date = None
        else:
            expiry = document.metadata.publish_date + timedelta(days=policy.retention_days)
            document.metadata.expiry_date = expiry

        return document

    def prepare_documents(self, documents: list[Document]) -> list[Document]:
        """Prepare documents by setting appropriate expiry dates.

        Args:
            documents: List of documents to prepare.

        Returns:
            Documents with expiry dates set.
        """
        return [self.set_expiry_for_document(doc) for doc in documents]

    def cleanup_expired(self) -> int:
        """Remove all expired documents from the vector store.

        Returns:
            Number of chunks removed.
        """
        return self.vector_store.delete_expired()

    def get_expiring_soon(self, days: int = 7) -> list[str]:
        """Get document IDs that will expire within the specified days.

        Args:
            days: Number of days to look ahead.

        Returns:
            List of document IDs expiring soon.
        """
        results = self.vector_store._collection.get(include=["metadatas"])

        if not results["metadatas"]:
            return []

        now = datetime.now()
        cutoff = now + timedelta(days=days)
        expiring_doc_ids = set()

        for metadata in results["metadatas"]:
            expiry_str = metadata.get("expiry_date")
            if expiry_str:
                try:
                    expiry_date = datetime.fromisoformat(expiry_str)
                    if now < expiry_date <= cutoff:
                        expiring_doc_ids.add(metadata.get("doc_id"))
                except (ValueError, TypeError):
                    pass

        return list(expiring_doc_ids)

    def extend_expiry(self, doc_id: str, days: int | None = None) -> bool:
        """Extend the expiry date of a document.

        Args:
            doc_id: Document ID to extend.
            days: Number of days to extend. If None, uses policy default.

        Returns:
            True if extension was successful.
        """
        # Get document chunks
        results = self.vector_store._collection.get(
            where={"doc_id": doc_id},
            include=["metadatas"],
        )

        if not results["ids"]:
            return False

        # Get document type and policy
        first_metadata = results["metadatas"][0]
        doc_type = DocumentType(first_metadata.get("document_type"))
        policy = self.type_policies.get_policy(doc_type)

        extension_days = days or policy.extension_days
        if extension_days <= 0:
            return False

        # Update expiry date in metadata
        for chunk_id, metadata in zip(results["ids"], results["metadatas"]):
            current_expiry_str = metadata.get("expiry_date")
            if current_expiry_str:
                try:
                    current_expiry = datetime.fromisoformat(current_expiry_str)
                    new_expiry = current_expiry + timedelta(days=extension_days)
                    metadata["expiry_date"] = new_expiry.isoformat()

                    # Update in ChromaDB
                    self.vector_store._collection.update(
                        ids=[chunk_id],
                        metadatas=[metadata],
                    )
                except (ValueError, TypeError):
                    pass

        return True

    def get_freshness_stats(self) -> dict:
        """Get statistics about document freshness.

        Returns:
            Dictionary with freshness statistics.
        """
        results = self.vector_store._collection.get(include=["metadatas"])

        if not results["metadatas"]:
            return {
                "total_chunks": 0,
                "expired_chunks": 0,
                "expiring_7_days": 0,
                "expiring_30_days": 0,
                "by_type": {},
            }

        now = datetime.now()
        expired = 0
        expiring_7 = 0
        expiring_30 = 0
        by_type: dict[str, dict] = {}

        for metadata in results["metadatas"]:
            doc_type = metadata.get("document_type", "unknown")

            if doc_type not in by_type:
                by_type[doc_type] = {"total": 0, "expired": 0, "expiring_soon": 0}

            by_type[doc_type]["total"] += 1

            expiry_str = metadata.get("expiry_date")
            if expiry_str:
                try:
                    expiry_date = datetime.fromisoformat(expiry_str)
                    if expiry_date < now:
                        expired += 1
                        by_type[doc_type]["expired"] += 1
                    elif expiry_date <= now + timedelta(days=7):
                        expiring_7 += 1
                        by_type[doc_type]["expiring_soon"] += 1
                    elif expiry_date <= now + timedelta(days=30):
                        expiring_30 += 1
                except (ValueError, TypeError):
                    pass

        return {
            "total_chunks": len(results["metadatas"]),
            "expired_chunks": expired,
            "expiring_7_days": expiring_7,
            "expiring_30_days": expiring_30,
            "by_type": by_type,
        }


class DocumentUpdateManager:
    """Manages document updates and re-indexing."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        freshness_manager: FreshnessManager,
    ):
        """Initialize the update manager.

        Args:
            vector_store: ChromaDB vector store instance.
            freshness_manager: Freshness manager instance.
        """
        self.vector_store = vector_store
        self.freshness_manager = freshness_manager

    def update_document(self, document: Document) -> int:
        """Update a document by re-indexing it.

        This removes all existing chunks and re-indexes the document.

        Args:
            document: Updated document to index.

        Returns:
            Number of new chunks indexed.
        """
        # Remove old chunks
        self.vector_store.delete_document(document.doc_id)

        # Set new expiry based on policy
        document = self.freshness_manager.set_expiry_for_document(document)

        # Re-index
        return self.vector_store.add_documents([document])

    def refresh_documents(
        self,
        document_fetcher: Callable[[str], Document | None],
        doc_ids: list[str] | None = None,
    ) -> dict[str, int]:
        """Refresh documents by fetching and re-indexing.

        Args:
            document_fetcher: Function that fetches a document by ID.
            doc_ids: Optional list of document IDs to refresh.
                    If None, refreshes documents expiring within 7 days.

        Returns:
            Dictionary with refresh statistics.
        """
        if doc_ids is None:
            doc_ids = self.freshness_manager.get_expiring_soon(days=7)

        stats = {"refreshed": 0, "failed": 0, "skipped": 0}

        for doc_id in doc_ids:
            try:
                document = document_fetcher(doc_id)
                if document:
                    self.update_document(document)
                    stats["refreshed"] += 1
                else:
                    stats["skipped"] += 1
            except Exception:
                stats["failed"] += 1

        return stats
