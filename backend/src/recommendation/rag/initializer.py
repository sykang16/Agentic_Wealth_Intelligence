"""RAG system initialization and startup indexing."""

import logging
from pathlib import Path

from .retrieval import RAGRetriever
from .freshness import FreshnessManager, DocumentTypePolicy

logger = logging.getLogger(__name__)


class RAGInitializer:
    """Handles RAG system initialization and document indexing at startup."""

    DEFAULT_PERSIST_DIR = "./data/chroma"
    DEFAULT_COLLECTION = "financial_knowledge"

    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str | None = None,
        embedding_model: str | None = None,
    ):
        """Initialize the RAG initializer.

        Args:
            persist_directory: Directory for ChromaDB persistence.
            collection_name: Name of the collection.
            embedding_model: Sentence transformer model name.
        """
        self.persist_directory = persist_directory or self.DEFAULT_PERSIST_DIR
        self.collection_name = collection_name or self.DEFAULT_COLLECTION
        self.embedding_model = embedding_model
        self._retriever: RAGRetriever | None = None
        self._freshness_manager: FreshnessManager | None = None

    @property
    def retriever(self) -> RAGRetriever:
        """Get or create the RAG retriever."""
        if self._retriever is None:
            self._retriever = RAGRetriever(
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
                embedding_model=self.embedding_model,
            )
        return self._retriever

    @property
    def freshness_manager(self) -> FreshnessManager:
        """Get or create the freshness manager."""
        if self._freshness_manager is None:
            self._freshness_manager = FreshnessManager(
                vector_store=self.retriever.vector_store,
                type_policies=DocumentTypePolicy(),
            )
        return self._freshness_manager

    def is_initialized(self) -> bool:
        """Check if the RAG system has documents indexed.

        Returns:
            True if documents are already indexed.
        """
        try:
            stats = self.retriever.get_stats()
            return stats["total_chunks"] > 0
        except Exception:
            return False

    def get_stats(self) -> dict:
        """Get current RAG system statistics.

        Returns:
            Dictionary with index statistics.
        """
        return self.retriever.get_stats()

    def initialize_with_sample_documents(self, force_reindex: bool = False) -> dict:
        """Initialize RAG system with sample financial documents.

        This method:
        1. Checks if documents are already indexed
        2. If not (or force_reindex=True), generates and indexes sample documents
        3. Returns statistics about the indexed documents

        Args:
            force_reindex: If True, clears existing data and reindexes.

        Returns:
            Dictionary with initialization results.
        """
        from backend.src.data_generation.document_generator import generate_all_sample_documents

        result = {
            "action": "none",
            "documents_indexed": 0,
            "chunks_created": 0,
            "already_initialized": False,
        }

        # Check if already initialized
        if self.is_initialized() and not force_reindex:
            stats = self.get_stats()
            result["action"] = "skipped"
            result["already_initialized"] = True
            result["documents_indexed"] = stats["unique_documents"]
            result["chunks_created"] = stats["total_chunks"]
            logger.info(
                f"RAG system already initialized with {stats['unique_documents']} documents "
                f"({stats['total_chunks']} chunks). Skipping indexing."
            )
            return result

        # Clear if force reindex
        if force_reindex and self.is_initialized():
            logger.info("Force reindex requested. Clearing existing data...")
            self.retriever.clear_index()
            result["action"] = "reindexed"
        else:
            result["action"] = "initialized"

        # Generate sample documents
        logger.info("Generating sample financial documents...")
        documents = generate_all_sample_documents()

        # Prepare documents with expiry dates
        logger.info("Setting document retention policies...")
        documents = self.freshness_manager.prepare_documents(documents)

        # Index documents
        logger.info(f"Indexing {len(documents)} documents...")
        num_chunks = self.retriever.index_documents(documents)

        result["documents_indexed"] = len(documents)
        result["chunks_created"] = num_chunks

        logger.info(
            f"RAG initialization complete: {len(documents)} documents, {num_chunks} chunks"
        )

        return result

    def search(self, query: str, top_k: int = 5) -> dict:
        """Search the RAG system.

        Args:
            query: Natural language query.
            top_k: Number of results to return.

        Returns:
            Dictionary with search results.
        """
        response = self.retriever.search(query, top_k=top_k)
        return {
            "query": query,
            "num_results": len(response.results),
            "results": [
                {
                    "content": r.chunk.content[:500] + "..." if len(r.chunk.content) > 500 else r.chunk.content,
                    "score": round(r.score, 3),
                    "source": r.chunk.metadata.source,
                    "document_type": r.chunk.metadata.document_type.value,
                    "ticker": r.chunk.metadata.ticker,
                }
                for r in response.results
            ],
            "context": response.get_context_string(max_chunks=top_k),
        }

    def add_documents(self, documents: list) -> int:
        """Add documents to the RAG index incrementally.

        This adds new documents without clearing existing ones.

        Args:
            documents: List of Document instances to add.

        Returns:
            Number of chunks indexed.
        """
        if not documents:
            return 0

        # Prepare documents with expiry dates
        prepared = self.freshness_manager.prepare_documents(documents)

        # Index documents
        return self.retriever.index_documents(prepared)

    def delete_documents_by_source(self, source: str) -> int:
        """Delete all documents from a specific source.

        Args:
            source: Source name to delete (e.g., "Alpha Vantage").

        Returns:
            Number of chunks deleted.
        """
        # Get all chunks from this source
        collection = self.retriever.vector_store._collection
        results = collection.get(
            where={"source": source},
            include=["metadatas"],
        )

        if not results["ids"]:
            return 0

        count = len(results["ids"])
        collection.delete(ids=results["ids"])
        return count

    def cleanup_expired(self) -> int:
        """Remove expired documents from the index.

        Returns:
            Number of chunks removed.
        """
        return self.freshness_manager.cleanup_expired()


# Singleton instance for app-wide use
_rag_instance: RAGInitializer | None = None


def get_rag_system(
    persist_directory: str | None = None,
    collection_name: str | None = None,
    auto_initialize: bool = True,
) -> RAGInitializer:
    """Get or create the global RAG system instance.

    This function provides a singleton pattern for the RAG system,
    ensuring only one instance is created and reused across the app.

    Args:
        persist_directory: Directory for ChromaDB persistence.
        collection_name: Name of the collection.
        auto_initialize: If True, automatically indexes sample documents if empty.

    Returns:
        RAGInitializer instance.
    """
    global _rag_instance

    if _rag_instance is None:
        _rag_instance = RAGInitializer(
            persist_directory=persist_directory,
            collection_name=collection_name,
        )

        if auto_initialize:
            _rag_instance.initialize_with_sample_documents()

    return _rag_instance


def reset_rag_system() -> None:
    """Reset the global RAG system instance.

    Use this to force reinitialization on next access.
    """
    global _rag_instance
    _rag_instance = None
