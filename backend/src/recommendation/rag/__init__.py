"""RAG (Retrieval Augmented Generation) module for financial document processing."""

from .schemas import (
    Document,
    DocumentChunk,
    DocumentMetadata,
    DocumentType,
    RetrievalResult,
    RetrievalResponse,
    SearchQuery,
)
from .embeddings import (
    DocumentChunker,
    EmbeddingPipeline,
    SentenceTransformerEmbeddings,
)
from .vector_store import ChromaVectorStore
from .retrieval import (
    HybridRetriever,
    RAGRetriever,
    ReRankingConfig,
    ReRankingStrategy,
)
from .freshness import (
    DocumentTypePolicy,
    DocumentUpdateManager,
    FreshnessManager,
    RetentionConfig,
    RetentionPolicy,
)
from .initializer import (
    RAGInitializer,
    get_rag_system,
    reset_rag_system,
)

__all__ = [
    # Schemas
    "Document",
    "DocumentChunk",
    "DocumentMetadata",
    "DocumentType",
    "RetrievalResult",
    "RetrievalResponse",
    "SearchQuery",
    # Embeddings
    "DocumentChunker",
    "EmbeddingPipeline",
    "SentenceTransformerEmbeddings",
    # Vector Store
    "ChromaVectorStore",
    # Retrieval
    "HybridRetriever",
    "RAGRetriever",
    "ReRankingConfig",
    "ReRankingStrategy",
    # Freshness
    "DocumentTypePolicy",
    "DocumentUpdateManager",
    "FreshnessManager",
    "RetentionConfig",
    "RetentionPolicy",
    # Initializer
    "RAGInitializer",
    "get_rag_system",
    "reset_rag_system",
]
