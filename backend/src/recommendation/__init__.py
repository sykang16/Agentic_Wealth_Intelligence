"""Recommendation module for personalized investment advice using RAG and MCP."""

from .rag import (
    # Schemas
    Document,
    DocumentChunk,
    DocumentMetadata,
    DocumentType,
    RetrievalResult,
    RetrievalResponse,
    SearchQuery,
    # Embeddings
    DocumentChunker,
    EmbeddingPipeline,
    SentenceTransformerEmbeddings,
    # Vector Store
    ChromaVectorStore,
    # Retrieval
    HybridRetriever,
    RAGRetriever,
    ReRankingConfig,
    ReRankingStrategy,
    # Freshness
    DocumentTypePolicy,
    DocumentUpdateManager,
    FreshnessManager,
    RetentionConfig,
    RetentionPolicy,
    # Initializer
    RAGInitializer,
    get_rag_system,
    reset_rag_system,
)
from .collectors import (
    # Base
    BaseCollector,
    CollectorResult,
    # Collectors
    AlphaVantageCollector,
    SECEdgarCollector,
    NewsAPICollector,
    # Manager
    DataCollectionManager,
)
from .engine import (
    # Schemas
    AggregatedContext,
    ConfidenceLevel,
    DataSource,
    Recommendation,
    RecommendationCategory,
    RecommendationPriority,
    RecommendationRequest,
    RecommendationResponse,
    RiskLevel,
    # Engine components
    ContextBuilder,
    RecommendationEngine,
    RecommendationGenerator,
    RecommendationRanker,
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
    # Collectors
    "BaseCollector",
    "CollectorResult",
    "AlphaVantageCollector",
    "SECEdgarCollector",
    "NewsAPICollector",
    "DataCollectionManager",
    # Engine
    "AggregatedContext",
    "ConfidenceLevel",
    "ContextBuilder",
    "DataSource",
    "Recommendation",
    "RecommendationCategory",
    "RecommendationEngine",
    "RecommendationGenerator",
    "RecommendationPriority",
    "RecommendationRanker",
    "RecommendationRequest",
    "RecommendationResponse",
    "RiskLevel",
]
