"""Pydantic schemas for RAG system documents and retrieval."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, computed_field


class DocumentType(str, Enum):
    """Types of financial documents in the RAG system."""

    ETF_FACTSHEET = "etf_factsheet"
    FOMC_MINUTES = "fomc_minutes"
    INVESTMENT_GUIDE = "investment_guide"
    MARKET_ANALYSIS = "market_analysis"
    RESEARCH_REPORT = "research_report"
    NEWS_ARTICLE = "news_article"


class DocumentMetadata(BaseModel):
    """Metadata for financial documents."""

    source: str = Field(..., description="Source of the document (e.g., 'Vanguard', 'Federal Reserve')")
    document_type: DocumentType = Field(..., description="Type of document")
    publish_date: datetime = Field(..., description="Publication date")
    expiry_date: datetime | None = Field(default=None, description="When document should be removed")
    tags: list[str] = Field(default_factory=list, description="Tags for filtering")
    author: str | None = Field(default=None, description="Author or institution")
    url: str | None = Field(default=None, description="Source URL if applicable")
    ticker: str | None = Field(default=None, description="Related ticker symbol (e.g., 'VOO', 'SPY')")
    sector: str | None = Field(default=None, description="Related market sector")
    extra: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class Document(BaseModel):
    """A financial document to be indexed in the RAG system."""

    doc_id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Full document content")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @computed_field
    @property
    def is_expired(self) -> bool:
        """Check if document has expired based on expiry_date."""
        if self.metadata.expiry_date is None:
            return False
        return datetime.now() > self.metadata.expiry_date


class DocumentChunk(BaseModel):
    """A chunk of a document for vector storage."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    doc_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Chunk text content")
    chunk_index: int = Field(..., ge=0, description="Position of chunk in document")
    start_char: int = Field(..., ge=0, description="Start character position in original doc")
    end_char: int = Field(..., ge=0, description="End character position in original doc")
    metadata: DocumentMetadata = Field(..., description="Inherited document metadata")
    embedding: list[float] | None = Field(default=None, description="Vector embedding")

    @computed_field
    @property
    def token_estimate(self) -> int:
        """Estimate token count (roughly 4 chars per token)."""
        return len(self.content) // 4


class SearchQuery(BaseModel):
    """Query parameters for RAG retrieval."""

    query_text: str = Field(..., description="Natural language query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    min_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum similarity score")
    document_types: list[DocumentType] | None = Field(
        default=None, description="Filter by document types"
    )
    tags: list[str] | None = Field(default=None, description="Filter by tags")
    tickers: list[str] | None = Field(default=None, description="Filter by ticker symbols")
    sectors: list[str] | None = Field(default=None, description="Filter by sectors")
    include_expired: bool = Field(default=False, description="Include expired documents")
    date_from: datetime | None = Field(default=None, description="Filter by publish date (from)")
    date_to: datetime | None = Field(default=None, description="Filter by publish date (to)")


class RetrievalResult(BaseModel):
    """A single result from RAG retrieval."""

    chunk: DocumentChunk = Field(..., description="Retrieved chunk")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    rank: int = Field(..., ge=1, description="Rank in result set")

    @computed_field
    @property
    def doc_id(self) -> str:
        """Convenience accessor for document ID."""
        return self.chunk.doc_id


class RetrievalResponse(BaseModel):
    """Response from RAG retrieval containing multiple results."""

    query: SearchQuery = Field(..., description="Original query")
    results: list[RetrievalResult] = Field(default_factory=list, description="Retrieved results")
    total_found: int = Field(default=0, ge=0, description="Total matching documents")
    processing_time_ms: float = Field(default=0.0, ge=0.0, description="Query processing time")

    @computed_field
    @property
    def has_results(self) -> bool:
        """Check if any results were found."""
        return len(self.results) > 0

    def get_context_string(self, max_chunks: int = 5, separator: str = "\n\n---\n\n") -> str:
        """Format results as a context string for LLM prompts.

        Args:
            max_chunks: Maximum number of chunks to include.
            separator: Separator between chunks.

        Returns:
            Formatted context string.
        """
        chunks_to_use = self.results[:max_chunks]
        formatted_chunks = []

        for result in chunks_to_use:
            chunk = result.chunk
            header = f"[{chunk.metadata.document_type.value}] {chunk.metadata.source}"
            if chunk.metadata.ticker:
                header += f" ({chunk.metadata.ticker})"
            formatted_chunks.append(f"{header}\n{chunk.content}")

        return separator.join(formatted_chunks)
