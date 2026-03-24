"""ChromaDB vector store implementation for RAG system."""

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from .schemas import (
    Document,
    DocumentChunk,
    DocumentMetadata,
    DocumentType,
    RetrievalResponse,
    RetrievalResult,
    SearchQuery,
)
from .embeddings import EmbeddingPipeline


class ChromaVectorStore:
    """Vector store implementation using ChromaDB."""

    DEFAULT_COLLECTION_NAME = "financial_documents"
    DEFAULT_PERSIST_DIR = "./data/chroma"

    def __init__(
        self,
        collection_name: str | None = None,
        persist_directory: str | None = None,
        embedding_model: str | None = None,
    ):
        """Initialize the ChromaDB vector store.

        Args:
            collection_name: Name of the ChromaDB collection.
            persist_directory: Directory to persist the database.
            embedding_model: Sentence transformer model for embeddings.
        """
        self.collection_name = collection_name or self.DEFAULT_COLLECTION_NAME
        self.persist_directory = persist_directory or os.getenv(
            "CHROMA_PATH", self.DEFAULT_PERSIST_DIR
        )

        # Initialize embedding pipeline
        self.embedding_pipeline = EmbeddingPipeline(model_name=embedding_model)

        # Initialize ChromaDB client
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the ChromaDB client and collection."""
        # Ensure persist directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # Create ChromaDB client with persistence
        self._client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Financial documents for RAG"},
        )

    def add_documents(self, documents: list[Document]) -> int:
        """Add documents to the vector store.

        Args:
            documents: List of documents to add.

        Returns:
            Number of chunks added.
        """
        # Process documents into embedded chunks
        chunks = self.embedding_pipeline.process_documents(documents)
        return self.add_chunks(chunks)

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Add pre-processed chunks to the vector store.

        Args:
            chunks: List of document chunks to add.

        Returns:
            Number of chunks added.
        """
        if not chunks:
            return 0

        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for chunk in chunks:
            if chunk.embedding is None:
                # Generate embedding if not present
                chunk.embedding = self.embedding_pipeline.embed_query(chunk.content)

            ids.append(chunk.chunk_id)
            embeddings.append(chunk.embedding)
            documents.append(chunk.content)
            metadatas.append(self._chunk_to_metadata(chunk))

        # Add to collection
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        return len(chunks)

    def search(self, query: SearchQuery) -> RetrievalResponse:
        """Search for relevant documents.

        Args:
            query: Search query parameters.

        Returns:
            RetrievalResponse with ranked results.
        """
        import time
        start_time = time.time()

        # Generate query embedding
        query_embedding = self.embedding_pipeline.embed_query(query.query_text)

        # Build where filter
        where_filter = self._build_where_filter(query)

        # Perform search
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=query.top_k,
            where=where_filter if where_filter else None,
            include=["documents", "metadatas", "distances"],
        )

        # Convert results to RetrievalResult objects
        retrieval_results = []
        if results["ids"] and results["ids"][0]:
            for i, (chunk_id, document, metadata, distance) in enumerate(
                zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ):
                # Convert distance to similarity score (ChromaDB returns L2 distance)
                # L2 distance: lower is better, convert to similarity
                score = 1.0 / (1.0 + distance)

                if score < query.min_score:
                    continue

                chunk = self._metadata_to_chunk(chunk_id, document, metadata)

                # Check expiry
                if not query.include_expired and chunk.metadata.expiry_date:
                    if datetime.now() > chunk.metadata.expiry_date:
                        continue

                retrieval_results.append(
                    RetrievalResult(
                        chunk=chunk,
                        score=score,
                        rank=len(retrieval_results) + 1,
                    )
                )

        processing_time = (time.time() - start_time) * 1000

        return RetrievalResponse(
            query=query,
            results=retrieval_results,
            total_found=len(retrieval_results),
            processing_time_ms=processing_time,
        )

    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks for a document.

        Args:
            doc_id: Document ID to delete.

        Returns:
            Number of chunks deleted.
        """
        # Find all chunks for this document
        results = self._collection.get(
            where={"doc_id": doc_id},
            include=["metadatas"],
        )

        if not results["ids"]:
            return 0

        count = len(results["ids"])
        self._collection.delete(ids=results["ids"])
        return count

    def delete_expired(self) -> int:
        """Delete all expired documents.

        Returns:
            Number of chunks deleted.
        """
        # Get all documents
        results = self._collection.get(include=["metadatas"])

        if not results["ids"]:
            return 0

        now = datetime.now()
        expired_ids = []

        for chunk_id, metadata in zip(results["ids"], results["metadatas"]):
            expiry_str = metadata.get("expiry_date")
            if expiry_str:
                try:
                    expiry_date = datetime.fromisoformat(expiry_str)
                    if now > expiry_date:
                        expired_ids.append(chunk_id)
                except (ValueError, TypeError):
                    pass

        if expired_ids:
            self._collection.delete(ids=expired_ids)

        return len(expired_ids)

    def get_document_count(self) -> int:
        """Get total number of chunks in the store.

        Returns:
            Number of chunks.
        """
        return self._collection.count()

    def get_unique_documents(self) -> list[str]:
        """Get list of unique document IDs.

        Returns:
            List of document IDs.
        """
        results = self._collection.get(include=["metadatas"])

        if not results["metadatas"]:
            return []

        doc_ids = set()
        for metadata in results["metadatas"]:
            doc_id = metadata.get("doc_id")
            if doc_id:
                doc_ids.add(doc_id)

        return list(doc_ids)

    def clear(self) -> None:
        """Clear all documents from the store."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Financial documents for RAG"},
        )

    def _chunk_to_metadata(self, chunk: DocumentChunk) -> dict[str, Any]:
        """Convert a DocumentChunk to ChromaDB metadata format.

        Args:
            chunk: Chunk to convert.

        Returns:
            Metadata dictionary.
        """
        metadata = {
            "doc_id": chunk.doc_id,
            "chunk_index": chunk.chunk_index,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "source": chunk.metadata.source,
            "document_type": chunk.metadata.document_type.value,
            "publish_date": chunk.metadata.publish_date.isoformat(),
        }

        # Add optional metadata
        if chunk.metadata.expiry_date:
            metadata["expiry_date"] = chunk.metadata.expiry_date.isoformat()
        if chunk.metadata.tags:
            metadata["tags"] = ",".join(chunk.metadata.tags)
        if chunk.metadata.author:
            metadata["author"] = chunk.metadata.author
        if chunk.metadata.url:
            metadata["url"] = chunk.metadata.url
        if chunk.metadata.ticker:
            metadata["ticker"] = chunk.metadata.ticker
        if chunk.metadata.sector:
            metadata["sector"] = chunk.metadata.sector

        return metadata

    def _metadata_to_chunk(
        self, chunk_id: str, content: str, metadata: dict[str, Any]
    ) -> DocumentChunk:
        """Convert ChromaDB metadata back to a DocumentChunk.

        Args:
            chunk_id: Chunk identifier.
            content: Chunk content.
            metadata: ChromaDB metadata.

        Returns:
            DocumentChunk instance.
        """
        # Parse tags
        tags = []
        if metadata.get("tags"):
            tags = metadata["tags"].split(",")

        # Parse dates
        publish_date = datetime.fromisoformat(metadata["publish_date"])
        expiry_date = None
        if metadata.get("expiry_date"):
            expiry_date = datetime.fromisoformat(metadata["expiry_date"])

        doc_metadata = DocumentMetadata(
            source=metadata["source"],
            document_type=DocumentType(metadata["document_type"]),
            publish_date=publish_date,
            expiry_date=expiry_date,
            tags=tags,
            author=metadata.get("author"),
            url=metadata.get("url"),
            ticker=metadata.get("ticker"),
            sector=metadata.get("sector"),
        )

        return DocumentChunk(
            chunk_id=chunk_id,
            doc_id=metadata["doc_id"],
            content=content,
            chunk_index=metadata["chunk_index"],
            start_char=metadata["start_char"],
            end_char=metadata["end_char"],
            metadata=doc_metadata,
        )

    def _build_where_filter(self, query: SearchQuery) -> dict[str, Any] | None:
        """Build ChromaDB where filter from query parameters.

        Args:
            query: Search query with filter parameters.

        Returns:
            ChromaDB where filter or None if no filters.
        """
        conditions = []

        # Filter by document types
        if query.document_types:
            type_values = [dt.value for dt in query.document_types]
            if len(type_values) == 1:
                conditions.append({"document_type": type_values[0]})
            else:
                conditions.append({"document_type": {"$in": type_values}})

        # Filter by tickers
        if query.tickers:
            if len(query.tickers) == 1:
                conditions.append({"ticker": query.tickers[0]})
            else:
                conditions.append({"ticker": {"$in": query.tickers}})

        # Filter by sectors
        if query.sectors:
            if len(query.sectors) == 1:
                conditions.append({"sector": query.sectors[0]})
            else:
                conditions.append({"sector": {"$in": query.sectors}})

        # Combine conditions
        if not conditions:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}
