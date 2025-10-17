"""
Abstract base class for text retrieval backends (BM25, Elasticsearch, etc.).

This module provides the abstraction layer for different text search backends:
- BM25 Serialize (current implementation)
- Elasticsearch (future migration)
- Other text search engines (extensible)

Design Pattern: Strategy Pattern
- Allows swapping text search backends without changing client code
- Provides consistent interface for all text search implementations
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from langchain.schema import Document


class BaseTextRetriever(ABC):
    """
    Abstract base class for text search retrieval backends.

    All text search implementations (BM25, Elasticsearch, etc.) must inherit
    from this class and implement the abstract methods.

    Usage:
        # Client code (pipeline.py)
        retriever = get_text_retriever(backend="bm25")
        await retriever.initialize()
        results = await retriever.search(query="revenue growth", top_k=20)
    """

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the search backend.

        This method should:
        - Load or build the search index
        - Connect to external services if needed
        - Prepare the retriever for search queries

        Called once when the retriever is first set up.
        """
        pass

    @abstractmethod
    async def index_documents(
        self,
        documents: List[Document],
        presentation_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Index documents for a presentation (ingestion phase).

        Args:
            documents: List of Document objects with page_content and metadata
            presentation_id: Unique identifier for the presentation
            metadata: Optional presentation-level metadata (title, slides, etc.)

        This method should:
        - Store the documents in the search backend
        - Build/update the search index
        - Save any necessary metadata for retrieval
        """
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search for relevant documents (retrieval phase).

        Args:
            query: Search query string
            top_k: Number of results to return
            filters: Optional filters (e.g., presentation_id, slide_number)

        Returns:
            List of Document objects ranked by relevance

        This method should:
        - Execute search query against the index
        - Apply any filters
        - Return top-k results sorted by relevance score
        """
        pass

    @abstractmethod
    async def load_presentation(self, presentation_id: str) -> None:
        """
        Load documents for a specific presentation (optional optimization).

        Args:
            presentation_id: Presentation to load

        Some backends (BM25) may benefit from loading a specific presentation
        into memory. Others (Elasticsearch) may not need this.

        Default behavior: No-op if not needed by backend
        """
        pass

    @abstractmethod
    async def get_all_documents(self) -> List[Document]:
        """
        Load all documents across all presentations (for cross-document search).

        Returns:
            List of all Document objects in the search backend

        Used for:
        - Cross-document search (query across all presentations)
        - Re-building search index
        - Migration to different backend
        """
        pass

    @abstractmethod
    async def delete_presentation(self, presentation_id: str) -> None:
        """
        Delete all documents for a presentation.

        Args:
            presentation_id: Presentation to delete

        This method should:
        - Remove documents from search backend
        - Clean up any associated metadata
        - Update the search index
        """
        pass

    @abstractmethod
    async def list_presentations(self) -> List[Dict[str, Any]]:
        """
        List all indexed presentations.

        Returns:
            List of presentation metadata dicts with:
            - presentation_id
            - name
            - title
            - total_slides
            - total_chunks
            - indexed_at

        Used for:
        - UI presentation selection
        - Admin/management tools
        """
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get search backend statistics.

        Returns:
            Dictionary with backend-specific stats:
            - total_documents
            - total_presentations
            - index_size (bytes)
            - backend_type (bm25, elasticsearch, etc.)
            - last_updated

        Used for monitoring and debugging.
        """
        pass

    async def health_check(self) -> bool:
        """
        Check if search backend is healthy and ready.

        Returns:
            True if backend is ready, False otherwise

        Optional method with default implementation.
        Backends can override for custom health checks.
        """
        try:
            stats = await self.get_stats()
            return stats.get("total_documents", 0) >= 0
        except Exception:
            return False


def get_text_retriever(backend: str = "bm25", **kwargs) -> BaseTextRetriever:
    """
    Factory function to create text search retriever instances.

    Args:
        backend: Backend type ("bm25" or "elasticsearch")
        **kwargs: Backend-specific configuration

    Returns:
        Concrete implementation of BaseTextRetriever

    Example:
        # BM25 backend
        retriever = get_text_retriever(
            backend="bm25",
            db_path="data/bm25/bm25_store.db",
            index_path="data/bm25/bm25_index.dill"
        )

        # Elasticsearch backend (future)
        retriever = get_text_retriever(
            backend="elasticsearch",
            es_url="http://localhost:9200",
            index_name="presentations"
        )
    """
    if backend == "bm25":
        from src.storage.bm25_serialize_retriever import BM25SerializeRetriever
        return BM25SerializeRetriever(**kwargs)

    elif backend == "elasticsearch":
        from src.storage.elasticsearch_retriever import ElasticsearchRetriever
        return ElasticsearchRetriever(**kwargs)

    else:
        raise ValueError(
            f"Unknown search backend: {backend}. "
            f"Supported backends: bm25, elasticsearch"
        )
