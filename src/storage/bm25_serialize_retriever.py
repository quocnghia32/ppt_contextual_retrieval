"""
BM25 Serialize Retriever - Search backend using serialized BM25 index.

This implementation:
1. Stores text chunks in SQLite (via BM25Store)
2. Builds BM25 index in memory
3. Serializes index to disk for fast startup (dill format)
4. Supports both single-presentation and cross-document search

Performance (60K chunks):
- Startup (load serialized): ~3s
- Startup (rebuild): ~10s
- Query: 10-50ms
- Storage: ~400 MB (SQLite + serialized index)

Based on design in: docs/CROSS_DOCUMENT_SEARCH_STRATEGY.md
"""

import dill
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import logging
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever

from src.storage.base_text_retriever import BaseTextRetriever
from src.storage.bm25_store import BM25Store

logger = logging.getLogger(__name__)


class BM25SerializeRetriever(BaseTextRetriever):
    """
    BM25 search backend with index serialization.

    Usage:
        # Initialize
        retriever = BM25SerializeRetriever(
            db_path="data/bm25/bm25_store.db",
            index_path="data/bm25/bm25_index.dill"
        )
        await retriever.initialize()

        # Ingestion phase
        await retriever.index_documents(
            documents=chunks,
            presentation_id="ppt-report-2024"
        )

        # Query phase
        results = await retriever.search(query="revenue growth", top_k=20)
    """

    INDEX_VERSION = "1.0.0"

    def __init__(
        self,
        db_path: str = "data/bm25/bm25_store.db",
        index_path: str = "data/bm25/bm25_index.dill",
        k: int = 20
    ):
        """
        Initialize BM25SerializeRetriever.

        Args:
            db_path: Path to SQLite database (BM25Store)
            index_path: Path to serialized BM25 index file
            k: Number of results to return per search
        """
        self.db_path = db_path
        self.index_path = index_path
        self.k = k

        self.store = BM25Store(db_path=db_path)
        self.bm25_retriever: Optional[BM25Retriever] = None
        self._index_loaded = False

        self._ensure_dir()

    def _ensure_dir(self):
        """Ensure directories exist."""
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> None:
        """
        Initialize BM25 retriever.

        Strategy:
        1. Try to load serialized index (fast: 2-3s)
        2. If load fails, rebuild from SQLite (slow: 10s)
        3. Save rebuilt index for next time
        """
        # Initialize store
        await self.store.initialize()

        # Try to load serialized index
        success = await self._load_serialized_index()

        if not success:
            # Rebuild from SQLite
            logger.warning("Failed to load serialized index, rebuilding...")
            await self._rebuild_index()

        logger.info("BM25SerializeRetriever initialized successfully")

    async def _load_serialized_index(self) -> bool:
        """
        Try to load serialized BM25 index from disk.

        Returns:
            True if successful, False otherwise
        """
        if not Path(self.index_path).exists():
            logger.info("No serialized index found")
            return False

        try:
            # Load in thread pool to avoid blocking
            index_data = await asyncio.to_thread(self._load_index_file)

            # Validate version
            if index_data.get("version") != self.INDEX_VERSION:
                logger.warning(
                    f"Index version mismatch: "
                    f"{index_data.get('version')} != {self.INDEX_VERSION}"
                )
                return False

            # Extract retriever
            self.bm25_retriever = index_data["retriever"]
            self.bm25_retriever.k = self.k  # Update k

            self._index_loaded = True

            logger.info(
                f"✅ Loaded serialized BM25 index "
                f"({index_data.get('total_documents')} docs, "
                f"saved {index_data.get('created_at')})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load serialized index: {e}")
            return False

    def _load_index_file(self) -> Dict[str, Any]:
        """Load serialized index (synchronous)."""
        with open(self.index_path, "rb") as f:
            return dill.load(f)

    async def _rebuild_index(self) -> None:
        """
        Rebuild BM25 index from SQLite.

        This is called when:
        - Serialized index doesn't exist
        - Serialized index is corrupted/outdated
        - After ingesting new documents
        """
        logger.info("🔨 Rebuilding BM25 index from SQLite...")

        # Load all documents
        documents = await self.store.load_all_chunks()

        if not documents:
            logger.warning("No documents found in SQLite")
            self.bm25_retriever = None
            self._index_loaded = False
            return

        # Build BM25 retriever in thread pool (CPU-intensive)
        self.bm25_retriever = await asyncio.to_thread(
            self._build_bm25,
            documents
        )

        self._index_loaded = True

        # Save serialized index for next time
        await self._save_serialized_index(len(documents))

        logger.info(f"✅ Rebuilt BM25 index with {len(documents)} documents")

    def _build_bm25(self, documents: List[Document]) -> BM25Retriever:
        """Build BM25 retriever (synchronous, CPU-intensive)."""
        retriever = BM25Retriever.from_documents(documents)
        retriever.k = self.k
        return retriever

    async def _save_serialized_index(self, total_documents: int) -> None:
        """
        Save BM25 index to disk (serialized).

        Args:
            total_documents: Number of documents in index
        """
        if not self.bm25_retriever:
            logger.warning("No BM25 retriever to save")
            return

        try:
            # Prepare index data with metadata
            index_data = {
                "version": self.INDEX_VERSION,
                "retriever": self.bm25_retriever,
                "total_documents": total_documents,
                "created_at": datetime.utcnow().isoformat()
            }

            # Save in thread pool (disk I/O)
            await asyncio.to_thread(self._save_index_file, index_data)

            logger.info(f"💾 Saved serialized BM25 index to {self.index_path}")

        except Exception as e:
            logger.error(f"Failed to save serialized index: {e}")

    def _save_index_file(self, index_data: Dict[str, Any]) -> None:
        """Save index to file (synchronous)."""
        with open(self.index_path, "wb") as f:
            dill.dump(index_data, f)

    async def index_documents(
        self,
        documents: List[Document],
        presentation_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Index documents for a presentation (ingestion phase).

        This method:
        1. Saves documents to SQLite (BM25Store)
        2. Rebuilds BM25 index with ALL documents (including new ones)
        3. Saves serialized index for fast startup

        Args:
            documents: List of Document objects
            presentation_id: Unique presentation identifier
            metadata: Optional presentation metadata
        """
        # Save to SQLite
        await self.store.save_presentation(
            presentation_id=presentation_id,
            documents=documents,
            metadata=metadata
        )

        logger.info(f"Indexed {len(documents)} documents for {presentation_id}")

        # Rebuild index with ALL documents (cross-document search)
        await self._rebuild_index()

    async def search(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search for relevant documents using BM25.

        Args:
            query: Search query string
            top_k: Number of results to return
            filters: Optional filters (e.g., presentation_id)

        Returns:
            List of Document objects ranked by BM25 score
        """
        if not self._index_loaded or not self.bm25_retriever:
            raise RuntimeError(
                "BM25 index not loaded. Call initialize() first."
            )

        # Update k if different
        original_k = self.bm25_retriever.k
        self.bm25_retriever.k = top_k

        # Search in thread pool (CPU-intensive)
        results = await asyncio.to_thread(
            self.bm25_retriever.get_relevant_documents,
            query
        )

        # Apply filters if provided
        if filters:
            results = self._apply_filters(results, filters)

        # Restore original k
        self.bm25_retriever.k = original_k

        return results

    def _apply_filters(
        self,
        documents: List[Document],
        filters: Dict[str, Any]
    ) -> List[Document]:
        """
        Apply metadata filters to search results.

        Args:
            documents: Search results
            filters: Filter criteria (e.g., {"presentation_id": "ppt-123"})

        Returns:
            Filtered documents
        """
        filtered = []
        for doc in documents:
            match = True
            for key, value in filters.items():
                if doc.metadata.get(key) != value:
                    match = False
                    break
            if match:
                filtered.append(doc)

        return filtered

    async def load_presentation(self, presentation_id: str) -> None:
        """
        Load documents for a specific presentation.

        Note: For BM25 serialize strategy, we always load ALL documents
        for cross-document search. This method is a no-op but provided
        for interface compatibility.

        Args:
            presentation_id: Presentation to load (ignored)
        """
        # For cross-document search, all documents already loaded
        logger.info(
            f"BM25SerializeRetriever already has all documents loaded "
            f"(cross-document search mode)"
        )

    async def get_all_documents(self) -> List[Document]:
        """
        Get all documents from SQLite.

        Returns:
            List of all Document objects
        """
        return await self.store.load_all_chunks()

    async def delete_presentation(self, presentation_id: str) -> None:
        """
        Delete presentation and rebuild index.

        Args:
            presentation_id: Presentation to delete
        """
        # Delete from SQLite
        await self.store.delete_presentation(presentation_id)

        # Rebuild index without deleted documents
        await self._rebuild_index()

        logger.info(f"Deleted presentation {presentation_id} and rebuilt index")

    async def list_presentations(self) -> List[Dict[str, Any]]:
        """
        List all indexed presentations.

        Returns:
            List of presentation metadata dicts
        """
        return await self.store.list_presentations()

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get search backend statistics.

        Returns:
            Dictionary with stats
        """
        store_stats = await self.store.get_stats()

        # Index file size
        index_size = 0
        if Path(self.index_path).exists():
            index_size = Path(self.index_path).stat().st_size

        return {
            "backend_type": "bm25_serialize",
            "total_documents": store_stats["total_chunks"],
            "total_presentations": store_stats["total_presentations"],
            "sqlite_size_mb": store_stats["db_size_mb"],
            "index_size_mb": round(index_size / (1024 * 1024), 2),
            "total_size_mb": round(
                store_stats["db_size_mb"] + (index_size / (1024 * 1024)),
                2
            ),
            "index_loaded": self._index_loaded,
            "index_version": self.INDEX_VERSION,
            "k": self.k
        }
