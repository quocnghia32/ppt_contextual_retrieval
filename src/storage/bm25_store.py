"""
BM25Store - SQLite storage for text chunks.

Stores text content for BM25 keyword search. This is separate from Pinecone
(which stores embeddings) and focuses solely on text for BM25 retrieval.

Storage Strategy:
- Store only text content + minimal metadata
- Rebuild BM25 index in memory on load (~50ms for 200 chunks)
- No serialization of BM25 index (see bm25_serialize_retriever.py for that)

Schema:
    presentations: Presentation-level metadata
    chunks: Individual text chunks for BM25
"""

import sqlite3
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)


class BM25Store:
    """
    SQLite storage for BM25 text chunks.

    Usage:
        store = BM25Store(db_path="data/bm25/bm25_store.db")
        await store.initialize()

        # Save documents during ingestion
        await store.save_presentation(
            presentation_id="ppt-report-2024",
            documents=chunks,
            metadata={"title": "Q4 Report", "total_slides": 100}
        )

        # Load documents during retrieval
        chunks = await store.load_chunks(presentation_id="ppt-report-2024")
    """

    def __init__(self, db_path: str = "data/bm25/bm25_store.db"):
        """
        Initialize BM25Store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_dir()

    def _ensure_dir(self):
        """Ensure database directory exists."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> None:
        """
        Initialize database schema.

        Creates tables if they don't exist.
        """
        # Run in thread pool to avoid blocking
        await asyncio.to_thread(self._create_tables)
        logger.info(f"BM25Store initialized at {self.db_path}")

    def _create_tables(self):
        """Create database tables (synchronous)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Presentations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS presentations (
                presentation_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                title TEXT,
                total_slides INTEGER,
                total_chunks INTEGER,
                pinecone_index_name TEXT,
                indexed_at TEXT NOT NULL
            )
        """)

        # Chunks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                presentation_id TEXT NOT NULL,
                content TEXT NOT NULL,
                slide_number INTEGER,
                FOREIGN KEY (presentation_id) REFERENCES presentations(presentation_id)
                    ON DELETE CASCADE
            )
        """)

        # Index for fast lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_presentation
            ON chunks(presentation_id)
        """)

        conn.commit()
        conn.close()

    async def save_presentation(
        self,
        presentation_id: str,
        documents: List[Document],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save presentation and its chunks.

        Args:
            presentation_id: Unique presentation identifier
            documents: List of Document objects
            metadata: Optional presentation metadata (title, slides, etc.)
        """
        metadata = metadata or {}

        # Run in thread pool
        await asyncio.to_thread(
            self._save_presentation_sync,
            presentation_id,
            documents,
            metadata
        )

        logger.info(
            f"Saved {len(documents)} chunks for presentation {presentation_id}"
        )

    def _save_presentation_sync(
        self,
        presentation_id: str,
        documents: List[Document],
        metadata: Dict[str, Any]
    ):
        """Save presentation (synchronous)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Save presentation metadata
            cursor.execute("""
                INSERT OR REPLACE INTO presentations
                (presentation_id, name, title, total_slides, total_chunks,
                 pinecone_index_name, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                presentation_id,
                metadata.get("name", presentation_id),
                metadata.get("title", ""),
                metadata.get("total_slides", 0),
                len(documents),
                metadata.get("pinecone_index_name", ""),
                datetime.utcnow().isoformat()
            ))

            # Save chunks
            for doc in documents:
                chunk_id = doc.metadata.get("chunk_id", "")
                slide_number = doc.metadata.get("slide_number", 0)

                cursor.execute("""
                    INSERT OR REPLACE INTO chunks
                    (chunk_id, presentation_id, content, slide_number)
                    VALUES (?, ?, ?, ?)
                """, (
                    chunk_id,
                    presentation_id,
                    doc.page_content,
                    slide_number
                ))

            conn.commit()

        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    async def load_chunks(self, presentation_id: str) -> List[Document]:
        """
        Load chunks for a specific presentation.

        Args:
            presentation_id: Presentation to load

        Returns:
            List of Document objects with page_content and metadata
        """
        # Run in thread pool
        documents = await asyncio.to_thread(
            self._load_chunks_sync,
            presentation_id
        )

        logger.info(f"Loaded {len(documents)} chunks for {presentation_id}")
        return documents

    def _load_chunks_sync(self, presentation_id: str) -> List[Document]:
        """Load chunks (synchronous)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT chunk_id, content, slide_number
            FROM chunks
            WHERE presentation_id = ?
            ORDER BY slide_number, chunk_id
        """, (presentation_id,))

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return []

        # Reconstruct Document objects
        documents = []
        for chunk_id, content, slide_number in rows:
            metadata = {
                "chunk_id": chunk_id,
                "presentation_id": presentation_id,
                "slide_number": slide_number,
            }

            doc = Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(doc)

        return documents

    async def load_all_chunks(self) -> List[Document]:
        """
        Load ALL chunks across all presentations.

        Used for cross-document search.

        Returns:
            List of all Document objects
        """
        documents = await asyncio.to_thread(self._load_all_chunks_sync)
        logger.info(f"Loaded {len(documents)} total chunks across all presentations")
        return documents

    def _load_all_chunks_sync(self) -> List[Document]:
        """Load all chunks (synchronous)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT chunk_id, presentation_id, content, slide_number
            FROM chunks
            ORDER BY presentation_id, slide_number, chunk_id
        """)

        rows = cursor.fetchall()
        conn.close()

        documents = []
        for chunk_id, presentation_id, content, slide_number in rows:
            metadata = {
                "chunk_id": chunk_id,
                "presentation_id": presentation_id,
                "slide_number": slide_number,
            }

            doc = Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(doc)

        return documents

    async def get_presentation_info(self, presentation_id: str) -> Dict[str, Any]:
        """
        Get presentation metadata.

        Args:
            presentation_id: Presentation ID

        Returns:
            Dictionary with presentation metadata
        """
        info = await asyncio.to_thread(
            self._get_presentation_info_sync,
            presentation_id
        )
        return info

    def _get_presentation_info_sync(self, presentation_id: str) -> Dict[str, Any]:
        """Get presentation info (synchronous)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT name, title, total_slides, total_chunks,
                   pinecone_index_name, indexed_at
            FROM presentations
            WHERE presentation_id = ?
        """, (presentation_id,))

        row = cursor.fetchone()
        conn.close()

        if not row:
            raise ValueError(f"Presentation not found: {presentation_id}")

        return {
            "presentation_id": presentation_id,
            "name": row[0],
            "title": row[1],
            "total_slides": row[2],
            "total_chunks": row[3],
            "pinecone_index_name": row[4],
            "indexed_at": row[5]
        }

    async def list_presentations(self) -> List[Dict[str, Any]]:
        """
        List all presentations.

        Returns:
            List of presentation metadata dicts
        """
        presentations = await asyncio.to_thread(self._list_presentations_sync)
        return presentations

    def _list_presentations_sync(self) -> List[Dict[str, Any]]:
        """List presentations (synchronous)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT presentation_id, name, title, total_slides,
                   total_chunks, pinecone_index_name, indexed_at
            FROM presentations
            ORDER BY indexed_at DESC
        """)

        rows = cursor.fetchall()
        conn.close()

        presentations = []
        for row in rows:
            presentations.append({
                "presentation_id": row[0],
                "name": row[1],
                "title": row[2],
                "total_slides": row[3],
                "total_chunks": row[4],
                "pinecone_index_name": row[5],
                "indexed_at": row[6]
            })

        return presentations

    async def delete_presentation(self, presentation_id: str) -> None:
        """
        Delete presentation and all its chunks.

        Args:
            presentation_id: Presentation to delete
        """
        await asyncio.to_thread(
            self._delete_presentation_sync,
            presentation_id
        )
        logger.info(f"Deleted presentation {presentation_id}")

    def _delete_presentation_sync(self, presentation_id: str):
        """Delete presentation (synchronous)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Delete chunks (CASCADE will handle this, but explicit is better)
            cursor.execute(
                "DELETE FROM chunks WHERE presentation_id = ?",
                (presentation_id,)
            )

            # Delete presentation
            cursor.execute(
                "DELETE FROM presentations WHERE presentation_id = ?",
                (presentation_id,)
            )

            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Dictionary with stats
        """
        stats = await asyncio.to_thread(self._get_stats_sync)
        return stats

    def _get_stats_sync(self) -> Dict[str, Any]:
        """Get stats (synchronous)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total presentations
        cursor.execute("SELECT COUNT(*) FROM presentations")
        total_presentations = cursor.fetchone()[0]

        # Total chunks
        cursor.execute("SELECT COUNT(*) FROM chunks")
        total_chunks = cursor.fetchone()[0]

        # Database size
        import os
        db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0

        conn.close()

        return {
            "total_presentations": total_presentations,
            "total_chunks": total_chunks,
            "db_size_bytes": db_size,
            "db_size_mb": round(db_size / (1024 * 1024), 2),
            "db_path": self.db_path
        }
