"""
Comprehensive UI utilities for Streamlit frontend.

Provides methods for:
- List presentations (from BM25Store)
- Upload and index presentations
- Query presentations (via RetrievalPipeline)
- Delete presentations (complete cleanup)
- Statistics and monitoring
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import shutil

from loguru import logger
from pinecone import Pinecone

from src.config import settings
from src.pipeline import PPTContextualRetrievalPipeline
from src.retrieval_pipeline import RetrievalPipeline
from src.storage.base_text_retriever import get_text_retriever


class PresentationManager:
    """
    Comprehensive presentation management for UI.

    Handles:
    - Listing presentations from persistent storage
    - Uploading and indexing new presentations
    - Querying indexed presentations
    - Deleting presentations completely
    """

    def __init__(self):
        """Initialize manager with text retriever."""
        self.text_retriever = get_text_retriever(
            backend=settings.search_backend,
            db_path=settings.bm25_db_path,
            index_path=settings.bm25_index_path
        )
        self._initialized = False
        self._retrieval_pipeline: Optional[RetrievalPipeline] = None

    async def initialize(self):
        """Initialize text retriever (load BM25 index)."""
        if not self._initialized:
            await self.text_retriever.initialize()
            self._initialized = True
            logger.info("PresentationManager initialized")

    async def list_presentations(self) -> List[Dict[str, Any]]:
        """
        List all indexed presentations from BM25Store.

        Returns:
            List of presentation dicts with:
            - presentation_id: str
            - name: str (filename)
            - title: str
            - total_slides: int
            - total_chunks: int
            - pinecone_index_name: str
            - indexed_at: str (ISO timestamp)

        Example:
            presentations = await manager.list_presentations()
            for pres in presentations:
                print(f"{pres['title']}: {pres['total_chunks']} chunks")
        """
        await self.initialize()
        presentations = await self.text_retriever.list_presentations()
        logger.info(f"Found {len(presentations)} indexed presentations")
        return presentations

    async def get_presentation_info(self, presentation_id: str) -> Dict[str, Any]:
        """
        Get detailed info for a specific presentation.

        Args:
            presentation_id: Presentation ID

        Returns:
            Presentation metadata dict

        Raises:
            ValueError: If presentation not found
        """
        await self.initialize()
        presentations = await self.list_presentations()

        for pres in presentations:
            if pres['presentation_id'] == presentation_id:
                return pres

        raise ValueError(f"Presentation not found: {presentation_id}")

    async def upload_and_index(
        self,
        file_path: str,
        use_contextual: bool = True,
        use_vision: bool = True,
        extract_images: bool = True,
        include_notes: bool = True,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Upload and index a presentation.

        Args:
            file_path: Path to .pptx file
            use_contextual: Enable contextual chunking
            use_vision: Enable vision analysis
            extract_images: Extract images from slides
            include_notes: Include speaker notes
            progress_callback: Optional callback(progress, message)

        Returns:
            Indexing statistics:
            {
                'presentation_id': str,
                'presentation': str (title),
                'slides': int,
                'chunks': int,
                'indexed': bool,
                'contextual': bool,
                'vision_analyzed': bool,
                'pinecone_index_name': str,
                'indexed_at': str (ISO timestamp)
            }

        Example:
            def progress(pct, msg):
                print(f"{pct}%: {msg}")

            stats = await manager.upload_and_index(
                "presentation.pptx",
                progress_callback=progress
            )
        """
        await self.initialize()

        if progress_callback:
            progress_callback(10, "Initializing ingestion pipeline...")

        # Create ingestion pipeline
        pipeline = PPTContextualRetrievalPipeline(
            use_contextual=use_contextual,
            use_vision=use_vision
        )

        if progress_callback:
            progress_callback(20, "Loading and analyzing presentation...")

        logger.info(f"Indexing presentation: {file_path}")

        # Index presentation
        stats = await pipeline.index_presentation(
            ppt_path=file_path,
            extract_images=extract_images,
            include_notes=include_notes
        )

        if progress_callback:
            progress_callback(100, "Indexing complete!")

        # Add timestamp
        stats['indexed_at'] = datetime.utcnow().isoformat()

        logger.info(f"✅ Indexed {stats['presentation_id']}: "
                   f"{stats['slides']} slides → {stats['chunks']} chunks")

        return stats

    async def query_presentation(
        self,
        query: str,
        presentation_id: Optional[str] = None,
        top_k: int = 20,
        use_reranking: bool = True,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Query a presentation or all presentations.

        Args:
            query: Question to ask
            presentation_id: Optional, query specific presentation
                           If None, queries ALL presentations
            top_k: Number of chunks to retrieve
            use_reranking: Enable Cohere reranking
            return_sources: Include source documents in response

        Returns:
            {
                'answer': str,
                'sources': List[Document],
                'formatted_sources': List[Dict],
                'metadata': Dict
            }

        Example:
            # Query specific presentation
            result = await manager.query_presentation(
                "What is the revenue?",
                presentation_id="ppt-report-2024"
            )

            # Query all presentations (cross-document)
            result = await manager.query_presentation(
                "Compare revenue across all reports"
            )
        """
        await self.initialize()

        # Create or reuse retrieval pipeline
        if self._retrieval_pipeline is None:
            logger.info("Creating retrieval pipeline...")
            self._retrieval_pipeline = RetrievalPipeline(
                use_reranking=use_reranking
            )
            await self._retrieval_pipeline.initialize()

        # Query
        logger.info(f"Querying: {query[:50]}...")
        result = await self._retrieval_pipeline.query(
            question=query,
            return_sources=return_sources
        )

        logger.info(f"✅ Query answered ({len(result.get('sources', []))} sources)")

        return result

    async def delete_presentation(
        self,
        presentation_id: str,
        delete_pinecone: bool = True,
        delete_images: bool = True
    ) -> Dict[str, Any]:
        """
        Delete a presentation completely.

        Args:
            presentation_id: Presentation to delete
            delete_pinecone: Also delete from Pinecone
            delete_images: Also delete extracted images

        Returns:
            {
                'deleted_from_bm25': bool,
                'deleted_from_pinecone': bool,
                'deleted_images': bool,
                'presentation_id': str
            }

        Example:
            result = await manager.delete_presentation(
                "ppt-old-report",
                delete_pinecone=True,
                delete_images=True
            )
        """
        await self.initialize()

        logger.info(f"Deleting presentation: {presentation_id}")

        result = {
            'presentation_id': presentation_id,
            'deleted_from_bm25': False,
            'deleted_from_pinecone': False,
            'deleted_images': False,
            'errors': []
        }

        # 1. Delete from BM25Store (also rebuilds index)
        try:
            await self.text_retriever.delete_presentation(presentation_id)
            result['deleted_from_bm25'] = True
            logger.info(f"✅ Deleted {presentation_id} from BM25")
        except Exception as e:
            logger.error(f"Failed to delete from BM25: {e}")
            result['errors'].append(f"BM25: {str(e)}")

        # 2. Delete from Pinecone
        if delete_pinecone:
            try:
                pc = Pinecone(api_key=settings.pinecone_api_key)
                index = pc.Index(settings.pinecone_index_name)
                index.delete(filter={"presentation_id": presentation_id})
                result['deleted_from_pinecone'] = True
                logger.info(f"✅ Deleted {presentation_id} from Pinecone")
            except Exception as e:
                logger.error(f"Failed to delete from Pinecone: {e}")
                result['errors'].append(f"Pinecone: {str(e)}")

        # 3. Delete extracted images
        if delete_images:
            try:
                image_dir = Path(settings.extracted_images_dir) / presentation_id
                if image_dir.exists():
                    shutil.rmtree(image_dir)
                    result['deleted_images'] = True
                    logger.info(f"✅ Deleted images: {image_dir}")
            except Exception as e:
                logger.error(f"Failed to delete images: {e}")
                result['errors'].append(f"Images: {str(e)}")

        logger.info(f"🎉 Completed deletion of {presentation_id}")

        return result

    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.

        Returns:
            {
                'backend_type': str,
                'total_documents': int,
                'total_presentations': int,
                'sqlite_size_mb': float,
                'index_size_mb': float,
                'total_size_mb': float,
                'index_loaded': bool,
                'pinecone_index': str,
                'presentations': List[Dict]
            }

        Example:
            stats = await manager.get_statistics()
            print(f"Total: {stats['total_presentations']} presentations")
            print(f"Storage: {stats['total_size_mb']} MB")
        """
        await self.initialize()

        # Get BM25 stats
        bm25_stats = await self.text_retriever.get_stats()

        # Get presentations list
        presentations = await self.list_presentations()

        # Combine stats
        stats = {
            **bm25_stats,
            'pinecone_index': settings.pinecone_index_name,
            'presentations': presentations,
            'total_slides': sum(p['total_slides'] for p in presentations),
        }

        logger.info(f"Statistics: {stats['total_presentations']} presentations, "
                   f"{stats['total_documents']} documents")

        return stats

    async def clear_chat_history(self):
        """Clear chat history in retrieval pipeline."""
        if self._retrieval_pipeline:
            self._retrieval_pipeline.clear_chat_history()
            logger.info("Chat history cleared")

    async def health_check(self) -> bool:
        """
        Check if system is healthy and ready.

        Returns:
            True if healthy, False otherwise
        """
        try:
            await self.initialize()
            healthy = await self.text_retriever.health_check()
            logger.info(f"Health check: {'✅ Healthy' if healthy else '❌ Unhealthy'}")
            return healthy
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


# Singleton instance for UI
_presentation_manager: Optional[PresentationManager] = None


def get_presentation_manager() -> PresentationManager:
    """
    Get singleton PresentationManager instance.

    Usage in Streamlit:
        manager = get_presentation_manager()
        presentations = asyncio.run(manager.list_presentations())
    """
    global _presentation_manager
    if _presentation_manager is None:
        _presentation_manager = PresentationManager()
    return _presentation_manager


# Convenience functions for Streamlit
def list_presentations_sync() -> List[Dict[str, Any]]:
    """
    Synchronous wrapper for list_presentations().

    Usage in Streamlit:
        presentations = list_presentations_sync()
        for pres in presentations:
            st.write(pres['title'])
    """
    manager = get_presentation_manager()
    return asyncio.run(manager.list_presentations())


def upload_and_index_sync(
    file_path: str,
    use_contextual: bool = True,
    use_vision: bool = True,
    extract_images: bool = True,
    include_notes: bool = True,
    progress_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Synchronous wrapper for upload_and_index().

    Usage in Streamlit:
        def update_progress(pct, msg):
            progress_bar.progress(pct, text=msg)

        stats = upload_and_index_sync(
            "file.pptx",
            progress_callback=update_progress
        )
    """
    manager = get_presentation_manager()
    return asyncio.run(manager.upload_and_index(
        file_path,
        use_contextual=use_contextual,
        use_vision=use_vision,
        extract_images=extract_images,
        include_notes=include_notes,
        progress_callback=progress_callback
    ))


def query_presentation_sync(
    query: str,
    presentation_id: Optional[str] = None,
    return_sources: bool = True
) -> Dict[str, Any]:
    """
    Synchronous wrapper for query_presentation().

    Usage in Streamlit:
        result = query_presentation_sync(
            "What is the revenue?",
            presentation_id="ppt-report-2024"
        )
        st.write(result['answer'])
    """
    manager = get_presentation_manager()
    return asyncio.run(manager.query_presentation(
        query,
        presentation_id=presentation_id,
        return_sources=return_sources
    ))


def delete_presentation_sync(
    presentation_id: str,
    delete_pinecone: bool = True,
    delete_images: bool = True
) -> Dict[str, Any]:
    """
    Synchronous wrapper for delete_presentation().

    Usage in Streamlit:
        result = delete_presentation_sync("ppt-old-report")
        if result['deleted_from_bm25']:
            st.success("Deleted successfully!")
    """
    manager = get_presentation_manager()
    return asyncio.run(manager.delete_presentation(
        presentation_id,
        delete_pinecone=delete_pinecone,
        delete_images=delete_images
    ))


def get_statistics_sync() -> Dict[str, Any]:
    """
    Synchronous wrapper for get_statistics().

    Usage in Streamlit:
        stats = get_statistics_sync()
        st.metric("Total Presentations", stats['total_presentations'])
        st.metric("Total Documents", stats['total_documents'])
    """
    manager = get_presentation_manager()
    return asyncio.run(manager.get_statistics())
