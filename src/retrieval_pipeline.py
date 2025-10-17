"""
Retrieval pipeline for PPT Contextual Retrieval.

Handles: Load Index → Setup Retriever → Query → Answer

Separate from ingestion pipeline for clear separation of concerns.
"""
from typing import Dict, List, Optional, Any
from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore
from loguru import logger

from src.config import settings
from src.retrievers.hybrid_retriever import create_hybrid_retriever
from src.chains.qa_chain import create_qa_chain
from src.utils.caching_azure import get_cached_embeddings_azure
from src.storage.base_text_retriever import get_text_retriever


class RetrievalPipeline:
    """
    Retrieval pipeline for querying indexed presentations.

    Handles:
    1. Load BM25 index from storage
    2. Connect to Pinecone vector store
    3. Create hybrid retriever (Vector + BM25 + RRF)
    4. Create QA chain
    5. Query with chat history

    Usage:
        # Initialize for a specific presentation
        pipeline = RetrievalPipeline()
        await pipeline.initialize()

        # Query
        result = await pipeline.query("What is the revenue?")
        print(result["answer"])

        # Clear history
        pipeline.clear_chat_history()
    """

    def __init__(
        self,
        index_name: Optional[str] = None,
        use_reranking: bool = True
    ):
        """
        Initialize retrieval pipeline.

        Args:
            presentation_id: Specific presentation to query (optional, for single-doc mode)
            index_name: Pinecone index name (defaults to PINECONE_INDEX_NAME from .env)
            use_reranking: Enable Cohere reranking
        """
        self.index_name = index_name or settings.pinecone_index_name
        self.use_reranking = use_reranking

        # Components (initialized in initialize())
        self.embeddings = None
        self.text_retriever = None
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None

        logger.info(
            f"Retrieval pipeline created: "
            f"index={self.index_name}, "
            f"reranking={use_reranking}"
        )

    async def initialize(self):
        """
        Initialize retrieval pipeline.

        This method:
        1. Initializes embeddings (with cache)
        2. Loads BM25 index from storage
        3. Connects to Pinecone vector store
        4. Creates hybrid retriever
        5. Creates QA chain

        Call this once before querying.
        """
        logger.info("Initializing retrieval pipeline...")

        # Step 1: Initialize embeddings
        logger.info("Step 1/5: Initializing embeddings...")
        self.embeddings = get_cached_embeddings_azure(model=settings.embedding_model)

        # Step 2: Initialize text retriever (load BM25 index)
        logger.info("Step 2/5: Loading BM25 index from storage...")
        self.text_retriever = get_text_retriever(
            backend=settings.search_backend,
            db_path=settings.bm25_db_path,
            index_path=settings.bm25_index_path,
            k=settings.top_k_retrieval
        )
        await self.text_retriever.initialize()

        # # Optional: Load specific presentation (for single-doc mode)
        # if self.presentation_id:
        #     logger.info(f"Loading presentation: {self.presentation_id}")
        #     await self.text_retriever.load_presentation(self.presentation_id)

        # Step 3: Connect to Pinecone vector store
        logger.info("Step 3/5: Connecting to Pinecone...")
        self.vector_store = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings,
            pinecone_api_key=settings.pinecone_api_key
        )

        # Step 4: Create hybrid retriever
        logger.info("Step 4/5: Creating hybrid retriever...")
        self.retriever = create_hybrid_retriever(
            index_name=self.index_name,
            bm25_retriever=self.text_retriever.bm25_retriever,
            use_contextual=True,
            use_reranking=self.use_reranking
        )

        # Step 5: Create QA chain
        logger.info("Step 5/5: Creating QA chain...")
        self.qa_chain = create_qa_chain(
            retriever=self.retriever,
            streaming=False,
            enable_memory=True
        )

        logger.info("✅ Retrieval pipeline ready for queries")

    async def query(
        self,
        question: str,
        return_sources: bool = True,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the indexed presentations.

        Args:
            question: User question
            return_sources: Return source documents
            filters: Optional filters (e.g., {"presentation_id": "ppt-123"})

        Returns:
            Answer with sources and metadata

        Example:
            result = await pipeline.query(
                "What is the revenue?",
                filters={"presentation_id": "ppt-report-2024"}
            )
            print(result["answer"])
            for source in result["formatted_sources"]:
                print(f"Slide {source['slide_number']}: {source['content']}")
        """
        if not self.qa_chain:
            raise ValueError(
                "Pipeline not initialized. Call initialize() first."
            )

        logger.info(f"Query: {question}")

        # Query QA chain
        result = await self.qa_chain.aquery(question, return_sources)

        # Format sources for display
        if return_sources:
            result["formatted_sources"] = self._format_sources(
                result.get("source_documents", [])
            )

        return result

    def _format_sources(self, sources: List[Document]) -> List[Dict]:
        """
        Format source documents for display.

        Args:
            sources: List of source Document objects

        Returns:
            List of formatted source dicts
        """
        formatted = []

        for doc in sources:
            metadata = doc.metadata
            formatted.append({
                "slide_number": metadata.get("slide_number"),
                "slide_title": metadata.get("slide_title", ""),
                "section": metadata.get("section", ""),
                "content": metadata.get("original_text", doc.page_content)[:300],
                "rrf_score": metadata.get("rrf_score"),
                "vector_rank": metadata.get("vector_rank"),
                "bm25_rank": metadata.get("bm25_rank"),
                "presentation_id": metadata.get("presentation_id", "")
            })

        return formatted

    def clear_chat_history(self):
        """Clear conversation history."""
        if self.qa_chain:
            self.qa_chain.clear_memory()
            logger.info("Chat history cleared")
        else:
            logger.warning("QA chain not initialized, nothing to clear")

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get retrieval pipeline statistics.

        Returns:
            Statistics about loaded index, presentations, etc.
        """
        if not self.text_retriever:
            return {"status": "not_initialized"}

        search_stats = await self.text_retriever.get_stats()

        return {
            "status": "ready" if self.qa_chain else "initializing",
            "backend": settings.search_backend,
            "pinecone_index": self.index_name,
            "presentation_id": self.presentation_id,
            "use_reranking": self.use_reranking,
            **search_stats
        }

    async def list_presentations(self) -> List[Dict[str, Any]]:
        """
        List all available presentations.

        Returns:
            List of presentation metadata dicts

        Example:
            presentations = await pipeline.list_presentations()
            for pres in presentations:
                print(f"{pres['presentation_id']}: {pres['title']}")
        """
        if not self.text_retriever:
            raise ValueError("Pipeline not initialized. Call initialize() first.")

        return await self.text_retriever.list_presentations()


# Convenience function for quick queries
async def quick_query(
    question: str,
    presentation_id: Optional[str] = None,
    index_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Quick function to query without manual setup.

    Args:
        question: Question to ask
        presentation_id: Optional specific presentation
        index_name: Optional Pinecone index name

    Returns:
        Query result with answer and sources

    Example:
        result = await quick_query(
            "What is the revenue?",
            presentation_id="ppt-report-2024"
        )
        print(result["answer"])
    """
    pipeline = RetrievalPipeline(
        index_name=index_name
    )
    await pipeline.initialize()
    result = await pipeline.query(question)

    return result
