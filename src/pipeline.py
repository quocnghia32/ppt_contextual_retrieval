"""
End-to-end pipeline for PPT Contextual Retrieval.

Integrates all components: loading, chunking, embedding, indexing, retrieval, QA.
"""
from typing import List, Dict, Optional, Any
from pathlib import Path
import asyncio
import json
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from loguru import logger

from src.get_all_text import whole_document_from_pptx
from src.config import settings
from src.loaders.ppt_loader import PPTLoader
from src.splitters.contextual_splitter import ContextualTextSplitter
from src.models.vision_analyzer import VisionAnalyzer
from src.retrievers.hybrid_retriever import create_hybrid_retriever
from src.chains.qa_chain import create_qa_chain
from src.utils.caching import get_cached_embeddings, caching_manager


class PPTContextualRetrievalPipeline:
    """
    Complete pipeline for PPT contextual retrieval.

    Handles: Load → Chunk → Embed → Index → Retrieve → Answer
    """

    def __init__(
        self,
        index_name: Optional[str] = None,
        use_contextual: bool = True,
        use_vision: bool = True,
        use_reranking: bool = True
    ):
        """
        Initialize pipeline.

        Args:
            index_name: Pinecone index name (defaults to PINECONE_INDEX_NAME from .env)
            use_contextual: Use contextual chunking
            use_vision: Use vision analysis for images
            use_reranking: Use Cohere reranking
        """
        # Use index from .env if not provided
        self.index_name = index_name or settings.pinecone_index_name
        self.use_contextual = use_contextual
        self.use_vision = use_vision
        self.use_reranking = use_reranking

        # Initialize components with caching
        self.embeddings = get_cached_embeddings(model=settings.embedding_model)

        # Storage
        self.documents = []
        self.chunks = []
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None

        logger.info(
            f"Pipeline initialized: "
            f"contextual={use_contextual}, vision={use_vision}, "
            f"reranking={use_reranking}"
        )

    async def index_presentation(
        self,
        ppt_path: str,
        extract_images: bool = True,
        include_notes: bool = True
    ) -> Dict[str, Any]:
        """
        Index a PowerPoint presentation.

        Complete flow: Load → Vision Analysis → Contextual Chunking → Embedding → Index

        Args:
            ppt_path: Path to .pptx file
            extract_images: Extract image metadata
            include_notes: Include speaker notes
            analyze_images: Analyze images with vision model (default: use_vision setting)

        Returns:
            Indexing statistics
        """
        logger.info(f"Starting indexing: {ppt_path}")

        # Step 1: Load PPT
        logger.info("Step 1/5: Loading presentation...")
        loader = PPTLoader(
            ppt_path,
            extract_images=extract_images,
            include_speaker_notes=include_notes,
            use_vision=self.use_vision
        )
        self.documents, self.overall_info = await loader.load()

        logger.info(f"Loaded {len(self.documents)} slides")

        # Step 2: Get the whole_document phase
        logger.info("Step 2/5: Getting alll text...")
        all_doc_text = whole_document_from_pptx(ppt_path)

        # # Step 2: Vision Analysis (if enabled)
        # if self.use_vision and self.vision_analyzer:
        #     logger.info("Step 2/5: Analyzing images...")
        #     await self._analyze_slide_images(ppt_path)
        # else:
        #     logger.info("Step 2/5: Skipping image analysis")

        # Step 3: Contextual Chunking
        logger.info("Step 3/5: Creating contextual chunks...")
        splitter = ContextualTextSplitter(add_context=self.use_contextual)

        self.chunks = await splitter.asplit_documents(all_doc_text, self.documents)
        # Overall info chunk has not been split
        self.chunks = [self.overall_info] + self.chunks

        logger.info(f"Created {len(self.chunks)} chunks")
        

        # Step 4: Create/Connect to Pinecone Index        logger.info("Step 4/5: Setting up vector store...")
        await self._setup_pinecone_index()

        # Step 5: Index chunks
        logger.info("Step 5/5: Indexing chunks to Pinecone...")
        await self._index_chunks()

        # Create retriever and QA chain
        self._setup_retrieval()

        stats = {
            "presentation": Path(ppt_path).name,
            "slides": len(self.documents),
            "chunks": len(self.chunks),
            "indexed": True,
            "contextual": self.use_contextual,
            "vision_analyzed":  self.use_vision
        }

        logger.info(f"Indexing complete: {stats}")
        return stats

    

    async def _setup_pinecone_index(self):
        """Create or connect to Pinecone index."""
        try:
            # Initialize Pinecone
            pc = Pinecone(api_key=settings.pinecone_api_key)

            # Check if index exists
            existing_indexes = pc.list_indexes().names()

            if self.index_name not in existing_indexes:
                # Create new index
                logger.info(f"Creating Pinecone index: {self.index_name}")

                pc.create_index(
                    name=self.index_name,
                    dimension=1536,  # text-embedding-3-small dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=settings.pinecone_region or "us-east-1"
                    )
                )

                logger.info(f"Index created: {self.index_name}")
            else:
                logger.info(f"Using existing index: {self.index_name}")

            # Connect to vector store
            self.vector_store = PineconeVectorStore(
                index_name=self.index_name,
                embedding=self.embeddings,
                pinecone_api_key=settings.pinecone_api_key
            )

        except Exception as e:
            logger.error(f"Failed to setup Pinecone: {e}")
            raise

    async def _index_chunks(self):
        """Index chunks to Pinecone."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")

        try:
            # Add documents to vector store
            # LangChain will handle embedding

            self.vector_store.add_documents(self.chunks)

            logger.info(f"Indexed {len(self.chunks)} chunks to Pinecone")

        except Exception as e:
            logger.error(f"Failed to index chunks: {e}")
            raise

    def _setup_retrieval(self):
        """Setup retriever and QA chain."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")

        # Create hybrid retriever
        self.retriever = create_hybrid_retriever(
            index_name=self.index_name,
            documents=self.chunks,
            use_contextual=self.use_contextual,
            use_reranking=self.use_reranking
        )

        # Create QA chain
        self.qa_chain = create_qa_chain(
            retriever=self.retriever,
            streaming=False,
            enable_memory=True
        )

        logger.info("Retrieval pipeline ready")

    async def query(
        self,
        question: str,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Query the indexed presentation.

        Args:
            question: User question
            return_sources: Return source documents

        Returns:
            Answer with sources and metadata
        """
        if not self.qa_chain:
            raise ValueError("Pipeline not initialized. Index a presentation first.")

        logger.info(f"Query: {question}")

        # Query QA chain
        result = await self.qa_chain.aquery(question, return_sources)

        # Format sources
        if return_sources:
            result["formatted_sources"] = self._format_sources(
                result.get("source_documents", [])
            )

        return result

    def _format_sources(self, sources: List[Document]) -> List[Dict]:
        """Format source documents for display."""
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
                "bm25_rank": metadata.get("bm25_rank")
            })

        return formatted

    def clear_chat_history(self):
        """Clear conversation history."""
        if self.qa_chain:
            self.qa_chain.clear_memory()


# Convenience functions
async def index_ppt_file(
    ppt_path: str,
    index_name: Optional[str] = None,
    **kwargs
) -> PPTContextualRetrievalPipeline:
    """
    Quick function to index a PPT file.

    Args:
        ppt_path: Path to .pptx file
        index_name: Pinecone index name (auto-generated if None)
        **kwargs: Additional pipeline arguments

    Returns:
        Configured pipeline ready for querying
    """
    if index_name is None:
        # Generate index name from file
        filename = Path(ppt_path).stem
        index_name = f"ppt-{filename}".lower().replace(" ", "-")[:50]

    # Create pipeline
    pipeline = PPTContextualRetrievalPipeline(
        index_name=index_name,
        **kwargs
    )

    # Index presentation
    await pipeline.index_presentation(ppt_path)

    return pipeline


async def quick_query(
    ppt_path: str,
    question: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Quick function to index and query a PPT file in one step.

    Args:
        ppt_path: Path to .pptx file
        question: Question to ask
        **kwargs: Additional pipeline arguments

    Returns:
        Query result
    """
    pipeline = await index_ppt_file(ppt_path, **kwargs)
    result = await pipeline.query(question)
    return result
