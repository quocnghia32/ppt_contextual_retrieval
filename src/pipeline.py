"""
Ingestion pipeline for PPT Contextual Retrieval.

Handles: Load → Chunk → Embed → Index (BM25 + Pinecone)

Retrieval phase is handled separately by UI/application layer.
"""
from typing import Dict, Optional, Any
from pathlib import Path
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from loguru import logger

from src.get_all_text import whole_document_from_pptx
from src.config import settings
from src.loaders.ppt_loader import PPTLoader
from src.splitters.contextual_splitter import ContextualTextSplitter
from src.utils.caching_azure import get_cached_embeddings_azure
from src.storage.base_text_retriever import get_text_retriever



class PPTContextualRetrievalPipeline:
    """
    Ingestion pipeline for PPT contextual retrieval.

    Handles ONLY ingestion: Load → Chunk → Embed → Index (BM25 + Pinecone)

    Retrieval phase should be handled separately by application layer.
    """

    def __init__(
        self,
        index_name: Optional[str] = None,
        use_contextual: bool = True,
        use_vision: bool = True
    ):
        """
        Initialize ingestion pipeline.

        Args:
            index_name: Pinecone index name (defaults to PINECONE_INDEX_NAME from .env)
            use_contextual: Use contextual chunking
            use_vision: Use vision analysis for images
        """
        # Use index from .env if not provided
        self.index_name = index_name or settings.pinecone_index_name
        self.use_contextual = use_contextual
        self.use_vision = use_vision

        # Initialize components with caching
        self.embeddings = get_cached_embeddings_azure(model=settings.embedding_model)

        # Initialize text retriever (BM25 abstraction layer)
        self.text_retriever = get_text_retriever(
            backend=settings.search_backend,
            db_path=settings.bm25_db_path,
            index_path=settings.bm25_index_path,
            k=settings.top_k_retrieval
        )

        # Storage (for ingestion phase only)
        self.documents = []
        self.chunks = []
        self.vector_store = None

        logger.info(
            f"Ingestion pipeline initialized: "
            f"backend={settings.search_backend}, contextual={use_contextual}, "
            f"vision={use_vision}"
        )

    async def index_presentation(
        self,
        ppt_path: str,
        extract_images: bool = True,
        include_notes: bool = True
    ) -> Dict[str, Any]:
        """
        Index a PowerPoint presentation (ingestion phase only).

        Complete flow:
        1. Load PPT → Extract whole document
        2. Contextual chunking (with vision analysis if enabled)
        3. Index to BM25Store (SQLite + serialized index)
        4. Index to Pinecone (vector embeddings)

        Args:
            ppt_path: Path to .pptx file
            extract_images: Extract image metadata
            include_notes: Include speaker notes

        Returns:
            Indexing statistics (slides, chunks, indexed status)
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

        # Step 3: Contextual Chunking
        logger.info("Step 3/5: Creating contextual chunks...")
        splitter = ContextualTextSplitter(add_context=self.use_contextual)

        self.chunks = await splitter.asplit_documents(all_doc_text, self.documents)
        # Overall info chunk has not been split
        self.chunks = [self.overall_info] + self.chunks

        logger.info(f"Created {len(self.chunks)} chunks")
        

        # Step 4: Initialize text retriever
        logger.info("Step 4/7: Initializing text retriever...")
        await self.text_retriever.initialize()

        # Step 5: Index chunks to text retriever (BM25 store)
        logger.info("Step 5/7: Indexing chunks to BM25 store...")
        presentation_id = Path(ppt_path).stem
        await self.text_retriever.index_documents(
            documents=self.chunks,
            presentation_id=presentation_id,
            metadata={
                "name": Path(ppt_path).name,
                "title": self.overall_info.metadata.get("title", ""),
                "total_slides": len(self.documents),
                "pinecone_index_name": self.index_name
            }
        )

        # Step 6: Create/Connect to Pinecone Index
        logger.info("Step 6/7: Setting up vector store...")
        await self._setup_pinecone_index()

        # Step 7: Index chunks to Pinecone
        logger.info("Step 7/7: Indexing chunks to Pinecone...")
        await self._index_chunks()

        stats = {
            "presentation_id": Path(ppt_path).stem,
            "presentation": Path(ppt_path).name,
            "slides": len(self.documents),
            "chunks": len(self.chunks),
            "indexed": True,
            "contextual": self.use_contextual,
            "vision_analyzed": self.use_vision,
            "pinecone_index": self.index_name,
            "bm25_backend": settings.search_backend
        }

        logger.info(f"✅ Ingestion complete: {stats}")
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
                    dimension=(1536 if settings.embedding_model == "text-embedding-3-small" else 3072),  # text-embedding-3-small or large dimension
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


# Convenience function for ingestion
async def index_ppt_file(
    ppt_path: str,
    index_name: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Quick function to index a PPT file (ingestion only).

    Args:
        ppt_path: Path to .pptx file
        index_name: Pinecone index name (auto-generated if None)
        **kwargs: Additional pipeline arguments (use_contextual, use_vision)

    Returns:
        Indexing statistics

    Example:
        stats = await index_ppt_file("presentation.pptx")
        print(f"Indexed {stats['chunks']} chunks")
    """
    if index_name is None:
        # Generate index name from file
        filename = Path(ppt_path).stem
        index_name = f"ppt-{filename}".lower().replace(" ", "-")[:50]

    # Create ingestion pipeline
    pipeline = PPTContextualRetrievalPipeline(
        index_name=index_name,
        **kwargs
    )

    # Index presentation
    stats = await pipeline.index_presentation(ppt_path)

    return stats
