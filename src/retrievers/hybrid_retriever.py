"""
Hybrid Retriever combining Vector Search + BM25 + Reciprocal Rank Fusion.

Implements the retrieval component of Contextual Retrieval approach.
"""
from typing import List, Dict, Optional, Any
from langchain.schema import Document, BaseRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from pydantic import Field
from loguru import logger

from src.config import settings
from src.utils.rate_limiter import rate_limiter


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining vector similarity and BM25 lexical search.

    Uses Reciprocal Rank Fusion (RRF) to merge results.
    """

    # Declare fields for Pydantic
    vector_store: Any = Field(default=None)
    vector_weight: float = Field(default=0.7)
    bm25_weight: float = Field(default=0.3)
    top_k: int = Field(default=None)
    bm25_retriever: Any = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        vector_store: PineconeVectorStore,
        documents: List[Document],
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        top_k: int = None,
        **kwargs
    ):
        """
        Initialize hybrid retriever.

        Args:
            vector_store: Pinecone vector store
            documents: All documents for BM25 index
            vector_weight: Weight for vector search (default: 0.7)
            bm25_weight: Weight for BM25 search (default: 0.3)
            top_k: Number of results to return
        """
        super().__init__(
            vector_store=vector_store,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            top_k=top_k or settings.top_k_retrieval,
            **kwargs
        )

        # Initialize BM25 retriever
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = self.top_k

        logger.info(
            f"HybridRetriever initialized: "
            f"vector_weight={vector_weight}, bm25_weight={bm25_weight}, "
            f"top_k={self.top_k}"
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Retrieve relevant documents using hybrid search.

        Combines vector search and BM25 using Reciprocal Rank Fusion.
        """
        # Get vector search results
        vector_docs = self.vector_store.similarity_search(
            query,
            k=self.top_k
        )

        # Get BM25 results
        bm25_docs = self.bm25_retriever.get_relevant_documents(query)

        # Merge using Reciprocal Rank Fusion
        merged_docs = self._reciprocal_rank_fusion(
            vector_docs,
            bm25_docs,
            query
        )

        logger.debug(
            f"Retrieved {len(merged_docs)} documents: "
            f"{len(vector_docs)} from vector, {len(bm25_docs)} from BM25"
        )

        return merged_docs[:self.top_k]

    def _reciprocal_rank_fusion(
        self,
        vector_docs: List[Document],
        bm25_docs: List[Document],
        query: str,
        k: int = 60
    ) -> List[Document]:
        """
        Merge results using Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank_i)) across all retrievers

        Args:
            vector_docs: Results from vector search
            bm25_docs: Results from BM25 search
            query: Original query
            k: RRF constant (default: 60)

        Returns:
            Merged and re-ranked documents
        """
        # Build score dictionary
        doc_scores = {}

        # Add vector search scores
        for rank, doc in enumerate(vector_docs, start=1):
            doc_id = self._get_doc_id(doc)
            score = self.vector_weight * (1.0 / (k + rank))
            doc_scores[doc_id] = {
                "doc": doc,
                "score": score,
                "vector_rank": rank
            }

        # Add BM25 scores
        for rank, doc in enumerate(bm25_docs, start=1):
            doc_id = self._get_doc_id(doc)
            bm25_score = self.bm25_weight * (1.0 / (k + rank))

            if doc_id in doc_scores:
                doc_scores[doc_id]["score"] += bm25_score
                doc_scores[doc_id]["bm25_rank"] = rank
            else:
                doc_scores[doc_id] = {
                    "doc": doc,
                    "score": bm25_score,
                    "bm25_rank": rank
                }

        # Sort by score
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        # Add RRF metadata to documents
        results = []
        for item in sorted_docs:
            doc = item["doc"]
            doc.metadata["rrf_score"] = item["score"]
            doc.metadata["vector_rank"] = item.get("vector_rank", None)
            doc.metadata["bm25_rank"] = item.get("bm25_rank", None)
            results.append(doc)

        return results

    def _get_doc_id(self, doc: Document) -> str:
        """
        Get unique ID for document.

        Uses chunk_id from metadata if available, otherwise creates one.
        """
        if "chunk_id" in doc.metadata:
            return doc.metadata["chunk_id"]

        # Fallback: create ID from content hash
        import hashlib
        content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:12]
        return f"doc_{content_hash}"

    async def aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Async version of retrieval (not fully implemented).

        For now, just calls sync version.
        """
        return self._get_relevant_documents(query, run_manager=run_manager)


class ContextualHybridRetriever(HybridRetriever):
    """
    Enhanced hybrid retriever with contextual embeddings.

    Assumes documents have been embedded with contextual chunks.
    """

    # Declare additional fields for Pydantic
    use_reranking: bool = Field(default=True)
    reranker_model: Optional[str] = Field(default=None)
    reranker: Any = Field(default=None)

    def __init__(
        self,
        vector_store: PineconeVectorStore,
        documents: List[Document],
        use_reranking: bool = True,
        reranker_model: str = None,
        **kwargs
    ):
        """
        Initialize contextual hybrid retriever.

        Args:
            vector_store: Pinecone vector store with contextual embeddings
            documents: Contextual documents
            use_reranking: Whether to use Cohere reranking
            reranker_model: Reranker model name
        """
        super().__init__(
            vector_store=vector_store,
            documents=documents,
            use_reranking=use_reranking,
            reranker_model=reranker_model,
            **kwargs
        )

        # Initialize reranker if enabled
        if use_reranking and settings.cohere_api_key:
            try:
                from langchain.retrievers.document_compressors import CohereRerank
                self.reranker = CohereRerank(
                    cohere_api_key=settings.cohere_api_key,
                    model=reranker_model or "rerank-english-v2.0",
                    top_n=settings.top_n_rerank
                )
                logger.info("Cohere reranker initialized")
            except ImportError:
                logger.warning("Cohere reranker not available, install langchain-cohere")
                self.use_reranking = False

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Retrieve with optional reranking.
        """
        # Get hybrid search results
        docs = super()._get_relevant_documents(query, run_manager=run_manager)

        # Apply reranking if enabled
        if self.use_reranking and self.reranker:
            try:
                docs = self.reranker.compress_documents(docs, query)
                logger.debug(f"Reranked to {len(docs)} documents")
            except Exception as e:
                logger.warning(f"Reranking failed: {e}, using hybrid results")

        return docs


# Helper function to create retriever
def create_hybrid_retriever(
    index_name: str,
    documents: List[Document],
    use_contextual: bool = True,
    use_reranking: bool = True,
    **kwargs
) -> HybridRetriever:
    """
    Create hybrid retriever with Pinecone vector store.

    Args:
        index_name: Pinecone index name
        documents: List of documents to index
        use_contextual: Use contextual retriever with reranking
        use_reranking: Enable Cohere reranking
        **kwargs: Additional retriever arguments

    Returns:
        Configured hybrid retriever
    """
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key
    )

    # Create or connect to Pinecone index
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=settings.pinecone_api_key
    )

    # Check if we need to index documents
    # (In production, check if index exists and has data)

    # Create retriever
    if use_contextual:
        retriever = ContextualHybridRetriever(
            vector_store=vector_store,
            documents=documents,
            use_reranking=use_reranking,
            **kwargs
        )
    else:
        retriever = HybridRetriever(
            vector_store=vector_store,
            documents=documents,
            **kwargs
        )

    logger.info(f"Hybrid retriever created for index: {index_name}")
    return retriever
