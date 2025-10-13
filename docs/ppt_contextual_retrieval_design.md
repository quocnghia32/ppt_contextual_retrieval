# PPT Contextual Retrieval System - Design Document

**Version**: 1.0
**Date**: 2025-10-10
**Author**: Design Document
**Status**: Draft

---

## 📋 Table of Contents
1. [Overview](#overview)
2. [Contextual Retrieval Approach](#contextual-retrieval-approach)
3. [PPT-Specific Challenges](#ppt-specific-challenges)
4. [Architecture Design](#architecture-design)
5. [Core Components](#core-components)
6. [Vector Store Abstraction Layer](#vector-store-abstraction-layer)
7. [Image Processing & OCR Strategy](#image-processing--ocr-strategy)
8. [Implementation Plan](#implementation-plan)
9. [Performance Optimization](#performance-optimization)
10. [Technical Stack](#technical-stack)
11. [Expected Performance](#expected-performance)

---

## 🎯 Overview

### Implementation Approaches

> **📄 LangChain Implementation**: See [`langchain_implementation.md`](./langchain_implementation.md) for complete LangChain-based architecture.

**Two Implementation Options**:

| Approach | Pros | Cons | Best For |
|----------|------|------|----------|
| **Custom** | Full control, optimized for PPT | More code, more maintenance | Specific requirements, performance-critical |
| **LangChain** ⭐ | Fast development, community support, pre-built components | Less flexible | Standard RAG, quick MVP |

**Recommendation**: Start with **LangChain** for faster development (3-5 days vs 2-3 weeks). This document covers architectural concepts applicable to both approaches.

### Problem Statement
Traditional RAG systems struggle với PowerPoint files vì:
- **Loss of Context**: Khi split slides thành chunks, mất context về vị trí slide trong presentation
- **Structural Information Loss**: Không capture được hierarchy (Section → Slide → Bullet Points)
- **Visual Context Ignored**: Hình ảnh, charts, tables không được link với text context
- **Presentation Flow Lost**: Không hiểu được flow logic giữa các slides

### Solution
Áp dụng **Contextual Retrieval** approach từ Anthropic, customized cho PowerPoint files:
- Generate contextual embeddings cho mỗi slide/chunk
- Preserve presentation structure và flow
- Combine vector search với BM25 và reranking
- Include visual elements context

### Success Metrics
- Giảm retrieval failure rate xuống **< 2%** (từ baseline ~6%)
- Improve retrieval precision by **50%+**
- Maintain query latency **< 500ms**

---

## 🧠 Contextual Retrieval Approach

### Core Concepts (from Anthropic Blog)

1. **Contextual Embeddings**
   - Prepend chunk-specific context (50-100 tokens) before embedding
   - Context generated bởi LLM (Claude) để explain chunk's role
   - Significantly improves retrieval accuracy

2. **Hybrid Search**
   - Vector similarity (embeddings)
   - Lexical matching (BM25)
   - Combined scoring

3. **Reranking**
   - Retrieve top-K candidates (e.g., 20)
   - Rerank using more sophisticated model
   - Return top-N (e.g., 5) most relevant

### Performance Results from Blog
| Technique | Failure Rate | Improvement |
|-----------|--------------|-------------|
| Baseline | 5.7% | - |
| + Contextual Embeddings | 3.7% | 35% ↓ |
| + Contextual BM25 | 2.9% | 49% ↓ |
| + Reranking | 1.9% | 67% ↓ |

---

## 🎨 PPT-Specific Challenges

### 1. Hierarchical Structure
```
Presentation
├── Section 1: Introduction
│   ├── Slide 1: Title Slide
│   ├── Slide 2: Agenda
│   └── Slide 3: Overview
├── Section 2: Main Content
│   ├── Slide 4: Topic A
│   │   ├── Title
│   │   ├── Bullet points
│   │   └── Image/Chart
│   └── Slide 5: Topic B
└── Section 3: Conclusion
```

### 2. Multi-Modal Content
- **Text**: Titles, body text, bullet points, notes
- **Visual**: Images, charts, diagrams, tables
- **Metadata**: Slide numbers, speaker notes, animations

### 3. Presentation Flow
- Slides có logical sequence
- Earlier slides provide context cho later slides
- Section boundaries matter

### 4. Chunking Strategy
**Challenge**: Nên chunk ở level nào?
- **Slide-level**: Giữ nguyên structure nhưng chunks có thể quá lớn
- **Element-level**: Mất structure nhưng chunks size consistent
- **Hybrid**: Best of both worlds

---

## 🏗️ Architecture Design

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     PPT Input Pipeline                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              PPT Parser & Structure Extractor                │
│  • Extract slides, sections, text, images                   │
│  • Parse speaker notes, metadata                             │
│  • Build hierarchy tree                                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Chunking Strategy Module                    │
│  • Hybrid chunking (slide + element level)                  │
│  • Preserve structural boundaries                            │
│  • Handle multi-modal content                                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│             Context Generation Module (Claude)               │
│  • Generate contextual descriptions for each chunk          │
│  • Include: position, section, prev/next slides              │
│  • Describe visual elements context                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Embedding & Indexing                        │
│  ┌─────────────────┐          ┌─────────────────┐          │
│  │ Pinecone Index  │          │ BM25 Index      │          │
│  │ (Vector DB)     │          │ (Local/Redis)   │          │
│  │ • Contextual    │          │ • Contextual    │          │
│  │   Embeddings    │          │   Text          │          │
│  │ • Metadata      │          │ • Fast lexical  │          │
│  └─────────────────┘          └─────────────────┘          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Query Processing                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Hybrid Retrieval Module                     │
│  • Vector similarity search (top 20)                        │
│  • BM25 lexical search (top 20)                             │
│  • Fusion scoring (RRF - Reciprocal Rank Fusion)            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Reranking Module                          │
│  • Cross-encoder reranking                                   │
│  • Context-aware scoring                                     │
│  • Return top-5 chunks                                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Response Generation                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 Core Components

### 1. PPT Parser & Structure Extractor

**Input**: PowerPoint file (.pptx)
**Output**: Structured data with hierarchy

```python
class PPTStructure:
    presentation_title: str
    sections: List[Section]
    total_slides: int
    metadata: Dict

class Section:
    section_title: str
    slides: List[Slide]
    position: int  # Section index

class Slide:
    slide_number: int
    title: str
    content: List[ContentElement]
    speaker_notes: str
    section: str
    visual_elements: List[VisualElement]

class ContentElement:
    type: str  # text, bullet, table
    content: str
    level: int  # For nested bullets
    position: int

class VisualElement:
    type: str  # image, chart, diagram
    description: str  # OCR or caption
    position: Tuple[int, int]
```

**Libraries**:
- `python-pptx`: Parse .pptx files
- `Pillow`: Extract và process images
- `pytesseract`: OCR cho images (optional)

### 2. Chunking Strategy Module

**Strategy**: Hybrid Approach

#### Option A: Slide-Level Chunking
```python
def create_slide_chunk(slide: Slide, context: PPTStructure) -> Chunk:
    """
    Each slide = 1 chunk
    Pros: Preserves slide integrity
    Cons: Variable chunk size
    """
    chunk = {
        "content": combine_slide_elements(slide),
        "metadata": {
            "slide_number": slide.slide_number,
            "section": slide.section,
            "title": slide.title,
            "total_slides": context.total_slides
        }
    }
    return chunk
```

#### Option B: Element-Level Chunking (Recommended)
```python
def create_element_chunks(slide: Slide) -> List[Chunk]:
    """
    Split slide into logical elements
    Pros: Consistent chunk size, better granularity
    Cons: Need careful context preservation
    """
    chunks = []

    # Title chunk
    if slide.title:
        chunks.append(create_chunk(slide.title, "title"))

    # Content chunks (group related bullets)
    content_groups = group_content_elements(slide.content)
    for group in content_groups:
        chunks.append(create_chunk(group, "content"))

    # Visual element chunks
    for visual in slide.visual_elements:
        chunks.append(create_chunk(visual, "visual"))

    return chunks
```

**Chunk Size Guidelines**:
- Target: 200-400 tokens per chunk
- Max: 600 tokens
- Min: 50 tokens

### 3. Context Generation Module

**Purpose**: Generate contextual description cho mỗi chunk

**Prompt Template**:
```
<document>
{presentation_structure}
</document>

Here is the chunk we want to situate within the whole presentation:
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall presentation to improve search retrieval of the chunk. Answer only with the succinct context and nothing else.

Additional context to include:
- Slide position: {slide_number} of {total_slides}
- Section: {section_name}
- Previous slide topic: {prev_slide_title}
- Next slide topic: {next_slide_title}
- Visual elements present: {visual_summary}
```

**Example Output**:
```
"This content is from slide 5 of 20 in the 'Market Analysis' section.
It discusses revenue growth trends following the company overview from
the previous slide and precedes the competitive landscape analysis.
The slide includes a bar chart showing YoY growth."
```

**Implementation**:
```python
async def generate_context(
    chunk: Chunk,
    presentation: PPTStructure,
    llm: AnthropicLLM
) -> str:
    """
    Use Claude to generate contextual description
    """
    prompt = build_context_prompt(chunk, presentation)
    context = await llm.generate(
        prompt=prompt,
        max_tokens=100,
        temperature=0.0  # Deterministic
    )
    return context.strip()
```

### 4. Embedding & Indexing

**Vector Embeddings**:
```python
def create_contextual_embedding(chunk: Chunk, context: str):
    """
    Combine context + chunk content before embedding
    """
    # Format: <context>\n\n<chunk_content>
    contextualized_text = f"{context}\n\n{chunk.content}"

    # Generate embedding
    embedding = embedding_model.encode(contextualized_text)

    return {
        "chunk_id": chunk.id,
        "embedding": embedding,
        "original_text": chunk.content,
        "context": context,
        "metadata": chunk.metadata
    }
```

**BM25 Index**:
```python
def create_bm25_index(chunks: List[Chunk], contexts: List[str]):
    """
    BM25 index with contextual text
    """
    bm25_corpus = []
    for chunk, context in zip(chunks, contexts):
        # Combine context + chunk for BM25 too
        doc = f"{context} {chunk.content}"
        bm25_corpus.append(tokenize(doc))

    bm25 = BM25Okapi(bm25_corpus)
    return bm25
```

**Vector Store: Pinecone**

**Why Pinecone**:
- ✅ Fully managed, serverless - no infrastructure to manage
- ✅ Fast similarity search at scale (millions/billions of vectors)
- ✅ Built-in metadata filtering
- ✅ Auto-scaling và high availability
- ✅ Multiple index types (pod-based, serverless)
- ✅ Simple API và excellent SDKs

**Pinecone Index Structure**:
```python
# Index configuration
index_config = {
    "name": "ppt-contextual-retrieval",
    "dimension": 768,  # Embedding dimension (depends on model)
    "metric": "cosine",  # or "dotproduct", "euclidean"
    "spec": {
        "serverless": {
            "cloud": "aws",
            "region": "us-east-1"
        }
    }
}

# Vector format stored in Pinecone
vector_record = {
    "id": "chunk_12345",  # Unique chunk ID
    "values": [...],  # 768-dim embedding vector
    "metadata": {
        # Original content
        "content": "Revenue increased from $1.2M to $2.1M...",
        "context": "This is from slide 5 of 20 in Market Analysis section...",

        # Slide metadata
        "slide_number": 5,
        "section": "Market Analysis",
        "presentation_title": "Q4 Business Review",
        "slide_title": "Revenue Growth Trends",

        # Content metadata
        "chunk_type": "content",  # title, content, visual, note
        "has_visual": True,
        "visual_type": "chart",

        # For filtering
        "presentation_id": "ppt_abc123",
        "created_at": "2025-10-10T10:00:00Z",

        # Token count for cost tracking
        "token_count": 256
    }
}
```

**Pinecone Implementation**:
```python
from pinecone import Pinecone, ServerlessSpec

class PineconeVectorStore:
    def __init__(self, api_key: str, index_name: str):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.index = None

    def create_index(self, dimension: int = 768):
        """Create Pinecone index if not exists"""
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        self.index = self.pc.Index(self.index_name)

    async def upsert_chunks(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        contexts: List[str]
    ):
        """Batch upsert chunks to Pinecone"""
        vectors = []
        for chunk, embedding, context in zip(chunks, embeddings, contexts):
            vectors.append({
                "id": chunk.id,
                "values": embedding,
                "metadata": {
                    "content": chunk.content,
                    "context": context,
                    "slide_number": chunk.metadata.slide_number,
                    "section": chunk.metadata.section,
                    "presentation_id": chunk.metadata.presentation_id,
                    "chunk_type": chunk.metadata.type,
                    # Add all relevant metadata
                    **chunk.metadata.to_dict()
                }
            })

        # Batch upsert (max 100 vectors per batch)
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            self.index.upsert(vectors=batch)

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 20,
        filter: Dict = None
    ) -> List[SearchResult]:
        """Search for similar vectors"""
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            filter=filter,  # e.g., {"presentation_id": "ppt_abc123"}
            include_metadata=True
        )

        return [
            SearchResult(
                id=match.id,
                score=match.score,
                content=match.metadata.get("content"),
                context=match.metadata.get("context"),
                metadata=match.metadata
            )
            for match in results.matches
        ]

    async def delete_presentation(self, presentation_id: str):
        """Delete all vectors for a presentation"""
        self.index.delete(
            filter={"presentation_id": presentation_id}
        )
```

**Metadata Filtering Examples**:
```python
# Search only in specific presentation
filter = {"presentation_id": "ppt_abc123"}

# Search only in specific section
filter = {"section": "Market Analysis"}

# Search only charts/diagrams
filter = {"chunk_type": "visual", "visual_type": {"$in": ["chart", "diagram"]}}

# Combine filters
filter = {
    "presentation_id": "ppt_abc123",
    "section": "Market Analysis",
    "has_visual": True
}

results = await vector_store.search(
    query_embedding=query_emb,
    top_k=20,
    filter=filter
)
```

**Storage Details**:
- Embeddings: 768-dim float32 vectors
- Metadata: Up to 40KB per vector (plenty for our use case)
- Original text và context stored in metadata for retrieval

### 5. Hybrid Retrieval Module

**Retrieval Process**:
```python
async def hybrid_retrieve(
    query: str,
    pinecone_store: PineconeVectorStore,
    bm25_index: BM25,
    top_k: int = 20,
    filter: Dict = None  # Optional Pinecone metadata filter
) -> List[Chunk]:
    """
    Combine Pinecone vector search and BM25 retrieval
    """
    # 1. Pinecone vector similarity search
    query_embedding = embedding_model.encode(query)
    vector_results = await pinecone_store.search(
        query_embedding=query_embedding,
        top_k=top_k,
        filter=filter  # e.g., filter by presentation_id
    )

    # 2. BM25 lexical search
    bm25_results = bm25_index.search(
        tokenize(query),
        top_k=top_k
    )

    # 3. Fusion using Reciprocal Rank Fusion (RRF)
    fused_results = reciprocal_rank_fusion(
        [vector_results, bm25_results],
        k=60  # RRF parameter
    )

    return fused_results[:top_k]
```

**Advanced Retrieval with Pinecone Filtering**:
```python
async def context_aware_retrieve(
    query: str,
    presentation_id: str,
    pinecone_store: PineconeVectorStore,
    section: str = None
) -> List[Chunk]:
    """
    Retrieval with context-specific filtering
    """
    # Build filter
    filter = {"presentation_id": presentation_id}
    if section:
        filter["section"] = section

    # Retrieve with filter
    results = await pinecone_store.search(
        query_embedding=embedding_model.encode(query),
        top_k=20,
        filter=filter
    )

    return results
```

**Reciprocal Rank Fusion (RRF)**:
```python
def reciprocal_rank_fusion(
    result_lists: List[List[Tuple[str, float]]],
    k: int = 60
) -> List[Tuple[str, float]]:
    """
    RRF scoring: score(d) = Σ 1/(k + rank_i(d))
    """
    scores = defaultdict(float)

    for results in result_lists:
        for rank, (doc_id, _) in enumerate(results):
            scores[doc_id] += 1.0 / (k + rank + 1)

    # Sort by score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked
```

### 6. Reranking Module

**Cross-Encoder Reranking**:
```python
async def rerank_results(
    query: str,
    candidates: List[Chunk],
    reranker: CrossEncoder,
    top_n: int = 5
) -> List[Chunk]:
    """
    Rerank candidates using cross-encoder
    """
    # Prepare query-document pairs
    pairs = [(query, chunk.content) for chunk in candidates]

    # Score with cross-encoder
    scores = reranker.predict(pairs)

    # Sort and return top N
    ranked_indices = np.argsort(scores)[::-1][:top_n]
    return [candidates[i] for i in ranked_indices]
```

**Reranker Model Options**:
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast, decent)
- `cross-encoder/ms-marco-TinyBERT-L-2-v2` (fastest)
- Cohere Rerank API (best quality, paid)

---

## 🔌 Vector Store Abstraction Layer

> **📄 Full Documentation**: See [`vector_store_abstraction_layer.md`](./vector_store_abstraction_layer.md) for complete design details.

### Overview

**Problem**: Tight coupling with Pinecone creates vendor lock-in risk:
- ❌ Hard to switch providers
- ❌ Difficult to test (requires real Pinecone instance)
- ❌ Cannot A/B test different vector databases

**Solution**: **Repository + Adapter Pattern** for database-agnostic architecture.

### Design Pattern: Repository + Adapter

```
Application Code (RAG Pipeline)
        ↓
   [Uses only abstract interface]
        ↓
VectorStoreRepository (ABC)
        ↑
    ┌───┼───┬───────┐
    │   │   │       │
Pinecone Qdrant ChromaDB InMemory
Adapter  Adapter Adapter  (Testing)
```

**Benefits**:
- ✅ Switch providers với 1-line config change
- ✅ Easy unit testing với mock implementations
- ✅ Support multiple providers simultaneously
- ✅ Future-proof, no vendor lock-in

### Abstract Interface

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

# Vendor-agnostic domain models
@dataclass
class VectorRecord:
    """Universal vector record format"""
    id: str
    values: List[float]  # Embedding
    metadata: Dict[str, Any]

@dataclass
class SearchResult:
    """Universal search result format"""
    id: str
    score: float
    metadata: Dict[str, Any]

# Abstract repository interface
class VectorStoreRepository(ABC):
    """
    Abstract base class for all vector store implementations.

    All providers (Pinecone, Qdrant, ChromaDB, etc.) must implement
    this interface to ensure consistent API across the application.
    """

    @abstractmethod
    async def create_index(
        self,
        index_name: str,
        dimension: int,
        metric: str = "cosine"
    ) -> None:
        """Create vector index"""
        pass

    @abstractmethod
    async def upsert(
        self,
        vectors: List[VectorRecord],
        namespace: Optional[str] = None
    ) -> Dict[str, Any]:
        """Insert or update vectors"""
        pass

    @abstractmethod
    async def search(
        self,
        query_vector: List[float],
        top_k: int = 20,
        filter: Optional[Dict] = None,
        namespace: Optional[str] = None
    ) -> List[SearchResult]:
        """Search for similar vectors"""
        pass

    @abstractmethod
    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Delete vectors"""
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        pass
```

### Pinecone Adapter Implementation

```python
from pinecone import Pinecone, ServerlessSpec

class PineconeAdapter(VectorStoreRepository):
    """Pinecone implementation of abstract interface"""

    def __init__(self, api_key: str, environment: str = "us-east-1"):
        self.client = Pinecone(api_key=api_key)
        self.environment = environment
        self.index = None

    async def create_index(self, index_name: str, dimension: int, metric: str = "cosine"):
        """Implement abstract method for Pinecone"""
        if index_name not in self.client.list_indexes().names():
            self.client.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud="aws", region=self.environment)
            )
        self.index = self.client.Index(index_name)

    async def upsert(self, vectors: List[VectorRecord], namespace: Optional[str] = None):
        """Transform VectorRecord to Pinecone format and upsert"""
        pinecone_vectors = [
            {"id": v.id, "values": v.values, "metadata": v.metadata}
            for v in vectors
        ]
        self.index.upsert(vectors=pinecone_vectors, namespace=namespace or "")
        return {"upserted_count": len(vectors)}

    async def search(self, query_vector: List[float], top_k: int = 20,
                    filter: Optional[Dict] = None, namespace: Optional[str] = None):
        """Search Pinecone and return universal SearchResult format"""
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            filter=filter,
            namespace=namespace or "",
            include_metadata=True
        )

        # Transform to universal format
        return [
            SearchResult(id=m.id, score=m.score, metadata=m.metadata)
            for m in results.matches
        ]

    # ... other methods
```

### Factory Pattern for Provider Selection

```python
from enum import Enum

class VectorStoreProvider(Enum):
    PINECONE = "pinecone"
    QDRANT = "qdrant"
    CHROMADB = "chromadb"
    INMEMORY = "inmemory"  # For testing

class VectorStoreFactory:
    """Factory to create vector store instances"""

    @staticmethod
    def create(provider: VectorStoreProvider, **kwargs) -> VectorStoreRepository:
        """Create appropriate adapter based on provider"""
        if provider == VectorStoreProvider.PINECONE:
            return PineconeAdapter(**kwargs)
        elif provider == VectorStoreProvider.QDRANT:
            return QdrantAdapter(**kwargs)
        elif provider == VectorStoreProvider.CHROMADB:
            return ChromaDBAdapter(**kwargs)
        elif provider == VectorStoreProvider.INMEMORY:
            return InMemoryVectorStore()
        else:
            raise ValueError(f"Unknown provider: {provider}")

    @staticmethod
    def create_from_env() -> VectorStoreRepository:
        """Create from environment variables"""
        import os
        provider = os.getenv("VECTOR_STORE_PROVIDER", "pinecone")

        if provider == "pinecone":
            return VectorStoreFactory.create(
                VectorStoreProvider.PINECONE,
                api_key=os.getenv("PINECONE_API_KEY"),
                environment=os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
            )
        # ... other providers
```

### Usage: Application Code (Provider-Agnostic!)

```python
# Application code depends ONLY on abstract interface
class PPTRetrievalService:
    def __init__(self, vector_store: VectorStoreRepository):
        """Dependency injection - accepts ANY vector store implementation"""
        self.vector_store = vector_store

    async def index_presentation(self, presentation: Presentation):
        """Index presentation - works with ANY provider!"""
        # Generate chunks and embeddings
        chunks = chunk_presentation(presentation)
        embeddings = generate_embeddings(chunks)

        # Create vendor-agnostic VectorRecord objects
        vectors = [
            VectorRecord(
                id=chunk.id,
                values=embedding,
                metadata={
                    "content": chunk.content,
                    "slide_number": chunk.slide_number,
                    "presentation_id": presentation.id
                }
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        # Upsert - works with Pinecone, Qdrant, ChromaDB, etc!
        await self.vector_store.upsert(vectors)

    async def search(self, query: str, presentation_id: str):
        """Search - provider agnostic"""
        query_embedding = generate_embedding(query)

        results = await self.vector_store.search(
            query_vector=query_embedding,
            top_k=20,
            filter={"presentation_id": presentation_id}
        )

        return results

# Configuration (via environment or config file)
vector_store = VectorStoreFactory.create_from_env()

# Create service - works with ANY provider!
service = PPTRetrievalService(vector_store)

# Use it
await service.index_presentation(ppt)
results = await service.search("revenue growth", "ppt_123")
```

### Easy Provider Switching

```python
# Development: Use in-memory (no setup, fast tests)
dev_store = VectorStoreFactory.create(VectorStoreProvider.INMEMORY)

# Staging: Use Qdrant (self-hosted, cost-effective)
staging_store = VectorStoreFactory.create(
    VectorStoreProvider.QDRANT,
    url="qdrant.staging.internal",
    port=6333
)

# Production: Use Pinecone (managed, scalable)
prod_store = VectorStoreFactory.create(
    VectorStoreProvider.PINECONE,
    api_key=os.getenv("PINECONE_API_KEY"),
    environment="us-east-1"
)

# Same service code works with ALL!
service = PPTRetrievalService(prod_store)  # Just swap the instance!
```

### Testing Benefits

```python
# Unit tests with mock - no external dependencies!
import pytest
from unittest.mock import AsyncMock

@pytest.fixture
def mock_vector_store():
    """Mock vector store for unit tests"""
    mock = AsyncMock(spec=VectorStoreRepository)
    mock.search.return_value = [
        SearchResult(id="test_1", score=0.95, metadata={"content": "test"})
    ]
    return mock

async def test_ppt_retrieval_service(mock_vector_store):
    """Test service without real vector database"""
    service = PPTRetrievalService(mock_vector_store)

    results = await service.search("test query", "ppt_123")

    assert len(results) == 1
    assert results[0].score == 0.95
    mock_vector_store.search.assert_called_once()

# Integration tests with InMemory store
@pytest.fixture
async def inmemory_vector_store():
    """Real in-memory store for integration tests"""
    store = InMemoryVectorStore()
    await store.create_index("test-index", dimension=768)
    yield store
    await store.delete_index("test-index")

async def test_end_to_end(inmemory_vector_store):
    """End-to-end test without external services"""
    service = PPTRetrievalService(inmemory_vector_store)

    # Index test data
    await service.index_presentation(test_presentation)

    # Search
    results = await service.search("test", "ppt_test")
    assert len(results) > 0
```

### Migration Strategy

```python
# Zero-downtime migration: Dual-write pattern
class DualWriteVectorStore(VectorStoreRepository):
    """Write to both old and new stores during migration"""

    def __init__(self, primary: VectorStoreRepository, secondary: VectorStoreRepository):
        self.primary = primary
        self.secondary = secondary

    async def upsert(self, vectors: List[VectorRecord], **kwargs):
        """Write to BOTH stores"""
        await asyncio.gather(
            self.primary.upsert(vectors, **kwargs),
            self.secondary.upsert(vectors, **kwargs)
        )

    async def search(self, query_vector: List[float], **kwargs):
        """Read from new store with fallback to old"""
        try:
            return await self.primary.search(query_vector, **kwargs)
        except Exception:
            logger.warning("Primary failed, using fallback")
            return await self.secondary.search(query_vector, **kwargs)

# Migration from Pinecone to Qdrant
old_store = PineconeAdapter(api_key="...")
new_store = QdrantAdapter(url="...")

# Use dual-write during migration
migration_store = DualWriteVectorStore(primary=new_store, secondary=old_store)
service = PPTRetrievalService(migration_store)

# After backfill complete, switch to new store only
service = PPTRetrievalService(new_store)
```

### Architecture Impact

```
┌─────────────────────────────────────────────────────────────┐
│                  Application Layer                           │
│  • PPTRetrievalService                                       │
│  • RAG Pipeline                                              │
│  • API Endpoints                                             │
│                                                              │
│  Depends ONLY on VectorStoreRepository interface            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│            VectorStoreRepository (ABC)                       │
│  • Defines contract for all vector stores                   │
│  • Vendor-agnostic interface                                │
└─────────────────────────────────────────────────────────────┘
                            ↑
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────┴──────────┐ ┌──────┴─────────┐ ┌──────┴─────────────┐
│ PineconeAdapter  │ │ QdrantAdapter  │ │ ChromaDBAdapter    │
│ • Pinecone API   │ │ • Qdrant API   │ │ • ChromaDB API     │
│ • Transform data │ │ • Transform    │ │ • Transform data   │
└──────────────────┘ └────────────────┘ └────────────────────┘
```

### Key Benefits

| Benefit | Description |
|---------|-------------|
| **Flexibility** | Switch from Pinecone → Qdrant → Weaviate with config change |
| **Testability** | Mock for unit tests, InMemory for integration tests |
| **Cost Optimization** | A/B test providers for best cost/performance |
| **Risk Mitigation** | Not locked into single vendor |
| **Local Development** | Use InMemory locally, Pinecone in production |

### Implementation Priority

**Phase 1** (Week 2):
- ✅ Define `VectorStoreRepository` interface
- ✅ Implement `PineconeAdapter`
- ✅ Implement `InMemoryVectorStore` for testing
- ✅ Create `VectorStoreFactory`

**Phase 2** (Week 5-6, optional):
- ✅ Implement `QdrantAdapter` (alternative option)
- ✅ Implement `ChromaDBAdapter` (local dev option)
- ✅ Add migration utilities

**Phase 3** (Future):
- ✅ Add more providers (Weaviate, Milvus, etc.)
- ✅ Performance benchmarking across providers
- ✅ Auto-selection based on workload characteristics

---

### 7. Image Processing & OCR Strategy

**Challenge**: PowerPoint slides chứa nhiều visual elements (images, charts, diagrams, tables) cần được process để extract information.

**Two Approaches**:

#### Option A: Traditional OCR Models

**Models**:
- **Tesseract OCR**: Open-source, mature, fast
- **PaddleOCR**: Better accuracy, supports multiple languages
- **EasyOCR**: Deep learning-based, good for complex layouts
- **Azure Computer Vision**: Cloud-based, high accuracy

**Pros** ✅:
- **Fast**: 50-200ms per image
- **Cost-effective**: Free (local) hoặc cheap (cloud)
- **High accuracy for clean text**: 95-99% accuracy
- **Deterministic**: Same input → same output
- **No rate limits**: Run locally unlimited
- **Low latency**: No API calls needed
- **Privacy**: Data stays local
- **Specialized**: Optimized specifically cho text extraction

**Cons** ❌:
- **Text-only**: Chỉ extract text, không hiểu context
- **Poor với complex layouts**: Struggles với multi-column, rotated text
- **No semantic understanding**: Không biết ý nghĩa của text
- **Can't describe visuals**: Không describe được charts, diagrams
- **Poor với handwriting**: Accuracy drops significantly
- **No multi-modal context**: Không link text với visual elements
- **Rigid output**: Just raw text, no interpretation

**Best For**:
- Clean, structured text
- Tables with clear boundaries
- Slide titles và bullet points
- Text-heavy slides
- High-volume processing
- Cost-sensitive applications

#### Option B: LLM Vision Models

**Models**:
- **GPT-4 Vision**: Best quality, expensive
- **Claude 3 (Opus/Sonnet)**: Excellent balance quality/cost
- **Gemini Pro Vision**: Good quality, competitive pricing
- **LLaVA**: Open-source alternative (can run locally)

**Pros** ✅:
- **Semantic understanding**: Hiểu context và meaning
- **Multi-modal**: Text + visual elements together
- **Describe visuals**: Describe charts, graphs, diagrams, images
- **Extract insights**: Identify trends, patterns, statistics
- **Flexible output**: Can format output as needed
- **Handle complexity**: Better với complex layouts, handwriting
- **Question answering**: Can answer specific questions về image
- **Contextual extraction**: Understand relationships between elements

**Cons** ❌:
- **Expensive**: $0.01-0.05 per image (depending on model)
- **Slower**: 1-5 seconds per image
- **May hallucinate**: Can generate incorrect information
- **Rate limits**: API throttling
- **Privacy concerns**: Data sent to cloud
- **Non-deterministic**: Same input có thể có slightly different outputs
- **Overkill for simple text**: Not efficient cho plain text extraction

**Best For**:
- Complex visual content (charts, diagrams)
- Infographics và data visualizations
- Images với embedded text
- Slides requiring interpretation
- Statistical analysis needed
- High-value content

#### Option C: Hybrid Approach ⭐ **RECOMMENDED**

**Strategy**: Combine both approaches để maximize benefits

```python
class HybridImageProcessor:
    """
    Intelligent routing: Traditional OCR for simple cases,
    LLM Vision for complex cases
    """

    def __init__(self):
        self.ocr_engine = PaddleOCR()
        self.vision_llm = ClaudeVision()
        self.classifier = ImageComplexityClassifier()

    async def process_image(
        self,
        image: Image,
        context: SlideContext
    ) -> ImageProcessingResult:
        """
        Route to appropriate processor based on image type
        """
        # 1. Classify image type
        image_type = self.classify_image(image)

        # 2. Route based on type
        if image_type in ["text_heavy", "table", "simple"]:
            # Use fast OCR
            result = await self.process_with_ocr(image)

        elif image_type in ["chart", "diagram", "infographic", "complex"]:
            # Use LLM Vision
            result = await self.process_with_vision_llm(image, context)

        elif image_type == "hybrid":
            # Use both: OCR for text + LLM for understanding
            ocr_result = await self.process_with_ocr(image)
            vision_result = await self.process_with_vision_llm(
                image,
                context,
                extracted_text=ocr_result.text
            )
            result = self.merge_results(ocr_result, vision_result)

        return result

    def classify_image(self, image: Image) -> str:
        """
        Classify image type to determine processing strategy
        """
        # Simple heuristics or ML classifier
        features = extract_features(image)

        if features.text_ratio > 0.8:
            return "text_heavy"
        elif features.has_table_structure:
            return "table"
        elif features.has_chart_elements:
            return "chart"
        elif features.complexity_score > 0.7:
            return "complex"
        else:
            return "simple"

    async def process_with_ocr(self, image: Image) -> OCRResult:
        """
        Fast text extraction with traditional OCR
        """
        result = self.ocr_engine.ocr(image)
        return OCRResult(
            text=extract_text(result),
            bounding_boxes=extract_boxes(result),
            confidence=calculate_confidence(result)
        )

    async def process_with_vision_llm(
        self,
        image: Image,
        context: SlideContext,
        extracted_text: str = None
    ) -> VisionResult:
        """
        Deep understanding with LLM Vision
        """
        prompt = f"""
        Analyze this slide image from a presentation.

        Context:
        - Slide {context.slide_number} of {context.total_slides}
        - Section: {context.section_name}
        - Previous slide: {context.prev_slide_title}

        {f"Extracted text (for reference): {extracted_text}" if extracted_text else ""}

        Please provide:
        1. Description of visual elements (charts, diagrams, images)
        2. Key data points and statistics shown
        3. Main insights or trends illustrated
        4. Any text not captured by OCR
        5. Overall purpose of this visual in the presentation

        Be concise but comprehensive.
        """

        result = await self.vision_llm.analyze(
            image=image,
            prompt=prompt
        )

        return VisionResult(
            description=result.description,
            insights=result.insights,
            statistics=result.statistics,
            visual_elements=result.elements
        )
```

**Decision Matrix**:

| Image Type | OCR | LLM Vision | Hybrid | Reason |
|------------|-----|------------|--------|--------|
| Plain text slide | ✅ | ❌ | ❌ | Fast, cheap, accurate |
| Table | ✅ | ❌ | ⚡ | OCR for structure, LLM for understanding (optional) |
| Bar/Line chart | ❌ | ✅ | ❌ | Need data extraction & interpretation |
| Pie chart | ❌ | ✅ | ❌ | Need percentages & insights |
| Diagram/Flowchart | ❌ | ✅ | ❌ | Need relationship understanding |
| Infographic | ❌ | ✅ | ⚡ | OCR for text, LLM for visual meaning |
| Screenshot | ⚡ | ✅ | ⚡ | Depends on content |
| Photo với caption | ✅ | ❌ | ❌ | OCR sufficient for caption |
| Complex mixed | ❌ | ❌ | ✅ | Best of both worlds |

**Implementation Priority**:

**Phase 1** (MVP):
```python
# Simple approach: OCR for all text-based, skip pure images
async def process_slide_images_v1(slide: Slide):
    for image in slide.visual_elements:
        if has_text(image):
            text = ocr_engine.extract(image)
            image.extracted_text = text
        # Skip pure images for now
```

**Phase 2** (Enhanced):
```python
# Add LLM Vision for charts/diagrams
async def process_slide_images_v2(slide: Slide):
    for image in slide.visual_elements:
        image_type = classify(image)

        if image_type == "text":
            image.content = await ocr_extract(image)
        elif image_type in ["chart", "diagram"]:
            image.content = await vision_llm_analyze(image)
```

**Phase 3** (Full Hybrid):
```python
# Intelligent routing với full hybrid support
async def process_slide_images_v3(slide: Slide):
    processor = HybridImageProcessor()
    for image in slide.visual_elements:
        result = await processor.process_image(
            image,
            context=slide.context
        )
        image.content = result
```

**Cost Comparison** (per 100-slide presentation):

| Approach | Images/PPT | Cost per Image | Total Cost | Processing Time |
|----------|------------|----------------|------------|-----------------|
| **OCR Only** | ~50 | Free | $0 | ~5s |
| **LLM Vision Only** | ~50 | $0.015 | $0.75 | ~150s |
| **Hybrid (30% LLM)** | ~50 | Mixed | $0.23 | ~25s |

**Recommendation** 🎯:

1. **Start with Hybrid Phase 2**:
   - OCR cho text và tables (70% of images)
   - LLM Vision cho charts/diagrams (30% of images)
   - Best cost/performance ratio

2. **Image Classification**:
   - Simple heuristics initially (text ratio, color distribution)
   - ML classifier later nếu cần

3. **Cost Control**:
   - Set budgets per presentation
   - Fallback to OCR nếu quota exceeded
   - Cache results aggressively

4. **Quality Monitoring**:
   - Track OCR confidence scores
   - Manual review for low confidence
   - A/B test OCR vs Vision results

**Example Outputs**:

**OCR Output** (Table):
```
Revenue Growth
Q1 2024: $1.2M
Q2 2024: $1.5M
Q3 2024: $1.8M
Q4 2024: $2.1M
```

**LLM Vision Output** (Same table):
```
This table shows quarterly revenue growth for 2024, demonstrating
a consistent upward trend. Revenue increased from $1.2M in Q1 to
$2.1M in Q4, representing 75% growth over the year. The quarter-
over-quarter growth rate is approximately 20-25%, indicating
strong business momentum. This data supports the company's scaling
narrative presented in earlier slides.
```

**Hybrid Output** (Best of both):
```
Raw Data (OCR):
Q1: $1.2M | Q2: $1.5M | Q3: $1.8M | Q4: $2.1M

Insights (LLM):
75% YoY growth with consistent 20-25% QoQ acceleration,
supporting the scaling narrative from previous slides.
```

---

## 📝 Implementation Plan

> **💡 Quick Start with LangChain**: For faster development (3-5 days instead of 7 weeks), see [`langchain_implementation.md`](./langchain_implementation.md). The plan below is for custom implementation.

**Choose Your Path**:
- **Fast Track (LangChain)**: 3-5 days to MVP - Use pre-built components
- **Custom Track**: 6-7 weeks - Full control and optimization

### Custom Implementation Timeline

### Phase 1: Foundation (Week 1-2)

**Sprint 1.1: PPT Parsing & Project Setup**
- [ ] Setup project structure (src/, tests/, docs/)
- [ ] Define `VectorStoreRepository` abstract interface
- [ ] Implement PPT parser using `python-pptx`
- [ ] Extract slides, text, speaker notes
- [ ] Build hierarchy structure (sections → slides → elements)
- [ ] Unit tests for parser

**Sprint 1.2: Chunking Strategy & Vector Store Abstraction**
- [ ] Implement slide-level chunking
- [ ] Implement element-level chunking
- [ ] Hybrid chunking logic
- [ ] Chunk size validation (50-600 tokens)
- [ ] Implement `InMemoryVectorStore` for testing
- [ ] Implement `VectorStoreFactory`
- [ ] Unit tests for chunking

**Deliverables**:
- ✅ PPT Parser module
- ✅ Chunking module
- ✅ Vector Store abstraction layer (interface + InMemory impl)
- ✅ Factory pattern for provider selection
- ✅ Test suite với sample PPT files

### Phase 2: Context Generation (Week 2-3)

**Sprint 2.1: Context Prompt Engineering**
- [ ] Design context generation prompt
- [ ] Test với Claude API
- [ ] Optimize prompt for conciseness (50-100 tokens)
- [ ] A/B test different prompt variations

**Sprint 2.2: Batch Processing**
- [ ] Implement async context generation
- [ ] Batch API calls để giảm cost
- [ ] Error handling và retry logic
- [ ] Cache context results
- [ ] Cost estimation và monitoring

**Deliverables**:
- ✅ Context generation module
- ✅ Optimized prompts
- ✅ Cost analysis report

### Phase 3: Embedding & Indexing (Week 3-4)

**Sprint 3.1: Embedding Pipeline**
- [ ] Choose embedding model (test: OpenAI, Voyage, Cohere)
- [ ] Implement contextual embedding generation
- [ ] Batch embedding processing
- [ ] Store embeddings với metadata

**Sprint 3.2: Index Building & Pinecone Adapter**
- [ ] Setup Pinecone account và create index
- [ ] Implement `PineconeAdapter` (implements VectorStoreRepository)
- [ ] Configure Pinecone index (dimension, metric, serverless spec)
- [ ] Implement data transformation (VectorRecord ↔ Pinecone format)
- [ ] Build BM25 index (local storage)
- [ ] Test upsert/search operations via abstraction layer
- [ ] Setup metadata filtering
- [ ] Environment-based configuration (.env support)

**Deliverables**:
- ✅ Embedding pipeline
- ✅ `PineconeAdapter` implementation
- ✅ Vector + BM25 indices
- ✅ Storage layer
- ✅ Integration tests với Pinecone

### Phase 4: Retrieval & Reranking (Week 4-5)

**Sprint 4.1: Hybrid Retrieval**
- [ ] Implement vector similarity search
- [ ] Implement BM25 search
- [ ] Reciprocal Rank Fusion algorithm
- [ ] Tuning fusion parameters

**Sprint 4.2: Reranking**
- [ ] Setup reranker model
- [ ] Implement reranking pipeline
- [ ] Optimize top-K, top-N parameters
- [ ] A/B testing different rerankers

**Deliverables**:
- ✅ Hybrid retrieval module
- ✅ Reranking module
- ✅ Parameter tuning results

### Phase 5: Evaluation & Optimization (Week 5-6)

**Sprint 5.1: Evaluation Framework**
- [ ] Build evaluation dataset (queries + ground truth)
- [ ] Implement metrics: Precision@K, Recall@K, MRR, NDCG
- [ ] Baseline measurements
- [ ] Ablation studies

**Sprint 5.2: Optimization**
- [ ] Query latency optimization
- [ ] Memory usage optimization
- [ ] Cost optimization (API calls, storage)
- [ ] Caching strategy

**Deliverables**:
- ✅ Evaluation framework
- ✅ Performance benchmark report
- ✅ Optimized system

### Phase 6: Integration & Deployment (Week 6-7)

**Sprint 6.1: API Development**
- [ ] REST API design
- [ ] FastAPI implementation
- [ ] API documentation (OpenAPI)
- [ ] Rate limiting, auth

**Sprint 6.2: Deployment**
- [ ] Dockerization
- [ ] CI/CD pipeline
- [ ] Monitoring & logging
- [ ] Load testing

**Deliverables**:
- ✅ Production-ready API
- ✅ Deployment documentation
- ✅ Monitoring dashboard

---

## ⚡ Performance Optimization

### 1. Caching Strategy

**Multi-Level Caching**:
```python
# Level 1: Context cache (persistent)
context_cache = {
    "ppt_file_hash": {
        "chunk_id": "generated_context"
    }
}

# Level 2: Embedding cache (persistent)
embedding_cache = {
    "context_hash": embedding_vector
}

# Level 3: Query result cache (TTL: 1 hour)
query_cache = LRUCache(maxsize=1000)
```

### 2. Batch Processing

**Context Generation**:
```python
async def batch_generate_contexts(
    chunks: List[Chunk],
    batch_size: int = 10
):
    """
    Batch API calls để giảm latency và cost
    """
    batches = [chunks[i:i+batch_size]
               for i in range(0, len(chunks), batch_size)]

    tasks = [generate_context_batch(batch) for batch in batches]
    results = await asyncio.gather(*tasks)

    return flatten(results)
```

**Embedding Generation**:
```python
def batch_embed(texts: List[str], batch_size: int = 32):
    """
    Batch embedding generation
    """
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = embedding_model.encode(batch)
        embeddings.extend(batch_embeddings)
    return embeddings
```

### 3. Pinecone-Specific Optimizations

**Pinecone handles ANN internally** - uses proprietary algorithm optimized for:
- Sub-100ms query latency at scale
- High recall (>95%) even with billions of vectors
- No manual tuning needed

**Optimization Strategies**:

```python
# 1. Batch operations for efficiency
async def batch_upsert_presentations(presentations: List[Presentation]):
    """
    Batch multiple presentations in single API call
    """
    all_vectors = []
    for ppt in presentations:
        vectors = process_presentation(ppt)
        all_vectors.extend(vectors)

    # Pinecone recommends batches of 100-1000 vectors
    batch_size = 500
    for i in range(0, len(all_vectors), batch_size):
        batch = all_vectors[i:i+batch_size]
        await pinecone_index.upsert(vectors=batch, async_req=True)

# 2. Use metadata filters to reduce search space
async def optimized_search(query: str, presentation_id: str):
    """
    Metadata filtering is 10-100x faster than post-filtering
    """
    results = await pinecone_store.search(
        query_embedding=embed(query),
        top_k=20,
        filter={"presentation_id": presentation_id}  # Pre-filter
    )
    # Much faster than retrieving all and filtering after

# 3. Namespace strategy for data isolation
class MultiTenantPineconeStore:
    """
    Use namespaces to isolate different customers/projects
    """
    async def search_tenant(self, tenant_id: str, query: str):
        # Each tenant gets own namespace
        results = self.index.query(
            vector=embed(query),
            top_k=20,
            namespace=f"tenant_{tenant_id}"
        )
        return results

# 4. Sparse-dense hybrid (Pinecone feature)
async def hybrid_pinecone_search(query: str):
    """
    Combine dense embeddings với sparse BM25-like vectors
    (Pinecone supports this natively)
    """
    dense_vector = embedding_model.encode(query)
    sparse_vector = bm25_to_sparse(query)  # Convert BM25 to sparse format

    results = pinecone_index.query(
        vector=dense_vector,
        sparse_vector=sparse_vector,
        top_k=20,
        include_metadata=True
    )
    return results
```

**Performance Tuning**:
- **top_k**: Keep ≤ 100 for best latency (we use 20)
- **Metadata size**: Keep < 10KB per vector (we use ~2KB)
- **Dimension**: Use 768 (good balance) or 384 (faster, slight accuracy loss)
- **Metric**: Cosine similarity (normalized vectors, best for text)

### 4. Query Optimization

**Semantic Caching**:
```python
def semantic_query_cache(query: str, threshold: float = 0.95):
    """
    Check if similar query exists in cache
    """
    query_embedding = embed(query)

    for cached_query, cached_results in query_cache.items():
        cached_embedding = embed(cached_query)
        similarity = cosine_similarity(query_embedding, cached_embedding)

        if similarity > threshold:
            return cached_results

    return None
```

---

## 🛠️ Technical Stack

> **💡 LangChain Alternative**: For LangChain-based implementation, see [`langchain_implementation.md`](./langchain_implementation.md) for different stack.

### Implementation Choice

**Option A: LangChain Framework** ⭐ (Recommended)
```bash
pip install langchain==0.1.0
pip install langchain-anthropic
pip install langchain-openai
pip install langchain-pinecone
pip install langsmith  # Observability
```

**Benefits**:
- ✅ Pre-built components (Chains, Retrievers, Memory)
- ✅ Prompt management via PromptTemplate
- ✅ Easy LLM switching
- ✅ LangSmith debugging
- ✅ Community support

**Option B: Custom Implementation**

### Core Libraries (Custom Implementation)

**Python 3.11+**

**PPT Processing**:
- `python-pptx`: Parse .pptx files
- `Pillow`: Image extraction and manipulation

**Image Processing & OCR**:
- `paddleocr`: Advanced OCR (recommended for production)
- `pytesseract`: Tesseract OCR wrapper (alternative)
- `easyocr`: Deep learning OCR (alternative)
- `opencv-python`: Image preprocessing
- Vision LLM APIs: Claude 3, GPT-4V, Gemini Pro Vision

**LLM & Embeddings**:
- `anthropic`: Claude API for context generation
- `openai`: Alternative embedding models
- `cohere`: Reranking API (optional)
- `sentence-transformers`: Local embedding models

**Vector Search**:
- `pinecone-client`: Pinecone vector database (production-grade, fully managed)
- `faiss-cpu`: Optional for local development/testing

**BM25 & Text Processing**:
- `rank-bm25`: BM25 implementation
- `nltk` or `spacy`: Tokenization
- `tiktoken`: Token counting

**Web Framework**:
- `fastapi`: REST API
- `uvicorn`: ASGI server
- `pydantic`: Data validation

**Utilities**:
- `asyncio`: Async processing
- `aiohttp`: Async HTTP client
- `redis`: Caching layer
- `prometheus-client`: Metrics

### Infrastructure

**Development**:
- Docker & Docker Compose
- Poetry or pip-tools (dependency management)
- pytest (testing)
- black, ruff (linting)

**Deployment**:
- Kubernetes or AWS ECS (application hosting)
- **Pinecone** (vector database - fully managed, no deployment needed)
- AWS S3 (PPT file storage)
- PostgreSQL (metadata storage, presentation info)
- Redis (caching layer for queries và results)
- Grafana + Prometheus (monitoring)

**Pinecone Deployment Benefits**:
- ✅ Zero infrastructure management
- ✅ Auto-scaling (handles traffic spikes)
- ✅ Built-in high availability (99.9% SLA)
- ✅ Global distribution (multi-region support)
- ✅ No capacity planning needed
- ✅ Automatic backups và disaster recovery

**Environment Setup**:
```bash
# Environment variables
PINECONE_API_KEY=your-api-key
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=ppt-contextual-retrieval
PINECONE_DIMENSION=768
```

**Multi-Environment Strategy**:
- **Development**: Separate Pinecone index (`ppt-dev`)
- **Staging**: Separate index (`ppt-staging`)
- **Production**: Production index (`ppt-prod`)
- Each environment isolated for safety

---

## 📊 Expected Performance

### Baseline vs. Contextual Retrieval

| Metric | Baseline RAG | + Contextual Embeddings | + BM25 | + Reranking |
|--------|-------------|------------------------|--------|-------------|
| **Precision@5** | 0.65 | 0.78 (+20%) | 0.82 (+26%) | 0.87 (+34%) |
| **Recall@20** | 0.72 | 0.84 (+17%) | 0.88 (+22%) | 0.91 (+26%) |
| **MRR** | 0.58 | 0.71 (+22%) | 0.76 (+31%) | 0.82 (+41%) |
| **Failure Rate** | 6.0% | 3.9% (35% ↓) | 3.0% (50% ↓) | 2.0% (67% ↓) |

### Latency Budget

| Component | Target Latency | Notes |
|-----------|----------------|-------|
| PPT Parsing | < 5s per file | One-time cost |
| Context Generation | < 30s per PPT | One-time, batchable |
| Embedding | < 10s per PPT | One-time |
| Query (Vector) | < 50ms | Per query |
| Query (BM25) | < 20ms | Per query |
| Reranking | < 100ms | Per query |
| **Total Query Latency** | **< 200ms** | End-to-end |

### Cost Estimation

**Per 100-slide Presentation**:

| Service | Usage | Cost |
|---------|-------|------|
| Claude API (context) | ~100 calls × 2K tokens | $0.60 |
| Embedding API | ~100 chunks × 500 tokens | $0.02 |
| Image Processing (Hybrid) | ~50 images (30% LLM Vision) | $0.23 |
| Storage (vectors) | ~100 × 768 dim × 4 bytes | $0.001 |
| **Total per PPT** | | **~$0.85** |

**Image Processing Breakdown**:
- OCR only (70% of images): ~35 images × $0 = $0
- LLM Vision (30% of images): ~15 images × $0.015 = $0.23
- Total image processing: $0.23

**Pinecone Storage Costs**:
- Serverless index: Pay per usage (reads/writes/storage)
- ~100 vectors per PPT × 768 dim × 4 bytes = ~300KB
- Storage cost: Negligible ($0.001 per PPT)
- One-time write: ~100 writes = $0.001

**Query Costs** (per 1000 queries):

| Service | Cost |
|---------|------|
| Pinecone vector search | $0.03 (serverless read units) |
| BM25 search | Free (local) |
| Reranking (Cohere) | $0.20 |
| **Total per 1000 queries** | **$0.23** |

**Pinecone Pricing Tiers**:
- **Serverless** (Recommended for MVP):
  - No fixed costs, pay-per-use
  - ~$0.0003 per 1000 read units
  - ~$2 per 1M read units
  - Best for variable/unpredictable load

- **Pod-Based** (For high-volume production):
  - Fixed monthly cost per pod
  - p1.x1: $70/month (unlimited queries up to capacity)
  - Better for consistent high load

**Example Monthly Costs** (Serverless):
- 10 presentations uploaded: $0.01
- 100K queries/month: $6.00
- Storage (1000 vectors): $0.01
- **Total**: ~$6.00/month + LLM costs

---

## 🎯 Success Criteria

### Functional Requirements
- ✅ Parse 95%+ of .pptx files successfully
- ✅ Generate contextual embeddings for all chunks
- ✅ Hybrid retrieval working end-to-end
- ✅ Reranking improves precision
- ✅ API response time < 500ms (p95)

### Performance Requirements
- ✅ Precision@5 > 0.85
- ✅ Recall@20 > 0.88
- ✅ Failure rate < 2%
- ✅ Query latency < 200ms (p50)
- ✅ System uptime > 99.5%

### Quality Requirements
- ✅ Context generation quality > 4/5 (human eval)
- ✅ Retrieved chunks relevant to query > 80%
- ✅ No hallucination in context generation
- ✅ Handles multi-modal content (text + images)

---

## 🔮 Future Enhancements

### Phase 2 Features

1. **Multi-Modal Retrieval**
   - Image understanding (CLIP embeddings)
   - Chart/table parsing and embedding
   - Visual question answering

2. **Advanced Context**
   - Cross-slide context (slide dependencies)
   - Temporal context (animation sequences)
   - Presenter notes integration

3. **Intelligent Chunking**
   - ML-based boundary detection
   - Semantic coherence scoring
   - Adaptive chunk sizing

4. **Query Understanding**
   - Query expansion
   - Intent classification
   - Multi-hop reasoning

5. **Personalization**
   - User feedback loop
   - Relevance learning
   - Custom ranking models

---

## 📚 References

1. [Anthropic - Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval)
2. [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
3. [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)
4. [HNSW Algorithm](https://arxiv.org/abs/1603.09320)
5. [Cross-Encoder Reranking](https://www.sbert.net/examples/applications/cross-encoder/README.html)

---

## 📞 Contact & Support

**Project Owner**: NghiaNQ
**Repository**: `/home/hungson175/users/NghiaNQ/ppt_context_retrieval`
**Documentation**: `docs/`

---

**Document End**
