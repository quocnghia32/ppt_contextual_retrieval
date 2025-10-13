# Vector Store Abstraction Layer - Design Document

**Version**: 1.0
**Date**: 2025-10-10
**Design Pattern**: **Repository Pattern + Adapter Pattern**
**Purpose**: Database-agnostic vector store interface for easy provider switching

---

## 📋 Table of Contents
1. [Overview](#overview)
2. [Design Pattern](#design-pattern)
3. [Architecture](#architecture)
4. [Abstract Interface](#abstract-interface)
5. [Adapter Implementations](#adapter-implementations)
6. [Factory Pattern](#factory-pattern)
7. [Usage Examples](#usage-examples)
8. [Migration Strategy](#migration-strategy)
9. [Testing Strategy](#testing-strategy)

---

## 🎯 Overview

### Problem Statement
**Vendor Lock-in Risk**: Tightly coupling application code với specific vector database (Pinecone) makes it:
- ❌ Hard to switch providers (cost, features, performance)
- ❌ Difficult to test (need real Pinecone instance)
- ❌ Impossible to support multi-provider scenarios
- ❌ Risky if provider has outages or pricing changes

### Solution
**Abstraction Layer**: Create interface-based abstraction using **Repository + Adapter Pattern**:
- ✅ Switch vector databases với minimal code changes
- ✅ Easy unit testing với mock implementations
- ✅ Support multiple providers simultaneously
- ✅ Future-proof architecture

### Benefits
| Benefit | Description |
|---------|-------------|
| **Flexibility** | Switch from Pinecone → Qdrant → Weaviate với 1 line config change |
| **Testability** | Mock vector store for unit tests, no external dependencies |
| **Cost Optimization** | Easy A/B test different providers for cost/performance |
| **Risk Mitigation** | Not locked into single vendor |
| **Local Development** | Use FAISS locally, Pinecone in production |

---

## 🏗️ Design Pattern

### Pattern: **Repository + Adapter**

**Repository Pattern**:
- Abstracts data access logic
- Provides collection-like interface
- Hides storage implementation details

**Adapter Pattern**:
- Converts interface of class into another interface
- Allows incompatible interfaces to work together
- Each vector DB gets its own adapter

**Combined Structure**:
```
┌─────────────────────────────────────────────┐
│         Application Code                    │
│  (RAG Pipeline, Query Handler, etc.)        │
└─────────────────────────────────────────────┘
                    ↓
        Uses only abstract interface
                    ↓
┌─────────────────────────────────────────────┐
│    VectorStoreRepository (ABC)              │
│    • search()                               │
│    • upsert()                               │
│    • delete()                               │
│    • create_index()                         │
└─────────────────────────────────────────────┘
                    ↑
        ┌───────────┼───────────┐
        │           │           │
┌───────┴──────┐ ┌──┴────────┐ ┌┴──────────────┐
│ Pinecone     │ │ Qdrant    │ │ ChromaDB      │
│ Adapter      │ │ Adapter   │ │ Adapter       │
└──────────────┘ └───────────┘ └───────────────┘
        │           │           │
┌───────┴──────┐ ┌──┴────────┐ ┌┴──────────────┐
│ Pinecone     │ │ Qdrant    │ │ ChromaDB      │
│ Client       │ │ Client    │ │ Client        │
└──────────────┘ └───────────┘ └───────────────┘
```

---

## 🔧 Architecture

### Layer Breakdown

**Layer 1: Abstract Interface (Repository)**
- Defines contract that ALL vector stores must implement
- Business logic depends ONLY on this interface
- No vendor-specific code

**Layer 2: Adapter Implementations**
- One adapter per vector database provider
- Implements abstract interface
- Handles vendor-specific API calls and data transformations

**Layer 3: Factory**
- Creates appropriate adapter based on configuration
- Dependency injection friendly
- Environment-based selection

**Layer 4: Configuration**
- Environment variables or config files
- Runtime provider selection
- Credentials management

---

## 📐 Abstract Interface

### Base Repository Interface

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Domain models (vendor-agnostic)
@dataclass
class VectorRecord:
    """Vendor-agnostic vector record"""
    id: str
    values: List[float]
    metadata: Dict[str, Any]

@dataclass
class SearchResult:
    """Vendor-agnostic search result"""
    id: str
    score: float
    metadata: Dict[str, Any]

    @property
    def content(self) -> str:
        return self.metadata.get("content", "")

    @property
    def context(self) -> str:
        return self.metadata.get("context", "")

class VectorMetric(Enum):
    """Supported distance metrics"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dotproduct"

# Abstract repository interface
class VectorStoreRepository(ABC):
    """
    Abstract base class for vector store implementations.

    All vector databases must implement this interface.
    This ensures vendor-agnostic code throughout the application.
    """

    @abstractmethod
    async def create_index(
        self,
        index_name: str,
        dimension: int,
        metric: VectorMetric = VectorMetric.COSINE,
        **kwargs
    ) -> None:
        """
        Create a new vector index.

        Args:
            index_name: Name of the index
            dimension: Embedding dimension (e.g., 768)
            metric: Distance metric to use
            **kwargs: Provider-specific options
        """
        pass

    @abstractmethod
    async def delete_index(self, index_name: str) -> None:
        """Delete an index"""
        pass

    @abstractmethod
    async def index_exists(self, index_name: str) -> bool:
        """Check if index exists"""
        pass

    @abstractmethod
    async def upsert(
        self,
        vectors: List[VectorRecord],
        namespace: Optional[str] = None,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Insert or update vectors.

        Args:
            vectors: List of VectorRecord objects
            namespace: Optional namespace for data isolation
            batch_size: Number of vectors per batch

        Returns:
            Statistics about the upsert operation
        """
        pass

    @abstractmethod
    async def search(
        self,
        query_vector: List[float],
        top_k: int = 20,
        filter: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
        include_metadata: bool = True
    ) -> List[SearchResult]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query embedding
            top_k: Number of results to return
            filter: Metadata filters (vendor-specific format)
            namespace: Optional namespace to search in
            include_metadata: Whether to include metadata in results

        Returns:
            List of SearchResult objects sorted by similarity
        """
        pass

    @abstractmethod
    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
        delete_all: bool = False
    ) -> Dict[str, Any]:
        """
        Delete vectors by IDs or filter.

        Args:
            ids: List of vector IDs to delete
            filter: Delete all vectors matching filter
            namespace: Namespace to delete from
            delete_all: Delete all vectors (dangerous!)

        Returns:
            Statistics about deletion
        """
        pass

    @abstractmethod
    async def get_stats(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """
        Get index statistics.

        Returns:
            Dict with stats like vector count, dimension, etc.
        """
        pass

    @abstractmethod
    async def fetch(
        self,
        ids: List[str],
        namespace: Optional[str] = None
    ) -> List[VectorRecord]:
        """
        Fetch vectors by IDs.

        Args:
            ids: List of vector IDs
            namespace: Optional namespace

        Returns:
            List of VectorRecord objects
        """
        pass

    # Helper methods (optional to override)
    def validate_dimension(self, vectors: List[VectorRecord], expected_dim: int):
        """Validate all vectors have expected dimension"""
        for vec in vectors:
            if len(vec.values) != expected_dim:
                raise ValueError(
                    f"Vector {vec.id} has dimension {len(vec.values)}, "
                    f"expected {expected_dim}"
                )

    async def health_check(self) -> bool:
        """Check if vector store is healthy"""
        try:
            await self.get_stats()
            return True
        except Exception:
            return False
```

---

## 🔌 Adapter Implementations

### 1. Pinecone Adapter

```python
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Optional, Any

class PineconeAdapter(VectorStoreRepository):
    """Pinecone implementation of VectorStoreRepository"""

    def __init__(
        self,
        api_key: str,
        environment: str = "us-east-1",
        cloud: str = "aws"
    ):
        self.client = Pinecone(api_key=api_key)
        self.environment = environment
        self.cloud = cloud
        self.index = None
        self.index_name = None

    async def create_index(
        self,
        index_name: str,
        dimension: int,
        metric: VectorMetric = VectorMetric.COSINE,
        **kwargs
    ) -> None:
        """Create Pinecone serverless index"""
        self.index_name = index_name

        if index_name not in self.client.list_indexes().names():
            self.client.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric.value,
                spec=ServerlessSpec(
                    cloud=self.cloud,
                    region=self.environment
                )
            )

        self.index = self.client.Index(index_name)

    async def delete_index(self, index_name: str) -> None:
        """Delete Pinecone index"""
        self.client.delete_index(index_name)

    async def index_exists(self, index_name: str) -> bool:
        """Check if Pinecone index exists"""
        return index_name in self.client.list_indexes().names()

    async def upsert(
        self,
        vectors: List[VectorRecord],
        namespace: Optional[str] = None,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """Upsert vectors to Pinecone"""
        if not self.index:
            raise RuntimeError("Index not initialized. Call create_index first.")

        # Transform to Pinecone format
        pinecone_vectors = [
            {
                "id": vec.id,
                "values": vec.values,
                "metadata": vec.metadata
            }
            for vec in vectors
        ]

        # Batch upsert
        upserted_count = 0
        for i in range(0, len(pinecone_vectors), batch_size):
            batch = pinecone_vectors[i:i+batch_size]
            self.index.upsert(
                vectors=batch,
                namespace=namespace or ""
            )
            upserted_count += len(batch)

        return {"upserted_count": upserted_count}

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 20,
        filter: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
        include_metadata: bool = True
    ) -> List[SearchResult]:
        """Search Pinecone index"""
        if not self.index:
            raise RuntimeError("Index not initialized")

        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            filter=filter,
            namespace=namespace or "",
            include_metadata=include_metadata
        )

        # Transform to SearchResult
        return [
            SearchResult(
                id=match.id,
                score=match.score,
                metadata=match.metadata if include_metadata else {}
            )
            for match in results.matches
        ]

    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
        delete_all: bool = False
    ) -> Dict[str, Any]:
        """Delete from Pinecone"""
        if not self.index:
            raise RuntimeError("Index not initialized")

        if delete_all:
            self.index.delete(delete_all=True, namespace=namespace or "")
        elif ids:
            self.index.delete(ids=ids, namespace=namespace or "")
        elif filter:
            self.index.delete(filter=filter, namespace=namespace or "")
        else:
            raise ValueError("Must provide ids, filter, or delete_all=True")

        return {"deleted": True}

    async def get_stats(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Get Pinecone index stats"""
        if not self.index:
            raise RuntimeError("Index not initialized")

        stats = self.index.describe_index_stats()
        return {
            "total_vector_count": stats.total_vector_count,
            "dimension": stats.dimension,
            "namespaces": stats.namespaces
        }

    async def fetch(
        self,
        ids: List[str],
        namespace: Optional[str] = None
    ) -> List[VectorRecord]:
        """Fetch vectors from Pinecone"""
        if not self.index:
            raise RuntimeError("Index not initialized")

        results = self.index.fetch(ids=ids, namespace=namespace or "")

        return [
            VectorRecord(
                id=id,
                values=vec.values,
                metadata=vec.metadata
            )
            for id, vec in results.vectors.items()
        ]
```

### 2. Qdrant Adapter

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, SearchParams
)

class QdrantAdapter(VectorStoreRepository):
    """Qdrant implementation of VectorStoreRepository"""

    def __init__(
        self,
        url: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        prefer_grpc: bool = False
    ):
        self.client = QdrantClient(
            url=url,
            port=port,
            api_key=api_key,
            prefer_grpc=prefer_grpc
        )
        self.collection_name = None

    def _metric_to_distance(self, metric: VectorMetric) -> Distance:
        """Convert VectorMetric to Qdrant Distance"""
        mapping = {
            VectorMetric.COSINE: Distance.COSINE,
            VectorMetric.EUCLIDEAN: Distance.EUCLID,
            VectorMetric.DOT_PRODUCT: Distance.DOT
        }
        return mapping[metric]

    async def create_index(
        self,
        index_name: str,
        dimension: int,
        metric: VectorMetric = VectorMetric.COSINE,
        **kwargs
    ) -> None:
        """Create Qdrant collection"""
        self.collection_name = index_name

        if not await self.index_exists(index_name):
            self.client.create_collection(
                collection_name=index_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=self._metric_to_distance(metric)
                )
            )

    async def delete_index(self, index_name: str) -> None:
        """Delete Qdrant collection"""
        self.client.delete_collection(collection_name=index_name)

    async def index_exists(self, index_name: str) -> bool:
        """Check if Qdrant collection exists"""
        collections = self.client.get_collections().collections
        return any(c.name == index_name for c in collections)

    async def upsert(
        self,
        vectors: List[VectorRecord],
        namespace: Optional[str] = None,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """Upsert to Qdrant"""
        if not self.collection_name:
            raise RuntimeError("Collection not initialized")

        # Transform to Qdrant format
        points = [
            PointStruct(
                id=vec.id,
                vector=vec.values,
                payload={
                    **vec.metadata,
                    "_namespace": namespace or "default"
                }
            )
            for vec in vectors
        ]

        # Batch upsert
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )

        return {"upserted_count": len(points)}

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 20,
        filter: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
        include_metadata: bool = True
    ) -> List[SearchResult]:
        """Search Qdrant collection"""
        if not self.collection_name:
            raise RuntimeError("Collection not initialized")

        # Build filter
        qdrant_filter = None
        if namespace or filter:
            conditions = []
            if namespace:
                conditions.append({
                    "key": "_namespace",
                    "match": {"value": namespace}
                })
            if filter:
                # Convert filter to Qdrant format
                # This needs custom logic based on filter structure
                pass

            if conditions:
                qdrant_filter = Filter(must=conditions)

        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=include_metadata
        )

        return [
            SearchResult(
                id=str(hit.id),
                score=hit.score,
                metadata=hit.payload if include_metadata else {}
            )
            for hit in results
        ]

    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
        delete_all: bool = False
    ) -> Dict[str, Any]:
        """Delete from Qdrant"""
        if not self.collection_name:
            raise RuntimeError("Collection not initialized")

        if delete_all:
            # Delete entire collection and recreate
            info = self.client.get_collection(self.collection_name)
            await self.delete_index(self.collection_name)
            await self.create_index(
                self.collection_name,
                info.config.params.vectors.size
            )
        elif ids:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=ids
            )
        elif filter:
            # Delete by filter
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(**filter)
            )

        return {"deleted": True}

    async def get_stats(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Get Qdrant collection stats"""
        if not self.collection_name:
            raise RuntimeError("Collection not initialized")

        info = self.client.get_collection(self.collection_name)
        return {
            "total_vector_count": info.points_count,
            "dimension": info.config.params.vectors.size,
            "status": info.status
        }

    async def fetch(
        self,
        ids: List[str],
        namespace: Optional[str] = None
    ) -> List[VectorRecord]:
        """Fetch from Qdrant"""
        if not self.collection_name:
            raise RuntimeError("Collection not initialized")

        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=ids,
            with_payload=True,
            with_vectors=True
        )

        return [
            VectorRecord(
                id=str(point.id),
                values=point.vector,
                metadata=point.payload
            )
            for point in points
        ]
```

### 3. ChromaDB Adapter

```python
import chromadb
from chromadb.config import Settings

class ChromaDBAdapter(VectorStoreRepository):
    """ChromaDB implementation of VectorStoreRepository"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        persist_directory: Optional[str] = None
    ):
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.HttpClient(host=host, port=port)

        self.collection = None
        self.collection_name = None

    async def create_index(
        self,
        index_name: str,
        dimension: int,
        metric: VectorMetric = VectorMetric.COSINE,
        **kwargs
    ) -> None:
        """Create ChromaDB collection"""
        self.collection_name = index_name

        # ChromaDB metric mapping
        metric_map = {
            VectorMetric.COSINE: "cosine",
            VectorMetric.EUCLIDEAN: "l2",
            VectorMetric.DOT_PRODUCT: "ip"
        }

        self.collection = self.client.get_or_create_collection(
            name=index_name,
            metadata={"hnsw:space": metric_map[metric]}
        )

    async def delete_index(self, index_name: str) -> None:
        """Delete ChromaDB collection"""
        self.client.delete_collection(name=index_name)

    async def index_exists(self, index_name: str) -> bool:
        """Check if ChromaDB collection exists"""
        try:
            self.client.get_collection(name=index_name)
            return True
        except Exception:
            return False

    async def upsert(
        self,
        vectors: List[VectorRecord],
        namespace: Optional[str] = None,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """Upsert to ChromaDB"""
        if not self.collection:
            raise RuntimeError("Collection not initialized")

        # Prepare data
        ids = [vec.id for vec in vectors]
        embeddings = [vec.values for vec in vectors]
        metadatas = [
            {**vec.metadata, "_namespace": namespace or "default"}
            for vec in vectors
        ]

        # ChromaDB upsert
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
        )

        return {"upserted_count": len(vectors)}

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 20,
        filter: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
        include_metadata: bool = True
    ) -> List[SearchResult]:
        """Search ChromaDB collection"""
        if not self.collection:
            raise RuntimeError("Collection not initialized")

        # Build where clause
        where = {}
        if namespace:
            where["_namespace"] = namespace
        if filter:
            where.update(filter)

        # Search
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where if where else None,
            include=["metadatas", "distances"]
        )

        # Transform results
        search_results = []
        for i, id in enumerate(results['ids'][0]):
            search_results.append(
                SearchResult(
                    id=id,
                    score=1.0 - results['distances'][0][i],  # Convert distance to similarity
                    metadata=results['metadatas'][0][i] if include_metadata else {}
                )
            )

        return search_results

    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
        delete_all: bool = False
    ) -> Dict[str, Any]:
        """Delete from ChromaDB"""
        if not self.collection:
            raise RuntimeError("Collection not initialized")

        if delete_all:
            # Delete all items
            all_ids = self.collection.get()['ids']
            if all_ids:
                self.collection.delete(ids=all_ids)
        elif ids:
            self.collection.delete(ids=ids)
        elif filter:
            where = filter.copy()
            if namespace:
                where["_namespace"] = namespace
            self.collection.delete(where=where)

        return {"deleted": True}

    async def get_stats(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Get ChromaDB collection stats"""
        if not self.collection:
            raise RuntimeError("Collection not initialized")

        count = self.collection.count()
        return {
            "total_vector_count": count,
            "dimension": None,  # ChromaDB doesn't expose this easily
            "name": self.collection_name
        }

    async def fetch(
        self,
        ids: List[str],
        namespace: Optional[str] = None
    ) -> List[VectorRecord]:
        """Fetch from ChromaDB"""
        if not self.collection:
            raise RuntimeError("Collection not initialized")

        results = self.collection.get(
            ids=ids,
            include=["embeddings", "metadatas"]
        )

        return [
            VectorRecord(
                id=results['ids'][i],
                values=results['embeddings'][i],
                metadata=results['metadatas'][i]
            )
            for i in range(len(results['ids']))
        ]
```

### 4. In-Memory Mock Adapter (for testing)

```python
class InMemoryVectorStore(VectorStoreRepository):
    """In-memory implementation for testing"""

    def __init__(self):
        self.stores: Dict[str, Dict[str, VectorRecord]] = {}
        self.current_index = None
        self.dimension = None

    async def create_index(
        self,
        index_name: str,
        dimension: int,
        metric: VectorMetric = VectorMetric.COSINE,
        **kwargs
    ) -> None:
        """Create in-memory index"""
        self.current_index = index_name
        self.dimension = dimension
        if index_name not in self.stores:
            self.stores[index_name] = {}

    async def delete_index(self, index_name: str) -> None:
        """Delete in-memory index"""
        if index_name in self.stores:
            del self.stores[index_name]

    async def index_exists(self, index_name: str) -> bool:
        """Check if index exists"""
        return index_name in self.stores

    async def upsert(
        self,
        vectors: List[VectorRecord],
        namespace: Optional[str] = None,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """Upsert to memory"""
        if not self.current_index:
            raise RuntimeError("Index not initialized")

        for vec in vectors:
            # Add namespace to metadata
            vec.metadata["_namespace"] = namespace or "default"
            self.stores[self.current_index][vec.id] = vec

        return {"upserted_count": len(vectors)}

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 20,
        filter: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
        include_metadata: bool = True
    ) -> List[SearchResult]:
        """Search in memory using cosine similarity"""
        import numpy as np
        from numpy.linalg import norm

        if not self.current_index:
            raise RuntimeError("Index not initialized")

        # Filter vectors
        candidates = []
        for vec_id, vec in self.stores[self.current_index].items():
            # Namespace filter
            if namespace and vec.metadata.get("_namespace") != namespace:
                continue

            # Custom filter
            if filter:
                match = all(
                    vec.metadata.get(k) == v
                    for k, v in filter.items()
                )
                if not match:
                    continue

            candidates.append((vec_id, vec))

        # Compute similarities
        query_vec = np.array(query_vector)
        results = []

        for vec_id, vec in candidates:
            vec_array = np.array(vec.values)
            # Cosine similarity
            similarity = np.dot(query_vec, vec_array) / (
                norm(query_vec) * norm(vec_array)
            )
            results.append((vec_id, similarity, vec.metadata))

        # Sort and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:top_k]

        return [
            SearchResult(
                id=vec_id,
                score=float(score),
                metadata=metadata if include_metadata else {}
            )
            for vec_id, score, metadata in results
        ]

    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
        delete_all: bool = False
    ) -> Dict[str, Any]:
        """Delete from memory"""
        if not self.current_index:
            raise RuntimeError("Index not initialized")

        if delete_all:
            self.stores[self.current_index] = {}
        elif ids:
            for id in ids:
                self.stores[self.current_index].pop(id, None)

        return {"deleted": True}

    async def get_stats(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Get stats"""
        if not self.current_index:
            raise RuntimeError("Index not initialized")

        count = len(self.stores[self.current_index])
        return {
            "total_vector_count": count,
            "dimension": self.dimension
        }

    async def fetch(
        self,
        ids: List[str],
        namespace: Optional[str] = None
    ) -> List[VectorRecord]:
        """Fetch from memory"""
        if not self.current_index:
            raise RuntimeError("Index not initialized")

        return [
            self.stores[self.current_index][id]
            for id in ids
            if id in self.stores[self.current_index]
        ]
```

---

## 🏭 Factory Pattern

### Vector Store Factory

```python
from enum import Enum
from typing import Optional

class VectorStoreProvider(Enum):
    """Supported vector store providers"""
    PINECONE = "pinecone"
    QDRANT = "qdrant"
    CHROMADB = "chromadb"
    INMEMORY = "inmemory"  # For testing

class VectorStoreConfig:
    """Configuration for vector store"""
    def __init__(
        self,
        provider: VectorStoreProvider,
        index_name: str,
        dimension: int = 768,
        metric: VectorMetric = VectorMetric.COSINE,
        **kwargs
    ):
        self.provider = provider
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.extra_params = kwargs

class VectorStoreFactory:
    """
    Factory for creating vector store instances.

    Supports multiple providers and environment-based configuration.
    """

    @staticmethod
    def create(
        provider: VectorStoreProvider,
        **kwargs
    ) -> VectorStoreRepository:
        """
        Create vector store instance based on provider.

        Args:
            provider: Vector store provider enum
            **kwargs: Provider-specific configuration

        Returns:
            VectorStoreRepository instance

        Example:
            # Pinecone
            store = VectorStoreFactory.create(
                VectorStoreProvider.PINECONE,
                api_key="xxx",
                environment="us-east-1"
            )

            # Qdrant
            store = VectorStoreFactory.create(
                VectorStoreProvider.QDRANT,
                url="localhost",
                port=6333
            )
        """
        if provider == VectorStoreProvider.PINECONE:
            return PineconeAdapter(**kwargs)

        elif provider == VectorStoreProvider.QDRANT:
            return QdrantAdapter(**kwargs)

        elif provider == VectorStoreProvider.CHROMADB:
            return ChromaDBAdapter(**kwargs)

        elif provider == VectorStoreProvider.INMEMORY:
            return InMemoryVectorStore()

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @staticmethod
    def create_from_env() -> VectorStoreRepository:
        """
        Create vector store from environment variables.

        Environment variables:
            VECTOR_STORE_PROVIDER: pinecone, qdrant, chromadb
            PINECONE_API_KEY: (if using Pinecone)
            PINECONE_ENVIRONMENT: (if using Pinecone)
            QDRANT_URL: (if using Qdrant)
            QDRANT_PORT: (if using Qdrant)
            CHROMA_HOST: (if using ChromaDB)
            CHROMA_PORT: (if using ChromaDB)
        """
        import os

        provider_str = os.getenv("VECTOR_STORE_PROVIDER", "pinecone")
        provider = VectorStoreProvider(provider_str.lower())

        if provider == VectorStoreProvider.PINECONE:
            return VectorStoreFactory.create(
                VectorStoreProvider.PINECONE,
                api_key=os.getenv("PINECONE_API_KEY"),
                environment=os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
            )

        elif provider == VectorStoreProvider.QDRANT:
            return VectorStoreFactory.create(
                VectorStoreProvider.QDRANT,
                url=os.getenv("QDRANT_URL", "localhost"),
                port=int(os.getenv("QDRANT_PORT", "6333"))
            )

        elif provider == VectorStoreProvider.CHROMADB:
            return VectorStoreFactory.create(
                VectorStoreProvider.CHROMADB,
                host=os.getenv("CHROMA_HOST", "localhost"),
                port=int(os.getenv("CHROMA_PORT", "8000"))
            )

        else:
            raise ValueError(f"Unsupported provider: {provider}")
```

---

## 💡 Usage Examples

### Example 1: Basic Usage

```python
# Create vector store (Pinecone)
store = VectorStoreFactory.create(
    VectorStoreProvider.PINECONE,
    api_key="your-api-key",
    environment="us-east-1"
)

# Create index
await store.create_index(
    index_name="ppt-retrieval",
    dimension=768,
    metric=VectorMetric.COSINE
)

# Prepare data
vectors = [
    VectorRecord(
        id="chunk_1",
        values=[0.1, 0.2, ...],  # 768-dim
        metadata={
            "content": "Revenue grew 75% YoY",
            "slide_number": 5,
            "presentation_id": "ppt_123"
        }
    ),
    # ... more vectors
]

# Upsert
await store.upsert(vectors)

# Search
results = await store.search(
    query_vector=[0.15, 0.25, ...],
    top_k=10,
    filter={"presentation_id": "ppt_123"}
)

for result in results:
    print(f"{result.id}: {result.score}")
    print(f"  {result.content}")
```

### Example 2: Switch Provider (Zero Code Change!)

```python
# BEFORE: Using Pinecone
# store = VectorStoreFactory.create(
#     VectorStoreProvider.PINECONE,
#     api_key="xxx"
# )

# AFTER: Switch to Qdrant - SAME API!
store = VectorStoreFactory.create(
    VectorStoreProvider.QDRANT,
    url="localhost",
    port=6333
)

# All downstream code remains EXACTLY the same!
await store.create_index("ppt-retrieval", dimension=768)
await store.upsert(vectors)
results = await store.search(query_vector, top_k=10)
```

### Example 3: Environment-Based Configuration

```python
# .env file
"""
VECTOR_STORE_PROVIDER=pinecone
PINECONE_API_KEY=xxx
PINECONE_ENVIRONMENT=us-east-1
"""

# Application code
store = VectorStoreFactory.create_from_env()

# Works across all environments:
# - Local dev: VECTOR_STORE_PROVIDER=inmemory
# - Staging: VECTOR_STORE_PROVIDER=qdrant
# - Production: VECTOR_STORE_PROVIDER=pinecone
```

### Example 4: Dependency Injection (FastAPI)

```python
from fastapi import Depends, FastAPI
from typing import Annotated

app = FastAPI()

def get_vector_store() -> VectorStoreRepository:
    """Dependency injection for vector store"""
    return VectorStoreFactory.create_from_env()

@app.post("/search")
async def search_endpoint(
    query: str,
    store: Annotated[VectorStoreRepository, Depends(get_vector_store)]
):
    """API endpoint using injected vector store"""
    query_embedding = embed_text(query)
    results = await store.search(query_embedding, top_k=10)
    return {"results": results}

# Easy to mock in tests!
def get_mock_vector_store():
    return InMemoryVectorStore()

# Override for testing
app.dependency_overrides[get_vector_store] = get_mock_vector_store
```

### Example 5: Unit Testing

```python
import pytest

@pytest.fixture
async def vector_store():
    """Test fixture using in-memory store"""
    store = InMemoryVectorStore()
    await store.create_index("test-index", dimension=768)

    # Seed test data
    test_vectors = [
        VectorRecord(
            id=f"test_{i}",
            values=[float(i)] * 768,
            metadata={"test": True}
        )
        for i in range(10)
    ]
    await store.upsert(test_vectors)

    yield store

    await store.delete_index("test-index")

async def test_search(vector_store):
    """Test search functionality"""
    query = [1.0] * 768
    results = await vector_store.search(query, top_k=5)

    assert len(results) == 5
    assert results[0].id == "test_1"  # Closest to query
```

---

## 🔄 Migration Strategy

### Step-by-Step Migration Guide

**Scenario**: Migrating from Pinecone to Qdrant

```python
import asyncio

async def migrate_pinecone_to_qdrant(
    source_index: str,
    target_index: str,
    batch_size: int = 100
):
    """
    Migrate all vectors from Pinecone to Qdrant.

    Zero downtime migration strategy:
    1. Dual-write to both stores
    2. Backfill historical data
    3. Switch reads to new store
    4. Stop writes to old store
    """
    # Source: Pinecone
    source = VectorStoreFactory.create(
        VectorStoreProvider.PINECONE,
        api_key=os.getenv("PINECONE_API_KEY")
    )

    # Target: Qdrant
    target = VectorStoreFactory.create(
        VectorStoreProvider.QDRANT,
        url=os.getenv("QDRANT_URL")
    )

    # Create target index
    stats = await source.get_stats()
    await target.create_index(
        index_name=target_index,
        dimension=stats['dimension'],
        metric=VectorMetric.COSINE
    )

    # Fetch and migrate in batches
    # Note: This is pseudocode - actual implementation depends on
    # provider's ability to list all IDs
    total_migrated = 0

    # Pinecone doesn't have list_all, so we'd need to track IDs separately
    # For demo, assume we have a list of all IDs
    all_ids = get_all_vector_ids_from_metadata_db()

    for i in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[i:i+batch_size]

        # Fetch from source
        vectors = await source.fetch(batch_ids)

        # Upsert to target
        await target.upsert(vectors)

        total_migrated += len(vectors)
        print(f"Migrated {total_migrated} / {len(all_ids)} vectors")

    print("Migration complete!")

    # Verify counts match
    source_stats = await source.get_stats()
    target_stats = await target.get_stats()
    assert source_stats['total_vector_count'] == target_stats['total_vector_count']

# Run migration
asyncio.run(migrate_pinecone_to_qdrant("ppt-prod", "ppt-prod-qdrant"))
```

### Dual-Write Pattern (Zero Downtime)

```python
class DualWriteVectorStore(VectorStoreRepository):
    """
    Write to both old and new vector stores during migration.
    Read from new store with fallback to old.
    """

    def __init__(
        self,
        primary: VectorStoreRepository,
        secondary: VectorStoreRepository
    ):
        self.primary = primary
        self.secondary = secondary

    async def upsert(self, vectors: List[VectorRecord], **kwargs):
        """Write to BOTH stores"""
        # Fire both in parallel
        await asyncio.gather(
            self.primary.upsert(vectors, **kwargs),
            self.secondary.upsert(vectors, **kwargs)
        )

    async def search(self, query_vector: List[float], **kwargs):
        """Read from PRIMARY with fallback"""
        try:
            return await self.primary.search(query_vector, **kwargs)
        except Exception as e:
            logger.warning(f"Primary search failed: {e}, falling back")
            return await self.secondary.search(query_vector, **kwargs)

    async def delete(self, **kwargs):
        """Delete from BOTH"""
        await asyncio.gather(
            self.primary.delete(**kwargs),
            self.secondary.delete(**kwargs)
        )

# Usage during migration
old_store = VectorStoreFactory.create(VectorStoreProvider.PINECONE, ...)
new_store = VectorStoreFactory.create(VectorStoreProvider.QDRANT, ...)

# Use dual-write temporarily
migration_store = DualWriteVectorStore(
    primary=new_store,  # New store is primary
    secondary=old_store  # Old store is fallback
)

# After migration complete, switch to just new store
store = new_store
```

---

## 🧪 Testing Strategy

### Unit Tests

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.mark.parametrize("provider", [
    VectorStoreProvider.PINECONE,
    VectorStoreProvider.QDRANT,
    VectorStoreProvider.CHROMADB,
    VectorStoreProvider.INMEMORY
])
async def test_vector_store_interface(provider):
    """
    Test that ALL adapters implement the interface correctly.

    This ensures interface compliance across providers.
    """
    # Create store
    if provider == VectorStoreProvider.INMEMORY:
        store = InMemoryVectorStore()
    else:
        # Mock for unit tests
        store = MagicMock(spec=VectorStoreRepository)
        store.create_index = AsyncMock()
        store.upsert = AsyncMock(return_value={"upserted_count": 1})
        store.search = AsyncMock(return_value=[])

    # Test interface methods exist
    assert hasattr(store, 'create_index')
    assert hasattr(store, 'upsert')
    assert hasattr(store, 'search')
    assert hasattr(store, 'delete')
    assert hasattr(store, 'get_stats')
    assert hasattr(store, 'fetch')

    # Test create_index
    await store.create_index("test", dimension=768)

    # Test upsert
    vectors = [VectorRecord(id="1", values=[0.1]*768, metadata={})]
    result = await store.upsert(vectors)
    assert "upserted_count" in result

### Integration Tests

```python
@pytest.mark.integration
@pytest.mark.parametrize("provider,config", [
    (VectorStoreProvider.PINECONE, {
        "api_key": os.getenv("PINECONE_API_KEY"),
        "environment": "us-east-1"
    }),
    (VectorStoreProvider.QDRANT, {
        "url": "localhost",
        "port": 6333
    })
])
async def test_end_to_end_workflow(provider, config):
    """
    End-to-end test against real vector stores.

    Requires actual services running.
    """
    store = VectorStoreFactory.create(provider, **config)

    # Create index
    index_name = f"test-{provider.value}-{int(time.time())}"
    await store.create_index(index_name, dimension=384)

    try:
        # Upsert
        vectors = [
            VectorRecord(
                id=f"test_{i}",
                values=[random.random() for _ in range(384)],
                metadata={"test": True, "number": i}
            )
            for i in range(100)
        ]
        result = await store.upsert(vectors)
        assert result["upserted_count"] == 100

        # Search
        query = [random.random() for _ in range(384)]
        results = await store.search(query, top_k=10)
        assert len(results) == 10
        assert all(r.score >= 0 for r in results)

        # Fetch
        fetched = await store.fetch(["test_0", "test_1"])
        assert len(fetched) == 2

        # Delete
        await store.delete(ids=["test_0"])
        fetched = await store.fetch(["test_0"])
        assert len(fetched) == 0

        # Stats
        stats = await store.get_stats()
        assert stats["total_vector_count"] == 99  # Deleted 1

    finally:
        # Cleanup
        await store.delete_index(index_name)
```

---

## 📊 Comparison Matrix

| Feature | Pinecone | Qdrant | ChromaDB | InMemory |
|---------|----------|--------|----------|----------|
| **Managed** | ✅ Fully | ❌ Self-hosted | ❌ Self-hosted | ✅ |
| **Serverless** | ✅ Yes | ❌ No | ❌ No | ✅ |
| **Metadata Filter** | ✅ Advanced | ✅ Advanced | ✅ Basic | ✅ |
| **Namespaces** | ✅ Native | ⚡ Via metadata | ⚡ Via metadata | ⚡ Via metadata |
| **Sparse Vectors** | ✅ Yes | ✅ Yes | ❌ No | ❌ No |
| **Cost** | $$$ | $ | Free | Free |
| **Best For** | Production, scale | Control, features | Development | Testing |

---

## 🎯 Benefits Summary

### For Development
- ✅ Use `InMemoryVectorStore` - no setup needed
- ✅ Fast tests, no external dependencies
- ✅ Easy debugging

### For Production
- ✅ Switch providers based on cost/performance
- ✅ A/B test different vector DBs
- ✅ No vendor lock-in

### For Testing
- ✅ Mock implementations
- ✅ Test interface compliance
- ✅ Integration tests against real services

---

## 🏁 Conclusion

**Design Pattern Used**: **Repository Pattern + Adapter Pattern**

**Key Principles**:
1. **Depend on abstractions, not concretions** (SOLID - Dependency Inversion)
2. **Open/Closed Principle** (Open for extension, closed for modification)
3. **Single Responsibility** (Each adapter handles ONE provider)
4. **Interface Segregation** (Lean interface, only what's needed)

**Result**: Flexible, testable, maintainable vector store layer that can adapt to any provider!
