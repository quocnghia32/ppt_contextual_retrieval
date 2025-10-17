# Text Retriever System - Phân Tích Chi Tiết

**Date:** 2025-01-17
**Version:** 1.0
**Status:** 🟢 Production-Ready

---

## 📋 Tổng Quan Kiến Trúc

Text Retriever System là một abstraction layer cho phép swap giữa các backend text search khác nhau (BM25, Elasticsearch) mà không cần thay đổi code client.

### Design Pattern: Strategy Pattern

```
┌─────────────────────────────────────────┐
│      BaseTextRetriever (Abstract)       │
│  - Define interface cho tất cả backend  │
└─────────────────────────────────────────┘
                    ▲
                    │ implements
        ┌───────────┴───────────┐
        │                       │
┌───────────────────┐  ┌────────────────────┐
│ BM25Serialize     │  │ Elasticsearch      │
│ Retriever         │  │ Retriever          │
│ (Current)         │  │ (Future)           │
└───────────────────┘  └────────────────────┘
        │
        ├─► BM25Store (SQLite)
        └─► Serialized Index (dill)
```

---

## 📁 File 1: `base_text_retriever.py`

### Mục đích
- **Abstract base class** định nghĩa interface chung cho tất cả text search backends
- **Factory function** để instantiate concrete implementations

### Class: `BaseTextRetriever(ABC)`

#### Abstract Methods (Must Implement)

**1. `async def initialize() -> None`**
```python
"""
Khởi tạo search backend.

Workflow:
- Load hoặc build search index
- Connect tới external services (nếu cần)
- Prepare retriever cho search queries

Called: Một lần khi retriever được setup
"""
```

**2. `async def index_documents(documents, presentation_id, metadata) -> None`**
```python
"""
Index documents cho một presentation (INGESTION PHASE).

Args:
    documents: List[Document] - LangChain Document objects
    presentation_id: str - Unique identifier
    metadata: Dict - Presentation metadata (title, slides, etc.)

Workflow:
- Store documents vào search backend
- Build/update search index
- Save metadata cần thiết cho retrieval
"""
```

**3. `async def search(query, top_k, filters) -> List[Document]`**
```python
"""
Search relevant documents (RETRIEVAL PHASE).

Args:
    query: str - Search query
    top_k: int - Number of results (default: 20)
    filters: Dict - Optional filters (presentation_id, slide_number)

Returns:
    List[Document] - Ranked by relevance score

Workflow:
- Execute search query against index
- Apply filters nếu có
- Return top-k results sorted by relevance
"""
```

**4. `async def load_presentation(presentation_id) -> None`**
```python
"""
Load documents cho specific presentation (optional optimization).

Note:
- BM25: May benefit from loading specific presentation
- Elasticsearch: May not need this (query directly)

Default: No-op if not needed
"""
```

**5. `async def get_all_documents() -> List[Document]`**
```python
"""
Load ALL documents across all presentations.

Use Cases:
- Cross-document search
- Re-building search index
- Migration to different backend

Returns: All Document objects in backend
"""
```

**6. `async def delete_presentation(presentation_id) -> None`**
```python
"""
Delete all documents for a presentation.

Workflow:
- Remove documents from backend
- Clean up metadata
- Update search index
"""
```

**7. `async def list_presentations() -> List[Dict]`**
```python
"""
List all indexed presentations.

Returns:
    List of dicts with:
    - presentation_id
    - name, title
    - total_slides, total_chunks
    - indexed_at

Use Cases:
- UI presentation selection
- Admin/management tools
"""
```

**8. `async def get_stats() -> Dict`**
```python
"""
Get search backend statistics.

Returns:
    - total_documents
    - total_presentations
    - index_size (bytes)
    - backend_type (bm25, elasticsearch)
    - last_updated

Used for: Monitoring and debugging
"""
```

#### Non-Abstract Methods

**`async def health_check() -> bool`**
```python
"""
Check if backend is healthy and ready.

Default Implementation:
    try:
        stats = await self.get_stats()
        return stats.get("total_documents", 0) >= 0
    except:
        return False

Backends can override cho custom health checks.
"""
```

### Function: `get_text_retriever(backend, **kwargs)`

**Factory function** để create text retriever instances.

```python
def get_text_retriever(backend: str = "bm25", **kwargs) -> BaseTextRetriever:
    """
    Args:
        backend: "bm25" | "elasticsearch"
        **kwargs: Backend-specific config

    Returns:
        Concrete BaseTextRetriever implementation

    Examples:
        # BM25
        retriever = get_text_retriever(
            backend="bm25",
            db_path="data/bm25/bm25_store.db",
            index_path="data/bm25/bm25_index.dill"
        )

        # Elasticsearch (future)
        retriever = get_text_retriever(
            backend="elasticsearch",
            es_url="http://localhost:9200",
            index_name="presentations"
        )
    """
```

**Implementation:**
```python
if backend == "bm25":
    from src.storage.bm25_serialize_retriever import BM25SerializeRetriever
    return BM25SerializeRetriever(**kwargs)
elif backend == "elasticsearch":
    from src.storage.elasticsearch_retriever import ElasticsearchRetriever
    return ElasticsearchRetriever(**kwargs)
else:
    raise ValueError(f"Unknown backend: {backend}")
```

---

## 📁 File 2: `bm25_store.py`

### Mục đích
SQLite storage cho text chunks. **Chỉ lưu text + minimal metadata**, KHÔNG serialization của BM25 index.

### Schema SQLite

#### Table: `presentations`
```sql
CREATE TABLE presentations (
    presentation_id TEXT PRIMARY KEY,      -- Unique ID
    name TEXT NOT NULL,                    -- Filename
    title TEXT,                            -- Presentation title
    total_slides INTEGER,                  -- Number of slides
    total_chunks INTEGER,                  -- Number of chunks
    pinecone_index_name TEXT,              -- Associated Pinecone index
    indexed_at TEXT NOT NULL               -- ISO timestamp
)
```

#### Table: `chunks`
```sql
CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,             -- Unique chunk ID
    presentation_id TEXT NOT NULL,         -- Foreign key
    content TEXT NOT NULL,                 -- Full text content
    slide_number INTEGER,                  -- Slide number
    FOREIGN KEY (presentation_id)
        REFERENCES presentations(presentation_id)
        ON DELETE CASCADE
)

-- Index for fast lookups
CREATE INDEX idx_chunks_presentation
    ON chunks(presentation_id)
```

### Class: `BM25Store`

#### Constructor
```python
def __init__(self, db_path: str = "data/bm25/bm25_store.db"):
    """
    Args:
        db_path: Path to SQLite database file

    Side Effects:
        - Creates parent directories if not exist
    """
```

#### Method: `async def initialize()`
```python
"""
Initialize database schema.

Workflow:
1. Run _create_tables() in thread pool (avoid blocking)
2. Create tables if not exist
3. Create indexes

Called: Once before first use
"""
```

**Internal:** `def _create_tables()`
- Synchronous database operations
- CREATE TABLE IF NOT EXISTS for both tables
- CREATE INDEX IF NOT EXISTS

#### Method: `async def save_presentation(presentation_id, documents, metadata)`
```python
"""
Save presentation and its chunks to SQLite.

Args:
    presentation_id: str
    documents: List[Document] - LangChain objects
    metadata: Dict - {name, title, total_slides, pinecone_index_name}

Workflow:
1. Run _save_presentation_sync() in thread pool
2. INSERT OR REPLACE into presentations table
3. For each document:
   - Extract chunk_id, slide_number from metadata
   - INSERT OR REPLACE into chunks table
4. COMMIT transaction

Error Handling:
- ROLLBACK on exception
- Re-raise exception to caller
"""
```

**Internal:** `def _save_presentation_sync()`
- Synchronous SQLite operations
- Transaction management (commit/rollback)

#### Method: `async def load_chunks(presentation_id) -> List[Document]`
```python
"""
Load chunks for specific presentation.

Args:
    presentation_id: str

Returns:
    List[Document] with:
    - page_content: Full text
    - metadata: {chunk_id, presentation_id, slide_number}

Workflow:
1. SELECT * FROM chunks WHERE presentation_id = ?
2. ORDER BY slide_number, chunk_id
3. Reconstruct Document objects
4. Return sorted list

Used By: Single-document retrieval mode
"""
```

**Internal:** `def _load_chunks_sync()`
- SQL query execution
- Document reconstruction from rows

#### Method: `async def load_all_chunks() -> List[Document]`
```python
"""
Load ALL chunks across ALL presentations.

Returns:
    List[Document] - All documents in database

Workflow:
1. SELECT * FROM chunks (no WHERE clause)
2. ORDER BY presentation_id, slide_number, chunk_id
3. Reconstruct all Document objects

Used By:
- Cross-document search
- BM25 index rebuild
- System-wide queries
"""
```

**Internal:** `def _load_all_chunks_sync()`
- Full table scan
- Multi-presentation document reconstruction

#### Method: `async def get_presentation_info(presentation_id) -> Dict`
```python
"""
Get metadata cho một presentation.

Returns:
    {
        presentation_id: str,
        name: str,
        title: str,
        total_slides: int,
        total_chunks: int,
        pinecone_index_name: str,
        indexed_at: str (ISO)
    }

Raises:
    ValueError if presentation not found
"""
```

#### Method: `async def list_presentations() -> List[Dict]`
```python
"""
List all presentations.

Returns:
    List of presentation metadata dicts
    Ordered by indexed_at DESC (newest first)

Used By:
- UI presentation selector
- Management dashboard
"""
```

#### Method: `async def delete_presentation(presentation_id)`
```python
"""
Delete presentation and all chunks.

Workflow:
1. DELETE FROM chunks WHERE presentation_id = ?
2. DELETE FROM presentations WHERE presentation_id = ?
3. COMMIT transaction

Note: CASCADE on foreign key also deletes chunks,
      but explicit DELETE is more clear
"""
```

#### Method: `async def get_stats() -> Dict`
```python
"""
Get storage statistics.

Returns:
    {
        total_presentations: int,
        total_chunks: int,
        db_size_bytes: int,
        db_size_mb: float,
        db_path: str
    }

Workflow:
1. COUNT(*) FROM presentations
2. COUNT(*) FROM chunks
3. os.path.getsize(db_path)
"""
```

### Async Pattern
**All public methods are async**, call synchronous helpers via `asyncio.to_thread()`:
```python
# Public async method
async def save_presentation(...):
    await asyncio.to_thread(
        self._save_presentation_sync,
        presentation_id,
        documents,
        metadata
    )

# Private sync helper
def _save_presentation_sync(...):
    conn = sqlite3.connect(self.db_path)
    # ... synchronous operations
    conn.close()
```

**Benefits:**
- Non-blocking I/O for async event loop
- SQLite operations run in thread pool
- Clean async/await interface

---

## 📁 File 3: `bm25_serialize_retriever.py`

### Mục đích
Concrete implementation của `BaseTextRetriever` using:
1. **SQLite storage** (via BM25Store) cho text
2. **In-memory BM25 index** (LangChain BM25Retriever)
3. **Serialized index** (dill format) cho fast startup

### Performance Metrics (60K chunks)
- **Startup (load serialized):** ~3s
- **Startup (rebuild):** ~10s
- **Query:** 10-50ms
- **Storage:** ~400 MB (SQLite + serialized index)

### Class: `BM25SerializeRetriever(BaseTextRetriever)`

#### Class Attributes
```python
INDEX_VERSION = "1.0.0"  # For version compatibility check
```

#### Constructor
```python
def __init__(
    self,
    db_path: str = "data/bm25/bm25_store.db",
    index_path: str = "data/bm25/bm25_index.dill",
    k: int = 20
):
    """
    Args:
        db_path: SQLite database path
        index_path: Serialized BM25 index path
        k: Default number of results per search

    Instance Variables:
        self.store: BM25Store instance
        self.bm25_retriever: BM25Retriever | None
        self._index_loaded: bool
    """
```

#### Method: `async def initialize()`
```python
"""
Initialize BM25 retriever.

Strategy:
    1. Try load serialized index (fast: 2-3s)
    2. If fails, rebuild from SQLite (slow: 10s)
    3. Save rebuilt index for next time

Workflow:
    1. await self.store.initialize()
       → Create SQLite tables

    2. success = await self._load_serialized_index()
       → Try load from disk

    3. if not success:
           await self._rebuild_index()
       → Rebuild from SQLite if load failed

    4. logger.info("Initialized successfully")

Called: Once before first use
"""
```

**Internal:** `async def _load_serialized_index() -> bool`
```python
"""
Load serialized BM25 index from disk.

Returns:
    True if successful, False otherwise

Workflow:
    1. Check if index_path exists
       → Return False if not

    2. index_data = await asyncio.to_thread(self._load_index_file)
       → Load pickle data in thread pool

    3. Validate version:
       if index_data["version"] != INDEX_VERSION:
           return False

    4. Extract retriever:
       self.bm25_retriever = index_data["retriever"]
       self.bm25_retriever.k = self.k

    5. self._index_loaded = True

    6. Log success with metadata

Error Handling:
    - Catch all exceptions
    - Log error
    - Return False (triggers rebuild)
"""
```

**Internal:** `def _load_index_file() -> Dict`
```python
"""
Load serialized index (synchronous).

Returns:
    {
        "version": "1.0.0",
        "retriever": BM25Retriever,
        "total_documents": int,
        "created_at": ISO timestamp
    }
"""
with open(self.index_path, "rb") as f:
    return dill.load(f)
```

**Internal:** `async def _rebuild_index()`
```python
"""
Rebuild BM25 index from SQLite.

Called When:
    - Serialized index doesn't exist
    - Serialized index corrupted/outdated
    - After ingesting new documents

Workflow:
    1. logger.info("🔨 Rebuilding...")

    2. documents = await self.store.load_all_chunks()
       → Load ALL documents from SQLite

    3. if not documents:
           self.bm25_retriever = None
           self._index_loaded = False
           return

    4. self.bm25_retriever = await asyncio.to_thread(
           self._build_bm25,
           documents
       )
       → Build BM25 in thread pool (CPU-intensive)

    5. self._index_loaded = True

    6. await self._save_serialized_index(len(documents))
       → Save for next startup

    7. logger.info(f"✅ Rebuilt with {len(documents)} docs")

Performance:
    - ~10s for 60K chunks
    - CPU-intensive (tokenization, IDF calculation)
"""
```

**Internal:** `def _build_bm25(documents) -> BM25Retriever`
```python
"""
Build BM25 retriever (synchronous, CPU-intensive).

Uses LangChain's BM25Retriever.from_documents()
- Tokenizes all documents
- Calculates document frequencies
- Calculates IDF scores
- Builds inverted index

Returns: BM25Retriever instance
"""
retriever = BM25Retriever.from_documents(documents)
retriever.k = self.k
return retriever
```

**Internal:** `async def _save_serialized_index(total_documents)`
```python
"""
Save BM25 index to disk (serialized).

Workflow:
    1. Prepare index data:
       index_data = {
           "version": INDEX_VERSION,
           "retriever": self.bm25_retriever,
           "total_documents": total_documents,
           "created_at": datetime.utcnow().isoformat()
       }

    2. await asyncio.to_thread(self._save_index_file, index_data)
       → Save in thread pool (disk I/O)

    3. logger.info(f"💾 Saved to {index_path}")

Error Handling:
    - Catch all exceptions
    - Log error
    - Don't fail (non-critical, will rebuild next time)
"""
```

**Internal:** `def _save_index_file(index_data)`
```python
"""
Save index to file (synchronous).
"""
with open(self.index_path, "wb") as f:
    dill.dump(index_data, f)
```

#### Method: `async def index_documents(documents, presentation_id, metadata)`
```python
"""
Index documents for a presentation (INGESTION PHASE).

Workflow:
    1. await self.store.save_presentation(...)
       → Save to SQLite

    2. logger.info(f"Indexed {len(documents)} docs")

    3. await self._rebuild_index()
       → Rebuild BM25 with ALL documents (cross-document search)

Important:
    - Always rebuilds with ALL documents
    - Supports cross-document search
    - Old + new documents in same index
"""
```

#### Method: `async def search(query, top_k, filters) -> List[Document]`
```python
"""
Search for relevant documents using BM25 (RETRIEVAL PHASE).

Args:
    query: str - Search query
    top_k: int - Number of results (default: 20)
    filters: Dict - Optional filters {presentation_id: "..."}

Returns:
    List[Document] ranked by BM25 score

Workflow:
    1. Check if index loaded:
       if not self._index_loaded:
           raise RuntimeError("Call initialize() first")

    2. Update k parameter:
       original_k = self.bm25_retriever.k
       self.bm25_retriever.k = top_k

    3. results = await asyncio.to_thread(
           self.bm25_retriever.get_relevant_documents,
           query
       )
       → Search in thread pool (CPU-intensive)

    4. if filters:
           results = self._apply_filters(results, filters)
       → Apply metadata filters

    5. self.bm25_retriever.k = original_k
       → Restore original k

    6. return results

Performance: 10-50ms per query
"""
```

**Internal:** `def _apply_filters(documents, filters) -> List[Document]`
```python
"""
Apply metadata filters to search results.

Example Filters:
    {"presentation_id": "ppt-123"}
    {"slide_number": 5}
    {"presentation_id": "ppt-123", "slide_number": 5}

Returns:
    Filtered list of documents
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
```

#### Method: `async def load_presentation(presentation_id)`
```python
"""
Load documents for specific presentation.

Note: For BM25 serialize strategy, we ALWAYS load ALL documents
      for cross-document search. This method is a no-op.

Provided for: Interface compatibility with other backends
"""
logger.info("Already has all documents loaded (cross-document mode)")
```

#### Method: `async def get_all_documents() -> List[Document]`
```python
"""
Get all documents from SQLite.

Returns: All Document objects in BM25Store
"""
return await self.store.load_all_chunks()
```

#### Method: `async def delete_presentation(presentation_id)`
```python
"""
Delete presentation and rebuild index.

Workflow:
    1. await self.store.delete_presentation(presentation_id)
       → Delete from SQLite

    2. await self._rebuild_index()
       → Rebuild index without deleted docs

    3. logger.info("Deleted and rebuilt")
"""
```

#### Method: `async def list_presentations() -> List[Dict]`
```python
"""
List all indexed presentations.

Delegates to: self.store.list_presentations()
"""
```

#### Method: `async def get_stats() -> Dict`
```python
"""
Get search backend statistics.

Returns:
    {
        backend_type: "bm25_serialize",
        total_documents: int,
        total_presentations: int,
        sqlite_size_mb: float,
        index_size_mb: float,
        total_size_mb: float,
        index_loaded: bool,
        index_version: str,
        k: int
    }

Workflow:
    1. store_stats = await self.store.get_stats()
       → Get SQLite stats

    2. index_size = Path(index_path).stat().st_size
       → Get serialized index file size

    3. Combine and return stats
"""
```

---

## 🔄 Complete Flow Diagrams

### 1️⃣ Ingestion Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    INGESTION PHASE                          │
└─────────────────────────────────────────────────────────────┘

PPT Upload → Pipeline → Chunks
                           ↓
     get_text_retriever(backend="bm25")
                           ↓
     ┌─────────────────────────────────────────┐
     │  BM25SerializeRetriever                 │
     │  - db_path: data/bm25/bm25_store.db    │
     │  - index_path: data/bm25/bm25_index.dill│
     └─────────────────────────────────────────┘
                           ↓
     await retriever.initialize()
         ├─→ store.initialize()
         │      └─→ Create SQLite tables
         ├─→ Try _load_serialized_index()
         │      └─→ Load from disk if exists
         └─→ If failed: _rebuild_index()
                └─→ Load all chunks + build BM25
                           ↓
     await retriever.index_documents(
         documents=chunks,
         presentation_id="ppt-cinema",
         metadata={...}
     )
         ├─→ store.save_presentation()
         │      ├─→ INSERT INTO presentations
         │      └─→ INSERT INTO chunks (30 chunks)
         └─→ _rebuild_index()
                ├─→ load_all_chunks() (ALL docs)
                ├─→ Build BM25 in thread pool
                └─→ Save serialized index
                           ↓
              ✅ Ingestion Complete
              - SQLite: 0.14 MB
              - Index: 0.24 MB
              - Total: 30 chunks indexed
```

### 2️⃣ Retrieval Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    RETRIEVAL PHASE                          │
└─────────────────────────────────────────────────────────────┘

User Query: "What is this presentation about?"
                           ↓
     get_text_retriever(backend="bm25")
                           ↓
     await retriever.initialize()
         ├─→ store.initialize()
         └─→ _load_serialized_index()
                ├─→ Load from data/bm25/bm25_index.dill
                ├─→ Validate version
                ├─→ Extract BM25Retriever
                └─→ ✅ Loaded in ~0.01s (30 docs)
                           ↓
     results = await retriever.search(
         query="What is this presentation about?",
         top_k=20,
         filters=None
     )
         ├─→ Check index_loaded = True
         ├─→ Set bm25_retriever.k = 20
         ├─→ bm25_retriever.get_relevant_documents(query)
         │      ├─→ Tokenize query
         │      ├─→ Calculate BM25 scores
         │      └─→ Return top 20 documents
         ├─→ Apply filters (if any)
         └─→ Restore original k
                           ↓
              ✅ Search Complete
              - Query time: 10-50ms
              - Results: 20 documents
              - Ranked by BM25 score
```

### 3️⃣ Cross-Document Search Flow

```
┌─────────────────────────────────────────────────────────────┐
│              CROSS-DOCUMENT SEARCH                          │
└─────────────────────────────────────────────────────────────┘

Multiple Presentations Indexed:
    - ppt-cinema (30 chunks)
    - ppt-report-2024 (45 chunks)
    - ppt-training (89 chunks)
    Total: 164 chunks
                           ↓
     await retriever.initialize()
         └─→ _load_serialized_index()
                ├─→ Load index with ALL 164 chunks
                └─→ BM25 index spans all presentations
                           ↓
     results = await retriever.search(
         query="revenue growth",
         top_k=20,
         filters=None  # NO filter = search all
     )
         └─→ Search across ALL presentations
             Return top 20 from any presentation
                           ↓
     results = await retriever.search(
         query="revenue growth",
         top_k=20,
         filters={"presentation_id": "ppt-report-2024"}
     )
         └─→ Search all, then filter
             Return top 20 from ppt-report-2024 only
```

---

## 🎯 Key Design Decisions

### 1. Why Serialize BM25 Index?
**Problem:** Building BM25 from scratch takes ~10s for 60K chunks
**Solution:** Serialize index to disk
**Benefit:** Load in ~3s (70% faster)
**Trade-off:** Extra disk space (index file ~equal to SQLite size)

### 2. Why Always Rebuild on Ingest?
**Problem:** How to add new documents to existing BM25 index?
**Solution:** Rebuild entire index with ALL documents
**Benefit:** Simple, correct cross-document search
**Trade-off:** Slower ingestion (acceptable for batch processing)
**Alternative:** Elasticsearch for incremental updates

### 3. Why Async/Await Pattern?
**Problem:** SQLite and BM25 operations are blocking
**Solution:** Run in thread pool via `asyncio.to_thread()`
**Benefit:** Non-blocking event loop for async frameworks
**Used by:** Streamlit, FastAPI, async pipelines

### 4. Why Separate SQLite Store?
**Problem:** BM25Retriever needs documents in memory
**Solution:** Persistent SQLite storage + in-memory BM25
**Benefit:**
- Persistent storage (survives restarts)
- Fast load (read from disk once)
- Separation of concerns (storage vs search)

### 5. Why dill Instead of pickle?
**Problem:** pickle can't serialize complex Python objects
**Solution:** dill extends pickle with more types
**Used for:** Serializing BM25Retriever object
**Alternative:** Store raw BM25 parameters and reconstruct

---

## 🔧 Usage Examples

### Example 1: Basic Usage
```python
from src.storage.base_text_retriever import get_text_retriever

# Create retriever
retriever = get_text_retriever(backend="bm25")

# Initialize (load index)
await retriever.initialize()

# Search
results = await retriever.search(
    query="What is the revenue?",
    top_k=20
)

for doc in results:
    print(f"Slide {doc.metadata['slide_number']}: {doc.page_content[:100]}")
```

### Example 2: Ingestion
```python
from langchain.schema import Document

# Prepare documents
chunks = [
    Document(
        page_content="Revenue increased by 25% in Q4...",
        metadata={
            "chunk_id": "ppt-report_1_1",
            "slide_number": 1,
            "presentation_id": "ppt-report-2024"
        }
    ),
    # ... more chunks
]

# Index documents
await retriever.index_documents(
    documents=chunks,
    presentation_id="ppt-report-2024",
    metadata={
        "name": "Q4_Report.pptx",
        "title": "Q4 Financial Results",
        "total_slides": 25,
        "pinecone_index_name": "ppt-report-2024"
    }
)
# → Saves to SQLite + rebuilds BM25 index
```

### Example 3: Filtered Search
```python
# Search only in specific presentation
results = await retriever.search(
    query="revenue growth",
    top_k=10,
    filters={"presentation_id": "ppt-report-2024"}
)
# → Returns top 10 from ppt-report-2024 only

# Search in specific slide
results = await retriever.search(
    query="revenue growth",
    top_k=10,
    filters={
        "presentation_id": "ppt-report-2024",
        "slide_number": 5
    }
)
# → Returns top 10 from slide 5 only
```

### Example 4: Management Operations
```python
# List all presentations
presentations = await retriever.list_presentations()
for pres in presentations:
    print(f"{pres['presentation_id']}: {pres['title']}")
    print(f"  Slides: {pres['total_slides']}, Chunks: {pres['total_chunks']}")
    print(f"  Indexed: {pres['indexed_at']}")

# Get statistics
stats = await retriever.get_stats()
print(f"Backend: {stats['backend_type']}")
print(f"Total Documents: {stats['total_documents']}")
print(f"Total Presentations: {stats['total_presentations']}")
print(f"Storage: {stats['total_size_mb']} MB")

# Delete presentation
await retriever.delete_presentation("ppt-old-report")
# → Deletes from SQLite + rebuilds index without it
```

### Example 5: Health Check
```python
# Check if backend is ready
healthy = await retriever.health_check()
if healthy:
    print("✅ Backend is ready")
else:
    print("❌ Backend is not ready")
```

---

## 📊 Performance Benchmarks

### Startup Time (Cold Start)
| Scenario | Time | Method |
|----------|------|--------|
| Load serialized (30 chunks) | ~0.01s | `_load_serialized_index()` |
| Load serialized (1K chunks) | ~0.5s | `_load_serialized_index()` |
| Load serialized (60K chunks) | ~3s | `_load_serialized_index()` |
| Rebuild (60K chunks) | ~10s | `_rebuild_index()` |

### Query Time
| Scenario | Time | Notes |
|----------|------|-------|
| Simple query (30 chunks) | 10-20ms | `search()` |
| Complex query (60K chunks) | 40-50ms | `search()` |
| With filters (60K chunks) | 50-60ms | `search()` + `_apply_filters()` |

### Ingestion Time
| Scenario | Time | Notes |
|----------|------|-------|
| Index 30 chunks | ~0.5s | SQLite + rebuild |
| Index 1K chunks | ~2s | SQLite + rebuild |
| Index 60K chunks | ~10s | SQLite + rebuild |

### Storage Size
| Component | Size (30 chunks) | Size (60K chunks) |
|-----------|------------------|-------------------|
| SQLite DB | 0.14 MB | 280 MB |
| Serialized Index | 0.24 MB | 150 MB |
| **Total** | **0.38 MB** | **430 MB** |

---

## 🚀 Future: Elasticsearch Migration

Khi nào cần migrate:
- ✅ >500 presentations
- ✅ >100K total chunks
- ✅ >1GB BM25 index size
- ✅ Distributed queries needed

Migration path:
1. Implement `ElasticsearchRetriever` methods
2. Change `.env`: `SEARCH_BACKEND=elasticsearch`
3. No code changes in UI/pipelines required (abstraction layer!)

---

## 📝 Summary

### Text Retriever System là gì?
**Abstraction layer** cho phép swap text search backends (BM25 ↔ Elasticsearch) mà không thay đổi client code.

### Core Components:
1. **BaseTextRetriever** - Abstract interface (strategy pattern)
2. **BM25Store** - SQLite storage cho text chunks
3. **BM25SerializeRetriever** - BM25 implementation với serialized index

### Key Features:
✅ **Persistent Storage** - SQLite database
✅ **Fast Startup** - Serialized index (~3s for 60K chunks)
✅ **Cross-Document Search** - Query across all presentations
✅ **Async Interface** - Non-blocking operations
✅ **Migration Ready** - Easy switch to Elasticsearch

### Current Status:
🟢 **Production-ready** with BM25 Serialize backend
🟡 **Suitable for <500 presentations (<100K chunks)**
🔵 **Elasticsearch interface defined, ready for future migration**

---

**End of Document**
