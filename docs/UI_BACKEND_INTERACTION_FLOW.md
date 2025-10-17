# UI-Backend Interaction Flow

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                          UI Layer                                │
│              (Streamlit / CLI / Web Application)                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │
          ┌───────────────────┴───────────────────┐
          │                                       │
          ▼                                       ▼
┌─────────────────────┐               ┌─────────────────────┐
│  Ingestion Backend  │               │  Retrieval Backend  │
│     (pipeline.py)   │               │(retrieval_pipeline) │
└─────────────────────┘               └─────────────────────┘
          │                                       │
          │                                       │
          ▼                                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Storage Backends                            │
│  ┌──────────────────┐           ┌────────────────────┐         │
│  │  BM25 Text Store │           │  Pinecone Vectors  │         │
│  │  (SQLite + dill) │           │  (Cloud Service)   │         │
│  └──────────────────┘           └────────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Ingestion Phase (Upload & Index)

### 1.1 Flow Diagram

```
UI Upload PPT File
    │
    ├─→ Create PPTContextualRetrievalPipeline
    │       - index_name (optional)
    │       - use_contextual (default: True)
    │       - use_vision (default: True)
    │
    ├─→ Call: await pipeline.index_presentation(ppt_path)
    │       │
    │       ├─→ Load PPT (PPTLoader)
    │       │       - Extract slides, images, notes
    │       │       - Vision analysis (if enabled)
    │       │       - Create overall_info document
    │       │
    │       ├─→ Extract whole document text
    │       │       - All slides content for context
    │       │
    │       ├─→ Contextual chunking
    │       │       - Split into ~400 token chunks
    │       │       - Generate LLM context for each chunk
    │       │       - Prepend context to chunk
    │       │
    │       ├─→ Index to BM25 Text Store
    │       │       - Save to SQLite (text_retriever.index_documents)
    │       │       - Build BM25 index
    │       │       - Serialize index to disk (dill)
    │       │       - presentation_id = filename stem
    │       │
    │       └─→ Index to Pinecone
    │               - Embed chunks (with cache)
    │               - Upload vectors to Pinecone
    │
    └─→ Return stats
            {
                "presentation_id": "ppt-report-2024",
                "presentation": "report-2024.pptx",
                "slides": 42,
                "chunks": 128,
                "indexed": True,
                "contextual": True,
                "vision_analyzed": True,
                "pinecone_index": "ppt-all-presentations",
                "bm25_backend": "bm25"
            }
```

### 1.2 UI Implementation Example

```python
import asyncio
from pathlib import Path
from src.pipeline import PPTContextualRetrievalPipeline

async def upload_and_index_presentation(ppt_file_path: str):
    """
    UI function to upload and index a PowerPoint presentation.

    Args:
        ppt_file_path: Path to uploaded .pptx file

    Returns:
        Indexing statistics
    """
    # Step 1: Create ingestion pipeline
    pipeline = PPTContextualRetrievalPipeline(
        # index_name will use default from .env (PINECONE_INDEX_NAME)
        use_contextual=True,  # Enable contextual chunking (recommended)
        use_vision=True       # Enable vision analysis for images
    )

    # Step 2: Index presentation
    try:
        stats = await pipeline.index_presentation(
            ppt_path=ppt_file_path,
            extract_images=True,
            include_notes=True
        )

        # Step 3: Display success
        print(f"✅ Indexed: {stats['presentation']}")
        print(f"   - Slides: {stats['slides']}")
        print(f"   - Chunks: {stats['chunks']}")
        print(f"   - Presentation ID: {stats['presentation_id']}")

        return stats

    except Exception as e:
        print(f"❌ Indexing failed: {str(e)}")
        raise

# Streamlit example
import streamlit as st

uploaded_file = st.file_uploader("Upload PowerPoint", type=['pptx'])
if uploaded_file:
    # Save uploaded file
    temp_path = f"temp/{uploaded_file.name}"
    Path("temp").mkdir(exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Index with progress
    with st.spinner("Indexing presentation..."):
        stats = asyncio.run(upload_and_index_presentation(temp_path))

    st.success(f"Indexed {stats['chunks']} chunks from {stats['slides']} slides")
```

### 1.3 Progress Tracking (Optional)

```python
# For real-time progress updates in UI
async def index_with_progress(ppt_path: str, progress_callback):
    """
    Index with progress callbacks for UI updates.

    Args:
        ppt_path: Path to PPT file
        progress_callback: Function to call with progress updates
    """
    pipeline = PPTContextualRetrievalPipeline()

    # Hook into pipeline stages (requires pipeline modification)
    # For now, just show loading states

    progress_callback("Loading presentation...", 0.1)
    # ... (pipeline would call this at each stage)

    stats = await pipeline.index_presentation(ppt_path)

    progress_callback("Indexing complete!", 1.0)
    return stats

# Streamlit usage
progress_bar = st.progress(0)
status_text = st.empty()

def update_progress(message, value):
    status_text.text(message)
    progress_bar.progress(value)

stats = asyncio.run(index_with_progress(ppt_path, update_progress))
```

---

## Phase 2: Retrieval Phase (Query & Answer)

### 2.1 Flow Diagram

```
UI Query Input
    │
    ├─→ Create RetrievalPipeline (ONCE per session)
    │       - index_name (optional: defaults from .env)
    │       - use_reranking (default: True)
    │       - Note: Always queries ALL presentations (cross-document search)
    │
    ├─→ Call: await retrieval.initialize() (ONCE)
    │       │
    │       ├─→ Load embeddings (with cache)
    │       │
    │       ├─→ Initialize text retriever
    │       │       - Load BM25 from SQLite (ALL documents)
    │       │       - Load serialized index (~3s for 60K chunks)
    │       │
    │       ├─→ Connect to Pinecone
    │       │       - PineconeVectorStore
    │       │
    │       ├─→ Create hybrid retriever
    │       │       - Combine Vector + BM25 + RRF
    │       │
    │       └─→ Create QA chain
    │               - LLM for answer generation
    │               - Chat memory for conversation
    │
    └─→ For each query:
            │
            ├─→ Call: await retrieval.query(question)
            │       │
            │       ├─→ Hybrid retrieval
            │       │       - Vector search (Pinecone)
            │       │       - BM25 search (text retriever)
            │       │       - RRF fusion
            │       │
            │       ├─→ Optional: Cohere reranking
            │       │
            │       ├─→ Generate answer
            │       │       - LLM with context
            │       │       - Include chat history
            │       │
            │       └─→ Return result
            │
            └─→ Display answer + sources
```

### 2.2 UI Implementation Example

```python
from src.retrieval_pipeline import RetrievalPipeline

class QuerySession:
    """
    UI session manager for querying indexed presentations.

    Note: Always queries ALL presentations (cross-document search).

    Usage:
        session = QuerySession()
        await session.initialize()
        result = await session.query("What is the revenue?")
    """

    def __init__(
        self,
        use_reranking: bool = True
    ):
        self.retrieval = RetrievalPipeline(
            use_reranking=use_reranking
        )
        self.initialized = False

    async def initialize(self):
        """Initialize retrieval pipeline (call once)."""
        if not self.initialized:
            await self.retrieval.initialize()
            self.initialized = True

    async def query(self, question: str, return_sources: bool = True):
        """
        Query the indexed presentations.

        Args:
            question: User question
            return_sources: Include source documents

        Returns:
            {
                "answer": "The revenue was $10M in Q4...",
                "source_documents": [...],
                "formatted_sources": [
                    {
                        "slide_number": 5,
                        "slide_title": "Q4 Results",
                        "content": "...",
                        "rrf_score": 0.85,
                        "presentation_id": "ppt-report-2024"
                    }
                ]
            }
        """
        if not self.initialized:
            raise RuntimeError("Call initialize() first")

        return await self.retrieval.query(question, return_sources)

    def clear_history(self):
        """Clear conversation history."""
        self.retrieval.clear_chat_history()

    async def get_stats(self):
        """Get retrieval statistics."""
        return await self.retrieval.get_stats()

    async def list_presentations(self):
        """List all indexed presentations."""
        return await self.retrieval.list_presentations()

# Streamlit example
import streamlit as st
import asyncio

# Initialize session (once per Streamlit session)
if "query_session" not in st.session_state:
    st.session_state.query_session = QuerySession()
    asyncio.run(st.session_state.query_session.initialize())

# Query input
question = st.text_input("Ask a question:")
if question:
    with st.spinner("Searching..."):
        result = asyncio.run(
            st.session_state.query_session.query(question)
        )

    # Display answer
    st.markdown("### Answer")
    st.write(result["answer"])

    # Display sources
    st.markdown("### Sources")
    for source in result["formatted_sources"]:
        with st.expander(
            f"Slide {source['slide_number']}: {source['slide_title']}"
        ):
            st.write(source["content"])
            st.caption(f"Score: {source['rrf_score']:.2f}")

# Clear history button
if st.button("Clear Chat History"):
    st.session_state.query_session.clear_history()
    st.success("History cleared")
```

### 2.3 Cross-Document Search (Query All Presentations)

**Note:** This is the ONLY query mode. The system always searches across ALL indexed presentations.

```python
async def query_all_presentations(question: str):
    """
    Query across ALL indexed presentations (cross-document search).

    Args:
        question: User question

    Returns:
        Query result with sources from multiple presentations
    """
    # Create retrieval pipeline (always queries all presentations)
    retrieval = RetrievalPipeline(
        use_reranking=True
    )

    # Initialize (loads BM25 for ALL presentations)
    await retrieval.initialize()

    # Query across all presentations
    result = await retrieval.query(question)

    # Group sources by presentation
    sources_by_pres = {}
    for source in result["formatted_sources"]:
        pres_id = source["presentation_id"]
        if pres_id not in sources_by_pres:
            sources_by_pres[pres_id] = []
        sources_by_pres[pres_id].append(source)

    return result, sources_by_pres

# Usage
result, grouped_sources = await query_all_presentations(
    "What are the revenue trends across all reports?"
)

# Display
print(result["answer"])
for pres_id, sources in grouped_sources.items():
    print(f"\nSources from {pres_id}:")
    for source in sources:
        print(f"  - Slide {source['slide_number']}: {source['content'][:100]}")
```

---

## Phase 3: Presentation Management

### 3.1 List All Presentations

```python
async def list_all_presentations():
    """
    Get list of all indexed presentations.

    Returns:
        [
            {
                "presentation_id": "ppt-report-2024",
                "name": "report-2024.pptx",
                "title": "Q4 Financial Report",
                "total_slides": 42,
                "total_chunks": 128,
                "indexed_at": "2025-01-17T10:30:00"
            },
            ...
        ]
    """
    # Create retrieval pipeline (just to access text_retriever)
    retrieval = RetrievalPipeline()
    await retrieval.initialize()

    # List presentations from BM25 store
    presentations = await retrieval.list_presentations()

    return presentations

# Streamlit UI for presentation listing
presentations = asyncio.run(list_all_presentations())

# Display all presentations (for user reference)
st.markdown("### 📚 Indexed Presentations")
for pres in presentations:
    st.write(f"- {pres['name']} ({pres['total_slides']} slides, {pres['total_chunks']} chunks)")

# Initialize session (always queries all presentations)
if "query_session" not in st.session_state:
    st.session_state.query_session = QuerySession()
    asyncio.run(st.session_state.query_session.initialize())

st.info("💡 Queries will search across ALL indexed presentations")
```

### 3.2 Get Statistics

```python
async def get_system_stats():
    """
    Get system-wide statistics.

    Returns:
        {
            "status": "ready",
            "backend": "bm25",
            "pinecone_index": "ppt-all-presentations",
            "total_presentations": 5,
            "total_documents": 640,
            "sqlite_size_mb": 15.3,
            "index_size_mb": 380.2,
            "index_loaded": True
        }
    """
    retrieval = RetrievalPipeline()
    await retrieval.initialize()

    stats = await retrieval.get_stats()

    return stats

# Display in UI
stats = asyncio.run(get_system_stats())
st.metric("Total Presentations", stats["total_presentations"])
st.metric("Total Chunks", stats["total_documents"])
st.metric("Storage Size", f"{stats['total_size_mb']} MB")
```

---

## Complete UI Example: Multi-Page Streamlit App

### app.py (Main)

```python
import streamlit as st

st.set_page_config(page_title="PPT RAG System", layout="wide")

# Sidebar navigation
page = st.sidebar.selectbox(
    "Navigation",
    ["Dashboard", "Ingest", "Query", "Manage"]
)

if page == "Dashboard":
    import pages.dashboard
    pages.dashboard.show()
elif page == "Ingest":
    import pages.ingest
    pages.ingest.show()
elif page == "Query":
    import pages.query
    pages.query.show()
elif page == "Manage":
    import pages.manage
    pages.manage.show()
```

### pages/ingest.py

```python
import streamlit as st
import asyncio
from pathlib import Path
from src.pipeline import PPTContextualRetrievalPipeline

def show():
    st.title("📤 Ingest Presentation")

    uploaded_file = st.file_uploader("Upload PowerPoint", type=['pptx'])

    col1, col2 = st.columns(2)
    use_contextual = col1.checkbox("Contextual Chunking", value=True)
    use_vision = col2.checkbox("Vision Analysis", value=True)

    if uploaded_file and st.button("Index Presentation"):
        # Save temp file
        temp_path = f"temp/{uploaded_file.name}"
        Path("temp").mkdir(exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Index
        with st.spinner("Indexing..."):
            pipeline = PPTContextualRetrievalPipeline(
                use_contextual=use_contextual,
                use_vision=use_vision
            )
            stats = asyncio.run(
                pipeline.index_presentation(temp_path)
            )

        # Success
        st.success("✅ Indexing complete!")
        st.json(stats)
```

### pages/query.py

```python
import streamlit as st
import asyncio
from src.retrieval_pipeline import RetrievalPipeline

def show():
    st.title("💬 Query Presentations")

    # Initialize session
    if "retrieval" not in st.session_state:
        st.session_state.retrieval = RetrievalPipeline()
        asyncio.run(st.session_state.retrieval.initialize())

    # Query input
    question = st.text_input("Ask a question:")

    if question:
        with st.spinner("Searching..."):
            result = asyncio.run(
                st.session_state.retrieval.query(question)
            )

        # Display answer
        st.markdown("### 💡 Answer")
        st.write(result["answer"])

        # Display sources
        st.markdown("### 📄 Sources")
        for source in result["formatted_sources"]:
            with st.expander(
                f"Slide {source['slide_number']}: {source['slide_title']}"
            ):
                st.write(source["content"])
                st.caption(
                    f"Score: {source['rrf_score']:.2f} | "
                    f"Presentation: {source['presentation_id']}"
                )

    # Clear history
    if st.button("Clear Chat History"):
        st.session_state.retrieval.clear_chat_history()
        st.success("History cleared")
```

### pages/manage.py

```python
import streamlit as st
import asyncio
from src.retrieval_pipeline import RetrievalPipeline

def show():
    st.title("🗂️ Manage Presentations")

    # Initialize
    if "retrieval_mgr" not in st.session_state:
        st.session_state.retrieval_mgr = RetrievalPipeline()
        asyncio.run(st.session_state.retrieval_mgr.initialize())

    # List presentations
    presentations = asyncio.run(
        st.session_state.retrieval_mgr.list_presentations()
    )

    # Display as table
    st.dataframe(presentations)

    # Statistics
    st.markdown("### 📊 Statistics")
    stats = asyncio.run(st.session_state.retrieval_mgr.get_stats())

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Presentations", stats["total_presentations"])
    col2.metric("Total Chunks", stats["total_documents"])
    col3.metric("Storage", f"{stats['total_size_mb']} MB")
```

---

## Error Handling

```python
from src.pipeline import PPTContextualRetrievalPipeline
from src.retrieval_pipeline import RetrievalPipeline

async def safe_index(ppt_path: str):
    """Index with error handling."""
    try:
        pipeline = PPTContextualRetrievalPipeline()
        stats = await pipeline.index_presentation(ppt_path)
        return {"success": True, "stats": stats}
    except FileNotFoundError:
        return {"success": False, "error": "File not found"}
    except ValueError as e:
        return {"success": False, "error": f"Invalid file: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"Indexing failed: {str(e)}"}

async def safe_query(question: str, retrieval: RetrievalPipeline):
    """Query with error handling."""
    try:
        result = await retrieval.query(question)
        return {"success": True, "result": result}
    except RuntimeError as e:
        return {"success": False, "error": "Pipeline not initialized"}
    except Exception as e:
        return {"success": False, "error": f"Query failed: {str(e)}"}
```

---

## Best Practices

### 1. Session Management
- **Initialize ONCE per session**: Create `RetrievalPipeline` once, reuse for multiple queries
- **Cache in Streamlit**: Use `st.session_state` to persist pipeline across reruns
- **Clear memory**: Call `clear_chat_history()` when switching presentations

### 2. Performance Optimization
- **Use async**: Always use `await` with async functions
- **Enable caching**: Set `ENABLE_EMBEDDING_CACHE=true` and `ENABLE_LLM_CACHE=true`
- **Batch queries**: If querying multiple times, reuse the same pipeline

### 3. Resource Management
- **Cleanup**: Close connections when done (pipelines don't require explicit cleanup currently)
- **Memory**: For large-scale deployments, consider separate workers for ingestion/retrieval

### 4. User Experience
- **Progress indicators**: Show loading states during indexing (can take 30-60s)
- **Error messages**: Display user-friendly errors, not raw exceptions
- **Source attribution**: Always show which slides/presentations answers came from

---

## Summary: Key API Surface

### Ingestion
```python
from src.pipeline import PPTContextualRetrievalPipeline

pipeline = PPTContextualRetrievalPipeline()
stats = await pipeline.index_presentation(ppt_path)
# Returns: {"presentation_id": "...", "slides": 42, "chunks": 128, ...}
```

### Retrieval
```python
from src.retrieval_pipeline import RetrievalPipeline

retrieval = RetrievalPipeline()
await retrieval.initialize()  # Call once

result = await retrieval.query(question)
# Returns: {"answer": "...", "formatted_sources": [...]}

retrieval.clear_chat_history()  # Reset conversation
```

### Management
```python
presentations = await retrieval.list_presentations()
# Returns: [{"presentation_id": "...", "name": "...", ...}]

stats = await retrieval.get_stats()
# Returns: {"total_presentations": 5, "total_documents": 640, ...}
```
