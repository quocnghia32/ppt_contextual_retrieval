# Codebase Understanding Guide

**Complete walkthrough để hiểu PPT Contextual Retrieval codebase từ A-Z**

---

## 📚 Table of Contents

1. [Quick Overview](#quick-overview)
2. [Codebase Structure](#codebase-structure)
3. [Core Concepts](#core-concepts)
4. [Data Flow - End to End](#data-flow---end-to-end)
5. [Reading Guide - Where to Start](#reading-guide---where-to-start)
6. [Component Deep Dive](#component-deep-dive)
7. [Common Scenarios](#common-scenarios)
8. [Debugging Guide](#debugging-guide)

---

## 🎯 Quick Overview

### What This System Does

**Input:** PowerPoint file (.pptx)
**Process:** Extract → Contextualize → Embed → Index → Retrieve → Answer
**Output:** Accurate answers to questions about the presentation

### The Big Innovation

**Traditional RAG:**
```
Slide text → Split into chunks → Embed → Search
❌ Loses context about where chunk is in presentation
```

**Our Approach (Contextual Retrieval):**
```
Slide text → Split into chunks → Generate context with LLM → Prepend context → Embed → Search
✅ Each chunk knows its role: "This is from Slide 5 in the Financial Results section..."
✅ 35% better accuracy
```

---

## 📁 Codebase Structure

```
ppt_context_retrieval/
├── src/                          # Core library code
│   ├── config.py                 # 🔑 Configuration (START HERE)
│   ├── pipeline.py               # 🔑 Main orchestrator (IMPORTANT)
│   │
│   ├── loaders/
│   │   └── ppt_loader.py         # Extract data from .pptx files
│   │
│   ├── splitters/
│   │   └── contextual_splitter.py # 🔑 THE CORE - Generate context for chunks
│   │
│   ├── models/
│   │   └── vision_analyzer.py    # Analyze images with GPT-4o mini
│   │
│   ├── retrievers/
│   │   └── hybrid_retriever.py   # Vector + BM25 + RRF search
│   │
│   ├── chains/
│   │   └── qa_chain.py           # Q&A with conversation memory
│   │
│   └── utils/
│       ├── rate_limiter.py       # Rate limiting with retry
│       └── caching.py            # 3-layer caching system
│
├── scripts/                      # CLI tools
│   ├── ingest.py                 # Index PPT files
│   ├── chat.py                   # Query indexed presentations
│   └── list_indexes.py           # View all indexes
│
├── streamlit_app/
│   └── app.py                    # Web UI
│
├── docs/                         # Documentation
│   ├── ppt_contextual_retrieval_design.md
│   ├── langchain_implementation.md
│   ├── prompts.md
│   ├── CACHING.md
│   └── CODEBASE_GUIDE.md         # This file
│
├── data/                         # Data directories
│   ├── uploads/                  # Uploaded PPT files
│   └── cache/                    # Embedding & LLM caches
│       ├── embeddings/
│       └── llm_responses/
│
├── .env                          # API keys (not in repo)
├── .env.example                  # Template
├── requirements.txt
└── README.md
```

---

## 🧠 Core Concepts

### 1. Contextual Retrieval (The Heart of the System)

**Problem:** When you split a presentation into chunks, each chunk loses context.

**Example:**
```
Slide 12: "Revenue increased by 23%"
```

Without context, search engines don't know:
- Is this Q1, Q2, Q3, or Q4?
- Is this total revenue or a specific segment?
- What was the previous quarter?

**Solution:** Generate LLM context before embedding:

```python
# Original chunk
chunk = "Revenue increased by 23%"

# LLM generates context
context = """
This is from Slide 12 'Q4 Financial Results' in the Financial Performance section.
Previous slide discussed Q3 results showing 15% growth.
Next slide breaks down revenue by segment.
"""

# Prepend context before embedding
to_embed = context + "\n\n" + chunk
```

**Implementation:** `src/splitters/contextual_splitter.py` - Read this file carefully!

### 2. Hybrid Search (Triple Retrieval)

**Three retrieval methods working together:**

```python
# 1. Vector Search (Semantic)
vector_results = pinecone.search(query_embedding)
# Finds: "revenue growth" matches "sales increase"

# 2. BM25 Search (Keyword)
bm25_results = bm25.search(query)
# Finds: Exact keyword matches "revenue"

# 3. Reciprocal Rank Fusion (Merge)
merged = RRF(vector_results, bm25_results)
# Combines strengths of both
```

**Why it works:** Vector catches semantics, BM25 catches exact terms. Together = best results.

**Implementation:** `src/retrievers/hybrid_retriever.py`

### 3. Three-Layer Caching

**Layer 1: Embedding Cache**
- Cache embeddings to disk
- Same text → Same embedding → Load from cache
- 75x faster on repeat

**Layer 2: LLM Response Cache**
- Cache all LLM outputs (context generation, answers)
- Same prompt → Same response → Load from cache
- 250x faster on repeat

**Layer 3: OpenAI Prompt Caching (Automatic)**
- Server-side caching for prompts >1024 tokens
- 50% discount on cached tokens
- Works automatically if prompts structured correctly

**Implementation:** `src/utils/caching.py`

### 4. Rate Limiting

**Why needed:** API providers have rate limits (requests/min, tokens/min)

**How it works:**
```python
# Before every API call
await rate_limiter.wait_if_needed(
    key="openai",
    estimated_tokens=300
)

# Checks:
# - Am I over requests/minute? → Wait
# - Am I over tokens/minute? → Wait
# - OK? → Proceed

# If rate limited by API:
@with_retry(max_attempts=3)  # Auto retry with backoff
async def call_api():
    ...
```

**Implementation:** `src/utils/rate_limiter.py`

---

## 🔄 Data Flow - End to End

### Flow 1: Ingestion (Index a PPT)

```
User runs: python scripts/ingest.py presentation.pptx

1️⃣ PPTLoader (src/loaders/ppt_loader.py)
   ├── Open .pptx file
   ├── Extract slides (text, images, tables, notes)
   ├── Detect sections (Introduction, Results, etc.)
   └── Create LangChain Document per slide

2️⃣ VisionAnalyzer (src/models/vision_analyzer.py) [if --vision enabled]
   ├── Find images in slides
   ├── Call GPT-4o mini to analyze each image
   ├── Extract: chart type, data points, insights
   └── Add to slide metadata

3️⃣ ContextualTextSplitter (src/splitters/contextual_splitter.py) ⭐ CRITICAL
   ├── Split each slide into chunks (~400 tokens)
   │   └── Sentence-based splitting with overlap
   │
   ├── For each chunk:
   │   ├── Build prompt with presentation context
   │   ├── Call LLM (OpenAI or Anthropic)
   │   ├── Generate 50-100 token context
   │   └── Prepend context to chunk
   │
   └── Batch process with rate limiting

4️⃣ Embeddings (via src/utils/caching.py)
   ├── get_cached_embeddings() → CacheBackedEmbeddings
   ├── For each chunk:
   │   ├── Check cache (disk)
   │   ├── If cached: Load instantly
   │   └── If not cached: Call OpenAI → Cache result
   └── Returns: List of vectors

5️⃣ Pinecone Indexing (src/pipeline.py)
   ├── Check if index exists
   │   └── If not: Create index (dimension=1536, metric=cosine)
   │
   ├── Add all chunks to Pinecone
   │   └── Each chunk: {id, vector, metadata}
   │
   └── Index ready for queries

✅ Done: Presentation indexed in Pinecone
```

### Flow 2: Query (Ask a Question)

```
User runs: python scripts/chat.py --index ppt-presentation --query "What was the revenue?"

1️⃣ Query Embeddings (src/utils/caching.py)
   ├── Embed user question
   ├── Check cache (same question asked before?)
   └── Get query vector

2️⃣ Hybrid Retrieval (src/retrievers/hybrid_retriever.py)
   ├── Vector Search:
   │   └── Pinecone.search(query_vector, top_k=20)
   │
   ├── BM25 Search:
   │   └── BM25.search(query_text, top_k=20)
   │
   └── Reciprocal Rank Fusion:
       ├── Merge results from both
       ├── Score = 1/(k + rank) for each retriever
       └── Re-rank by combined score

3️⃣ Reranking [if Cohere enabled]
   ├── Take top 20 from hybrid search
   ├── Cohere.rerank(query, documents, top_n=5)
   └── Return top 5 best matches

4️⃣ QA Chain (src/chains/qa_chain.py)
   ├── Build prompt:
   │   ├── System: "You are a PPT assistant..."
   │   ├── Context: Retrieved chunks
   │   ├── History: Previous Q&A (if conversation)
   │   └── Question: User's question
   │
   ├── Call LLM (OpenAI or Anthropic)
   │   ├── Check cache first
   │   └── Generate answer
   │
   └── Quality Check:
       ├── Has answer? ✓
       ├── Cites sources? ✓
       ├── Confident? ✓
       └── Return quality score

5️⃣ Format Response
   ├── Answer text
   ├── Source documents (with slide numbers)
   ├── Quality metrics
   └── Ranking scores (RRF, vector rank, BM25 rank)

✅ Return: Complete answer with sources
```

---

## 📖 Reading Guide - Where to Start

### Path 1: "I want to understand the full system quickly"

**Read in this order:**

1. **`src/config.py`** (5 min)
   - See all configuration options
   - Understand provider selection (OpenAI vs Anthropic)
   - See caching settings

2. **`src/pipeline.py`** (15 min)
   - Read `PPTContextualRetrievalPipeline` class
   - Focus on `index_presentation()` method
   - Focus on `query()` method
   - This shows the full orchestration

3. **`src/splitters/contextual_splitter.py`** (20 min) ⭐ MOST IMPORTANT
   - Read `ContextualTextSplitter` class
   - Understand `asplit_documents()` method
   - See `_generate_context_async()` - this is the magic
   - Look at prompt template - see how context is generated

4. **`scripts/ingest.py`** (10 min)
   - See how CLI uses the pipeline
   - Understand ingestion flow

5. **`scripts/chat.py`** (10 min)
   - See how CLI uses the pipeline for queries
   - Understand query flow

**Total: ~60 minutes to understand 80% of the system**

### Path 2: "I want to understand a specific component"

#### Understanding Contextual Retrieval

**Read:**
1. `src/splitters/contextual_splitter.py` - Implementation
2. `docs/prompts.md` - Section 1.1 (Context Generation Prompts)
3. `docs/ppt_contextual_retrieval_design.md` - Conceptual overview

#### Understanding Caching

**Read:**
1. `src/utils/caching.py` - Implementation
2. `docs/CACHING.md` - Complete guide
3. `CACHING_IMPLEMENTATION_SUMMARY.md` - Summary

#### Understanding Hybrid Search

**Read:**
1. `src/retrievers/hybrid_retriever.py` - Implementation
2. `docs/langchain_implementation.md` - Section on retrievers

#### Understanding Rate Limiting

**Read:**
1. `src/utils/rate_limiter.py` - Implementation
2. See usage in `contextual_splitter.py` and `vision_analyzer.py`

---

## 🔍 Component Deep Dive

### Component 1: PPTLoader

**File:** `src/loaders/ppt_loader.py`

**What it does:** Converts .pptx file → LangChain Documents

**Key method:**
```python
def load(self) -> List[Document]:
    prs = Presentation(self.file_path)

    for slide in prs.slides:
        # Extract text from all shapes
        slide_text = self._extract_slide_text(slide)

        # Extract speaker notes
        speaker_notes = slide.notes_slide.notes_text_frame.text

        # Extract images metadata
        images_info = self._extract_images_info(slide)

        # Extract tables
        tables_info = self._extract_tables(slide)

        # Build metadata
        metadata = {
            'slide_number': slide_num,
            'slide_title': title,
            'section': section,
            'speaker_notes': speaker_notes,
            'images': images_info,
            'tables': tables_info
        }

        # Create Document
        doc = Document(
            page_content=slide_text,
            metadata=metadata
        )
```

**Output:** List of Documents (one per slide)

**Key insight:** Each Document has rich metadata - this metadata is used later for context generation.

### Component 2: ContextualTextSplitter

**File:** `src/splitters/contextual_splitter.py`

**What it does:** Split slides → chunks → add LLM-generated context

**Key flow:**
```python
async def asplit_documents(self, documents: List[Document]):
    for doc in documents:  # Each slide
        # 1. Split slide text into chunks
        text_chunks = self._split_text(doc.page_content)

        for chunk_text in text_chunks:
            # 2. Build prompt for context generation
            prompt = self.context_prompt.format(
                presentation_title=doc.metadata['presentation_title'],
                slide_number=doc.metadata['slide_number'],
                section=doc.metadata['section'],
                chunk_content=chunk_text,
                ...
            )

            # 3. Generate context via LLM
            context = await self._generate_context_async(chunk)
            # → "This is from Slide 12 in Financial Results..."

            # 4. Prepend context to chunk
            final_content = f"{context}\n\n{chunk_text}"

            # 5. Create new Document with context
            chunk_doc = Document(
                page_content=final_content,
                metadata={...}
            )
```

**Critical detail:** The `_generate_context_async()` method:
```python
async def _generate_context_async(self, chunk: dict) -> str:
    # 1. Rate limiting
    await rate_limiter.wait_if_needed(key="anthropic")

    # 2. Build prompt (static first, dynamic last for caching)
    prompt = build_optimized_prompt(chunk)

    # 3. Call LLM
    response = await self.llm.ainvoke(prompt)

    # 4. Return context (50-100 tokens)
    return response.content.strip()
```

**Why important:** This is THE innovation. Every other component is supporting this.

### Component 3: Caching System

**File:** `src/utils/caching.py`

**Class:** `OpenAICachingManager`

**Three cache types:**

**1. Embedding Cache:**
```python
def create_cached_embeddings(self, model: str):
    # Underlying embeddings (makes API calls)
    underlying = OpenAIEmbeddings(model=model)

    # Wrap with cache
    cached = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=underlying,
        document_embedding_cache=LocalFileStore(cache_dir),
        query_embedding_store=LocalFileStore(cache_dir)
    )

    # Now:
    # First call: API request → Cache result
    # Second call: Load from cache (instant)
    return cached
```

**2. LLM Response Cache:**
```python
def _setup_llm_cache(self):
    # Global LangChain cache
    set_llm_cache(SQLiteCache(database_path))

    # Now ALL LLM calls automatically cached:
    # llm.invoke("What is 2+2?")  # → API call
    # llm.invoke("What is 2+2?")  # → From cache
```

**3. Prompt Cache Optimization:**
```python
class PromptOptimizer:
    @staticmethod
    def structure_for_caching(static_instructions, dynamic_content):
        # Static first (cached by OpenAI)
        # Dynamic last (not cached)
        return f"{static_instructions}\n\n{dynamic_content}"
```

### Component 4: Hybrid Retriever

**File:** `src/retrievers/hybrid_retriever.py`

**Key method:**
```python
def _get_relevant_documents(self, query: str):
    # 1. Vector search
    vector_docs = self.vector_store.similarity_search(query, k=20)

    # 2. BM25 search
    bm25_docs = self.bm25_retriever.get_relevant_documents(query)

    # 3. Reciprocal Rank Fusion
    merged_docs = self._reciprocal_rank_fusion(
        vector_docs,
        bm25_docs
    )

    return merged_docs[:self.top_k]
```

**RRF Algorithm:**
```python
def _reciprocal_rank_fusion(self, vector_docs, bm25_docs, k=60):
    doc_scores = {}

    # Score from vector search
    for rank, doc in enumerate(vector_docs, start=1):
        doc_id = get_id(doc)
        doc_scores[doc_id] = 0.7 * (1.0 / (k + rank))

    # Score from BM25
    for rank, doc in enumerate(bm25_docs, start=1):
        doc_id = get_id(doc)
        doc_scores[doc_id] += 0.3 * (1.0 / (k + rank))

    # Sort by combined score
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_docs
```

**Key insight:** RRF doesn't need score calibration. Works with just rankings.

### Component 5: QA Chain

**File:** `src/chains/qa_chain.py`

**Class:** `PPTQAChain`

**Key flow:**
```python
async def aquery(self, question: str):
    # 1. Rate limiting
    await rate_limiter.wait_if_needed(key="anthropic")

    # 2. Run retrieval + QA chain
    result = await self.chain.ainvoke({"question": question})

    # 3. Extract results
    answer = result['answer']
    source_docs = result['source_documents']

    # 4. Quality check
    quality_score = self._check_answer_quality(question, answer, source_docs)

    # 5. Return formatted result
    return {
        'answer': answer,
        'source_documents': source_docs,
        'quality_score': quality_score
    }
```

**Quality checking:**
```python
def _check_answer_quality(self, question, answer, sources):
    quality = {
        'has_answer': len(answer) > 50,
        'has_sources': len(sources) > 0,
        'cites_slides': 'slide' in answer.lower(),
        'not_uncertain': "don't have" not in answer.lower()
    }

    # Calculate score (0.0 to 1.0)
    score = sum([
        0.3 if quality['has_answer'] else 0,
        0.3 if quality['has_sources'] else 0,
        0.2 if quality['cites_slides'] else 0,
        0.2 if quality['not_uncertain'] else 0
    ])

    quality['score'] = score
    return quality
```

---

## 🎬 Common Scenarios

### Scenario 1: User Uploads PPT via Streamlit

**Code path:**

```
streamlit_app/app.py
└── show_upload_page()
    └── st.button("🚀 Start Indexing")
        └── Create PPTContextualRetrievalPipeline
            └── pipeline.index_presentation(ppt_path)
                ├── PPTLoader.load()                    # Load slides
                ├── VisionAnalyzer.analyze_slide_images() # Analyze images
                ├── ContextualTextSplitter.asplit_documents() # Add context
                ├── get_cached_embeddings().embed_documents() # Embed
                └── PineconeVectorStore.add_documents()  # Index
```

### Scenario 2: User Runs CLI Ingestion

**Code path:**

```
scripts/ingest.py
└── ingest_single_file(ppt_path)
    └── [Same as above: PPTContextualRetrievalPipeline.index_presentation()]
```

### Scenario 3: User Asks Question via CLI

**Code path:**

```
scripts/chat.py
└── ChatSession.query(question)
    └── pipeline.query(question)
        ├── embeddings.embed_query(question)         # Embed query
        ├── HybridRetriever.get_relevant_documents() # Retrieve
        │   ├── Pinecone.search()                   # Vector search
        │   ├── BM25.search()                       # BM25 search
        │   └── RRF.merge()                         # Merge results
        ├── CohereRerank.compress_documents()       # Rerank (optional)
        └── PPTQAChain.aquery()                     # Generate answer
            └── ChatOpenAI.ainvoke(prompt)          # LLM call
```

### Scenario 4: Cache Hit on Repeated Query

**Code path:**

```
User asks: "What was the revenue?"

1. Embed query
   └── get_cached_embeddings().embed_query()
       └── Check cache → HIT! (instant, $0)

2. Retrieve documents
   └── Pinecone.search(cached_embedding)

3. Generate answer
   └── ChatOpenAI.ainvoke(prompt)
       └── LangChain checks SQLite cache
           └── HIT! (instant, $0)

Total: ~0.01s, $0
```

### Scenario 5: First-Time Ingestion with Full Options

**Step-by-step:**

```
python scripts/ingest.py presentation.pptx

Time: 0s  | PPTLoader: Opening file...
Time: 2s  | PPTLoader: Loaded 45 slides
Time: 3s  | VisionAnalyzer: Found 12 images
Time: 8s  | VisionAnalyzer: Analyzed 12 images ($0.18)
Time: 10s | ContextualSplitter: Splitting into chunks...
Time: 12s | ContextualSplitter: Created 187 chunks
Time: 15s | ContextualSplitter: Generating contexts (batch 1/38)...
Time: 45s | ContextualSplitter: Context generation complete ($0.35)
Time: 47s | Embeddings: Embedding 187 chunks...
Time: 52s | Embeddings: Complete ($0.02, all cached for next time)
Time: 54s | Pinecone: Creating index...
Time: 56s | Pinecone: Uploading 187 vectors...
Time: 60s | ✅ Complete!

Total: 60s, $0.55
```

---

## 🐛 Debugging Guide

### Enable Verbose Logging

**Option 1: Environment variable**
```bash
export VERBOSE_LOGGING=true
python scripts/ingest.py file.pptx
```

**Option 2: Code**
```python
# src/config.py
verbose_logging = True

# All components will log more details
```

### Check Cache Status

```python
from src.utils.caching import caching_manager

stats = caching_manager.get_cache_stats()
print(stats)
# {
#   'embedding_cache_size_mb': 45.2,
#   'llm_cache_size_mb': 12.8,
#   'embedding_namespaces': 3
# }
```

### Check Rate Limiter Stats

```python
from src.utils.rate_limiter import rate_limiter

stats = rate_limiter.get_stats("openai")
print(stats)
# {
#   'requests_in_last_minute': 15,
#   'tokens_in_last_minute': 45000,
#   'request_limit': 50,
#   'token_limit': 100000
# }
```

### Trace LLM Calls (LangSmith)

**Enable in `.env`:**
```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key
LANGCHAIN_PROJECT=ppt-debug
```

**View traces:** https://smith.langchain.com

### Debug Specific Component

**Example: Debug contextual splitter**
```python
# Run in Python REPL
from src.loaders.ppt_loader import PPTLoader
from src.splitters.contextual_splitter import ContextualTextSplitter
import asyncio

# Load PPT
loader = PPTLoader("test.pptx")
documents = loader.load()
print(f"Loaded {len(documents)} slides")

# Split one document
splitter = ContextualTextSplitter(add_context=True)
chunks = asyncio.run(splitter.asplit_documents([documents[0]]))

# Inspect first chunk
chunk = chunks[0]
print("Content:", chunk.page_content)
print("Metadata:", chunk.metadata)
```

### Common Issues & Solutions

**Issue: "Context generation too slow"**
```python
# Solution 1: Increase batch size
splitter = ContextualTextSplitter(batch_size=10)  # Default: 5

# Solution 2: Use faster model
# .env: CONTEXT_GENERATION_MODEL=gpt-3.5-turbo

# Solution 3: Disable context for testing
splitter = ContextualTextSplitter(add_context=False)
```

**Issue: "Rate limit errors"**
```python
# Solution 1: Increase limits in .env
MAX_REQUESTS_PER_MINUTE=100
MAX_TOKENS_PER_MINUTE=200000

# Solution 2: Check rate limiter
from src.utils.rate_limiter import rate_limiter
stats = rate_limiter.get_stats("openai")
# Are you hitting limits?

# Solution 3: Reduce batch size
splitter = ContextualTextSplitter(batch_size=3)
```

**Issue: "Cache not working"**
```python
# Check cache is enabled
from src.config import settings
print(settings.enable_embedding_cache)  # Should be True
print(settings.enable_llm_cache)        # Should be True

# Check cache directory exists
import os
print(os.path.exists("data/cache/embeddings"))
print(os.path.exists("data/cache/llm_responses"))

# Clear cache if corrupted
from src.utils.caching import clear_all_caches
clear_all_caches()
```

---

## 🎓 Next Steps After Understanding

### For Development

1. **Add tests:** Create `tests/test_contextual_splitter.py`
2. **Add monitoring:** Integrate Prometheus metrics
3. **Add new provider:** Support Google Gemini, etc.
4. **Improve prompts:** Experiment in `docs/prompts.md`

### For Deployment

1. **Dockerize:** Create `Dockerfile` and `docker-compose.yml`
2. **Scale:** Use Redis for distributed caching
3. **Monitor:** Add logging aggregation (ELK, DataDog)
4. **Secure:** Use secret management (AWS Secrets Manager, etc.)

### For Research

1. **Experiment with models:** Compare Claude vs GPT-4o for context
2. **Tune retrieval:** Optimize TOP_K, TOP_N, RRF weights
3. **Analyze prompts:** Which prompt structure works best?
4. **Measure quality:** A/B test with and without contextual retrieval

---

## 📚 Further Reading

**In this repo:**
- `docs/ppt_contextual_retrieval_design.md` - Original design
- `docs/langchain_implementation.md` - LangChain architecture
- `docs/prompts.md` - All prompts with explanations
- `docs/CACHING.md` - Caching deep dive
- `CLAUDE.md` - Development guide

**External:**
- [Anthropic Contextual Retrieval Blog](https://www.anthropic.com/engineering/contextual-retrieval)
- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching)

---

## ✅ Checklist: "I Understand the Codebase"

- [ ] I can explain what contextual retrieval is
- [ ] I understand the ingestion flow (PPT → Documents → Chunks → Context → Embeddings → Pinecone)
- [ ] I understand the query flow (Question → Embed → Retrieve → Rerank → Answer)
- [ ] I know where the "magic" happens (`contextual_splitter.py`)
- [ ] I understand why we have 3 layers of caching
- [ ] I can run ingestion and query via CLI
- [ ] I can debug issues using logs and cache stats
- [ ] I understand how to modify prompts
- [ ] I know where to add new features

**If you checked all boxes:** Congratulations! You understand the codebase. 🎉

---

**Happy Coding! 🚀**
