# 📚 Hướng dẫn Index PowerPoint File

## Bước 1: Chuẩn bị môi trường

### 1.1. Kiểm tra virtual environment đã được activate
```bash
# Activate venv
source venv/bin/activate

# Kiểm tra Python version
python --version  # Should be 3.11+
```

### 1.2. Cấu hình file `.env`
```bash
# Copy từ template nếu chưa có
cp .env.example .env

# Edit file .env
nano .env
```

**Required environment variables:**
```bash
# API Keys (REQUIRED)
OPENAI_API_KEY=sk-your-openai-key-here
PINECONE_API_KEY=your-pinecone-key-here

# Optional API Keys
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key  # Optional
COHERE_API_KEY=your-cohere-key                # Optional (for reranking)

# Pinecone Configuration
PINECONE_INDEX_NAME=ppt-contextual-retrieval  # Index name
PINECONE_ENVIRONMENT=us-east-1

# Model Configuration (sử dụng defaults hoặc customize)
EMBEDDING_MODEL=text-embedding-3-small
CONTEXT_GENERATION_MODEL=gpt-4o-mini
ANSWER_GENERATION_MODEL=gpt-4o
VISION_MODEL=gpt-4o-mini
```

---

## Bước 2: Chuẩn bị file PowerPoint

### 2.1. Đặt file vào thư mục phù hợp
```bash
# Tạo folder để lưu presentations
mkdir -p data/presentations

# Copy hoặc move file PPT vào folder
cp /path/to/your/presentation.pptx data/presentations/
```

### 2.2. Kiểm tra file
```bash
# Kiểm tra file tồn tại
ls -lh data/presentations/

# Verify file format
file data/presentations/presentation.pptx
# Output: Microsoft PowerPoint 2007+
```

---

## Bước 3: Chạy Ingestion

### 3.1. Basic Ingestion (Recommended - Full Quality)

**Command:**
```bash
python scripts/ingest.py data/presentations/presentation.pptx
```

**Output mẫu:**
```
2025-10-10 10:30:15 | INFO | Starting ingestion: presentation.pptx
2025-10-10 10:30:15 | INFO | Index name: ppt-contextual-retrieval (from .env)
2025-10-10 10:30:15 | INFO | Options: contextual=True, vision=True

Step 1/5: Loading presentation...
✅ Loaded 45 slides

Step 2/5: Analyzing images with GPT-4o mini...
✅ Analyzed 23 images

Step 3/5: Generating contextual descriptions...
Processing chunks: 100%|████████████| 187/187 [01:23<00:00]
✅ Generated 187 contextual chunks

Step 4/5: Creating embeddings...
✅ Embedded 187 chunks

Step 5/5: Indexing to Pinecone...
✅ Indexed 187 vectors

==============================================================
INGESTION COMPLETE
==============================================================

✅ Presentation: Q4 Financial Report
📊 Slides: 45
📝 Chunks: 187
⏱️  Time: 92.3s
🗂️  Index: ppt-contextual-retrieval (from .env)
✨ Contextual: Yes
👁️  Vision: Yes
==============================================================
```

### 3.2. Fast Mode (Faster, Lower Quality)

**Khi nào dùng:** Testing, prototyping, hoặc không cần accuracy cao

```bash
# Tắt context generation và vision analysis
python scripts/ingest.py data/presentations/presentation.pptx \
  --no-context \
  --no-vision
```

**Performance:**
- ⏱️ Time: ~15s (vs 92s full mode)
- 💰 Cost: ~$0.20 (vs $0.85 full mode)
- 📊 Quality: ⭐⭐⭐ (vs ⭐⭐⭐⭐⭐ full mode)

### 3.3. Balanced Mode

**Khi nào dùng:** Cần balance giữa speed và quality

```bash
# Giữ context, tắt vision
python scripts/ingest.py data/presentations/presentation.pptx \
  --no-vision
```

**Performance:**
- ⏱️ Time: ~45s
- 💰 Cost: ~$0.45
- 📊 Quality: ⭐⭐⭐⭐

---

## Bước 4: Batch Ingestion (Multiple Files)

### 4.1. Index nhiều files cùng lúc

```bash
# Index tất cả PPT files trong folder
python scripts/ingest.py data/presentations/*.pptx --batch
```

**Output mẫu:**
```
============================================================
Processing 1/3: q1_report.pptx
============================================================
...
✅ Success: 45 slides, 187 chunks

============================================================
Processing 2/3: q2_report.pptx
============================================================
...
✅ Success: 52 slides, 213 chunks

============================================================
Processing 3/3: q3_report.pptx
============================================================
...
✅ Success: 48 slides, 195 chunks

============================================================
INGESTION SUMMARY
============================================================

✅ Successful: 3/3
❌ Failed: 0/3

📊 Statistics:
  - Total slides: 145
  - Total chunks: 595
  - Total time: 276.5s
  - Avg time per file: 92.2s

✅ Indexed Presentations:
  - Q1 Financial Report → ppt-contextual-retrieval
  - Q2 Financial Report → ppt-contextual-retrieval
  - Q3 Financial Report → ppt-contextual-retrieval
============================================================
```

---

## Bước 5: Verify Ingestion

### 5.1. Kiểm tra index đã được tạo

```bash
# List tất cả indexes
python scripts/list_indexes.py
```

**Output:**
```
==============================================================
📊 PINECONE INDEXES
==============================================================

📁 ppt-contextual-retrieval
   Dimension: 1536
   Metric: cosine
   Vectors: 187
   Namespaces: 1

==============================================================
Total indexes: 1
==============================================================
```

### 5.2. Test query ngay

```bash
# Quick test query
python scripts/chat.py --query "What is this presentation about?"
```

---

## Bước 6: Advanced Options

### 6.1. Tất cả CLI options

```bash
python scripts/ingest.py presentation.pptx \
  --no-context      # Tắt contextual descriptions
  --no-vision       # Tắt image analysis
  --no-reranking    # Tắt Cohere reranking
  --no-images       # Skip image extraction
  --no-notes        # Skip speaker notes
  --batch           # Batch mode cho multiple files
```

### 6.2. Configuration qua .env

Thay vì dùng CLI flags, có thể config trong `.env`:

```bash
# Model selection
CONTEXT_GENERATION_MODEL=gpt-3.5-turbo  # Faster, cheaper
VISION_MODEL=gpt-4o                     # Better quality

# Chunking
MAX_CHUNK_SIZE=500       # Larger chunks
CHUNK_OVERLAP=100        # More overlap

# Retrieval
TOP_K_RETRIEVAL=30       # Retrieve more candidates
TOP_N_RERANK=10          # Keep more after reranking
```

---

## 📊 Performance & Cost Estimate

### Single 100-slide Presentation:

| Configuration | Time | Cost | Quality |
|---------------|------|------|---------|
| **Full (default)** | ~60s | $0.85 | ⭐⭐⭐⭐⭐ |
| **No vision** | ~45s | $0.45 | ⭐⭐⭐⭐ |
| **No context** | ~30s | $0.30 | ⭐⭐⭐ |
| **Fast mode** | ~15s | $0.20 | ⭐⭐ |

**Cost breakdown:**
- Embeddings: ~$0.05
- Context generation: ~$0.30
- Vision analysis: ~$0.40
- Indexing: ~$0.10

---

## 🔍 Troubleshooting

### Issue 1: "OPENAI_API_KEY not found"
```bash
# Kiểm tra .env file
cat .env | grep OPENAI_API_KEY

# Set temporary nếu cần
export OPENAI_API_KEY=sk-your-key-here
```

### Issue 2: "File not found"
```bash
# Dùng absolute path
python scripts/ingest.py /full/path/to/presentation.pptx

# Hoặc check current directory
pwd
ls data/presentations/
```

### Issue 3: "Pinecone index already exists"
```bash
# Index sẽ được reused - OK!
# Nếu muốn delete và create lại, vào Pinecone console
```

### Issue 4: Rate limit errors
```bash
# Tăng rate limits trong .env
MAX_REQUESTS_PER_MINUTE=30   # Giảm xuống
MAX_TOKENS_PER_MINUTE=50000  # Giảm xuống
```

### Issue 5: Memory errors (large presentations)
```bash
# Process in smaller batches
python scripts/ingest.py large_presentation.pptx --no-vision

# Or increase chunk size to reduce total chunks
# In .env:
MAX_CHUNK_SIZE=600
```

### Issue 6: Slow ingestion
```bash
# Check network connection to Pinecone
ping api.pinecone.io

# Use fast mode for testing
python scripts/ingest.py file.pptx --no-context --no-vision

# Check API key rate limits on OpenAI dashboard
```

---

## 🎯 What Happens During Ingestion?

### Step-by-Step Process:

1. **Load PPT** (`src/loaders/ppt_loader.py`)
   - Extract text from slides
   - Extract images and metadata
   - Extract speaker notes
   - Build slide hierarchy

2. **Vision Analysis** (`src/models/vision_analyzer.py`)
   - Send images to GPT-4o mini
   - Generate descriptions of charts, diagrams, tables
   - Extract data insights from visuals

3. **Contextual Chunking** (`src/splitters/contextual_splitter.py`)
   - Split slides into semantic chunks
   - Generate contextual descriptions for each chunk
   - Preserve document structure and flow

4. **Embedding** (`src/utils/caching.py`)
   - Generate vector embeddings using OpenAI
   - Cache embeddings for reuse
   - Dimension: 1536 (text-embedding-3-small)

5. **Indexing** (`src/pipeline.py`)
   - Create or connect to Pinecone index
   - Upload vectors with metadata
   - Build BM25 index for hybrid search

### Metadata Stored:

Each chunk includes:
```python
{
    "source": "presentation.pptx",
    "slide_number": 12,
    "slide_title": "Q4 Revenue Breakdown",
    "section": "Financial Results",
    "chunk_index": 2,
    "total_chunks": 5,
    "has_image": True,
    "has_table": False,
    "contextual_description": "This chunk discusses..."
}
```

---

## 💡 Best Practices

### 1. **Naming Conventions**
```bash
# Good names (descriptive, lowercase, dashes)
q4-2024-financial-report.pptx
product-roadmap-2025.pptx
sales-training-module-1.pptx

# Avoid (spaces, special chars, too generic)
"Q4 Report (Final).pptx"
"Presentation#1.pptx"
"New Presentation.pptx"
```

### 2. **File Organization**
```bash
data/
├── presentations/
│   ├── financial/
│   │   ├── q1-2024.pptx
│   │   ├── q2-2024.pptx
│   │   └── q3-2024.pptx
│   ├── product/
│   │   └── roadmap-2025.pptx
│   └── training/
│       └── onboarding.pptx
```

### 3. **Batch Processing Strategy**
```bash
# Process by category
python scripts/ingest.py data/presentations/financial/*.pptx --batch

# Process incrementally
python scripts/ingest.py data/presentations/new/*.pptx --batch

# Move processed files
mv data/presentations/new/*.pptx data/presentations/processed/
```

### 4. **Cost Optimization**
```bash
# Development: Use fast mode
python scripts/ingest.py test.pptx --no-context --no-vision

# Staging: Balanced mode
python scripts/ingest.py staging.pptx --no-vision

# Production: Full quality
python scripts/ingest.py prod.pptx
```

### 5. **Quality Assurance**
```bash
# After indexing, test with queries
python scripts/chat.py --query "Summarize the presentation"
python scripts/chat.py --query "List all key metrics"
python scripts/chat.py --query "What are the main recommendations?"

# Check source citations
python scripts/chat.py --query "What was the revenue?" --json | jq '.sources'
```

---

## 🔄 Re-Indexing Strategy

### When to Re-Index:

1. **Presentation Updated**
   - Content changed significantly
   - New slides added
   - Data updated

2. **Configuration Changed**
   - Changed chunking parameters
   - Switched embedding models
   - Updated context generation prompts

3. **Quality Issues**
   - Poor retrieval results
   - Missing information
   - Incorrect citations

### How to Re-Index:

```bash
# Option 1: Index over existing (appends)
python scripts/ingest.py updated_presentation.pptx

# Option 2: Delete and recreate index
# Go to Pinecone console → Delete index
# Then re-run ingestion
python scripts/ingest.py presentation.pptx

# Option 3: Use different index name
# Edit .env
PINECONE_INDEX_NAME=ppt-contextual-retrieval-v2
# Then ingest
python scripts/ingest.py presentation.pptx
```

---

## 📈 Monitoring & Logging

### Check Logs:

```bash
# Logs are output to console by default
# To save logs to file:
python scripts/ingest.py file.pptx 2>&1 | tee ingestion.log

# To see verbose logging, set in .env:
VERBOSE_LOGGING=true
```

### Track Costs:

```bash
# After ingestion, check stats in output
# Example cost calculation for 100-slide presentation:
# - Embeddings: $0.05 (187 chunks * $0.0004/1K tokens)
# - Context: $0.30 (187 chunks * 200 tokens * $0.002/1K tokens)
# - Vision: $0.40 (23 images * $0.017/image)
# Total: ~$0.85
```

---

## ✅ Quick Start Summary

```bash
# 1. Setup
source venv/bin/activate
cp .env.example .env
nano .env  # Add API keys

# 2. Ingest
python scripts/ingest.py presentation.pptx

# 3. Verify
python scripts/list_indexes.py

# 4. Query
python scripts/chat.py --query "Summarize this presentation"
```

---

## 🎯 Next Steps

Sau khi ingest thành công:

1. **Query presentations:** `python scripts/chat.py`
2. **Interactive chat:** `python scripts/chat.py` (no --query flag)
3. **See full docs:** `cat scripts/README.md`
4. **Try Streamlit UI:** `streamlit run streamlit_app/app.py`

---

## 📚 Related Documentation

- [CLI Scripts Guide](../scripts/README.md) - Complete CLI documentation
- [Codebase Guide](CODEBASE_GUIDE.md) - Understanding the system architecture
- [Caching Guide](CACHING.md) - Performance optimization
- [Main README](../README.md) - Project overview

---

**Last Updated:** 2025-10-10
**Status:** ✅ Production Ready
