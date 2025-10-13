# CLI Scripts Usage Guide

Separate scripts để run ingestion và chat phases independently.

---

## 📤 Ingestion Phase (Index PPT Files)

### Single File

```bash
# Basic ingestion
python scripts/ingest.py presentation.pptx

# With custom options
python scripts/ingest.py presentation.pptx \
  --no-context \
  --no-vision \
  --no-reranking
```

### Batch Ingestion

```bash
# Index multiple files
python scripts/ingest.py *.pptx --batch

# Index folder
python scripts/ingest.py data/presentations/*.pptx --batch

# Fast mode (no context, no vision)
python scripts/ingest.py *.pptx --batch --no-context --no-vision
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--batch` | Batch process multiple files | Off |
| `--no-context` | Skip contextual descriptions | Context enabled |
| `--no-vision` | Skip image analysis | Vision enabled |
| `--no-reranking` | Disable Cohere reranking | Reranking enabled |
| `--no-images` | Skip image extraction | Images extracted |
| `--no-notes` | Skip speaker notes | Notes included |

**Note:** Index name comes from `PINECONE_INDEX_NAME` in `.env` file.

### Example Output

```
==============================================================
INGESTION COMPLETE
==============================================================

✅ Presentation: Q4 Financial Report
📊 Slides: 45
📝 Chunks: 187
⏱️  Time: 42.3s
🗂️  Index: ppt-q4-financial-report
✨ Contextual: Yes
👁️  Vision: Yes
==============================================================
```

---

## 💬 Chat Phase (Query Indexed Presentations)

**Note:** Index name is configured via `PINECONE_INDEX_NAME` in `.env` file.

### Interactive Mode

```bash
# Start interactive chat (uses PINECONE_INDEX_NAME from .env)
python scripts/chat.py

# Chat without sources
python scripts/chat.py --no-sources
```

**Interactive Commands:**
- Type your question and press Enter
- `quit` or `exit` - End session
- `clear` - Clear chat history
- `sources on/off` - Toggle source display

### Single Query Mode

```bash
# Ask single question
python scripts/chat.py --query "What is the revenue?"

# JSON output (for automation)
python scripts/chat.py --query "Summarize key points" --json
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--query "Q"` | Single query | Interactive mode |
| `--interactive` | Force interactive mode | Auto |
| `--no-sources` | Hide source documents | Sources shown |
| `--no-reranking` | Disable reranking | Reranking enabled |
| `--json` | JSON output | Human-readable |

**Note:** Index name comes from `PINECONE_INDEX_NAME` in `.env` file.

### Example Output

```
==============================================================
💡 ANSWER
==============================================================
According to Slide 12, the Q4 revenue was $45.2M, representing
a 23% year-over-year growth. The main drivers were:

1. Enterprise sales: +35% ($18.5M)
2. SMB segment: +15% ($12.3M)
3. International expansion: +42% ($14.4M)

------------------------------------------------------------
📊 Quality Score: 100%
  ✅ Has sources
  ✅ Cites slides
  ✅ Confident answer

==============================================================
📚 SOURCES (3 documents)
==============================================================

[1] Slide 12: Q4 Revenue Breakdown
    Section: Financial Results
    Content: Q4 revenue reached $45.2M, up 23% YoY...
    RRF Score: 0.8234

[2] Slide 13: Revenue by Segment
    Section: Financial Results
    Content: Enterprise segment led growth with $18.5M...
    RRF Score: 0.7891
==============================================================
```

---

## 📊 List Indexes

```bash
# View all indexed presentations
python scripts/list_indexes.py
```

**Output:**
```
==============================================================
📊 PINECONE INDEXES
==============================================================

📁 ppt-q4-financial-report
   Dimension: 1536
   Metric: cosine
   Vectors: 187
   Namespaces: 1

📁 ppt-product-roadmap
   Dimension: 1536
   Metric: cosine
   Vectors: 342
   Namespaces: 1

==============================================================
Total indexes: 2
==============================================================
```

---

## 🚀 Common Workflows

### Workflow 1: Index and Query

```bash
# Step 1: Index presentation (uses PINECONE_INDEX_NAME from .env)
python scripts/ingest.py my_presentation.pptx

# Step 2: Query it (uses same index from .env)
python scripts/chat.py
```

### Workflow 2: Batch Ingest Multiple Files

```bash
# Index all presentations in a folder (all go into same index)
python scripts/ingest.py data/presentations/*.pptx --batch

# Check what was indexed
python scripts/list_indexes.py

# Query the index
python scripts/chat.py
```

### Workflow 3: Fast Ingestion (No Context/Vision)

```bash
# Quick ingestion for testing
python scripts/ingest.py test.pptx --no-context --no-vision

# Query normally
python scripts/chat.py --query "Summary?"
```

### Workflow 4: Automated Pipeline

```bash
#!/bin/bash
# Automated batch processing

# Ingest all new presentations (into single index from .env)
python scripts/ingest.py data/new/*.pptx --batch

# Query the index for summaries
python scripts/chat.py \
  --query "Summarize all presentations in 3 bullet points each" \
  --json > "summaries/all_presentations.json"
```

---

## 🔍 Troubleshooting

### Issue: "Index not found"

```bash
# Check if index exists
python scripts/list_indexes.py

# If not found, index the presentation first
python scripts/ingest.py your_file.pptx
```

### Issue: "File not found"

```bash
# Use absolute path
python scripts/ingest.py /full/path/to/presentation.pptx

# Or relative from project root
cd /path/to/ppt_context_retrieval
python scripts/ingest.py data/presentations/file.pptx
```

### Issue: Slow ingestion

```bash
# Disable context and vision for faster processing
python scripts/ingest.py file.pptx --no-context --no-vision

# Trade-off: Lower quality retrieval
```

---

## 💡 Tips

### Performance

- **Context Generation**: Adds ~30% time but improves accuracy by 35%
- **Vision Analysis**: Adds ~20% time, essential for charts/diagrams
- **Caching**: Second run on same file is 75x faster (embeddings cached)

### Cost Optimization

```bash
# Cheapest: No context, no vision
python scripts/ingest.py file.pptx --no-context --no-vision
# Cost: ~$0.20/100 slides

# Balanced: Context only
python scripts/ingest.py file.pptx --no-vision
# Cost: ~$0.45/100 slides

# Best Quality: Everything enabled (default)
python scripts/ingest.py file.pptx
# Cost: ~$0.85/100 slides
```

### Automation

```python
# Python script to automate ingestion
import subprocess
import glob

files = glob.glob("data/presentations/*.pptx")
for file in files:
    subprocess.run([
        "python", "scripts/ingest.py", file,
        "--no-vision"  # Save cost
    ])
```

---

## 📚 Environment Variables

Make sure `.env` is configured:

```bash
# Required for ingestion
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...

# Required for chat
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...

# Optional
ANTHROPIC_API_KEY=sk-...  # If using Anthropic models
COHERE_API_KEY=...        # If using reranking
```

---

## 🎯 Quick Reference

| Task | Command |
|------|---------|
| Index single file | `python scripts/ingest.py file.pptx` |
| Index batch | `python scripts/ingest.py *.pptx --batch` |
| List indexes | `python scripts/list_indexes.py` |
| Interactive chat | `python scripts/chat.py` |
| Single query | `python scripts/chat.py --query "Q?"` |
| JSON output | `python scripts/chat.py --query "Q?" --json` |

**Note:** All commands use `PINECONE_INDEX_NAME` from `.env` file.

---

**See main README.md for more information.**
