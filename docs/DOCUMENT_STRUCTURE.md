# Document Structure Analysis Report

**Date:** 2025-10-10
**System:** PPT Contextual Retrieval System
**Author:** Technical Documentation

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Document Structure](#document-structure)
3. [Field-by-Field Analysis](#field-by-field-analysis)
4. [Data Flow Pipeline](#data-flow-pipeline)
5. [Retrieval Phase Usage](#retrieval-phase-usage)
6. [Code Examples](#code-examples)
7. [Best Practices](#best-practices)

---

## Overview

In this system, a **Document** is a LangChain `Document` object that represents a chunk of text from a PowerPoint presentation with rich metadata. Each Document contains:

- **`page_content`**: The actual text content (with optional contextual prefix)
- **`metadata`**: A dictionary of 15-20 fields containing information about the source, position, and enrichments

Documents flow through multiple stages:
```
PPT File → PPTLoader → Vision Analysis → Contextual Splitting → Embedding → Pinecone
```

---

## Document Structure

### Schema Overview

```python
Document(
    page_content: str,           # Main text content
    metadata: {                  # Rich metadata (15-20 fields)
        # === Source Information ===
        "source": str,
        "presentation_id": str,
        "presentation_title": str,

        # === Position & Navigation ===
        "slide_number": int,
        "total_slides": int,
        "slide_title": str,
        "section": str,

        # === Content Type Flags ===
        "type": str,
        "has_images": bool,
        "has_tables": bool,

        # === Detailed Content ===
        "speaker_notes": str,
        "images": str,              # JSON string
        "tables": str,              # JSON string
        "image_analyses": str,      # JSON string (added by Vision Analyzer)

        # === Chunking Metadata ===
        "chunk_index": int,         # Added by ContextualTextSplitter
        "chunk_id": str,            # Added by ContextualTextSplitter
        "chunk_size": int,          # Added by ContextualTextSplitter

        # === Contextual Retrieval ===
        "context": str,             # Added by ContextualTextSplitter
        "original_text": str,       # Added by ContextualTextSplitter
    }
)
```

---

## Field-by-Field Analysis

### 1. **page_content** (Main Content)

#### What it is:
The primary text content of the document chunk. Contains the actual text that will be embedded and searched.

#### How it's extracted:
```python
# Stage 1: Initial extraction (PPTLoader)
full_content = self._build_full_content(
    slide_text,         # Text from all shapes on slide
    speaker_notes,      # Notes from speaker notes panel
    tables_info         # Extracted table data
)

# Stage 2: After contextual chunking
if self.add_context and "context" in chunk:
    # Context prepended for better embedding
    content = f"{chunk['context']}\n\n{chunk['text']}"
else:
    content = chunk["text"]
```

**Source code:** `src/loaders/ppt_loader.py:112-116`, `src/splitters/contextual_splitter.py:188-193`

#### Format example:
```
This chunk discusses the Vietnam case study, which shows that successful
film communities can thrive on existing platforms like Facebook and YouTube
rather than building proprietary platforms.

Vietnam case study

Case Việt Nam không có nền tảng riêng mà sống ký sinh trên big platform
và rất thành công...
```

#### Role in retrieval:
- **Primary search target**: This is what gets embedded and matched against queries
- **Context prefix improves accuracy**: Contextual description helps LLM understand chunk's position in presentation
- **Source for LLM answers**: Extracted text is shown to user as evidence

---

### 2. **source** (File Path)

#### What it is:
Absolute file path to the original PowerPoint file.

#### How it's extracted:
```python
metadata = {
    "source": self.file_path,  # From PPTLoader.__init__(file_path)
    ...
}
```

**Source code:** `src/loaders/ppt_loader.py:94`

#### Format example:
```
"/home/user/presentations/Cinema - Desk - Community.pptx"
```

#### Role in retrieval:
- **Source attribution**: Shows which file the answer came from
- **Multi-file disambiguation**: When multiple presentations are indexed, identifies the source
- **File-level filtering**: Can filter results by specific presentation

---

### 3. **presentation_id** (Unique Identifier)

#### What it is:
MD5 hash of the presentation file path, used as unique identifier.

#### How it's extracted:
```python
presentation_id = hashlib.md5(self.file_path.encode()).hexdigest()[:8]
```

**Source code:** `src/loaders/ppt_loader.py:55`

#### Format example:
```
"a3f2c9d1"
```

#### Role in retrieval:
- **Deduplication**: Prevents indexing same file multiple times
- **Chunk grouping**: Links all chunks from same presentation
- **Analytics**: Track which presentations are queried most

---

### 4. **presentation_title** (Presentation Name)

#### What it is:
Title of the presentation, extracted from first slide or filename.

#### How it's extracted:
```python
# Get title from first slide or fallback to filename
title = self._get_presentation_title(prs)

def _get_presentation_title(self, prs):
    first_slide = prs.slides[0]
    if first_slide.shapes.title:
        return first_slide.shapes.title.text.strip()
    return Path(self.file_path).stem
```

**Source code:** `src/loaders/ppt_loader.py:56`, `src/loaders/ppt_loader.py:259-270`

#### Format example:
```
"July 2020"  (from slide title)
or
"Cinema - Desk - Community"  (from filename)
```

#### Role in retrieval:
- **User-friendly display**: Show readable presentation name instead of file path
- **Contextual understanding**: LLM knows which presentation context it's in
- **Presentation-level queries**: "What is the July 2020 presentation about?"

---

### 5. **slide_number** (Slide Position)

#### What it is:
1-based index indicating which slide this chunk comes from.

#### How it's extracted:
```python
for slide_num, slide in enumerate(prs.slides, start=1):
    metadata = {
        "slide_number": slide_num,
        ...
    }
```

**Source code:** `src/loaders/ppt_loader.py:63`, `src/loaders/ppt_loader.py:97`

#### Format example:
```
12  (for Slide 12)
```

#### Role in retrieval:
- **Precise citations**: "According to Slide 12..."
- **Ordering**: Sort results by slide order
- **Navigation**: Direct user to specific slide
- **Slide-specific queries**: "What's on slide 5?"

---

### 6. **total_slides** (Presentation Length)

#### What it is:
Total number of slides in the presentation.

#### How it's extracted:
```python
metadata = {
    "total_slides": len(prs.slides),
    ...
}
```

**Source code:** `src/loaders/ppt_loader.py:98`

#### Format example:
```
29
```

#### Role in retrieval:
- **Context awareness**: LLM knows if slide is at beginning/middle/end
- **Relative positioning**: "This is slide 12 of 29 total slides"
- **Progress indicators**: Show "Slide 5/29" in UI

---

### 7. **slide_title** (Slide Header)

#### What it is:
Title text from the slide's title placeholder.

#### How it's extracted:
```python
slide_title = ""
if slide.shapes.title and slide.shapes.title.text:
    slide_title = slide.shapes.title.text.strip()
```

**Source code:** `src/loaders/ppt_loader.py:68-70`

#### Format example:
```
"Deep dive - Vietnam case study"
```

#### Role in retrieval:
- **Topic identification**: Quickly identify what slide is about
- **Search relevance**: Match queries to slide titles
- **Source citations**: "From slide 'Vietnam case study'..."
- **Navigation**: Table of contents generation

---

### 8. **section** (Presentation Section)

#### What it is:
Hierarchical section name that groups related slides.

#### How it's extracted:
```python
sections = self._extract_sections(prs)
section = self._get_section_for_slide(slide_num, sections, prs)

def _extract_sections(self, prs):
    """Extract section headers from presentation."""
    sections = []
    for idx, slide in enumerate(prs.slides, start=1):
        # Check if slide is a section divider
        if self._is_section_slide(slide):
            sections.append({
                "slide_num": idx,
                "title": slide.shapes.title.text.strip()
            })
    return sections
```

**Source code:** `src/loaders/ppt_loader.py:88`, `src/loaders/ppt_loader.py:233-257`

#### Format example:
```
"Deep Dive"
"Implication"
"Phụ lục"
```

#### Role in retrieval:
- **Hierarchical context**: "This is in the 'Deep Dive' section"
- **Section-level queries**: "Summarize the Deep Dive section"
- **Improved understanding**: LLM knows broader context
- **Filtering**: Filter results by section

---

### 9. **speaker_notes** (Presenter Notes)

#### What it is:
Text from the speaker notes panel below each slide.

#### How it's extracted:
```python
speaker_notes = ""
if self.extract_notes and slide.has_notes_slide:
    notes_slide = slide.notes_slide
    if notes_slide.notes_text_frame:
        speaker_notes = notes_slide.notes_text_frame.text.strip()
```

**Source code:** `src/loaders/ppt_loader.py:72-76`

#### Format example:
```
"Remember to emphasize the 6:1 ratio of peer vs expert reviews.
Mention the Facebook strategy as key differentiator."
```

#### Role in retrieval:
- **Hidden knowledge**: Information not visible on slides
- **Contextual hints**: Speaker's intended emphasis
- **Richer answers**: Additional details for comprehensive responses
- **Search breadth**: Match queries against notes text

---

### 10. **has_images** (Image Flag)

#### What it is:
Boolean flag indicating if slide contains images.

#### How it's extracted:
```python
images_info = self._extract_images_info(slide, slide_num)

metadata = {
    "has_images": len(images_info) > 0,
    ...
}
```

**Source code:** `src/loaders/ppt_loader.py:80`, `src/loaders/ppt_loader.py:102`

#### Format example:
```
true
```

#### Role in retrieval:
- **Content type filtering**: "Find slides with charts"
- **Vision analysis trigger**: Determines if vision model should process slide
- **UI indicators**: Show 📊 icon for visual slides
- **Quality signals**: Slides with images may be more important

---

### 11. **image_count** (Number of Images)

#### What it is:
Count of images/shapes on the slide.

#### How it's extracted:
```python
metadata = {
    "image_count": len(images_info),
    ...
}
```

**Source code:** `src/loaders/ppt_loader.py:103`

#### Format example:
```
6  (6 images on slide)
```

#### Role in retrieval:
- **Complexity indicator**: Slides with many images are often data-heavy
- **Sorting/ranking**: Prioritize visually rich slides
- **Resource planning**: Estimate vision analysis cost

---

### 12. **images** (Image Metadata Array)

#### What it is:
JSON string containing array of image metadata objects.

#### How it's extracted:
```python
images_info = self._extract_images_info(slide, slide_num)

def _extract_images_info(self, slide, slide_num: int) -> List[dict]:
    images = []
    for shape in slide.shapes:
        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
            images.append({
                "type": "picture",
                "width": shape.width,
                "height": shape.height,
                "left": shape.left,
                "top": shape.top,
                "shape_id": shape.shape_id
            })
    return images

# Convert to JSON for Pinecone compatibility
metadata = {
    "images": json.dumps(images_info),
    ...
}
```

**Source code:** `src/loaders/ppt_loader.py:80`, `src/loaders/ppt_loader.py:157-198`, `src/loaders/ppt_loader.py:104`

#### Format example:
```json
"[{\"type\":\"picture\",\"width\":4572000,\"height\":3048000,\"left\":1524000,\"top\":1143000,\"shape_id\":5}]"
```

#### Role in retrieval:
- **Image reconstruction**: Potentially re-extract specific images
- **Layout understanding**: Position and size of visual elements
- **Advanced queries**: "Find slides with large images"

---

### 13. **has_tables** (Table Flag)

#### What it is:
Boolean indicating if slide contains tables.

#### How it's extracted:
```python
tables_info = []
if self.extract_tables:
    tables_info = self._extract_tables(slide)

metadata = {
    "has_tables": len(tables_info) > 0,
    ...
}
```

**Source code:** `src/loaders/ppt_loader.py:83-85`, `src/loaders/ppt_loader.py:105`

#### Format example:
```
true
```

#### Role in retrieval:
- **Content filtering**: "Show slides with data tables"
- **Data-heavy signals**: Tables often contain key metrics
- **Response formatting**: Format table data appropriately

---

### 14. **table_count** (Number of Tables)

#### What it is:
Count of tables on the slide.

#### How it's extracted:
```python
metadata = {
    "table_count": len(tables_info),
    ...
}
```

**Source code:** `src/loaders/ppt_loader.py:106`

#### Format example:
```
2
```

#### Role in retrieval:
- **Complexity metric**: Multiple tables = dense information
- **Prioritization**: Weight table-heavy slides higher for data queries

---

### 15. **tables** (Table Data Array)

#### What it is:
JSON string containing extracted table structures.

#### How it's extracted:
```python
tables_info = self._extract_tables(slide)

def _extract_tables(self, slide) -> List[dict]:
    tables = []
    for shape in slide.shapes:
        if shape.shape_type == MSO_SHAPE_TYPE.TABLE:
            table_data = []
            for row in shape.table.rows:
                row_data = [cell.text.strip() for cell in row.cells]
                table_data.append(row_data)

            tables.append({
                "col_count": len(shape.table.columns),
                "row_count": len(shape.table.rows),
                "data": table_data
            })
    return tables

metadata = {
    "tables": json.dumps(tables_info),
    ...
}
```

**Source code:** `src/loaders/ppt_loader.py:83-85`, `src/loaders/ppt_loader.py:200-224`, `src/loaders/ppt_loader.py:107`

#### Format example:
```json
"[{\"col_count\":3,\"row_count\":4,\"data\":[[\"Platform\",\"Users\",\"Growth\"],[\"TIX\",\"500K\",\"25%\"]]}]"
```

#### Role in retrieval:
- **Structured data**: Parse tables for specific values
- **Rich context**: Include table data in embeddings
- **Data queries**: "What are the growth numbers?"

---

### 16. **type** (Content Type)

#### What it is:
Classification of the document type (always "slide" for PPT).

#### How it's extracted:
```python
metadata = {
    "type": "slide",
    ...
}
```

**Source code:** `src/loaders/ppt_loader.py:108`

#### Format example:
```
"slide"
```

#### Role in retrieval:
- **Multi-source systems**: Distinguish PPT slides from PDF pages, web articles, etc.
- **Type-specific processing**: Handle slides differently from other content
- **Filtering**: "Only search presentation slides"

---

### 17. **image_analyses** (Vision Analysis Results)

#### What it is:
JSON string containing AI-generated descriptions of images/charts.

#### How it's extracted:
```python
# In pipeline.py, after loading documents
if doc.metadata.get("has_images", False):
    analyses = await self.vision_analyzer.analyze_slide_images(
        ppt_path,
        slide_num,
        doc.metadata
    )

    if analyses:
        doc.metadata["image_analyses"] = json.dumps(analyses)

# Vision analyzer uses GPT-4o mini
async def analyze_chart(self, image_data: str) -> str:
    response = await self.llm.ainvoke([
        SystemMessage(content=self.vision_prompt),
        HumanMessage(content=[
            {"type": "text", "text": "Analyze this image:"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
        ])
    ])
    return response.content
```

**Source code:** `src/pipeline.py:150-167`, `src/models/vision_analyzer.py:85-126`

#### Format example:
```json
"[{\"data\":\"\",\"image_analysis\":\"Bar chart showing growth metrics across Q1-Q4. Vietnam shows 42% YoY growth, highest among markets.\"}]"
```

#### Role in retrieval:
- **Visual understanding**: Make charts/diagrams searchable
- **Data extraction**: Find specific metrics from images
- **Rich context**: Include visual insights in answers
- **Multimodal search**: "Show charts about growth"

---

### 18. **chunk_index** (Chunk Position)

#### What it is:
0-based index of this chunk within the parent slide.

#### How it's extracted:
```python
# In ContextualTextSplitter.asplit_documents()
for chunk_idx, chunk_text in enumerate(text_chunks):
    chunk_metadata = {
        **doc.metadata,  # Inherit all slide metadata
        "chunk_index": chunk_idx,
        ...
    }
```

**Source code:** `src/splitters/contextual_splitter.py:163`

#### Format example:
```
0  (first chunk)
2  (third chunk)
```

#### Role in retrieval:
- **Ordering**: Reassemble chunks in correct order
- **Deduplication**: Distinguish chunks from same slide
- **Context awareness**: "This is the 2nd of 5 chunks from this slide"

---

### 19. **chunk_id** (Unique Chunk Identifier)

#### What it is:
Unique identifier combining presentation ID, slide number, and chunk index.

#### How it's extracted:
```python
chunk_metadata = {
    "chunk_id": (
        f"{doc.metadata.get('presentation_id', 'unknown')}_"
        f"{doc.metadata.get('slide_number', 0)}_"
        f"{chunk_idx}"
    ),
    ...
}
```

**Source code:** `src/splitters/contextual_splitter.py:167-171`

#### Format example:
```
"a3f2c9d1_12_2"
(presentation_a3f2c9d1, slide_12, chunk_2)
```

#### Role in retrieval:
- **Global uniqueness**: Unique across all presentations
- **Deduplication**: Prevent duplicate indexing
- **Traceability**: Track specific chunk through pipeline
- **Cache keys**: Use as key for caching systems

---

### 20. **chunk_size** (Chunk Token Count)

#### What it is:
Approximate word count of the chunk.

#### How it's extracted:
```python
chunk_metadata = {
    "chunk_size": len(chunk_text.split()),
    ...
}
```

**Source code:** `src/splitters/contextual_splitter.py:172`

#### Format example:
```
156  (156 words)
```

#### Role in retrieval:
- **Size validation**: Ensure chunks fit in context window
- **Quality metrics**: Very small chunks may be low-quality
- **Token estimation**: Estimate API costs

---

### 21. **context** (Contextual Description)

#### What it is:
LLM-generated 50-100 token description explaining what this chunk is about and how it relates to the broader presentation.

**This is THE KEY INNOVATION of Contextual Retrieval!**

#### How it's extracted:
```python
# Generate context using LLM
prompt = self.context_prompt.format(
    presentation_title=metadata.get("presentation_title"),
    slide_number=metadata.get("slide_number"),
    total_slides=metadata.get("total_slides"),
    section=metadata.get("section"),
    slide_title=metadata.get("slide_title"),
    chunk_content=chunk["text"][:500],
    prev_slide_title=prev_slide,
    next_slide_title=next_slide
)

response = await self.llm.ainvoke(prompt)
context = response.content.strip()

# Store in metadata AND prepend to page_content
chunk["metadata"]["context"] = context
```

**Source code:** `src/splitters/contextual_splitter.py:247-269`, `src/splitters/contextual_splitter.py:188-191`

#### Format example:
```
"This chunk discusses the Vietnam case study within the Deep Dive section (Slide 11 of 29).
It explains how Vietnamese film platforms succeed by leveraging existing social platforms
like Facebook and YouTube rather than building proprietary infrastructure."
```

#### Role in retrieval:
- **Accuracy boost**: 35% improvement in retrieval accuracy (Anthropic study)
- **Disambiguation**: Helps distinguish similar chunks from different contexts
- **Better embeddings**: Context + chunk embedded together
- **LLM understanding**: Provides situational awareness for answer generation

---

### 22. **original_text** (Pre-Context Chunk Text)

#### What it is:
The chunk text BEFORE contextual description is prepended.

#### How it's extracted:
```python
if self.add_context and "context" in chunk:
    # Context prepended to page_content
    content = f"{chunk['context']}\n\n{chunk['text']}"
    chunk["metadata"]["context"] = chunk["context"]
    chunk["metadata"]["original_text"] = chunk["text"]  # ← Store original
```

**Source code:** `src/splitters/contextual_splitter.py:188-191`

#### Format example:
```
"Vietnam case study

Case Việt Nam không có nền tảng riêng mà sống ký sinh trên big platform..."
```

#### Role in retrieval:
- **Source display**: Show clean text to user without context prefix
- **Debugging**: Compare original vs contextualized versions
- **Re-processing**: Can regenerate contexts from original text
- **Quote extraction**: Extract exact quotes from slides

---

## Data Flow Pipeline

### Stage 1: PPTLoader → Initial Documents

```python
# Input: presentation.pptx
# Output: List[Document] - one per slide

Document(
    page_content="Full slide text + speaker notes + tables",
    metadata={
        "source": "/path/to/file.pptx",
        "presentation_id": "a3f2c9d1",
        "presentation_title": "July 2020",
        "slide_number": 12,
        "total_slides": 29,
        "slide_title": "Vietnam case study",
        "section": "Deep Dive",
        "speaker_notes": "...",
        "has_images": true,
        "image_count": 2,
        "images": "[{...}]",
        "has_tables": false,
        "table_count": 0,
        "tables": "[]",
        "type": "slide"
    }
)
```

---

### Stage 2: Vision Analysis → Image Enrichment

```python
# Adds vision analysis to documents with images

doc.metadata["image_analyses"] = json.dumps([
    {
        "data": "",
        "image_analysis": "Bar chart showing 42% YoY growth in Vietnam market..."
    }
])
```

---

### Stage 3: ContextualTextSplitter → Chunked Documents

```python
# Splits long slides into chunks, adds contextual descriptions

Document(
    page_content="""
This chunk discusses the Vietnam case study within the Deep Dive section...

Vietnam case study
Case Việt Nam không có nền tảng riêng...
    """,
    metadata={
        # All previous metadata +
        "chunk_index": 0,
        "chunk_id": "a3f2c9d1_12_0",
        "chunk_size": 156,
        "context": "This chunk discusses the Vietnam case study...",
        "original_text": "Vietnam case study\nCase Việt Nam..."
    }
)
```

---

### Stage 4: Embedding → Vector Store

```python
# Documents are embedded and stored in Pinecone
# Metadata is preserved as-is (with JSON strings for complex fields)

vector_store.add_documents(chunks)
# → Creates 1536-dim embeddings
# → Stores all metadata
```

---

## Retrieval Phase Usage

### Phase 1: Query → Initial Retrieval (Top 20)

```python
# Hybrid search: Vector + BM25
results = await retriever.aget_relevant_documents(query)

# Each result is a Document with full metadata
for doc in results:
    print(f"Slide {doc.metadata['slide_number']}: {doc.metadata['slide_title']}")
    print(f"Content: {doc.page_content}")
```

**Metadata usage:**
- `slide_number`, `slide_title`: Quick identification
- `section`: Group related results
- `has_images`, `image_analyses`: Include visual insights
- `context`: Helps LLM understand chunk position

---

### Phase 2: Reranking → Top 5

```python
# Cohere reranks top 20 to top 5 most relevant
reranked = await reranker.compress_documents(results, query)
```

**Metadata usage:**
- Preserved unchanged during reranking
- Used for final source citations

---

### Phase 3: Answer Generation

```python
# LLM receives top 5 chunks with metadata
prompt = f"""
Answer based on these sources:

Source 1: Slide {doc.metadata['slide_number']} - {doc.metadata['slide_title']}
Section: {doc.metadata['section']}
Content: {doc.page_content}

...
"""

response = await llm.ainvoke(prompt)
```

**Metadata usage:**
- `slide_number`, `slide_title`: For citations ("According to Slide 12...")
- `section`: For context ("In the Deep Dive section...")
- `presentation_title`: For multi-file disambiguation
- `context`: Helps LLM understand broader narrative
- `image_analyses`: Include visual insights in answer
- `speaker_notes`: Additional hidden knowledge

---

### Phase 4: Source Attribution

```python
# Display sources to user
for doc in sources:
    print(f"[{i}] Slide {doc.metadata['slide_number']}: {doc.metadata['slide_title']}")
    print(f"    Section: {doc.metadata['section']}")
    print(f"    Content: {doc.metadata['original_text'][:200]}...")
```

**Metadata usage:**
- `slide_number`, `slide_title`: User-friendly citation
- `section`: Context for the slide
- `original_text`: Clean text without context prefix
- `source`: Link to original file

---

## Code Examples

### Example 1: Accessing Metadata in Retrieval

```python
# src/chains/qa_chain.py (simplified)

async def aquery(self, query: str, filters: dict = None):
    # Retrieve documents
    docs = await self.retriever.aget_relevant_documents(query)

    # Build context with metadata
    context_parts = []
    for i, doc in enumerate(docs, 1):
        slide_num = doc.metadata.get('slide_number', '?')
        slide_title = doc.metadata.get('slide_title', 'Untitled')
        section = doc.metadata.get('section', 'Unknown')

        context_parts.append(
            f"[Source {i}] Slide {slide_num}: {slide_title} (Section: {section})\n"
            f"{doc.page_content}\n"
        )

    context = "\n\n".join(context_parts)

    # Generate answer
    prompt = f"Answer based on:\n\n{context}\n\nQuestion: {query}"
    response = await self.llm.ainvoke(prompt)

    return {
        "answer": response.content,
        "sources": docs
    }
```

---

### Example 2: Filtering by Metadata

```python
# Filter for slides with images in specific section
results = vector_store.similarity_search(
    query="growth metrics",
    filter={
        "section": "Deep Dive",
        "has_images": True
    },
    k=10
)
```

---

### Example 3: Reconstructing Slide Context

```python
def get_full_slide_context(chunk_doc):
    """Get all chunks from the same slide."""
    presentation_id = chunk_doc.metadata['presentation_id']
    slide_num = chunk_doc.metadata['slide_number']

    # Query for all chunks from this slide
    all_chunks = vector_store.similarity_search(
        query="",  # Empty query
        filter={
            "presentation_id": presentation_id,
            "slide_number": slide_num
        },
        k=100
    )

    # Sort by chunk_index
    all_chunks.sort(key=lambda d: d.metadata['chunk_index'])

    # Combine original texts
    full_text = "\n\n".join(
        d.metadata.get('original_text', d.page_content)
        for d in all_chunks
    )

    return {
        "slide_title": all_chunks[0].metadata['slide_title'],
        "slide_number": slide_num,
        "full_content": full_text
    }
```

---

## Best Practices

### 1. **Always Use Contextual Descriptions**

```python
# ✅ GOOD: With context
splitter = ContextualTextSplitter(add_context=True)

# ❌ BAD: Without context (35% lower accuracy)
splitter = ContextualTextSplitter(add_context=False)
```

**Why:** Context improves retrieval accuracy by 35% (Anthropic study)

---

### 2. **Include Vision Analysis for Visual Slides**

```python
# ✅ GOOD: Analyze images
pipeline = PPTContextualRetrievalPipeline(
    enable_vision=True
)

# ❌ BAD: Skip images (lose important data)
pipeline = PPTContextualRetrievalPipeline(
    enable_vision=False
)
```

**Why:** Charts and diagrams often contain key metrics

---

### 3. **Use Metadata for Smart Filtering**

```python
# ✅ GOOD: Filter before LLM
results = retriever.get_relevant_documents(
    query="What are the metrics?",
    filter={
        "has_tables": True,  # Only slides with data
        "section": {"$ne": "Appendix"}  # Exclude appendix
    }
)

# ❌ BAD: No filtering (LLM sees irrelevant content)
results = retriever.get_relevant_documents(query)
```

---

### 4. **Preserve Original Text for Citations**

```python
# ✅ GOOD: Show clean text to user
source_text = doc.metadata.get('original_text', doc.page_content)
print(f"Quote: {source_text}")

# ❌ BAD: Show text with context prefix (confusing)
print(f"Quote: {doc.page_content}")
# "This chunk discusses Vietnam... Vietnam case study..."
```

---

### 5. **Use Chunk IDs for Deduplication**

```python
# ✅ GOOD: Deduplicate by chunk_id
seen_chunks = set()
unique_docs = []
for doc in results:
    chunk_id = doc.metadata['chunk_id']
    if chunk_id not in seen_chunks:
        seen_chunks.add(chunk_id)
        unique_docs.append(doc)
```

---

## Summary

### Critical Fields for Retrieval:

| Field | Importance | Use Case |
|-------|------------|----------|
| `page_content` | ⭐⭐⭐⭐⭐ | Primary search target, answer source |
| `context` | ⭐⭐⭐⭐⭐ | 35% accuracy boost, situational awareness |
| `slide_number` | ⭐⭐⭐⭐⭐ | Citations, navigation |
| `slide_title` | ⭐⭐⭐⭐ | Quick identification, user display |
| `section` | ⭐⭐⭐⭐ | Hierarchical context, filtering |
| `image_analyses` | ⭐⭐⭐⭐ | Visual understanding, multimodal search |
| `chunk_id` | ⭐⭐⭐ | Deduplication, traceability |
| `original_text` | ⭐⭐⭐ | Clean citations, debugging |
| `speaker_notes` | ⭐⭐⭐ | Hidden knowledge, richer context |

### Key Takeaways:

1. **Contextual descriptions** are the most important innovation - they dramatically improve retrieval accuracy

2. **Vision analysis** makes charts and diagrams searchable - critical for data-heavy presentations

3. **Rich metadata** enables smart filtering, precise citations, and better LLM understanding

4. **Hierarchical structure** (presentation → section → slide → chunk) preserves document organization

5. **Original text preservation** allows clean user-facing citations while benefiting from contextual embeddings

---

**Last Updated:** 2025-10-10
**System Version:** 1.0.0
**Status:** ✅ Production Ready
