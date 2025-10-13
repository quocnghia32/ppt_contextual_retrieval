# Prompt Usage Guide - PPT Contextual Retrieval

**Version**: 1.0
**Date**: 2025-10-10
**Purpose**: Comprehensive guide on when and how to use each prompt

---

## 📋 Table of Contents
1. [Overview](#overview)
2. [Indexing Pipeline Prompts](#indexing-pipeline-prompts)
3. [Query Pipeline Prompts](#query-pipeline-prompts)
4. [Response Generation Prompts](#response-generation-prompts)
5. [Utility Prompts](#utility-prompts)
6. [Decision Trees](#decision-trees)
7. [Best Practices](#best-practices)

---

## 🎯 Overview

### Prompt Categories by Pipeline Stage

```
┌─────────────────────────────────────────────────────────────┐
│                    INDEXING PIPELINE                         │
├─────────────────────────────────────────────────────────────┤
│ 1. PPT Upload                                                │
│ 2. Parse & Extract → Slides, Text, Images                  │
│ 3. PROMPTS USED:                                            │
│    → Section Detection (5.1)                                │
│    → Metadata Extraction (5.2)                              │
│    → Image Analysis (2.1, 2.2, 2.3)                         │
│ 4. Chunk Creation                                            │
│ 5. PROMPTS USED:                                            │
│    → Context Generation (1.1, 1.2)                          │
│    → Visual Element Context (1.3)                           │
│ 6. Generate Embeddings                                       │
│ 7. Store in Vector DB                                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    QUERY PIPELINE                            │
├─────────────────────────────────────────────────────────────┤
│ 1. User Query Input                                          │
│ 2. PROMPTS USED:                                            │
│    → Query Intent Classification (3.2)                      │
│    → Query Reformulation (3.3) [if needed]                 │
│    → Query Expansion (3.1) [optional]                       │
│ 3. Generate Query Embedding                                  │
│ 4. Vector Search + BM25                                      │
│ 5. Reranking                                                 │
│ 6. PROMPTS USED:                                            │
│    → RAG Answer Generation (4.1)                            │
│    → Multi-hop Reasoning (4.2) [if complex]                │
│    → Comparison Answer (4.5) [if comparison]               │
│ 7. QUALITY CHECK:                                           │
│    → Answer Quality Check (7.1)                             │
│    → Hallucination Detection (7.2)                          │
│ 8. Return Answer + Sources                                   │
│ 9. PROMPTS USED:                                            │
│    → Follow-up Questions (6.3)                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📥 Indexing Pipeline Prompts

### When: During PPT upload and processing (ONE-TIME per presentation)

### 1. Section Detection (Prompt 5.1)

**Use Case**: Automatically identify logical sections in presentation

**When to Use**:
- ✅ For ALL presentations during initial processing
- ✅ When presentation doesn't have explicit section markers
- ✅ To improve context generation (knowing section boundaries)

**Input Required**:
- All slide titles
- Brief content from each slide (first 50 words)

**Output Used For**:
- Storing section metadata with each chunk
- Generating better chunk contexts
- Helping users navigate ("Show me the Financial Results section")

**Example**:
```python
# During PPT parsing
def process_presentation(ppt_file):
    slides = parse_ppt(ppt_file)

    # Extract titles
    slide_titles = [slide.title for slide in slides]

    # Run section detection
    sections = llm.run_prompt(
        prompt_id="5.1_section_detection",
        slide_titles=slide_titles
    )

    # Store section info with slides
    for slide in slides:
        slide.section = sections.get_section_for_slide(slide.number)
```

**Cost**: ~$0.01 per presentation (run once)

**Alternative**: Manual section tagging (more accurate but requires user input)

---

### 2. Presentation Metadata Extraction (Prompt 5.2)

**Use Case**: Extract key metadata from presentation

**When to Use**:
- ✅ For ALL presentations during upload
- ✅ Before chunking (metadata used in context generation)

**Input Required**:
- Title slide content
- First 2-3 slides
- Last slide

**Output Used For**:
- Presentation search/filtering
- Context generation (presenter name, date, organization)
- Display in UI

**Example**:
```python
def extract_metadata(ppt):
    metadata = llm.run_prompt(
        prompt_id="5.2_metadata_extraction",
        slide_1=ppt.slides[0].content,
        slides_2_3=ppt.slides[1:3].content,
        last_slide=ppt.slides[-1].content
    )

    return {
        "title": metadata.title,
        "presenter": metadata.presenter,
        "date": metadata.date,
        "organization": metadata.organization
    }
```

**Cost**: ~$0.005 per presentation

**When to Skip**: If metadata already provided by user during upload

---

### 3. Image Analysis (Prompts 2.1, 2.2, 2.3, 2.4)

**Use Case**: Extract information from visual elements

**Decision Tree**:
```
Image/Visual Element Detected
    |
    ├─ Is it text-heavy? (>80% text pixels)
    │   └─> Use OCR only (no prompt needed)
    |
    ├─ Is it a chart/graph?
    │   └─> Use Prompt 2.1 (Chart Analysis) with LLM Vision
    |
    ├─ Is it a table?
    │   ├─ Simple table with clear structure?
    │   │   └─> Use OCR + Prompt 2.2 (Table Analysis)
    │   └─ Complex table?
    │       └─> Use LLM Vision + Prompt 2.2
    |
    ├─ Is it an infographic (mixed text/visual)?
    │   └─> Use Prompt 2.3 (Infographic Analysis) with LLM Vision
    |
    └─ Is it an image with caption?
        └─> Use Prompt 2.4 (Simple OCR + context)
```

**When to Use Each**:

#### 2.1 Chart/Diagram Analysis
- **Use for**: Bar charts, line graphs, pie charts, flowcharts, diagrams
- **Cost**: $0.015 per image (LLM Vision)
- **Value**: High - extracts data AND insights
- **Example images**: Revenue trend chart, process flowchart, org chart

```python
if image_type == "chart" or image_type == "diagram":
    analysis = llm_vision.run_prompt(
        prompt_id="2.1_chart_analysis",
        image=image,
        slide_context=slide.context
    )

    # Store extracted data
    chunk.metadata["chart_data"] = analysis.data
    chunk.metadata["insights"] = analysis.insights
```

#### 2.2 Table Analysis
- **Use for**: Data tables, comparison tables, pricing tables
- **Cost**: OCR = free, LLM Vision = $0.015
- **Decision**: Try OCR first, use Vision if OCR confidence < 80%
- **Example**: Q1-Q4 revenue comparison table

```python
if image_type == "table":
    # Try OCR first
    ocr_result = ocr.extract(image)

    if ocr_result.confidence > 0.8:
        # Use OCR + simple parsing
        table_data = parse_table(ocr_result.text)
    else:
        # Fall back to LLM Vision
        table_data = llm_vision.run_prompt(
            prompt_id="2.2_table_analysis",
            image=image
        )
```

#### 2.3 Infographic Analysis
- **Use for**: Complex infographics with text, icons, and data
- **Cost**: $0.015 per image
- **When**: Always use LLM Vision (too complex for OCR)
- **Example**: Market landscape infographic, product feature comparison

#### 2.4 Image with Caption
- **Use for**: Photos with captions, screenshots, logos
- **Cost**: Free (OCR only)
- **When**: Image is primarily visual, text is just caption

**Cost Optimization**:
```python
# Hybrid strategy
def process_image(image, slide):
    image_type = classify_image(image)

    if image_type in ["text_heavy", "simple_caption"]:
        return ocr_extract(image)  # Free
    elif image_type in ["chart", "diagram", "infographic"]:
        return llm_vision_analyze(image)  # $0.015
    else:
        # Try OCR first, fall back to Vision
        ocr_result = ocr_extract(image)
        if ocr_result.confidence < 0.8:
            return llm_vision_analyze(image)
        return ocr_result
```

---

### 4. Chunk Context Generation (Prompts 1.1, 1.2, 1.3)

**Use Case**: Generate contextual descriptions for each chunk

**CRITICAL**: This is the MOST IMPORTANT prompt in the system - it's what makes contextual retrieval work!

**When to Use**:

#### 1.1 Chunk Context Generation (Primary)
- **Use for**: EVERY text chunk in element-level chunking
- **Frequency**: Once per chunk during indexing
- **Cost**: ~$0.006 per chunk (with batching)
- **Why critical**: Improves retrieval accuracy by 35%+

**Input**:
```python
chunk = "Revenue increased from $1.2M to $2.1M"

context_input = {
    "presentation_structure": full_outline,
    "chunk_content": chunk.text,
    "slide_number": 10,
    "total_slides": 20,
    "section_name": "Financial Results",
    "prev_slide_title": "Company Overview",
    "next_slide_title": "Expense Breakdown",
    "visual_summary": "Bar chart showing revenue trend"
}
```

**Output**:
```
"This content is from slide 10 of 20 in the 'Financial Results'
section. It presents key revenue growth metrics following the
company overview and precedes expense analysis. The slide includes
a bar chart visualizing the quarterly revenue trend."
```

**Implementation**:
```python
async def generate_chunk_contexts(chunks, presentation):
    contexts = []

    # Batch for efficiency
    batch_size = 10
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]

        # Parallel API calls
        batch_contexts = await asyncio.gather(*[
            llm.run_prompt(
                prompt_id="1.1_chunk_context",
                presentation_structure=presentation.outline,
                chunk_content=chunk.content,
                slide_number=chunk.slide_number,
                total_slides=presentation.total_slides,
                section_name=chunk.section,
                prev_slide_title=chunk.prev_slide.title,
                next_slide_title=chunk.next_slide.title,
                visual_summary=chunk.visual_summary
            )
            for chunk in batch
        ])

        contexts.extend(batch_contexts)

    return contexts
```

**Cost Optimization**:
- Batch requests (10-20 at a time)
- Use async/await for parallel processing
- Cache contexts (keyed by chunk hash)
- Use lower-cost model (Claude Haiku, not Opus)

#### 1.2 Slide-Level Context
- **Use for**: Slide-level chunking strategy
- **When**: Alternative to 1.1 when using slide-level chunks
- **Trade-off**: Fewer API calls but larger chunks

#### 1.3 Visual Element Context
- **Use for**: Standalone visual elements that become separate chunks
- **When**: Chart/diagram is chunked separately from slide text
- **Example**: Large infographic that's its own chunk

**Decision: Which Context Prompt?**
```python
if chunking_strategy == "element_level":
    # Use 1.1 for text chunks
    # Use 1.3 for visual chunks
    for chunk in chunks:
        if chunk.type == "text":
            context = generate_context_1_1(chunk)
        elif chunk.type == "visual":
            context = generate_context_1_3(chunk)

elif chunking_strategy == "slide_level":
    # Use 1.2 for entire slides
    for slide in slides:
        context = generate_context_1_2(slide)
```

---

## 🔍 Query Pipeline Prompts

### When: During user query processing (EVERY query)

### 5. Query Intent Classification (Prompt 3.2)

**Use Case**: Understand what user is asking for

**When to Use**:
- ✅ For EVERY user query (very fast, cheap)
- ✅ Before retrieval to optimize search strategy

**Why Important**:
- Different intents need different retrieval strategies
- Helps select the right answer generation prompt
- Improves answer quality

**Intent Types & Actions**:

| Intent | Example Query | Action |
|--------|---------------|--------|
| FACT_FINDING | "What was Q4 revenue?" | Precise retrieval, short answer |
| TREND_ANALYSIS | "How did sales change?" | Retrieve time-series data, analytical answer |
| COMPARISON | "Sales vs expenses?" | Retrieve both entities, comparison table |
| EXPLANATION | "Why did revenue drop?" | Multi-hop retrieval, reasoning answer |
| SUMMARY | "Summarize the presentation" | Retrieve key slides, summary prompt |
| LOCATION | "Which slide talks about X?" | Return slide numbers, not full answer |

**Implementation**:
```python
async def process_query(user_query):
    # 1. Classify intent
    intent = await llm.run_prompt(
        prompt_id="3.2_intent_classification",
        user_query=user_query
    )

    # 2. Adjust retrieval based on intent
    if intent == "FACT_FINDING":
        retrieval_params = {"top_k": 5, "precision": "high"}
    elif intent == "SUMMARY":
        retrieval_params = {"top_k": 20, "diversity": "high"}
    elif intent == "COMPARISON":
        retrieval_params = {"top_k": 10, "filter": "data_slides"}

    # 3. Retrieve
    results = await retrieve(user_query, **retrieval_params)

    # 4. Select answer prompt based on intent
    if intent == "COMPARISON":
        answer_prompt = "4.5_comparison_answer"
    elif intent == "EXPLANATION":
        answer_prompt = "4.2_multi_hop_reasoning"
    else:
        answer_prompt = "4.1_rag_answer"

    # 5. Generate answer
    answer = await llm.run_prompt(
        prompt_id=answer_prompt,
        user_query=user_query,
        retrieved_chunks=results
    )

    return answer
```

**Cost**: ~$0.001 per query (very cheap!)

**When to Skip**: Never - always classify intent

---

### 6. Query Reformulation (Prompt 3.3)

**Use Case**: Fix poorly formed queries

**When to Use**:
- ⚡ Only when query is ambiguous or incomplete
- ⚡ Based on intent classification confidence

**Decision Logic**:
```python
intent_result = classify_intent(query)

if intent_result.confidence == "low":
    # Query is unclear - needs reformulation

    # Option 1: Ask user for clarification
    if interactive_mode:
        return ask_clarification(query, presentation)  # Prompt 6.4

    # Option 2: Auto-reformulate
    else:
        query = reformulate_query(query, presentation)  # Prompt 3.3
```

**Examples of When to Use**:

| Original Query | Issue | Action |
|----------------|-------|--------|
| "sales" | Too vague | Reformulate → "What were the sales figures?" |
| "what about revenue" | Incomplete | Reformulate → "What was the revenue in Q4?" |
| "the numbers" | No context | Ask clarification (6.4) |
| "slide 5" | Wrong intent | Don't reformulate, user wants specific slide |

**Implementation**:
```python
def should_reformulate(query, intent_result):
    if intent_result.confidence == "low":
        return True
    if len(query.split()) < 3:  # Very short query
        return True
    if query.lower() in ["revenue", "sales", "profit"]:  # Single word
        return True
    return False

async def process_ambiguous_query(query, presentation):
    if should_reformulate(query, intent):
        reformulated = await llm.run_prompt(
            prompt_id="3.3_query_reformulation",
            user_query=query,
            presentation_context=presentation.metadata
        )
        return reformulated
    else:
        return query
```

**Cost**: ~$0.002 per reformulation

**When to Skip**: Clear, well-formed queries (>80% of queries)

---

### 7. Query Expansion (Prompt 3.1)

**Use Case**: Generate query variants for better recall

**When to Use**:
- ⚡ Optional optimization
- ⚡ When initial retrieval returns < 5 good results
- ⚡ For high-stakes queries (e.g., enterprise search)

**Trade-offs**:
- ✅ Improves recall (find more relevant chunks)
- ❌ Increases cost (multiple retrievals)
- ❌ May reduce precision

**Strategy**:
```python
# Strategy A: Always expand (high recall)
expanded_queries = expand_query(user_query)  # Returns 5 variants
results = []
for query in [user_query] + expanded_queries:
    results.extend(retrieve(query, top_k=5))
results = deduplicate_and_rerank(results)

# Strategy B: Conditional expansion (balanced)
initial_results = retrieve(user_query, top_k=10)
if len(initial_results) < 5 or max_score < 0.7:
    # Low quality - try expansion
    expanded = expand_query(user_query)
    for q in expanded:
        results.extend(retrieve(q, top_k=5))

# Strategy C: No expansion (fast, cheap)
results = retrieve(user_query, top_k=20)
```

**Recommendation**: Start with Strategy C, add expansion if recall issues

**Cost**: ~$0.001 per query expansion

---

## 💬 Response Generation Prompts

### When: After retrieval, before returning answer to user

### 8. RAG Answer Generation (Prompt 4.1) - PRIMARY

**Use Case**: Generate final answer from retrieved chunks

**When to Use**:
- ✅ For 80%+ of queries
- ✅ When retrieved context is sufficient
- ✅ Default answer generation prompt

**Input**:
```python
answer = await llm.run_prompt(
    prompt_id="4.1_rag_answer",
    presentation_title="Q4 Business Review",
    total_slides=20,
    section_list="Intro, Market Analysis, Financial Results, Strategy",
    retrieved_chunks=top_5_chunks,
    user_query="What was the revenue growth in 2024?"
)
```

**Output Structure**:
```
Direct answer: Revenue grew 75% from $1.2M to $2.1M in 2024.

Details:
- Revenue increased from $1.2M in Q1 to $2.1M in Q4 (Slide 10)
- This 75% growth exceeded the 50% target (Slide 12)
- Growth was consistent at 20-25% QoQ (Slide 10)
```

**Temperature**: 0.3 (slightly creative but grounded)

**Max Tokens**: 400 (concise answers preferred)

---

### 9. Multi-Hop Reasoning (Prompt 4.2)

**Use Case**: Answer complex questions requiring synthesis

**When to Use**:
- ⚡ When query intent = "EXPLANATION"
- ⚡ When query has "why", "how", or causal language
- ⚡ When answer requires connecting multiple chunks

**Decision Logic**:
```python
if intent == "EXPLANATION" or "why" in query.lower() or "how" in query.lower():
    answer_prompt = "4.2_multi_hop_reasoning"
else:
    answer_prompt = "4.1_rag_answer"
```

**Example Queries**:
- "Why did revenue increase?" (needs cause-effect reasoning)
- "How does the strategy address market challenges?" (synthesis)
- "What's the relationship between X and Y?" (connection)

**Not for**:
- "What was the revenue?" (simple fact)
- "List the key points" (no reasoning needed)

---

### 10. Comparison Answer (Prompt 4.5)

**Use Case**: Answer comparison questions

**When to Use**:
- ⚡ When query intent = "COMPARISON"
- ⚡ When query contains "vs", "compared to", "difference between"

**Example Queries**:
- "Q3 vs Q4 revenue"
- "Difference between Plan A and Plan B"
- "Compare sales across regions"

**Output Format**: Always use table structure for clarity

---

### 11. Slide Summary (Prompt 4.3)

**Use Case**: Summarize specific slide(s)

**When to Use**:
- ⚡ When user asks "summarize slide X"
- ⚡ For preview/tooltip in UI
- ⚡ When location intent detected

**Not for**: Answering content questions (use 4.1 instead)

---

### 12. Presentation Summary (Prompt 4.4)

**Use Case**: Generate executive summary

**When to Use**:
- ⚡ When intent = "SUMMARY"
- ⚡ When query asks for "overview", "summary", "main points"
- ⚡ First-time presentation view

**Retrieval Strategy**:
```python
# Don't just retrieve top-K chunks
# Instead, retrieve representative chunks from each section
summary_chunks = []
for section in presentation.sections:
    section_chunks = retrieve_from_section(
        section=section,
        query="main points and key takeaways",
        top_k=2
    )
    summary_chunks.extend(section_chunks)

summary = generate_summary(summary_chunks)
```

---

## 🛠️ Utility Prompts

### 13. Answer Quality Check (Prompt 7.1)

**Use Case**: Self-check answer before returning to user

**When to Use**:
- ✅ In production for ALL answers (if budget allows)
- ✅ During development/testing (always)
- ⚡ For high-stakes queries (enterprise, sensitive data)

**Implementation**:
```python
async def generate_checked_answer(query, chunks):
    # 1. Generate answer
    answer = await generate_answer(query, chunks)

    # 2. Quality check
    quality_check = await llm.run_prompt(
        prompt_id="7.1_answer_quality_check",
        user_query=query,
        generated_answer=answer,
        source_context=chunks
    )

    # 3. Handle failures
    if quality_check.overall_quality == "fail":
        # Log issue
        logger.warning(f"Quality check failed: {quality_check.issues}")

        # Option A: Regenerate with fixes
        answer = await regenerate_answer(
            query, chunks,
            guidance=quality_check.suggested_improvement
        )

        # Option B: Return safe fallback
        # answer = "I found some information but cannot provide a confident answer..."

    return answer
```

**Cost**: ~$0.003 per check (double the cost but worth it!)

**When to Skip**: Low-stakes applications, tight budget

---

### 14. Hallucination Detection (Prompt 7.2)

**Use Case**: Detect if answer contains unsupported claims

**When to Use**:
- ✅ Critical applications (medical, legal, financial)
- ⚡ Randomly sample 10% of answers in production
- ✅ During development

**Implementation**:
```python
async def safe_answer_generation(query, chunks):
    answer = await generate_answer(query, chunks)

    # Check for hallucinations
    hallucination_check = await llm.run_prompt(
        prompt_id="7.2_hallucination_detection",
        source_context=chunks,
        generated_answer=answer
    )

    if hallucination_check.hallucination_detected:
        # Log for review
        logger.error(f"Hallucination detected: {hallucination_check.hallucinated_claims}")

        # Remove hallucinated parts
        cleaned_answer = remove_unsupported_claims(
            answer,
            hallucination_check.hallucinated_claims
        )

        return cleaned_answer

    return answer
```

**Alternative**: Use stricter prompt instructions (cheaper)
```
"IMPORTANT: Only use information from the provided context.
If you don't know something, say 'I don't know' rather than guessing."
```

---

### 15. Follow-up Questions (Prompt 6.3)

**Use Case**: Suggest next questions to user

**When to Use**:
- ✅ After every successful answer (enhance UX)
- ✅ In chatbot/interactive applications

**Implementation**:
```python
async def generate_response_with_suggestions(query, answer, presentation):
    # Generate follow-ups
    suggestions = await llm.run_prompt(
        prompt_id="6.3_followup_questions",
        user_query=query,
        answer_provided=answer,
        presentation_topics=presentation.topics
    )

    return {
        "answer": answer,
        "follow_up_questions": suggestions.questions,
        "sources": chunks
    }
```

**Cost**: ~$0.001 per query

**UX Impact**: High - keeps users engaged

---

### 16. Clarification Request (Prompt 6.4)

**Use Case**: Ask user to clarify vague queries

**When to Use**:
- ⚡ When query is too vague to answer
- ⚡ When intent classification fails (confidence < 30%)
- ✅ In interactive applications

**Decision Flow**:
```python
def handle_vague_query(query, presentation):
    intent = classify_intent(query)

    if intent.confidence < 0.3:
        # Too vague - ask for clarification
        clarification = generate_clarification_request(
            query, presentation
        )  # Prompt 6.4
        return clarification

    elif intent.confidence < 0.6:
        # Somewhat vague - reformulate
        query = reformulate_query(query, presentation)  # Prompt 3.3
        return process_query(query)

    else:
        # Clear enough - proceed
        return process_query(query)
```

**Cost**: ~$0.001

---

## 📊 Decision Trees

### Decision Tree 1: Indexing Pipeline

```
New PPT Uploaded
    |
    ├─> Parse PPT → Extract slides, text, images
    |
    ├─> Run Section Detection (5.1) → Get sections
    |
    ├─> Run Metadata Extraction (5.2) → Get title, presenter, etc.
    |
    ├─> For each IMAGE:
    │   |
    │   ├─> Classify image type
    │   |   |
    │   │   ├─> Chart/Diagram → Run 2.1 (LLM Vision)
    │   │   ├─> Table → Try OCR, fallback to 2.2 (LLM Vision)
    │   │   ├─> Infographic → Run 2.3 (LLM Vision)
    │   │   └─> Image + Caption → OCR only (2.4)
    │   |
    │   └─> Store extracted data
    |
    ├─> Chunk slides
    │   |
    │   ├─> Element-level?
    │   │   ├─> Text chunks → Generate context (1.1)
    │   │   └─> Visual chunks → Generate context (1.3)
    │   |
    │   └─> Slide-level?
    │       └─> Generate context (1.2)
    |
    ├─> Generate embeddings
    |
    └─> Store in Vector DB
```

---

### Decision Tree 2: Query Pipeline

```
User Query Received
    |
    ├─> Classify Intent (3.2) → ALWAYS
    |   |
    │   ├─> High confidence (>80%) → Proceed
    │   ├─> Medium confidence (50-80%) → Reformulate (3.3)
    │   └─> Low confidence (<50%) → Ask Clarification (6.4)
    |
    ├─> Expand Query? (3.1)
    │   |
    │   ├─> High-stakes query → YES
    │   ├─> Initial retrieval poor → YES
    │   └─> Otherwise → NO
    |
    ├─> Retrieve Chunks
    │   |
    │   ├─> Adjust top_k based on intent:
    │   │   ├─> FACT_FINDING → top_k=5
    │   │   ├─> SUMMARY → top_k=20
    │   │   └─> Others → top_k=10
    │   |
    │   └─> Vector search + BM25 + Rerank
    |
    ├─> Select Answer Prompt based on Intent:
    │   |
    │   ├─> COMPARISON → Use 4.5
    │   ├─> EXPLANATION → Use 4.2
    │   ├─> SUMMARY → Use 4.4
    │   ├─> LOCATION → Use 6.1 (Slide Recommendation)
    │   └─> Others → Use 4.1 (RAG)
    |
    ├─> Generate Answer
    |
    ├─> Quality Check?
    │   |
    │   ├─> Production + budget → YES (7.1)
    │   ├─> Critical application → YES + Hallucination Check (7.2)
    │   └─> Development → ALWAYS
    |
    ├─> Generate Follow-ups (6.3) → If interactive
    |
    └─> Return Response
```

---

### Decision Tree 3: Image Processing

```
Image Detected in Slide
    |
    ├─> Classify Image
    │   |
    │   ├─> Text Ratio > 80%
    │   │   └─> Pure text → OCR only, no prompt
    │   |
    │   ├─> Has chart elements (bars, lines, axes)
    │   │   └─> Chart → LLM Vision + Prompt 2.1
    │   │       Cost: $0.015
    │   |
    │   ├─> Has table structure
    │   │   ├─> Simple table
    │   │   │   └─> OCR + Prompt 2.2
    │   │   └─> Complex table
    │   │       └─> LLM Vision + Prompt 2.2
    │   |
    │   ├─> Mixed visual + text + icons
    │   │   └─> Infographic → LLM Vision + Prompt 2.3
    │   |
    │   └─> Photo/Logo with caption
    │       └─> OCR + Prompt 2.4
    |
    └─> Check OCR confidence (if using OCR)
        |
        ├─> Confidence > 80% → Use OCR result
        |
        └─> Confidence < 80% → Fallback to LLM Vision
            Cost: $0.015
```

---

## 💡 Best Practices

### 1. Cost Optimization

**Expensive Prompts** (use sparingly):
- Image Analysis (2.1, 2.2, 2.3): $0.015 each
- Multi-hop Reasoning (4.2): Uses more tokens
- Answer Quality Check (7.1): Doubles generation cost

**Cheap Prompts** (use freely):
- Intent Classification (3.2): $0.001
- Query Expansion (3.1): $0.001
- Follow-up Questions (6.3): $0.001

**Cost Reduction Strategies**:
```python
# Strategy 1: Batch similar operations
contexts = await generate_contexts_batch(chunks, batch_size=20)

# Strategy 2: Use cheaper models for simple tasks
intent = await claude_haiku.classify(query)  # Haiku for classification
answer = await claude_sonnet.generate(query, chunks)  # Sonnet for generation

# Strategy 3: Cache aggressively
@cache(ttl=3600)
def generate_context(chunk):
    ...

# Strategy 4: Hybrid approach for images
if image.is_simple_table():
    data = ocr_extract(image)  # Free
else:
    data = llm_vision_analyze(image)  # $0.015
```

---

### 2. Quality Optimization

**High-Quality Pipeline** (recommended for production):
```python
async def high_quality_pipeline(query, presentation):
    # 1. Intent classification
    intent = await classify_intent(query)

    # 2. Reformulate if needed
    if intent.confidence < 0.7:
        query = await reformulate_query(query, presentation)

    # 3. Expand for better recall
    expanded = await expand_query(query)

    # 4. Retrieve with fusion
    chunks = await hybrid_retrieve([query] + expanded)

    # 5. Select best answer prompt
    prompt = select_answer_prompt(intent)

    # 6. Generate answer
    answer = await generate_answer(query, chunks, prompt)

    # 7. Quality check
    quality = await check_answer_quality(answer, chunks)
    if quality.overall_quality == "fail":
        answer = await regenerate_with_guidance(query, chunks, quality)

    # 8. Hallucination check for critical apps
    if is_critical_application:
        hallucination = await detect_hallucination(answer, chunks)
        if hallucination.detected:
            answer = clean_hallucinations(answer, hallucination)

    # 9. Generate follow-ups
    followups = await generate_followups(query, answer, presentation)

    return {
        "answer": answer,
        "sources": chunks,
        "follow_up_questions": followups
    }
```

---

### 3. Performance Optimization

**Latency Targets**:
- Intent Classification: < 200ms
- Retrieval: < 100ms
- Answer Generation: < 2s
- Total Query Time: < 3s

**Optimization Techniques**:

```python
# 1. Parallel execution
async def fast_pipeline(query):
    # Run classification and retrieval in parallel
    intent_task = classify_intent(query)
    retrieval_task = retrieve(query)

    intent, chunks = await asyncio.gather(intent_task, retrieval_task)

    # Generate answer
    answer = await generate_answer(query, chunks)
    return answer

# 2. Streaming responses
async def streaming_pipeline(query):
    chunks = await retrieve(query)

    # Stream answer as it's generated
    async for token in generate_answer_streaming(query, chunks):
        yield token

# 3. Caching
@lru_cache(maxsize=1000)
def get_presentation_context(presentation_id):
    # Cache presentation metadata, sections, etc.
    ...

# 4. Precompute when possible
# During indexing, precompute:
# - Section summaries
# - Key topics
# - Metadata
# So queries don't need to compute on-the-fly
```

---

### 4. Prompt Selection Matrix

| Query Type | Intent | Prompts Used | Cost | Latency |
|------------|--------|--------------|------|---------|
| **Simple fact** | FACT_FINDING | 3.2 → 4.1 | $0.002 | 1-2s |
| **Comparison** | COMPARISON | 3.2 → 4.5 | $0.003 | 2-3s |
| **Why/How** | EXPLANATION | 3.2 → 4.2 | $0.004 | 3-4s |
| **Summary** | SUMMARY | 3.2 → 4.4 | $0.005 | 3-5s |
| **Vague query** | Unknown | 3.2 → 3.3 → 4.1 | $0.004 | 2-3s |
| **Complex** | EXPLANATION | 3.2 → 3.1 → 4.2 → 7.1 | $0.010 | 4-6s |

---

### 5. Testing Checklist

**Before Production**:

- [ ] Test each prompt with 10+ examples
- [ ] Measure success rate (target: >90%)
- [ ] A/B test prompt variations
- [ ] Test edge cases:
  - [ ] Very short queries
  - [ ] Very long queries
  - [ ] Ambiguous queries
  - [ ] Out-of-scope queries
- [ ] Measure latency (target: <3s p95)
- [ ] Measure cost per query
- [ ] Test quality checks
- [ ] Verify hallucination detection

---

## 📈 Monitoring & Metrics

### Key Metrics to Track

**Per Prompt**:
- Usage count
- Average latency
- Cost per execution
- Success rate
- User satisfaction (if available)

**Quality Metrics**:
- Answer accuracy (manual eval)
- Hallucination rate
- Citation accuracy
- User thumbs up/down

**Cost Metrics**:
- Cost per query
- Cost per presentation indexed
- Monthly LLM spend by prompt type

---

**End of Usage Guide**
