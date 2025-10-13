# PPT Contextual Retrieval - Prompt Library

**Version**: 1.0
**Date**: 2025-10-10
**Purpose**: Collection of all LLM prompts used in the system

---

## 📋 Table of Contents
1. [Context Generation Prompts](#context-generation-prompts)
2. [Image Analysis Prompts](#image-analysis-prompts)
3. [Query Processing Prompts](#query-processing-prompts)
4. [Answer Generation Prompts](#answer-generation-prompts)
5. [Metadata Extraction Prompts](#metadata-extraction-prompts)

---

## 1️⃣ Context Generation Prompts

### 1.1 Chunk Context Generation (Primary)

**Purpose**: Generate contextual description for each text chunk to improve retrieval

**Template**:
```xml
<document>
{presentation_structure}
</document>

Here is the chunk we want to situate within the whole presentation:
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall presentation to improve search retrieval of the chunk. Answer only with the succinct context and nothing else.

Context to include:
- Slide position: {slide_number} of {total_slides}
- Section: {section_name}
- Previous slide topic: {prev_slide_title}
- Next slide topic: {next_slide_title}
- Visual elements present: {visual_summary}

Keep the context between 50-100 tokens. Focus on the role and relationship of this content within the broader presentation narrative.
```

**Variables**:
- `{presentation_structure}`: High-level outline of entire presentation
- `{chunk_content}`: The actual text chunk to contextualize
- `{slide_number}`: Current slide number
- `{total_slides}`: Total number of slides
- `{section_name}`: Section title this slide belongs to
- `{prev_slide_title}`: Title of previous slide
- `{next_slide_title}`: Title of next slide
- `{visual_summary}`: Brief description of charts/images on slide

**Example Input**:
```
Presentation Structure: "Q4 Business Review: Introduction (slides 1-3), Market Analysis (slides 4-8), Financial Results (slides 9-15), Future Strategy (slides 16-20)"

Chunk: "Revenue increased from $1.2M in Q1 to $2.1M in Q4, representing 75% growth year-over-year with consistent 20-25% quarter-over-quarter acceleration."

Slide: 10 of 20
Section: Financial Results
Previous: Company Overview
Next: Expense Breakdown
Visuals: Bar chart showing quarterly revenue trend
```

**Example Output**:
```
This content is from slide 10 of 20 in the 'Financial Results' section. It presents the key revenue growth metrics following the company overview and precedes a detailed expense analysis. The slide includes a bar chart visualizing the quarterly revenue trend that supports these growth statistics.
```

---

### 1.2 Slide-Level Context Generation

**Purpose**: Generate context for entire slide (when using slide-level chunking)

**Template**:
```xml
<document>
Presentation: {presentation_title}
Total Slides: {total_slides}
Sections: {section_list}
Current Section: {current_section}
</document>

<slide>
Slide {slide_number}: {slide_title}

Content:
{slide_content}

Speaker Notes:
{speaker_notes}

Visual Elements:
{visual_elements_description}
</slide>

Generate a concise context (75-125 tokens) that situates this slide within the presentation. Include:
1. The slide's role in the presentation flow
2. How it connects to previous and next slides
3. The main message or purpose of this slide
4. Significance of any visual elements

Output only the context, nothing else.
```

**Variables**:
- `{presentation_title}`: Title of presentation
- `{total_slides}`: Total slides
- `{section_list}`: Comma-separated list of sections
- `{current_section}`: Current section name
- `{slide_number}`: Slide number
- `{slide_title}`: Title of this slide
- `{slide_content}`: All text content from slide
- `{speaker_notes}`: Speaker notes (if any)
- `{visual_elements_description}`: Description of images/charts

---

### 1.3 Visual Element Context Generation

**Purpose**: Generate context for visual elements (charts, diagrams)

**Template**:
```xml
<presentation_context>
Presentation: {presentation_title}
Slide {slide_number} of {total_slides}
Section: {section_name}
Slide Title: {slide_title}
</presentation_context>

<visual_element>
Type: {visual_type}
Description: {visual_description}
Accompanying Text: {accompanying_text}
</visual_element>

Generate a brief context (30-50 tokens) explaining this visual element's role in the presentation narrative. Focus on:
- What it illustrates or supports
- How it relates to the presentation's main argument
- Its connection to surrounding content

Context only, no preamble.
```

---

## 2️⃣ Image Analysis Prompts

### 2.1 Chart/Diagram Analysis (LLM Vision)

**Purpose**: Extract data and insights from charts, diagrams, infographics

**Template**:
```xml
<presentation_context>
Presentation: {presentation_title}
Slide {slide_number} of {total_slides}
Section: {section_name}
Slide Title: {slide_title}
Previous Slide: {prev_slide_title}
Next Slide: {next_slide_title}
</presentation_context>

<task>
Analyze this chart/diagram image from the presentation above.

Provide a structured analysis:

1. **Type**: Identify the type (bar chart, line graph, pie chart, flowchart, diagram, etc.)

2. **Data Points**: Extract all visible data points, labels, and values
   - For charts: List all data series, axes labels, values
   - For diagrams: Describe all components and their relationships

3. **Key Insights**: What are the 2-3 main takeaways?
   - Trends (increasing, decreasing, stable)
   - Comparisons (biggest, smallest, differences)
   - Patterns or anomalies

4. **Narrative**: How does this visual support the presentation's story?

5. **Text Content**: Any text embedded in the image (titles, labels, annotations)

Be concise but comprehensive. Output in this exact format:

Type: [chart type]

Data:
[bullet points of data]

Insights:
[2-3 key insights]

Narrative:
[1-2 sentences on role in presentation]

Text:
[any text found in image]
</task>
```

**Example for Bar Chart**:
```
Type: Vertical bar chart with quarterly data

Data:
- X-axis: Q1 2024, Q2 2024, Q3 2024, Q4 2024
- Y-axis: Revenue in millions ($M)
- Q1: $1.2M
- Q2: $1.5M (25% increase)
- Q3: $1.8M (20% increase)
- Q4: $2.1M (17% increase)

Insights:
- Consistent upward revenue trend throughout 2024
- 75% total growth from Q1 to Q4
- Growth rate slightly decelerating (25% → 20% → 17%)

Narrative:
This chart supports the company's growth story presented in earlier slides, providing concrete evidence of sustained revenue expansion while hinting at market saturation discussed in later strategy slides.

Text:
- Title: "2024 Revenue Growth"
- Y-axis label: "Revenue ($M)"
- Annotation: "Target: $2M achieved in Q4"
```

---

### 2.2 Table Analysis (LLM Vision or OCR + LLM)

**Purpose**: Extract structured data from tables

**Template**:
```xml
<context>
Slide {slide_number}: {slide_title}
Section: {section_name}
</context>

<task>
Extract and analyze the table in this image.

Provide:

1. **Structure**: Number of rows and columns, headers

2. **Data**: Complete table data in markdown format

3. **Summary**: Key findings from the table (2-3 sentences)

4. **Comparisons**: Notable comparisons or patterns

Format:

Structure:
[description]

Data:
| Header 1 | Header 2 | Header 3 |
|----------|----------|----------|
| ...      | ...      | ...      |

Summary:
[2-3 sentences]

Comparisons:
[bullet points]
</task>
```

---

### 2.3 Infographic Analysis

**Purpose**: Analyze complex infographics with mixed text/visual

**Template**:
```xml
Analyze this infographic from slide {slide_number}: "{slide_title}"

Extract:

1. **Main Message**: What is the central message? (1 sentence)

2. **Components**: List all distinct sections/components

3. **Data & Statistics**: All numbers, percentages, metrics

4. **Visual Metaphors**: Describe any icons, symbols, or visual metaphors used

5. **Flow/Structure**: How is information organized? (top-to-bottom, circular, etc.)

6. **Text Content**: All readable text

Be thorough. This will be used for search and retrieval.

Output format:

Main Message:
[1 sentence]

Components:
1. [component name]: [brief description]
2. ...

Data:
- [statistic 1]
- [statistic 2]
...

Visual Elements:
- [description]

Flow:
[description]

Text:
[all text content]
```

---

### 2.4 Image with Caption

**Purpose**: Simple OCR + context for images with text captions

**Template**:
```xml
This is an image from slide {slide_number} in the "{section_name}" section.

Task:
1. Extract any visible text (captions, labels, watermarks)
2. Describe the image content (1 sentence)
3. Infer the purpose of this image in the presentation context

Format:

Text: [extracted text]
Description: [1 sentence]
Purpose: [1 sentence]
```

---

## 3️⃣ Query Processing Prompts

### 3.1 Query Expansion

**Purpose**: Expand user query to improve retrieval recall

**Template**:
```xml
User Query: "{user_query}"

Generate 3-5 semantically similar queries that would help retrieve relevant content from a presentation.

Consider:
- Synonyms and alternative phrasings
- More specific variants
- More general variants
- Related concepts

Output format (one per line):
1. [query variant 1]
2. [query variant 2]
3. [query variant 3]
4. [query variant 4]
5. [query variant 5]

No explanations, just the queries.
```

**Example**:
```
User Query: "revenue growth"

Output:
1. revenue increase
2. sales growth trends
3. financial performance improvement
4. year-over-year revenue expansion
5. income and earnings growth
```

---

### 3.2 Query Intent Classification

**Purpose**: Classify query intent to optimize retrieval strategy

**Template**:
```xml
Query: "{user_query}"

Classify this query's intent. Choose ONE:

1. FACT_FINDING - Seeking specific data, numbers, facts
2. TREND_ANALYSIS - Looking for patterns, trends, changes over time
3. COMPARISON - Comparing entities, time periods, options
4. EXPLANATION - Seeking to understand why or how something works
5. SUMMARY - Wanting overview or high-level summary
6. LOCATION - Finding where information is (which slide/section)

Output format:
Intent: [INTENT_TYPE]
Confidence: [high/medium/low]
Reasoning: [1 sentence why]

Example Query: "What was the revenue in Q4?"
Output:
Intent: FACT_FINDING
Confidence: high
Reasoning: User seeks a specific numerical data point.
```

---

### 3.3 Query Reformulation

**Purpose**: Reformulate ambiguous queries for better retrieval

**Template**:
```xml
<context>
Presentation: {presentation_title}
Sections: {section_list}
</context>

User Query: "{user_query}"

This query is ambiguous or poorly formed. Reformulate it into a clear, well-formed query that would retrieve the most relevant information from this presentation.

Consider:
- What is the user really asking?
- What presentation sections are most relevant?
- What specific information are they seeking?

Output only the reformulated query, nothing else.
```

**Example**:
```
Input Query: "what about sales"

Context: Presentation: "Q4 Business Review", Sections: "Market Analysis, Financial Results, Future Strategy"

Output: "What were the Q4 sales figures and sales growth trends?"
```

---

## 4️⃣ Answer Generation Prompts

### 4.1 RAG Answer Generation (Primary)

**Purpose**: Generate final answer from retrieved chunks

**Template**:
```xml
<context>
You are an AI assistant helping users understand presentation content. Answer questions accurately based ONLY on the provided context from the presentation.
</context>

<presentation_info>
Title: {presentation_title}
Total Slides: {total_slides}
Sections: {section_list}
</presentation_info>

<retrieved_context>
{retrieved_chunks}
</retrieved_context>

<user_query>
{user_query}
</user_query>

<instructions>
Based on the retrieved context above, answer the user's question.

Requirements:
1. **Answer directly and concisely**
2. **Cite slide numbers** when referencing specific information
3. **If context is insufficient**, say "Based on the available slides, I cannot find information about [topic]. This may be covered in slides not retrieved."
4. **Do NOT hallucinate** - only use information from the provided context
5. **Structure**: Start with direct answer, then provide supporting details
6. **Include relevant numbers/data** when available

Format:
[Direct answer in 1-2 sentences]

Details:
- [Supporting point 1 from slide X]
- [Supporting point 2 from slide Y]
- [etc.]

If the query asks about trends/comparisons, organize your answer accordingly.
</instructions>

Answer:
```

**Example**:
```
User Query: "What was the revenue growth in 2024?"

Retrieved Context:
- Slide 10: "Revenue increased from $1.2M in Q1 to $2.1M in Q4, representing 75% growth year-over-year."
- Slide 12: "This growth exceeded our 50% target and positions us well for 2025."

Answer:
The company achieved 75% revenue growth in 2024, growing from $1.2M in Q1 to $2.1M in Q4.

Details:
- Revenue grew from $1.2M to $2.1M throughout 2024 (Slide 10)
- This 75% growth exceeded the initial 50% target (Slide 12)
- Growth was consistent across quarters with 20-25% QoQ acceleration (Slide 10)
```

---

### 4.2 Multi-Hop Reasoning Answer

**Purpose**: Answer complex queries requiring synthesis across multiple chunks

**Template**:
```xml
<context>
The user's question requires synthesizing information from multiple parts of the presentation.
</context>

<retrieved_chunks>
{retrieved_chunks_with_metadata}
</retrieved_chunks>

<user_query>
{user_query}
</user_query>

<instructions>
This question requires connecting information across multiple slides.

Process:
1. Identify relevant information from each chunk
2. Determine the logical flow/connections
3. Synthesize into a coherent answer
4. Show your reasoning

Format your answer as:

**Answer**: [Direct answer to the question]

**Reasoning**:
1. From slide X: [key point]
2. From slide Y: [key point]
3. Connection: [how they relate]
4. Therefore: [conclusion]

**Supporting Evidence**:
- Slide X: [quote or data]
- Slide Y: [quote or data]

Be explicit about how you connected the information.
</instructions>

Answer:
```

---

### 4.3 Slide Summary Generation

**Purpose**: Summarize specific slide(s)

**Template**:
```xml
<slide_content>
{slide_content}
</slide_content>

Generate a concise summary of this slide.

Include:
1. Main topic (1 sentence)
2. Key points (3-5 bullet points)
3. Important data/metrics (if any)
4. Conclusion or takeaway (1 sentence)

Format:

**Topic**: [main topic]

**Key Points**:
- [point 1]
- [point 2]
- [point 3]

**Data**: [key metrics/numbers]

**Takeaway**: [conclusion]

Keep it under 100 words total.
```

---

### 4.4 Presentation Summary Generation

**Purpose**: Generate executive summary of entire presentation

**Template**:
```xml
<presentation>
Title: {presentation_title}
Slides: {total_slides}

Section Summaries:
{section_summaries}

Key Slides Content:
{key_slides_content}
</presentation>

Generate an executive summary of this presentation.

Structure:

# {presentation_title} - Executive Summary

## Overview
[2-3 sentences on what this presentation is about]

## Key Findings
[3-5 most important points, with slide references]

## Data Highlights
[Important numbers, metrics, achievements]

## Sections Covered
1. [Section 1]: [1 sentence summary]
2. [Section 2]: [1 sentence summary]
...

## Conclusion
[Main takeaway in 1-2 sentences]

Keep total length under 300 words.
```

---

### 4.5 Comparison Answer

**Purpose**: Answer comparison questions

**Template**:
```xml
<query>
{comparison_query}
</query>

<context>
{retrieved_chunks}
</context>

The user is asking for a comparison. Structure your answer to clearly show the comparison.

Format:

**Comparison: [Entity A] vs [Entity B]**

| Aspect | {Entity A} | {Entity B} |
|--------|-----------|-----------|
| [Aspect 1] | [Value/Description] | [Value/Description] |
| [Aspect 2] | [Value/Description] | [Value/Description] |
| ... | ... | ... |

**Key Differences**:
1. [Major difference 1]
2. [Major difference 2]

**Similarities**:
1. [Similarity 1]

**Conclusion**: [Which is better/different and why, based on data]

**Sources**: Slides [X, Y, Z]
```

---

## 5️⃣ Metadata Extraction Prompts

### 5.1 Section Detection

**Purpose**: Automatically detect section boundaries in presentation

**Template**:
```xml
<slides>
{slide_titles_and_content}
</slides>

Analyze these slide titles and group them into logical sections.

A section is a group of related slides covering one major topic.

Output format:

Section 1: [Section Name]
- Slide X: [Title]
- Slide Y: [Title]
...

Section 2: [Section Name]
- Slide A: [Title]
- Slide B: [Title]
...

Rules:
- Section names should be descriptive (not just "Section 1")
- Each section should have 2-8 slides
- First slide is usually "Introduction" or presentation title
- Last section is usually "Conclusion" or "Next Steps"
```

---

### 5.2 Presentation Metadata Extraction

**Purpose**: Extract key metadata from presentation

**Template**:
```xml
<presentation>
Slide 1 (Title Slide):
{slide_1_content}

Slide 2-3 (Early slides):
{slides_2_3_content}

Last Slide:
{last_slide_content}
</presentation>

Extract presentation metadata:

1. **Title**: The presentation title
2. **Presenter(s)**: Name(s) of presenter(s) if mentioned
3. **Date**: Date or time period covered
4. **Organization**: Company/organization name
5. **Topic**: Main topic/theme (1-2 sentences)
6. **Audience**: Intended audience (if inferable)

Output format:

Title: [title]
Presenter: [name or "Unknown"]
Date: [date or "Unknown"]
Organization: [org or "Unknown"]
Topic: [description]
Audience: [audience or "Unknown"]

If information is not present, use "Unknown".
```

---

### 5.3 Key Topics Extraction

**Purpose**: Extract main topics/themes from presentation

**Template**:
```xml
<presentation_content>
{all_slides_text_content}
</presentation_content>

Identify the 5-10 main topics or themes discussed in this presentation.

For each topic:
1. Topic name (2-4 words)
2. Brief description (1 sentence)
3. Slides where it appears
4. Importance (high/medium/low)

Output format:

1. **[Topic Name]**
   - Description: [1 sentence]
   - Slides: [slide numbers]
   - Importance: [high/medium/low]

2. **[Topic Name]**
   ...

Order by importance (most important first).
```

---

### 5.4 Actionable Items Extraction

**Purpose**: Extract action items, next steps, recommendations

**Template**:
```xml
<presentation_content>
{presentation_content}
</presentation_content>

Extract all actionable items, recommendations, or next steps mentioned in this presentation.

For each item:
- The action/recommendation
- Who should do it (if specified)
- Timeline (if specified)
- Which slide mentions it

Output format:

Actionable Items:

1. [Action description]
   - Owner: [who or "Not specified"]
   - Timeline: [when or "Not specified"]
   - Source: Slide [X]

2. [Action description]
   ...

If no actionable items found, output: "No explicit action items found."
```

---

## 6️⃣ Special Purpose Prompts

### 6.1 Slide Recommendation

**Purpose**: Recommend which slides user should look at

**Template**:
```xml
<user_query>
{user_query}
</user_query>

<all_slides_summary>
{slides_with_titles_and_brief_content}
</all_slides_summary>

Based on the user's question, recommend the top 3-5 slides they should review.

For each recommendation:
- Slide number and title
- Relevance (high/medium)
- Why this slide is relevant (1 sentence)

Output format:

Recommended Slides:

1. **Slide X: [Title]** (High Relevance)
   - Why: [reason]

2. **Slide Y: [Title]** (High Relevance)
   - Why: [reason]

3. **Slide Z: [Title]** (Medium Relevance)
   - Why: [reason]

Order by relevance.
```

---

### 6.2 Missing Information Detection

**Purpose**: Detect when presentation lacks information user is asking about

**Template**:
```xml
<user_query>
{user_query}
</user_query>

<retrieved_context>
{retrieved_chunks}
</retrieved_context>

<all_slides_titles>
{all_slide_titles}
</all_slides_titles>

Determine if the presentation contains information to answer this query.

Analysis:
1. **Can the query be answered?** (yes/no/partial)
2. **What information IS available?** (if any)
3. **What information is MISSING?** (if incomplete)
4. **Suggested response** (what to tell the user)

Output format:

Answerable: [yes/no/partial]

Available Information:
[What can be answered from the presentation]

Missing Information:
[What's not covered that the user is asking about]

Suggested Response:
[How to respond to the user - either provide available info and note what's missing, or clearly state it's not covered]
```

---

### 6.3 Follow-up Question Generation

**Purpose**: Suggest relevant follow-up questions

**Template**:
```xml
<user_query>
{user_query}
</user_query>

<answer_provided>
{answer_provided}
</answer_provided>

<presentation_context>
{presentation_topics}
</presentation_context>

Based on the user's question and the answer provided, suggest 3-5 relevant follow-up questions they might want to ask.

Questions should:
- Be natural follow-ups to their original query
- Explore related topics in the presentation
- Help them understand the topic more deeply
- Be actually answerable from the presentation

Output format:

Suggested Follow-up Questions:

1. [Question 1]
2. [Question 2]
3. [Question 3]
4. [Question 4]
5. [Question 5]

No explanations, just the questions.
```

---

### 6.4 Clarification Request

**Purpose**: Ask for clarification when query is too vague

**Template**:
```xml
<user_query>
{user_query}
</user_query>

<presentation_context>
Title: {presentation_title}
Sections: {section_list}
Key Topics: {key_topics}
</presentation_context>

This query is too vague or broad. Generate a helpful clarification request.

Include:
1. Acknowledge what they're asking about
2. List 2-4 specific aspects they might be interested in
3. Ask them to clarify

Keep it friendly and helpful.

Output format:

I can help you with information about {topic}. This presentation covers several aspects:

- {aspect 1}
- {aspect 2}
- {aspect 3}
- {aspect 4}

Which aspect would you like to know more about? Or feel free to ask a more specific question.
```

---

## 7️⃣ Validation & Quality Prompts

### 7.1 Answer Quality Check

**Purpose**: Self-check answer quality before returning to user

**Template**:
```xml
<user_query>
{user_query}
</user_query>

<generated_answer>
{generated_answer}
</generated_answer>

<source_context>
{source_chunks}
</source_context>

Evaluate this generated answer on these criteria:

1. **Accuracy**: Is the answer factually correct based on source context? (yes/no)
2. **Completeness**: Does it fully answer the query? (yes/no/partial)
3. **Relevance**: Is the answer directly relevant to the query? (yes/no)
4. **Grounding**: Is every claim backed by the source context? (yes/no)
5. **Clarity**: Is the answer clear and well-structured? (yes/no)

Output format:

Accuracy: [yes/no] - [brief reason if no]
Completeness: [yes/no/partial] - [what's missing if partial]
Relevance: [yes/no] - [reason if no]
Grounding: [yes/no] - [ungrounded claims if no]
Clarity: [yes/no] - [issues if no]

Overall Quality: [pass/fail]

If "fail", provide:
Issues: [list of issues]
Suggested Improvement: [how to fix]
```

---

### 7.2 Hallucination Detection

**Purpose**: Detect if generated answer contains hallucinations

**Template**:
```xml
<source_context>
{source_chunks}
</source_context>

<generated_answer>
{generated_answer}
</generated_answer>

Check if the generated answer contains any hallucinations (information not present in source context).

For each claim in the answer:
1. Identify the claim
2. Find supporting text in source context
3. Mark as [SUPPORTED] or [UNSUPPORTED]

Output format:

Claim Analysis:

1. "[Claim 1 text]"
   - Status: [SUPPORTED/UNSUPPORTED]
   - Source: [quote from context] or "NOT FOUND"

2. "[Claim 2 text]"
   - Status: [SUPPORTED/UNSUPPORTED]
   - Source: [quote from context] or "NOT FOUND"

...

Hallucination Detected: [yes/no]
Hallucinated Claims: [list if any]
```

---

## 8️⃣ Prompt Engineering Best Practices

### General Guidelines

1. **Use XML tags** for structure - easier for LLM to parse
2. **Be explicit** about output format - always show examples
3. **Set constraints** - token limits, bullet counts, etc.
4. **One task per prompt** - don't mix multiple objectives
5. **Provide context** - always include relevant metadata
6. **Use examples** - show desired output format
7. **Think step-by-step** - for complex reasoning tasks
8. **Specify edge cases** - handle unknowns explicitly

### Token Limits by Prompt Type

| Prompt Type | Max Input Tokens | Target Output Tokens |
|-------------|-----------------|---------------------|
| Context Generation | 3000 | 50-100 |
| Image Analysis | 2000 + image | 150-250 |
| Query Expansion | 200 | 100 |
| Answer Generation | 8000 | 200-400 |
| Summary | 10000 | 150-300 |

### Temperature Settings

| Task | Recommended Temperature |
|------|------------------------|
| Context Generation | 0.0 (deterministic) |
| Data Extraction | 0.0 |
| Answer Generation | 0.3 (slightly creative) |
| Query Expansion | 0.5 (more creative) |
| Summarization | 0.3 |

---

## 9️⃣ Prompt Versioning

**Version History**:

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-10-10 | Initial prompt library |
| | | - Context generation prompts |
| | | - Image analysis prompts |
| | | - Query processing prompts |
| | | - Answer generation prompts |

**Testing Protocol**:
- Test each prompt with 10+ examples
- Measure quality metrics (accuracy, relevance)
- A/B test prompt variations
- Track performance in production

---

**End of Prompt Library**
