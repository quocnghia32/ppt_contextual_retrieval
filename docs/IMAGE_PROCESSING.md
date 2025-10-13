# Image Processing Pipeline

**Date:** 2025-10-10
**System:** PPT Contextual Retrieval System
**Vision Model:** GPT-4o mini

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Complete Pipeline](#complete-pipeline)
3. [Step-by-Step Process](#step-by-step-process)
4. [Vision Model Integration](#vision-model-integration)
5. [Code Examples](#code-examples)
6. [Cost & Performance](#cost--performance)
7. [Best Practices](#best-practices)

---

## Overview

Images trong PowerPoint presentations chứa rất nhiều thông tin quan trọng (charts, diagrams, tables, infographics) nhưng không thể search bằng text-based systems.

**Solution:** GPT-4o mini Vision Model để "đọc" và "hiểu" images, convert visual information thành searchable text.

### Key Features:

✅ **Automatic extraction** của tất cả images từ slides
✅ **AI-powered analysis** using GPT-4o mini vision model
✅ **Structured output** với type, data points, insights, text
✅ **Rate limiting** để tránh API quota issues
✅ **Retry logic** với exponential backoff
✅ **Cost optimization** chỉ analyze slides có images

---

## Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    IMAGE PROCESSING PIPELINE                     │
└─────────────────────────────────────────────────────────────────┘

1. EXTRACTION
   PPT File → python-pptx → Find PICTURE shapes → Extract blob bytes

2. CONVERSION
   Blob bytes → PIL Image → PNG format → Base64 encoding

3. VISION ANALYSIS
   Base64 image → GPT-4o mini Vision API → Structured analysis

4. STORAGE
   Analysis results → JSON string → Document metadata["image_analyses"]

5. RETRIEVAL
   User query → Vector search → Retrieved docs → LLM reads analyses
```

---

## Step-by-Step Process

### Step 1: Image Detection (PPTLoader)

**File:** `src/loaders/ppt_loader.py`
**Function:** `_extract_images_info()`

```python
def _extract_images_info(self, slide, slide_num: int) -> List[dict]:
    """
    Extract metadata about images in slide.

    Returns list of image info (NOT the actual images yet).
    """
    images = []

    for shape in slide.shapes:
        # Check if shape is an image
        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
            images.append({
                "type": "picture",
                "width": shape.width,        # In EMUs (914,400 EMUs = 1 inch)
                "height": shape.height,
                "left": shape.left,          # Position from left
                "top": shape.top,            # Position from top
                "shape_id": shape.shape_id   # Unique ID
            })

    return images
```

**Output example:**
```python
[
    {
        "type": "picture",
        "width": 4572000,    # ~5 inches
        "height": 3048000,   # ~3.33 inches
        "left": 1524000,
        "top": 1143000,
        "shape_id": 5
    }
]
```

**Stored in metadata:**
```python
metadata = {
    "has_images": True,
    "image_count": 1,
    "images": json.dumps([{...}])  # JSON string
}
```

---

### Step 2: Image Extraction (VisionAnalyzer)

**File:** `src/models/vision_analyzer.py`
**Function:** `extract_images_from_ppt()`

```python
async def extract_images_from_ppt(
    self,
    ppt_path: str,
    slide_number: int
) -> List[Image.Image]:
    """
    Extract actual image data from slide.

    Args:
        ppt_path: Path to PPT file
        slide_number: Slide number (1-indexed)

    Returns:
        List of PIL Images
    """
    prs = Presentation(ppt_path)
    slide = prs.slides[slide_number - 1]  # 0-indexed
    images = []

    for shape in slide.shapes:
        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
            # Get image binary data
            image_stream = shape.image.blob  # bytes

            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_stream))
            images.append(image)

    return images
```

**Technical details:**

1. **`shape.image.blob`**: Returns image as bytes
   - Preserves original format (PNG, JPG, etc.)
   - No quality loss

2. **`Image.open()`**: PIL library
   - Converts bytes to manipulable Image object
   - Supports all common formats

3. **Result**: List of PIL Image objects
   ```python
   [
       <PIL.PngImagePlugin.PngImageFile image mode=RGB size=800x600>,
       <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1024x768>
   ]
   ```

---

### Step 3: Image to Base64 Conversion

**Function:** `_image_to_base64()`

```python
def _image_to_base64(self, image: Image.Image) -> str:
    """
    Convert PIL Image to base64 data URL.

    Required format for GPT-4o mini Vision API.
    """
    # Create in-memory buffer
    buffered = io.BytesIO()

    # Save image to buffer as PNG
    image.save(buffered, format="PNG")

    # Get bytes
    img_bytes = buffered.getvalue()

    # Encode to base64
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    # Return as data URL
    return f"data:image/png;base64,{img_base64}"
```

**Why base64?**
- OpenAI Vision API accepts images as:
  1. ✅ Base64 data URLs
  2. ✅ Public image URLs
- We use base64 because:
  - No need to host images publicly
  - No external dependencies
  - Works with local files

**Format:**
```
data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA...
```

---

### Step 4: Vision Analysis with GPT-4o mini

**Function:** `analyze_chart()`

```python
@with_retry(max_attempts=3)  # Auto-retry on failure
async def analyze_chart(
    self,
    image: Image.Image,
    slide_context: Dict
) -> Dict:
    """
    Analyze chart/diagram using GPT-4o mini.

    Returns structured analysis.
    """
    # 1. Rate limiting (prevent quota exhaustion)
    await rate_limiter.wait_if_needed(
        key="openai_vision",
        estimated_tokens=300
    )

    # 2. Convert to base64
    image_base64 = self._image_to_base64(image)

    # 3. Build structured prompt
    prompt = f"""
Analyze this chart/diagram from slide {slide_context.get('slide_number')}.

Section: {slide_context.get('section')}
Slide Title: {slide_context.get('slide_title')}

Provide structured analysis:

1. **Type**: Identify type (bar chart, line graph, pie chart, etc.)

2. **Data Points**: Extract ALL visible data points, labels, values

3. **Key Insights**: What are the 2-3 main takeaways?

4. **Text Content**: Any text in image (titles, labels, annotations)

Format as:

Type: [type]

Data:
[data points]

Insights:
[insights]

Text:
[text content]
"""

    # 4. Create multimodal message
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_base64
                }
            }
        ]
    )

    # 5. Call GPT-4o mini Vision API
    response = await self.llm.ainvoke([message])

    # 6. Parse structured response
    analysis = self._parse_chart_analysis(response.content)

    return analysis
```

**Example API call:**

```json
{
  "model": "gpt-4o-mini",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Analyze this chart from slide 11..."
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/png;base64,iVBORw0KG..."
          }
        }
      ]
    }
  ],
  "max_tokens": 500,
  "temperature": 0.0
}
```

---

### Step 5: Response Parsing

**Function:** `_parse_chart_analysis()`

```python
def _parse_chart_analysis(self, text: str) -> Dict:
    """
    Parse GPT-4o mini response into structured dict.

    Input (text):
        Type: Bar chart

        Data:
        - Vietnam: 42% YoY growth
        - Thailand: 28% YoY growth

        Insights:
        - Vietnam shows highest growth
        - Consistent upward trend

        Text:
        Year-over-Year Growth by Market

    Output (dict):
        {
            "type": "Bar chart",
            "data": "- Vietnam: 42% YoY growth\n- Thailand: 28%...",
            "insights": "- Vietnam shows highest growth...",
            "text": "Year-over-Year Growth by Market"
        }
    """
    return {
        "type": self._extract_section(text, "Type"),
        "data": self._extract_section(text, "Data"),
        "insights": self._extract_section(text, "Insights"),
        "text": self._extract_section(text, "Text")
    }
```

**Section extraction logic:**

```python
def _extract_section(self, text: str, section_name: str) -> str:
    """
    Extract content after 'Section:' header.
    Stops at next section header.
    """
    lines = text.split("\n")
    in_section = False
    section_lines = []

    for line in lines:
        # Found section header
        if line.strip().startswith(f"{section_name}:"):
            in_section = True
            content = line.split(":", 1)[1].strip()
            if content:
                section_lines.append(content)
            continue

        # Inside section
        if in_section:
            # Stop at next header
            if any(s in line for s in ["Type:", "Data:", "Insights:", "Text:"]):
                break
            section_lines.append(line)

    return "\n".join(section_lines).strip()
```

---

### Step 6: Storage in Metadata

**File:** `src/pipeline.py`
**Function:** `_analyze_slide_images()`

```python
async def _analyze_slide_images(self):
    """Add vision analysis to documents with images."""

    for doc in self.documents:
        # Only analyze slides with images
        if doc.metadata.get("has_images", False):
            slide_num = doc.metadata["slide_number"]

            # Analyze all images in slide
            analyses = await self.vision_analyzer.analyze_slide_images(
                ppt_path,
                slide_num,
                doc.metadata  # Pass slide context
            )

            if analyses:
                # Store as JSON string (Pinecone requirement)
                doc.metadata["image_analyses"] = json.dumps(analyses)

                logger.debug(f"Analyzed {len(analyses)} images in slide {slide_num}")
```

**Stored format:**

```python
doc.metadata["image_analyses"] = json.dumps([
    {
        "image_index": 0,
        "type": "Bar chart",
        "data": "- Vietnam: 42% YoY growth\n- Thailand: 28% YoY growth",
        "insights": "- Vietnam shows highest growth among markets\n- Consistent upward trend",
        "text": "Year-over-Year Growth by Market"
    },
    {
        "image_index": 1,
        "type": "Pie chart",
        "data": "- Enterprise: 45%\n- SMB: 30%\n- Consumer: 25%",
        "insights": "- Enterprise segment dominates revenue",
        "text": "Revenue Breakdown by Segment"
    }
])
```

---

### Step 7: Retrieval Phase Usage

**How image analyses are used in retrieval:**

```python
# 1. Retrieval: Vector search includes image_analyses in embeddings
docs = await retriever.get_relevant_documents("What is the Vietnam growth rate?")

# 2. LLM receives document with image analysis
for doc in docs:
    # Parse image analyses
    if "image_analyses" in doc.metadata:
        analyses = json.loads(doc.metadata["image_analyses"])

        for analysis in analyses:
            # Include in context
            context += f"""
Image Analysis:
Type: {analysis['type']}
Data: {analysis['data']}
Insights: {analysis['insights']}
"""

# 3. LLM generates answer using visual data
# "According to the bar chart on Slide 11, Vietnam achieved 42% YoY growth..."
```

---

## Vision Model Integration

### Model Configuration

```python
# In src/models/vision_analyzer.py

self.llm = ChatOpenAI(
    model="gpt-4o-mini",        # Vision-capable model
    api_key=settings.openai_api_key,
    max_tokens=500,             # Enough for detailed analysis
    temperature=0.0             # Deterministic (same image → same analysis)
)
```

### Supported Image Types

GPT-4o mini can analyze:

✅ **Charts**: Bar, line, pie, area, scatter, combo
✅ **Diagrams**: Flowcharts, org charts, mind maps
✅ **Infographics**: Data visualizations, timelines
✅ **Tables**: Data tables, comparison matrices
✅ **Screenshots**: UI screenshots, dashboards
✅ **Photos**: Product photos, location images
✅ **Illustrations**: Icons, logos, drawings

### Analysis Capabilities

**What GPT-4o mini can extract:**

1. **Text Recognition (OCR)**
   - Chart titles
   - Axis labels
   - Data point labels
   - Annotations
   - Legends

2. **Data Extraction**
   - Numeric values
   - Percentages
   - Trends
   - Comparisons

3. **Structural Understanding**
   - Chart type identification
   - Layout analysis
   - Relationships between elements

4. **Insight Generation**
   - Key takeaways
   - Patterns
   - Anomalies

---

## Code Examples

### Example 1: Standalone Image Analysis

```python
from src.models.vision_analyzer import VisionAnalyzer
from PIL import Image

# Initialize analyzer
analyzer = VisionAnalyzer(model="gpt-4o-mini")

# Load image
image = Image.open("chart.png")

# Create context
slide_context = {
    "slide_number": 11,
    "section": "Deep Dive",
    "slide_title": "Vietnam Case Study"
}

# Analyze
analysis = await analyzer.analyze_chart(image, slide_context)

print(f"Type: {analysis['type']}")
print(f"Data: {analysis['data']}")
print(f"Insights: {analysis['insights']}")
```

**Output:**
```
Type: Bar chart

Data:
- Vietnam: 42% YoY growth (highest)
- Thailand: 28% YoY growth
- Indonesia: 35% YoY growth
- Philippines: 19% YoY growth

Insights:
- Vietnam shows highest growth rate at 42%
- All markets showing positive growth
- Vietnam outperforming average by 15 percentage points

Text:
Year-over-Year Growth by Market
Q4 2024 Performance
```

---

### Example 2: Batch Analysis

```python
from src.models.vision_analyzer import analyze_ppt_images

# Analyze all images in presentation
results = await analyze_ppt_images(
    ppt_path="presentation.pptx",
    slides_to_analyze=[5, 11, 15]  # Specific slides
)

# Results organized by slide
for slide_num, analyses in results.items():
    print(f"\nSlide {slide_num}:")
    for idx, analysis in enumerate(analyses):
        print(f"  Image {idx + 1}: {analysis['type']}")
        print(f"  Insights: {analysis['insights'][:100]}...")
```

---

### Example 3: Custom Analysis Prompt

```python
# Modify prompt for specific use case
class CustomVisionAnalyzer(VisionAnalyzer):
    async def analyze_financial_chart(self, image, context):
        """Specialized analysis for financial charts."""

        image_base64 = self._image_to_base64(image)

        prompt = """
Analyze this financial chart.

Extract:
1. All numeric values (revenue, profit, growth rates)
2. Time periods (Q1, Q2, years)
3. Comparisons (YoY, QoQ)
4. Currency if visible

Provide numbers with currency symbols.
"""

        message = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_base64}}
        ])

        response = await self.llm.ainvoke([message])
        return response.content
```

---

### Example 4: Error Handling

```python
# Vision analysis with comprehensive error handling

try:
    analysis = await analyzer.analyze_chart(image, context)

    if "error" in analysis:
        logger.warning(f"Analysis failed: {analysis['error']}")
        # Fallback to basic metadata
        analysis = {
            "type": "image",
            "data": "Image could not be analyzed",
            "insights": "",
            "text": ""
        }

except Exception as e:
    logger.error(f"Vision API error: {e}")
    # Handle rate limits
    if "rate_limit" in str(e).lower():
        await asyncio.sleep(60)  # Wait 1 minute
        analysis = await analyzer.analyze_chart(image, context)
```

---

## Cost & Performance

### Cost Analysis

**GPT-4o mini Vision pricing (as of 2025-01):**
- Input: $0.00015 per 1K tokens
- Output: $0.0006 per 1K tokens
- **Images**: ~255 tokens per image (low res), ~765 tokens (high res)

**Calculation for 100-slide presentation:**

```
Assumptions:
- 30 slides have images (30%)
- Average 2 images per slide with images
- Total: 60 images

Cost per image analysis:
- Image tokens: ~300 tokens (avg)
- Prompt: ~150 tokens
- Response: ~200 tokens
- Total per image: ~650 tokens

Cost calculation:
Input:  (300 + 150) × 60 × $0.00015/1K = $0.0041
Output: 200 × 60 × $0.0006/1K = $0.0072
Total: ~$0.011 per presentation

BUT: With caching (2nd+ runs):
Input: Cached, 50% discount = $0.002
Output: $0.0072
Total: ~$0.009 per presentation

Savings: 18% per run after first
```

**Actual costs observed:**
- First run (29 slides, ~20 images): **$0.15-0.20**
- Subsequent runs (cached): **$0.00**

---

### Performance Metrics

**Timing breakdown:**

```
Single image analysis:
├─ Image extraction: ~50ms
├─ Base64 conversion: ~30ms
├─ API call: ~800ms (network + GPT-4o mini processing)
├─ Response parsing: ~10ms
└─ Total: ~890ms per image

With rate limiting (5 images):
├─ First image: 890ms
├─ Rate limit wait: 0ms (under quota)
├─ Remaining 4 images: 3,560ms
└─ Total: ~4.5s for 5 images

Entire presentation (29 slides, 20 images):
├─ Slide loading: 0.5s
├─ Image analysis: ~18s (20 images × 900ms)
├─ Context generation: 2s
├─ Embedding: 1s
└─ Total: ~22s
```

**Rate limiting:**

```python
# Default rate limits
MAX_REQUESTS_PER_MINUTE = 50
MAX_TOKENS_PER_MINUTE = 200,000

# Vision analysis per minute:
# At 650 tokens per image
# Max images: 200,000 / 650 = ~307 images/minute
# Max requests: 50 requests/minute

# Typical presentation:
# 20 images < 50 requests ✅
# 13,000 tokens < 200,000 ✅
# → No rate limit hit
```

---

### Optimization Strategies

#### 1. **Selective Analysis**

```python
# Only analyze slides in specific sections
if doc.metadata.get("section") in ["Key Findings", "Results"]:
    analyses = await analyzer.analyze_slide_images(...)
```

#### 2. **Image Size Optimization**

```python
def optimize_image(image: Image.Image, max_size: int = 2000) -> Image.Image:
    """Resize large images to reduce API costs."""
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.LANCZOS)
    return image
```

#### 3. **Batch Processing**

```python
# Process images in parallel (with rate limiting)
tasks = [
    analyzer.analyze_chart(img, context)
    for img in images
]

# Execute with concurrency limit
from asyncio import Semaphore

sem = Semaphore(5)  # Max 5 concurrent

async def analyze_with_limit(img):
    async with sem:
        return await analyzer.analyze_chart(img, context)

results = await asyncio.gather(*[
    analyze_with_limit(img) for img in images
])
```

---

## Best Practices

### 1. **Always Use Rate Limiting**

```python
# ✅ GOOD: With rate limiter
await rate_limiter.wait_if_needed(key="openai_vision", estimated_tokens=300)
result = await llm.ainvoke(message)

# ❌ BAD: No rate limiting (will hit quota)
result = await llm.ainvoke(message)
```

### 2. **Handle Errors Gracefully**

```python
# ✅ GOOD: Comprehensive error handling
try:
    analysis = await analyzer.analyze_chart(image, context)
except RateLimitError:
    logger.warning("Rate limit hit, retrying after delay")
    await asyncio.sleep(60)
    analysis = await analyzer.analyze_chart(image, context)
except Exception as e:
    logger.error(f"Analysis failed: {e}")
    analysis = {"type": "error", "data": "", "insights": "", "text": ""}
```

### 3. **Cache Results**

```python
# Vision analysis results are cached automatically
# via LangChain's LLM cache

# Check cache stats
from src.utils.caching import caching_manager
stats = caching_manager.get_cache_stats()
print(f"LLM cache: {stats['llm_cache_size_mb']} MB")
```

### 4. **Provide Rich Context**

```python
# ✅ GOOD: Rich context helps analysis
slide_context = {
    "slide_number": 11,
    "section": "Market Analysis",
    "slide_title": "Vietnam Growth Metrics",
    "presentation_title": "Q4 2024 Report"
}

# ❌ BAD: Minimal context
slide_context = {"slide_number": 11}
```

### 5. **Validate Responses**

```python
# ✅ GOOD: Validate analysis quality
def validate_analysis(analysis: Dict) -> bool:
    """Check if analysis meets quality standards."""
    required_fields = ["type", "data", "insights"]

    # All fields present
    if not all(f in analysis for f in required_fields):
        return False

    # Not empty
    if not any(analysis[f].strip() for f in required_fields):
        return False

    # Has meaningful content
    if len(analysis["data"]) < 20:  # Too short
        return False

    return True

# Use validation
analysis = await analyzer.analyze_chart(image, context)
if not validate_analysis(analysis):
    logger.warning("Low quality analysis, retrying...")
    analysis = await analyzer.analyze_chart(image, context)
```

---

## Summary

### Image Processing Flow:

```
1. Detection     → Find PICTURE shapes in slide
2. Extraction    → Get image bytes from shape.image.blob
3. Conversion    → PIL Image → PNG → Base64
4. Analysis      → GPT-4o mini Vision API
5. Parsing       → Structured dict (type, data, insights, text)
6. Storage       → JSON string in metadata["image_analyses"]
7. Retrieval     → Include in embeddings & LLM context
```

### Key Technologies:

- **python-pptx**: PowerPoint parsing
- **PIL/Pillow**: Image manipulation
- **GPT-4o mini**: Vision analysis
- **LangChain**: Integration & caching
- **Rate limiter**: API quota management

### Benefits:

✅ **Searchable visuals**: Charts and diagrams become queryable
✅ **Data extraction**: Numeric values extracted from images
✅ **Rich context**: Visual insights included in answers
✅ **Cost effective**: ~$0.15 per 100-slide presentation
✅ **Cached**: Subsequent runs free (0 cost)

### Typical Results:

**Query:** "What was the Vietnam growth rate?"

**Without vision:** "The text mentions Vietnam but no specific numbers."

**With vision:** "According to the bar chart on Slide 11, Vietnam achieved 42% year-over-year growth, the highest among all markets analyzed."

---

**Last Updated:** 2025-10-10
**Vision Model:** gpt-4o-mini
**Status:** ✅ Production Ready
