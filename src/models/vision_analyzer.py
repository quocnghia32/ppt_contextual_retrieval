"""
Vision model integration for analyzing images from PPT slides.

Uses GPT-4o mini for image analysis with rate limiting.
"""
from typing import Dict, List, Optional, Union
import base64
import io
from pathlib import Path
from PIL import Image
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from loguru import logger

from src.config import settings
from src.utils.rate_limiter import rate_limiter, with_retry
from src.prompts import generate_context_from_image


class VisionAnalyzer:
    """
    Analyze images using GPT-4o mini vision model.

    Handles rate limiting and provides structured analysis.
    """

    def __init__(self, model: str = None):
        """
        Initialize vision analyzer.

        Args:
            model: Vision model to use (default: from settings)
        """
        self.model_name = model or settings.vision_model
        self.llm = ChatOpenAI(
            model=self.model_name,
            api_key=settings.openai_api_key,
            max_tokens=1000,  # Increased for comprehensive analysis with OCR
            temperature=0.0  # Deterministic
        )
        logger.info(f"Vision analyzer initialized with model: {self.model_name}")

    def _image_to_base64(self, image: Union[Image.Image, str, Path]) -> str:
        """
        Convert image to base64 data URL with proper MIME type.

        Args:
            image: PIL Image object, file path string, or Path object

        Returns:
            Data URL string with proper MIME type
        """
        # If image is a path (string or Path), load it and detect MIME type
        if isinstance(image, (str, Path)):
            image_path = Path(image)

            # Detect MIME type based on extension
            mime_types = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
                ".gif": "image/gif",
                ".bmp": "image/bmp"
            }
            mime = mime_types.get(image_path.suffix.lower(), "application/octet-stream")

            # Read file and encode
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        # If image is PIL Image, convert to PNG
        else:
            buffered = io.BytesIO()
            # Detect format from PIL Image if available
            image_format = getattr(image, 'format', 'PNG')
            if image_format is None:
                image_format = 'PNG'

            image.save(buffered, format=image_format)
            image_bytes = buffered.getvalue()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')

            # Map PIL format to MIME type
            format_to_mime = {
                'PNG': 'image/png',
                'JPEG': 'image/jpeg',
                'JPG': 'image/jpeg',
                'WEBP': 'image/webp',
                'GIF': 'image/gif',
                'BMP': 'image/bmp'
            }
            mime = format_to_mime.get(image_format.upper(), 'image/png')

        # Create proper data URL
        data_url = f"data:{mime};base64,{image_base64}"
        return data_url

    @with_retry(max_attempts=3)
    async def analyze_chart(
        self,
        image: Image.Image,
        slide_context: Dict
    ) -> Dict:
        """
        Analyze chart/diagram using vision model.

        Args:
            image: PIL Image object
            slide_context: Context about the slide

        Returns:
            Dict with analysis results
        """
        # Wait for rate limit (increased estimate for comprehensive analysis)
        await rate_limiter.wait_if_needed(
            key="openai_vision",
            estimated_tokens=500  # Increased for comprehensive prompt + OCR output
        )

        # Convert image to base64 data URL with proper MIME type
        data_url = self._image_to_base64(image)

        # Build comprehensive prompt combining context and detailed analysis
        prompt_text = f"""
{generate_context_from_image}

**Context Information:**
- Slide Number: {slide_context.get('slide_number')}
- Section: {slide_context.get('section', 'Unknown')}
- Slide Title: {slide_context.get('slide_title', 'Unknown')}
- Total Slides: {slide_context.get('total_slides', 'Unknown')}

Please analyze this image and provide the response in the exact format specified above.
"""

        try:
            # Create message with proper format for OpenAI vision API
            msg = [
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ]
                )
            ]

            # Call vision model
            response = await self.llm.ainvoke(msg)

            # Parse response with new comprehensive structure
            analysis = self._parse_comprehensive_analysis(response.content)

            logger.debug(f"Image analyzed: {analysis.get('description', 'unknown')[:100]}...")

            return analysis

        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            return {
                "description": "NO INFORMATION",
                "key_statistics": "NO INFORMATION",
                "key_message": "NO INFORMATION",
                "context_insight": "NO INFORMATION",
                "summary": "NO INFORMATION",
                "ocr_results": "NO INFORMATION",
                "error": str(e)
            }

    @with_retry(max_attempts=3)
    async def analyze_table(
        self,
        image: Image.Image,
        slide_context: Dict
    ) -> Dict:
        """
        Analyze table in image.

        Args:
            image: PIL Image object
            slide_context: Slide context

        Returns:
            Table analysis
        """
        await rate_limiter.wait_if_needed(key="openai_vision", estimated_tokens=300)

        # Convert image to base64 data URL with proper MIME type
        data_url = self._image_to_base64(image)

        prompt_text = f"""
Extract and analyze the table in this image from slide {slide_context.get('slide_number')}.

Provide:

1. **Structure**: Number of rows and columns, headers

2. **Data**: Complete table data in markdown format

3. **Summary**: Key findings (2-3 sentences)

Format as:

Structure:
[description]

Data:
| Header 1 | Header 2 |
|----------|----------|
| ...      | ...      |

Summary:
[summary]
"""

        try:
            # Create message with proper format for OpenAI vision API
            msg = [
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ]
                )
            ]

            response = await self.llm.ainvoke(msg)

            return {
                "structure": self._extract_section(response.content, "Structure"),
                "data": self._extract_section(response.content, "Data"),
                "summary": self._extract_section(response.content, "Summary")
            }

        except Exception as e:
            logger.error(f"Table analysis failed: {e}")
            return {"error": str(e)}

    def _parse_chart_analysis(self, text: str) -> Dict:
        """Parse structured analysis from LLM response (legacy format)."""
        return {
            "type": self._extract_section(text, "Type"),
            "data": self._extract_section(text, "Data"),
            "insights": self._extract_section(text, "Insights"),
            "text": self._extract_section(text, "Text")
        }

    def _parse_comprehensive_analysis(self, text: str) -> Dict:
        """
        Parse comprehensive analysis with all fields from generate_context_from_image prompt.

        Returns:
            Dict with keys: description, key_statistics, key_message,
                           context_insight, summary, ocr_results
        """
        return {
            "description": self._extract_section(text, "Description"),
            "key_statistics": self._extract_section(text, "Key Statistics"),
            "key_message": self._extract_section(text, "Key Message"),
            "context_insight": self._extract_section(text, "Context/Insight"),
            "summary": self._extract_section(text, "Summary"),
            "ocr_results": self._extract_section(text, "OCR Results")
        }

    def _extract_section(self, text: str, section_name: str) -> str:
        """Extract a section from structured response."""
        lines = text.split("\n")
        in_section = False
        section_lines = []

        # List of all possible section headers (for stopping condition)
        all_sections = [
            "Description:", "Key Statistics:", "Key Message:",
            "Context/Insight:", "Summary:", "OCR Results:",
            "Type:", "Data:", "Insights:", "Text:",
            "Structure:", "Summary:"
        ]

        for line in lines:
            # Check if this line starts with our target section
            # Handle both "Section:" and "**Section**:" formats
            line_stripped = line.strip()
            target_patterns = [
                f"{section_name}:",
                f"**{section_name}**:",
                f"- **{section_name}**:"
            ]

            if any(line_stripped.startswith(pattern) for pattern in target_patterns):
                in_section = True
                # Get content after colon if on same line
                for pattern in target_patterns:
                    if line_stripped.startswith(pattern):
                        content = line_stripped.split(":", 1)[1].strip() if ":" in line_stripped else ""
                        if content:
                            section_lines.append(content)
                        break
                continue

            if in_section:
                # Stop at next section header
                if any(section in line_stripped for section in all_sections):
                    break

                # Add line to current section
                section_lines.append(line)

        result = "\n".join(section_lines).strip()

        # Return "NO INFORMATION" if section is empty
        return result if result else "NO INFORMATION"

    def _get_presentation_folder(self, ppt_path: str) -> Path:
        """
        Get folder path for storing extracted images from a presentation.

        Args:
            ppt_path: Path to PPT file

        Returns:
            Path to presentation-specific folder
        """
        from src.config import settings

        # Get PPT filename without extension
        ppt_name = Path(ppt_path).stem

        # Create presentation-specific folder
        presentation_folder = Path(settings.extracted_images_dir) / ppt_name
        presentation_folder.mkdir(parents=True, exist_ok=True)

        return presentation_folder

    async def extract_images_from_ppt(
        self,
        ppt_path: str,
        slide_number: int,
        save_to_disk: bool = True
    ) -> List[Dict[str, any]]:
        """
        Extract images from a specific slide and optionally save to disk.

        Args:
            ppt_path: Path to PPT file
            slide_number: Slide number (1-indexed)
            save_to_disk: Whether to save images to disk (default: True)

        Returns:
            List of dicts with image info:
            {
                'image': PIL.Image,
                'image_path': str (if saved),
                'image_index': int,
                'format': str
            }
        """
        try:
            prs = Presentation(ppt_path)
            slide = prs.slides[slide_number - 1]
            images_info = []

            # Get presentation folder for saving
            if save_to_disk:
                presentation_folder = self._get_presentation_folder(ppt_path)

            for shape_idx, shape in enumerate(slide.shapes):
                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    try:
                        # Get image bytes
                        image_stream = shape.image.blob
                        image = Image.open(io.BytesIO(image_stream))

                        # Detect image format
                        image_format = image.format or 'PNG'

                        image_info = {
                            'image': image,
                            'image_index': shape_idx,
                            'format': image_format,
                            'slide_number': slide_number
                        }

                        # Save to disk if requested
                        if save_to_disk:
                            # Create filename: slide_XX_image_Y.ext
                            filename = f"slide_{slide_number:02d}_image_{shape_idx}.{image_format.lower()}"
                            image_path = presentation_folder / filename

                            # Save image
                            image.save(str(image_path))
                            image_info['image_path'] = str(image_path)

                            logger.debug(f"Saved image to: {image_path}")

                        images_info.append(image_info)

                    except Exception as e:
                        logger.warning(f"Failed to extract image {shape_idx} from slide {slide_number}: {e}")

            logger.info(f"Extracted {len(images_info)} images from slide {slide_number}")
            if save_to_disk and images_info:
                logger.info(f"Images saved to: {presentation_folder}")

            return images_info

        except Exception as e:
            logger.error(f"Failed to extract images from PPT: {e}")
            return []

    async def analyze_slide_images(
        self,
        ppt_path: str,
        slide_number: int,
        slide_context: Dict,
        save_images: bool = True
    ) -> List[Dict]:
        """
        Analyze all images in a slide.

        Args:
            ppt_path: Path to PPT file
            slide_number: Slide number
            slide_context: Slide metadata
            save_images: Whether to save extracted images to disk (default: True)

        Returns:
            List of image analyses with metadata
        """
        # Extract images (now returns list of dicts with image info)
        images_info = await self.extract_images_from_ppt(
            ppt_path,
            slide_number,
            save_to_disk=save_images
        )
        analyses = []

        for img_info in images_info:
            image = img_info['image']
            idx = img_info['image_index']

            logger.info(f"Analyzing image {idx + 1}/{len(images_info)} from slide {slide_number}")

            # Analyze the image
            analysis = await self.analyze_chart(image, slide_context)

            # Add image metadata to analysis
            analysis["image_index"] = idx
            analysis["image_format"] = img_info.get('format', 'unknown')

            # Add image path if saved to disk
            if 'image_path' in img_info:
                analysis["image_path"] = img_info['image_path']
                logger.debug(f"Analysis includes image path: {img_info['image_path']}")

            analyses.append(analysis)

        return analyses


# Convenience function
async def analyze_ppt_images(
    ppt_path: str,
    slides_to_analyze: Optional[List[int]] = None
) -> Dict[int, List[Dict]]:
    """
    Analyze images in PPT slides.

    Args:
        ppt_path: Path to PPT file
        slides_to_analyze: List of slide numbers to analyze (None = all)

    Returns:
        Dict mapping slide numbers to image analyses
    """
    analyzer = VisionAnalyzer()
    prs = Presentation(ppt_path)

    if slides_to_analyze is None:
        slides_to_analyze = list(range(1, len(prs.slides) + 1))

    results = {}

    for slide_num in slides_to_analyze:
        slide = prs.slides[slide_num - 1]

        # Build context
        slide_context = {
            "slide_number": slide_num,
            "total_slides": len(prs.slides),
            "slide_title": slide.shapes.title.text if slide.shapes.title else "",
            "section": "Main Content"  # Would need section detection
        }

        # Analyze images
        analyses = await analyzer.analyze_slide_images(
            ppt_path,
            slide_num,
            slide_context
        )

        if analyses:
            results[slide_num] = analyses

    return results
