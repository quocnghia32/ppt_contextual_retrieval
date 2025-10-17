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
        # self.llm = ChatOpenAI(
        #     model=self.model_name,
        #     api_key=settings.openai_api_key,
        #     max_tokens=1000,  # Increased for comprehensive analysis with OCR
        #     temperature=0.0  # Deterministic
        # )
        from langchain_openai import AzureChatOpenAI
        self.llm = AzureChatOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            azure_deployment=self.model_name,
            api_version=settings.azure_openai_api_version_chat,
            temperature=0.0
        )
        logger.info(f"Vision analyzer initialized with model: {self.model_name}")


    @with_retry(max_attempts=3)
    async def analyze_image(
        self,
        image_base64: str,
        image_format: str,
    ) -> str:
        """
        Analyze chart/diagram using vision model.

        Args:
            image_base64: Base64-encoded image
            image_format: Image format
        Returns:
            str with analysis results
        """
        # Wait for rate limit (increased estimate for comprehensive analysis)
        await rate_limiter.wait_if_needed(
            key="openai_vision",
            estimated_tokens=500  # Increased for comprehensive prompt + OCR output
        )

        # Convert image to base64 data URL with proper MIME type
        mime = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "webp": "image/webp",
        }.get(image_format.lower(), "application/octet-stream")

        data_url = f"data:{mime};base64,{image_base64}"

        try:
            # Create message with proper format for OpenAI vision API
            msg = [
                HumanMessage(
                    content=[
                        {"type": "text", "text": generate_context_from_image},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ]
                )
            ]

            # Call vision model
            response = await self.llm.ainvoke(msg)

            # Parse response with new comprehensive structure
            #analysis = self._parse_comprehensive_analysis(response.content)
            response.content

            logger.debug(f"Image analyzed: {response.content[:100]}...")

            return response.content

        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            return """
                "description": "NO INFORMATION",
                "key_statistics": "NO INFORMATION",
                "key_message": "NO INFORMATION",
                "context_insight": "NO INFORMATION",
                "summary": "NO INFORMATION",
                "ocr_results": "NO INFORMATION",
                "error": str(e)
            """

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

    # def _parse_chart_analysis(self, text: str) -> Dict:
    #     """Parse structured analysis from LLM response (legacy format)."""
    #     return {
    #         "type": self._extract_section(text, "Type"),
    #         "data": self._extract_section(text, "Data"),
    #         "insights": self._extract_section(text, "Insights"),
    #         "text": self._extract_section(text, "Text")
    #     }

    # def _parse_comprehensive_analysis(self, text: str) -> Dict:
    #     """
    #     Parse comprehensive analysis with all fields from generate_context_from_image prompt.

    #     Returns:
    #         Dict with keys: description, key_statistics, key_message,
    #                        context_insight, summary, ocr_results
    #     """
    #     return {
    #         "description": self._extract_section(text, "Description"),
    #         "key_statistics": self._extract_section(text, "Key Statistics"),
    #         "key_message": self._extract_section(text, "Key Message"),
    #         "context_insight": self._extract_section(text, "Context/Insight"),
    #         "summary": self._extract_section(text, "Summary"),
    #         "ocr_results": self._extract_section(text, "OCR Results")
    #     }

    # def _extract_section(self, text: str, section_name: str) -> str:
    #     """Extract a section from structured response."""
    #     lines = text.split("\n")
    #     in_section = False
    #     section_lines = []

    #     # List of all possible section headers (for stopping condition)
    #     all_sections = [
    #         "Description:", "Key Statistics:", "Key Message:",
    #         "Context/Insight:", "Summary:", "OCR Results:",
    #         "Type:", "Data:", "Insights:", "Text:",
    #         "Structure:", "Summary:"
    #     ]

    #     for line in lines:
    #         # Check if this line starts with our target section
    #         # Handle both "Section:" and "**Section**:" formats
    #         line_stripped = line.strip()
    #         target_patterns = [
    #             f"{section_name}:",
    #             f"**{section_name}**:",
    #             f"- **{section_name}**:"
    #         ]

    #         if any(line_stripped.startswith(pattern) for pattern in target_patterns):
    #             in_section = True
    #             # Get content after colon if on same line
    #             for pattern in target_patterns:
    #                 if line_stripped.startswith(pattern):
    #                     content = line_stripped.split(":", 1)[1].strip() if ":" in line_stripped else ""
    #                     if content:
    #                         section_lines.append(content)
    #                     break
    #             continue

    #         if in_section:
    #             # Stop at next section header
    #             if any(section in line_stripped for section in all_sections):
    #                 break

    #             # Add line to current section
    #             section_lines.append(line)

    #     result = "\n".join(section_lines).strip()

    #     # Return "NO INFORMATION" if section is empty
    #     return result if result else "NO INFORMATION"

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

    async def process_image(
        self,
        ppt_path: str,
        img_info: Dict,
        image_bytes: bytes,
        save_images: bool = True
    ) -> str:
        
        # Extract images (now returns list of dicts with image info)
        slide_number = img_info["slide_number"]

        idx = img_info['image_index']
        image_format = img_info['image_format']

        logger.info(f"Analyzing image {idx + 1} from slide {slide_number}")

        # Analyze the image
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        vision_description = await self.analyze_image(image_base64, image_format)

        if save_images:
            # Create filename: slide_XX_image_Y.ext
            filename = f"slide_{slide_number:02d}_image_{idx}.{image_format.lower()}"
            presentation_folder = self._get_presentation_folder(ppt_path)
            image_path = presentation_folder / filename

            # Save image
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            img_info['image_path'] = str(image_path)

            logger.debug(f"Saved image to: {image_path}")
        
        return vision_description