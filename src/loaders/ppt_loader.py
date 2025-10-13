"""
Custom LangChain document loader for PowerPoint files.
"""
from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document
from typing import List, Optional
import hashlib
import json
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from PIL import Image
import io
from loguru import logger


class PPTLoader(BaseLoader):
    """
    LangChain Document Loader for PowerPoint (.pptx) files.

    Extracts slides with metadata and optionally extracts images.
    """

    def __init__(
        self,
        file_path: str,
        extract_images: bool = True,
        include_speaker_notes: bool = True,
        extract_tables: bool = True
    ):
        """
        Initialize PPT Loader.

        Args:
            file_path: Path to .pptx file
            extract_images: Whether to extract images from slides
            include_speaker_notes: Whether to include speaker notes
            extract_tables: Whether to extract table data
        """
        self.file_path = file_path
        self.extract_images = extract_images
        self.include_speaker_notes = include_speaker_notes
        self.extract_tables = extract_tables

    def load(self) -> List[Document]:
        """
        Load PPT and return list of LangChain Documents.

        Each slide becomes a Document with comprehensive metadata.
        """
        try:
            prs = Presentation(self.file_path)
            documents = []

            # Extract presentation metadata
            presentation_id = hashlib.md5(self.file_path.encode()).hexdigest()[:12]
            title = self._extract_presentation_title(prs)

            logger.info(f"Loading presentation: {title} ({len(prs.slides)} slides)")

            # Detect sections (if any)
            sections = self._detect_sections(prs)

            for slide_idx, slide in enumerate(prs.slides):
                slide_num = slide_idx + 1

                # Extract slide content
                slide_text = self._extract_slide_text(slide)
                slide_title = self._extract_slide_title(slide)

                # Extract speaker notes
                speaker_notes = ""
                if self.include_speaker_notes and slide.has_notes_slide:
                    try:
                        speaker_notes = slide.notes_slide.notes_text_frame.text
                    except Exception as e:
                        logger.warning(f"Failed to extract notes from slide {slide_num}: {e}")

                # Extract images
                images_info = []
                if self.extract_images:
                    images_info = self._extract_images_info(slide, slide_num)

                # Extract tables
                tables_info = []
                if self.extract_tables:
                    tables_info = self._extract_tables(slide)

                # Determine section
                section = self._get_section_for_slide(slide_num, sections, prs)

                # Build comprehensive metadata
                # Convert complex objects to JSON strings for Pinecone compatibility
                metadata = {
                    "source": self.file_path,
                    "presentation_id": presentation_id,
                    "presentation_title": title,
                    "slide_number": slide_num,
                    "total_slides": len(prs.slides),
                    "slide_title": slide_title,
                    "section": section,
                    "speaker_notes": speaker_notes,
                    "has_images": len(images_info) > 0,
                    "image_count": len(images_info),
                    "images": json.dumps(images_info),  # Convert to JSON string
                    "has_tables": len(tables_info) > 0,
                    "table_count": len(tables_info),
                    "tables": json.dumps(tables_info),  # Convert to JSON string
                    "type": "slide"
                }

                # Combine all text content
                full_content = self._build_full_content(
                    slide_text,
                    speaker_notes,
                    tables_info
                )

                # Create Document
                doc = Document(
                    page_content=full_content,
                    metadata=metadata
                )
                documents.append(doc)

            logger.info(f"Successfully loaded {len(documents)} slides")
            return documents

        except Exception as e:
            logger.error(f"Failed to load PPT {self.file_path}: {e}")
            raise

    def _extract_presentation_title(self, prs: Presentation) -> str:
        """Extract presentation title from first slide."""
        if len(prs.slides) > 0:
            first_slide = prs.slides[0]
            if first_slide.shapes.title:
                return first_slide.shapes.title.text.strip()
        return "Untitled Presentation"

    def _extract_slide_title(self, slide) -> str:
        """Extract title from slide."""
        try:
            if slide.shapes.title:
                return slide.shapes.title.text.strip()
        except Exception:
            pass
        return ""

    def _extract_slide_text(self, slide) -> str:
        """Extract all text content from slide."""
        text_parts = []

        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                text_parts.append(shape.text.strip())

        return "\n\n".join(text_parts)

    def _extract_images_info(self, slide, slide_num: int) -> List[dict]:
        """
        Extract information about images in slide.

        Returns list of image metadata (not the actual images yet).
        """
        images = []

        for shape_idx, shape in enumerate(slide.shapes):
            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                try:
                    # Get image info
                    image_info = {
                        "slide_number": slide_num,
                        "shape_index": shape_idx,
                        "type": "picture",
                        "has_image": True,
                        # Store reference for later processing
                        "shape_id": shape.shape_id
                    }

                    # Try to get dimensions
                    if hasattr(shape, "width") and hasattr(shape, "height"):
                        image_info["width"] = shape.width
                        image_info["height"] = shape.height

                    images.append(image_info)

                except Exception as e:
                    logger.warning(f"Failed to process image in slide {slide_num}: {e}")

            # Check for charts (treated as images for vision analysis)
            elif shape.shape_type == MSO_SHAPE_TYPE.CHART:
                images.append({
                    "slide_number": slide_num,
                    "shape_index": shape_idx,
                    "type": "chart",
                    "has_image": True,
                    "shape_id": shape.shape_id
                })

        return images

    def _extract_tables(self, slide) -> List[dict]:
        """Extract table data from slide."""
        tables = []

        for shape in slide.shapes:
            if shape.shape_type == MSO_SHAPE_TYPE.TABLE:
                try:
                    table = shape.table
                    rows = []

                    for row in table.rows:
                        row_data = []
                        for cell in row.cells:
                            row_data.append(cell.text.strip())
                        rows.append(row_data)

                    tables.append({
                        "rows": rows,
                        "row_count": len(table.rows),
                        "col_count": len(table.columns)
                    })

                except Exception as e:
                    logger.warning(f"Failed to extract table: {e}")

        return tables

    def _detect_sections(self, prs: Presentation) -> List[dict]:
        """
        Detect section boundaries based on slide titles.

        Simple heuristic: Look for slides that look like section headers.
        """
        sections = []

        section_keywords = [
            "introduction", "overview", "agenda",
            "background", "conclusion", "summary",
            "results", "analysis", "recommendations"
        ]

        for idx, slide in enumerate(prs.slides):
            title = self._extract_slide_title(slide).lower()

            # Check if this looks like a section header
            is_section_header = False

            # Heuristic 1: Contains section keywords
            if any(keyword in title for keyword in section_keywords):
                is_section_header = True

            # Heuristic 2: Slide has only title (section divider)
            text = self._extract_slide_text(slide)
            if title and len(text.split()) < 20:  # Very short content
                is_section_header = True

            if is_section_header:
                sections.append({
                    "title": self._extract_slide_title(slide),
                    "start_slide": idx + 1
                })

        return sections

    def _get_section_for_slide(
        self,
        slide_num: int,
        sections: List[dict],
        prs: Presentation
    ) -> str:
        """Determine which section a slide belongs to."""
        if not sections:
            # Default sections
            total_slides = len(prs.slides)
            if slide_num == 1:
                return "Introduction"
            elif slide_num == total_slides:
                return "Conclusion"
            else:
                return "Main Content"

        # Find the section this slide belongs to
        current_section = "Introduction"
        for section in sections:
            if slide_num >= section["start_slide"]:
                current_section = section["title"]
            else:
                break

        return current_section

    def _build_full_content(
        self,
        slide_text: str,
        speaker_notes: str,
        tables: List[dict]
    ) -> str:
        """Build full content string from all text sources."""
        parts = []

        # Add main slide text
        if slide_text:
            parts.append(slide_text)

        # Add table data
        for table in tables:
            table_text = self._format_table(table["rows"])
            parts.append(f"\n[Table]\n{table_text}")

        # Add speaker notes
        if speaker_notes:
            parts.append(f"\n[Speaker Notes]\n{speaker_notes}")

        return "\n\n".join(parts)

    def _format_table(self, rows: List[List[str]]) -> str:
        """Format table as markdown."""
        if not rows:
            return ""

        # Simple markdown table
        lines = []
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")

        # Add header separator after first row
        if len(lines) > 1:
            col_count = len(rows[0])
            separator = "| " + " | ".join(["---"] * col_count) + " |"
            lines.insert(1, separator)

        return "\n".join(lines)
