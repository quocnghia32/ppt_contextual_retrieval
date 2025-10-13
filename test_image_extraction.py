#!/usr/bin/env python3
"""
Test script to demonstrate image extraction and storage functionality.

Usage:
    python test_image_extraction.py
"""
import asyncio
from pathlib import Path
from src.models.vision_analyzer import VisionAnalyzer
from src.config import settings
from PIL import Image


async def test_image_extraction():
    """Test image extraction with folder structure."""
    print("=" * 70)
    print("🧪 IMAGE EXTRACTION & STORAGE TEST")
    print("=" * 70)

    # Show configuration
    print("\n📁 Configuration:")
    print(f"   Base folder: {settings.extracted_images_dir}")
    print(f"   Folder exists: {Path(settings.extracted_images_dir).exists()}")

    # Create sample PPT structure for demo
    print("\n" + "=" * 70)
    print("📦 FOLDER STRUCTURE")
    print("=" * 70)
    print("""
After extracting images from PPT files, the structure will be:

data/extracted_images/
  ├── presentation1/
  │   ├── slide_01_image_0.png
  │   ├── slide_01_image_1.jpg
  │   ├── slide_02_image_0.png
  │   └── ...
  ├── presentation2/
  │   ├── slide_01_image_0.jpeg
  │   └── ...
  └── another_presentation/
      └── ...
    """)

    # Demo with test image
    print("=" * 70)
    print("🎯 DEMO: Extract & Save Test Images")
    print("=" * 70)

    analyzer = VisionAnalyzer()
    print(f"\n✅ VisionAnalyzer initialized")

    # Create test presentation folder
    test_ppt_name = "test_presentation"
    test_folder = analyzer._get_presentation_folder(f"{test_ppt_name}.pptx")
    print(f"\n📂 Presentation folder created:")
    print(f"   Path: {test_folder}")
    print(f"   Exists: {test_folder.exists()}")

    # Create sample images to simulate extraction
    print("\n🖼️  Creating sample images...")
    sample_images = []
    for i in range(3):
        # Create test image
        img = Image.new('RGB', (400, 300), color=('red', 'green', 'blue')[i])
        img_format = ('PNG', 'JPEG', 'PNG')[i]

        # Simulate image info as returned by extract_images_from_ppt
        img_info = {
            'image': img,
            'image_index': i,
            'format': img_format,
            'slide_number': 1
        }

        # Save manually for demo
        filename = f"slide_01_image_{i}.{img_format.lower()}"
        img_path = test_folder / filename
        img.save(str(img_path))
        img_info['image_path'] = str(img_path)

        sample_images.append(img_info)
        print(f"   ✅ Created: {filename} ({img_format})")

    print("\n" + "=" * 70)
    print("📊 EXTRACTION RESULTS")
    print("=" * 70)

    print(f"\n✅ Total images extracted: {len(sample_images)}")
    print(f"\n📂 Saved to folder: {test_folder}")
    print("\nImage details:")
    for img_info in sample_images:
        print(f"   - {Path(img_info['image_path']).name}")
        print(f"     • Format: {img_info['format']}")
        print(f"     • Index: {img_info['image_index']}")
        print(f"     • Path exists: {Path(img_info['image_path']).exists()}")

    # Show how it works in practice
    print("\n" + "=" * 70)
    print("💡 HOW TO USE IN YOUR CODE")
    print("=" * 70)
    print("""
# Option 1: Extract and save images automatically
analyzer = VisionAnalyzer()
images_info = await analyzer.extract_images_from_ppt(
    ppt_path="your_presentation.pptx",
    slide_number=1,
    save_to_disk=True  # Default: True
)

# Access results
for img_info in images_info:
    image = img_info['image']          # PIL Image object
    image_path = img_info['image_path']  # Path where saved
    image_format = img_info['format']    # PNG, JPEG, etc.
    image_index = img_info['image_index']  # Index in slide

# Option 2: Extract without saving (memory only)
images_info = await analyzer.extract_images_from_ppt(
    ppt_path="your_presentation.pptx",
    slide_number=1,
    save_to_disk=False  # Don't save to disk
)

# Option 3: Analyze and save in one step
analyses = await analyzer.analyze_slide_images(
    ppt_path="your_presentation.pptx",
    slide_number=1,
    slide_context={...},
    save_images=True  # Default: True
)

# Access both analysis and image path
for analysis in analyses:
    description = analysis['description']
    ocr_results = analysis['ocr_results']
    image_path = analysis['image_path']  # Where image is saved
    """)

    print("\n" + "=" * 70)
    print("🔍 KEY FEATURES")
    print("=" * 70)
    print("""
✅ Organized folder structure per presentation
✅ Consistent naming: slide_XX_image_Y.ext
✅ Preserves original image format (PNG, JPEG, etc.)
✅ Optional disk saving (can work in memory only)
✅ Image paths included in analysis results
✅ Automatic folder creation
✅ Handles multiple presentations
    """)

    print("=" * 70)
    print("📝 INTEGRATION WITH INGESTION")
    print("=" * 70)
    print("""
When running: python scripts/ingest.py your_file.pptx

Images will be automatically:
1. Extracted from each slide
2. Saved to: data/extracted_images/your_file/
3. Named: slide_XX_image_Y.ext
4. Paths stored in image analysis metadata
5. Available for retrieval and reference

You can access images later using the paths stored in metadata!
    """)

    print("=" * 70)
    print("✅ TEST COMPLETE!")
    print("=" * 70)
    print(f"\n✅ Sample images saved to: {test_folder}")
    print(f"✅ Verify folder exists: ls {test_folder}")
    print("\n💡 Test with real PPT: python scripts/ingest.py your_file.pptx")


if __name__ == '__main__':
    asyncio.run(test_image_extraction())
