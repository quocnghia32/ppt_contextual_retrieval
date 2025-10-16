#!/usr/bin/env python3
"""
PPT Ingestion Script - Index PowerPoint presentations

Index name is configured via PINECONE_INDEX_NAME in .env file.

Usage:
    python scripts/ingest.py path/to/presentation.pptx
    python scripts/ingest.py path/to/folder/*.pptx --batch
    python scripts/ingest.py presentation.pptx --no-context --no-vision
"""
import sys
import asyncio
import argparse
from pathlib import Path
from typing import List
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import PPTContextualRetrievalPipeline
from src.config import settings
from loguru import logger


async def ingest_single_file(
    ppt_path: str,
    use_contextual: bool = True,
    use_vision: bool = True,
    use_reranking: bool = True,
    extract_images: bool = True,
    include_notes: bool = True
) -> dict:
    """
    Ingest a single PowerPoint file.

    Args:
        ppt_path: Path to .pptx file
        use_contextual: Add contextual descriptions
        use_vision: Analyze images with GPT-4o mini
        use_reranking: Enable Cohere reranking
        extract_images: Extract image metadata
        include_notes: Include speaker notes

    Returns:
        Indexing statistics
    """
    ppt_file = Path(ppt_path)

    if not ppt_file.exists():
        raise FileNotFoundError(f"File not found: {ppt_path}")

    if not ppt_file.suffix.lower() == '.pptx':
        raise ValueError(f"Not a PowerPoint file: {ppt_path}")

    logger.info(f"Starting ingestion: {ppt_file.name}")
    logger.info(f"Index name: {settings.pinecone_index_name} (from .env)")
    logger.info(f"Options: contextual={use_contextual}, vision={use_vision}")

    # Create pipeline (uses index from .env)
    pipeline = PPTContextualRetrievalPipeline(
        use_contextual=use_contextual,
        use_vision=use_vision,
        use_reranking=use_reranking
    )

    # Index presentation
    start_time = time.time()

    stats = await pipeline.index_presentation(
        str(ppt_file.absolute()),
        extract_images=extract_images,
        include_notes=include_notes,
    )

    elapsed = time.time() - start_time

    stats['index_name'] = settings.pinecone_index_name
    stats['elapsed_seconds'] = round(elapsed, 2)
    stats['file_path'] = str(ppt_file.absolute())

    return stats


async def ingest_batch(
    ppt_files: List[str],
    **kwargs
) -> List[dict]:
    """
    Ingest multiple PowerPoint files.

    Args:
        ppt_files: List of .pptx file paths
        **kwargs: Arguments passed to ingest_single_file

    Returns:
        List of indexing statistics
    """
    results = []
    total = len(ppt_files)

    logger.info(f"Batch ingestion: {total} files")

    for idx, ppt_path in enumerate(ppt_files, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {idx}/{total}: {Path(ppt_path).name}")
        logger.info(f"{'='*60}")

        try:
            stats = await ingest_single_file(ppt_path, **kwargs)
            results.append(stats)

            logger.info(f"✅ Success: {stats['slides']} slides, {stats['chunks']} chunks")

        except Exception as e:
            logger.error(f"❌ Failed: {e}")
            results.append({
                'file_path': ppt_path,
                'error': str(e),
                'success': False
            })

    return results


def print_summary(results: List[dict]):
    """Print ingestion summary."""
    print("\n" + "="*60)
    print("INGESTION SUMMARY")
    print("="*60)

    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]

    print(f"\n✅ Successful: {len(successful)}/{len(results)}")
    print(f"❌ Failed: {len(failed)}/{len(results)}")

    if successful:
        print("\n📊 Statistics:")
        total_slides = sum(r['slides'] for r in successful)
        total_chunks = sum(r['chunks'] for r in successful)
        total_time = sum(r['elapsed_seconds'] for r in successful)

        print(f"  - Total slides: {total_slides}")
        print(f"  - Total chunks: {total_chunks}")
        print(f"  - Total time: {total_time:.1f}s")
        print(f"  - Avg time per file: {total_time/len(successful):.1f}s")

    if successful:
        print("\n✅ Indexed Presentations:")
        for r in successful:
            print(f"  - {r['presentation']} → {r['index_name']}")

    if failed:
        print("\n❌ Failed Files:")
        for r in failed:
            print(f"  - {Path(r['file_path']).name}: {r['error']}")

    print("="*60)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Index PowerPoint presentations for contextual retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index single file (uses PINECONE_INDEX_NAME from .env)
  python scripts/ingest.py presentation.pptx

  # Index with custom settings
  python scripts/ingest.py presentation.pptx --no-context --no-vision

  # Batch index multiple files
  python scripts/ingest.py *.pptx --batch

Note:
  - Index name is configured via PINECONE_INDEX_NAME in .env
  - All presentations are indexed into the same Pinecone index
  - Use metadata to distinguish between different presentations
        """
    )

    parser.add_argument(
        'files',
        nargs='+',
        help='PowerPoint file(s) to index (.pptx)'
    )

    parser.add_argument(
        '--batch',
        action='store_true',
        help='Batch mode: process multiple files'
    )

    parser.add_argument(
        '--no-context',
        action='store_true',
        help='Disable contextual descriptions (faster, lower quality)'
    )

    parser.add_argument(
        '--no-vision',
        action='store_true',
        help='Disable vision analysis (faster, no image insights)'
    )

    parser.add_argument(
        '--no-reranking',
        action='store_true',
        help='Disable Cohere reranking'
    )

    parser.add_argument(
        '--no-images',
        action='store_true',
        help='Skip image extraction'
    )

    parser.add_argument(
        '--no-notes',
        action='store_true',
        help='Skip speaker notes'
    )

    args = parser.parse_args()

    # Prepare kwargs
    kwargs = {
        'use_contextual': not args.no_context,
        'use_vision': not args.no_vision,
        'use_reranking': not args.no_reranking,
        'extract_images': not args.no_images,
        'include_notes': not args.no_notes,
    }

    # Run ingestion
    try:
        if len(args.files) == 1 and not args.batch:
            # Single file
            result = asyncio.run(ingest_single_file(args.files[0], **kwargs))

            print("\n" + "="*60)
            print("INGESTION COMPLETE")
            print("="*60)
            print(f"\n✅ Presentation: {result['presentation']}")
            print(f"📊 Slides: {result['slides']}")
            print(f"📝 Chunks: {result['chunks']}")
            print(f"⏱️  Time: {result['elapsed_seconds']}s")
            print(f"🗂️  Index: {result['index_name']} (from .env)")
            print(f"✨ Contextual: {'Yes' if result['contextual'] else 'No'}")
            print(f"👁️  Vision: {'Yes' if result['vision_analyzed'] else 'No'}")
            print("="*60)

        else:
            # Batch mode
            results = asyncio.run(ingest_batch(args.files, **kwargs))
            print_summary(results)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    file_path = "data/presentations/ITPM_Scrum_Training.pptx"
    result = asyncio.run(ingest_single_file(file_path))

    print("\n" + "="*60)
    print("INGESTION COMPLETE")
    print("="*60)
    print(f"\n✅ Presentation: {result['presentation']}")
    print(f"📊 Slides: {result['slides']}")
    print(f"📝 Chunks: {result['chunks']}")
    print(f"⏱️  Time: {result['elapsed_seconds']}s")
    print(f"🗂️  Index: {result['index_name']} (from .env)")
    print(f"✨ Contextual: {'Yes' if result['contextual'] else 'No'}")
    print(f"👁️  Vision: {'Yes' if result['vision_analyzed'] else 'No'}")
    print("="*60)
     
    #main()
