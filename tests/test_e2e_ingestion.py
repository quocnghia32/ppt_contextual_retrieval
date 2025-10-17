"""
End-to-End Ingestion Test

Tests the complete ingestion pipeline with a real PPT file.
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import PPTContextualRetrievalPipeline
from src.config import settings


async def test_ingest_cinema_ppt():
    """
    Test ingestion of Cinema - Desk - Community.pptx

    This test:
    1. Creates ingestion pipeline
    2. Indexes the presentation
    3. Validates results
    4. Reports statistics
    """
    print("=" * 80)
    print("🧪 E2E INGESTION TEST")
    print("=" * 80)
    print()

    # File to test
    ppt_path = "/home/hungson175/users/NghiaNQ/ppt_context_retrieval/data/presentations/Cinema - Desk - Community.pptx"

    # Verify file exists
    if not Path(ppt_path).exists():
        print(f"❌ File not found: {ppt_path}")
        return None

    print(f"📁 File: {Path(ppt_path).name}")
    print(f"📏 Size: {Path(ppt_path).stat().st_size / 1024:.2f} KB")
    print()

    # Configuration
    print("⚙️ Configuration:")
    print(f"   - Contextual chunking: True")
    print(f"   - Vision analysis: True")
    print(f"   - Pinecone index: {settings.pinecone_index_name}")
    print(f"   - BM25 backend: {settings.search_backend}")
    print(f"   - Context provider: {settings.context_generation_provider}")
    print(f"   - Context model: {settings.context_generation_model}")
    print()

    # Create pipeline
    print("🔧 Creating ingestion pipeline...")
    start_time = datetime.now()

    pipeline = PPTContextualRetrievalPipeline(
        use_contextual=True,
        use_vision=True
    )

    print("✅ Pipeline created")
    print()

    # Index presentation
    print("🚀 Starting ingestion...")
    print("-" * 80)

    try:
        stats = await pipeline.index_presentation(
            ppt_path=ppt_path,
            extract_images=True,
            include_notes=True
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print("-" * 80)
        print()
        print("✅ INGESTION SUCCESSFUL!")
        print()

        # Display results
        print("📊 RESULTS:")
        print(f"   Presentation ID: {stats['presentation_id']}")
        print(f"   Presentation: {stats['presentation']}")
        print(f"   Total Slides: {stats['slides']}")
        print(f"   Total Chunks: {stats['chunks']}")
        print(f"   Indexed: {stats['indexed']}")
        print(f"   Contextual: {stats['contextual']}")
        print(f"   Vision Analyzed: {stats['vision_analyzed']}")
        print(f"   Pinecone Index: {stats['pinecone_index']}")
        print(f"   BM25 Backend: {stats['bm25_backend']}")
        print()

        # Performance metrics
        print("⚡ PERFORMANCE:")
        print(f"   Total Duration: {duration:.2f}s")
        print(f"   Time per Slide: {duration / stats['slides']:.2f}s")
        print(f"   Time per Chunk: {duration / stats['chunks']:.2f}s")
        print()

        # Validation
        print("✔️ VALIDATION:")
        assert stats['indexed'] == True, "Indexing failed"
        assert stats['slides'] > 0, "No slides found"
        assert stats['chunks'] > 0, "No chunks created"
        print(f"   ✓ Indexing status: OK")
        print(f"   ✓ Slides count: OK ({stats['slides']} slides)")
        print(f"   ✓ Chunks count: OK ({stats['chunks']} chunks)")
        print()

        print("=" * 80)
        print("🎉 TEST PASSED!")
        print("=" * 80)

        return stats

    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print("-" * 80)
        print()
        print("❌ INGESTION FAILED!")
        print()
        print(f"Error: {str(e)}")
        print(f"Duration before failure: {duration:.2f}s")
        print()
        print("=" * 80)
        print("💥 TEST FAILED!")
        print("=" * 80)

        raise


async def verify_storage():
    """
    Verify that data was stored correctly.
    """
    print()
    print("🔍 STORAGE VERIFICATION")
    print("-" * 80)

    from src.storage.base_text_retriever import get_text_retriever

    # Initialize text retriever
    text_retriever = get_text_retriever(
        backend=settings.search_backend,
        db_path=settings.bm25_db_path,
        index_path=settings.bm25_index_path,
        k=settings.top_k_retrieval
    )

    await text_retriever.initialize()

    # Get stats
    stats = await text_retriever.get_stats()

    print(f"BM25 Store Statistics:")
    print(f"   Backend Type: {stats['backend_type']}")
    print(f"   Total Documents: {stats['total_documents']}")
    print(f"   Total Presentations: {stats['total_presentations']}")
    print(f"   SQLite Size: {stats['sqlite_size_mb']} MB")
    print(f"   Index Size: {stats['index_size_mb']} MB")
    print(f"   Total Size: {stats['total_size_mb']} MB")
    print(f"   Index Loaded: {stats['index_loaded']}")
    print()

    # List presentations
    presentations = await text_retriever.list_presentations()
    print(f"Indexed Presentations ({len(presentations)}):")
    for pres in presentations:
        print(f"   - {pres['presentation_id']}: {pres['name']}")
        print(f"     Slides: {pres['total_slides']}, Chunks: {pres['total_chunks']}")
        print(f"     Indexed at: {pres['indexed_at']}")
    print()

    print("-" * 80)
    print("✅ Storage verification complete")


if __name__ == "__main__":
    # Run test
    stats = asyncio.run(test_ingest_cinema_ppt())

    # Verify storage
    if stats:
        asyncio.run(verify_storage())
