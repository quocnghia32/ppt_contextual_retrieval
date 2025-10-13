#!/usr/bin/env python3
"""
List all Pinecone indexes.

Usage:
    python scripts/list_indexes.py
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pinecone import Pinecone
from src.config import settings
from loguru import logger


def list_indexes():
    """List all Pinecone indexes."""
    try:
        pc = Pinecone(api_key=settings.pinecone_api_key)

        # Get all indexes
        indexes = pc.list_indexes()

        if not indexes.names():
            print("\n⚠️  No indexes found")
            print("\nTo create an index, run:")
            print("  python scripts/ingest.py your_presentation.pptx")
            return

        print("\n" + "="*60)
        print("📊 PINECONE INDEXES")
        print("="*60)

        for idx_name in indexes.names():
            try:
                # Get index info
                index_info = pc.describe_index(idx_name)

                print(f"\n📁 {idx_name}")
                print(f"   Dimension: {index_info.dimension}")
                print(f"   Metric: {index_info.metric}")
                print(f"   Host: {index_info.host}")

                # Get stats
                index = pc.Index(idx_name)
                stats = index.describe_index_stats()

                print(f"   Vectors: {stats.total_vector_count:,}")
                print(f"   Namespaces: {len(stats.namespaces)}")

            except Exception as e:
                logger.warning(f"Could not get details for {idx_name}: {e}")
                print(f"\n📁 {idx_name}")
                print(f"   (Details unavailable)")

        print("\n" + "="*60)
        print(f"\nTotal indexes: {len(indexes.names())}")
        print("\nTo query an index, run:")
        print("  python scripts/chat.py --index <index-name>")
        print("="*60 + "\n")

    except Exception as e:
        logger.error(f"Failed to list indexes: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    list_indexes()
