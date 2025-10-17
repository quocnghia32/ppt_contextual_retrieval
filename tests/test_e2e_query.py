"""
End-to-End Query Test

Tests the complete retrieval pipeline with Cinema PPT.
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval_pipeline import RetrievalPipeline
from src.config import settings


# Test questions about Cinema presentation
TEST_QUESTIONS = [
    "What is this presentation about?",
    "What are the main topics covered?",
    "Tell me about IMDb",
    "What movie communities are mentioned?",
    "What are the key findings about cinema?",
]


async def test_query_cinema_ppt():
    """
    Test querying Cinema - Desk - Community PPT.

    This test:
    1. Creates retrieval pipeline
    2. Initializes (loads indexes)
    3. Queries with test questions
    4. Reports results
    """
    print("=" * 80)
    print("🧪 E2E QUERY TEST")
    print("=" * 80)
    print()

    presentation_id = "Cinema - Desk - Community"

    print(f"📦 Target: {presentation_id}")
    print(f"📊 Index: {settings.pinecone_index_name}")
    print(f"🔍 BM25 Backend: {settings.search_backend}")
    print(f"🤖 Answer Model: {settings.answer_generation_model}")
    print()

    # Create retrieval pipeline
    print("🔧 Creating retrieval pipeline...")
    start_time = datetime.now()

    retrieval = RetrievalPipeline(
        use_reranking=False    # Disable reranking for speed
    )

    print("✅ Pipeline created")
    print()

    # Initialize
    print("🚀 Initializing retrieval pipeline...")
    print("   - Loading BM25 index from SQLite...")
    print("   - Connecting to Pinecone...")
    print("   - Creating hybrid retriever...")
    print("   - Creating QA chain...")
    print()

    try:
        await retrieval.initialize()

        init_time = datetime.now()
        init_duration = (init_time - start_time).total_seconds()

        print(f"✅ Initialization complete ({init_duration:.2f}s)")
        print()

        # Get statistics
        print("📊 RETRIEVAL STATISTICS:")
        stats = await retrieval.get_stats()
        print(f"   Status: {stats['status']}")
        print(f"   Backend: {stats['backend']}")
        print(f"   Pinecone Index: {stats['pinecone_index']}")
        print(f"   Total Documents: {stats.get('total_documents', 'N/A')}")
        print(f"   Total Presentations: {stats.get('total_presentations', 'N/A')}")
        print()

        # List presentations
        print("📚 INDEXED PRESENTATIONS:")
        presentations = await retrieval.list_presentations()
        for pres in presentations:
            print(f"   - {pres['presentation_id']}: {pres['name']}")
            print(f"     Slides: {pres['total_slides']}, Chunks: {pres['total_chunks']}")
        print()

        # Run test queries
        print("=" * 80)
        print("🔍 RUNNING TEST QUERIES")
        print("=" * 80)
        print()

        results_summary = []

        for idx, question in enumerate(TEST_QUESTIONS, 1):
            print(f"[{idx}/{len(TEST_QUESTIONS)}] Question: {question}")
            print("-" * 80)

            query_start = datetime.now()

            try:
                result = await retrieval.query(question, return_sources=True)

                query_end = datetime.now()
                query_duration = (query_end - query_start).total_seconds()

                # Display answer
                print()
                print("💡 Answer:")
                print(result["answer"])
                print()

                # Display sources
                if "formatted_sources" in result and result["formatted_sources"]:
                    print("📄 Sources:")
                    for src_idx, source in enumerate(result["formatted_sources"][:3], 1):
                        print(f"   {src_idx}. Slide {source['slide_number']}: {source['slide_title']}")
                        print(f"      Content: {source['content'][:100]}...")
                        print(f"      Score: {source.get('rrf_score', 'N/A')}")
                    print()

                # Performance
                print(f"⚡ Query time: {query_duration:.2f}s")
                print()

                results_summary.append({
                    "question": question,
                    "answer": result["answer"],
                    "sources_count": len(result.get("source_documents", [])),
                    "duration": query_duration,
                    "success": True
                })

            except Exception as e:
                query_end = datetime.now()
                query_duration = (query_end - query_start).total_seconds()

                print()
                print(f"❌ Query failed: {str(e)}")
                print(f"⚡ Time before failure: {query_duration:.2f}s")
                print()

                results_summary.append({
                    "question": question,
                    "error": str(e),
                    "duration": query_duration,
                    "success": False
                })

            print()

        # Summary
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        print("=" * 80)
        print("📊 TEST SUMMARY")
        print("=" * 80)
        print()

        successful_queries = [r for r in results_summary if r["success"]]
        failed_queries = [r for r in results_summary if not r["success"]]

        print(f"Total Queries: {len(TEST_QUESTIONS)}")
        print(f"Successful: {len(successful_queries)}")
        print(f"Failed: {len(failed_queries)}")
        print()

        if successful_queries:
            avg_duration = sum(r["duration"] for r in successful_queries) / len(successful_queries)
            print(f"Average Query Time: {avg_duration:.2f}s")
            avg_sources = sum(r["sources_count"] for r in successful_queries) / len(successful_queries)
            print(f"Average Sources per Query: {avg_sources:.1f}")
        print()

        print(f"Total Test Duration: {total_duration:.2f}s")
        print(f"Initialization Time: {init_duration:.2f}s")
        print(f"Query Phase Time: {total_duration - init_duration:.2f}s")
        print()

        if failed_queries:
            print("❌ FAILED QUERIES:")
            for fail in failed_queries:
                print(f"   - {fail['question']}")
                print(f"     Error: {fail.get('error', 'Unknown')}")
            print()

        if len(successful_queries) == len(TEST_QUESTIONS):
            print("=" * 80)
            print("🎉 ALL QUERIES PASSED!")
            print("=" * 80)
            return results_summary
        else:
            print("=" * 80)
            print("⚠️ SOME QUERIES FAILED")
            print("=" * 80)
            return results_summary

    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print()
        print("❌ RETRIEVAL PIPELINE FAILED!")
        print()
        print(f"Error: {str(e)}")
        print(f"Duration before failure: {duration:.2f}s")
        print()
        print("=" * 80)
        print("💥 TEST FAILED!")
        print("=" * 80)

        raise


if __name__ == "__main__":
    # Run test
    results = asyncio.run(test_query_cinema_ppt())

    # Save results for documentation
    import json
    with open("test_query_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print()
    print("💾 Results saved to: test_query_results.json")
