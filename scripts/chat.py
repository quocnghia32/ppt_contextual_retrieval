#!/usr/bin/env python3
"""
PPT Chat Script - Query indexed presentations

Index name is configured via PINECONE_INDEX_NAME in .env file.

Usage:
    python scripts/chat.py
    python scripts/chat.py --interactive
    python scripts/chat.py --query "What is the revenue?"
"""
import sys
import asyncio
import argparse
from pathlib import Path
from typing import Optional
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import PPTContextualRetrievalPipeline
from src.config import settings
from loguru import logger


class ChatSession:
    """Interactive chat session with indexed presentation."""

    def __init__(
        self,
        use_reranking: bool = True,
        show_sources: bool = True
    ):
        """
        Initialize chat session.

        Args:
            use_reranking: Enable reranking
            show_sources: Display source documents
        """
        self.index_name = settings.pinecone_index_name
        self.show_sources = show_sources
        self.pipeline = None
        self.use_reranking = use_reranking

    async def initialize(self):
        """Initialize pipeline (connect to existing index from .env)."""
        logger.info(f"Connecting to index: {self.index_name} (from .env)")

        # Create pipeline (will connect to existing index)
        self.pipeline = PPTContextualRetrievalPipeline(
            use_contextual=True,  # Already indexed with context
            use_vision=False,  # No need for vision in query phase
            use_reranking=self.use_reranking
        )

        # Setup retrieval without indexing
        from langchain_pinecone import PineconeVectorStore
        #from src.utils.caching import get_cached_embeddings
        from src.utils.caching_azure import get_cached_embeddings_azure
        from src.retrievers.hybrid_retriever import create_hybrid_retriever
        from src.chains.qa_chain import create_qa_chain

        # Connect to vector store
        embeddings = get_cached_embeddings_azure(model=settings.embedding_model)
        vector_store = PineconeVectorStore(
            index_name=self.index_name,
            embedding=embeddings,
            pinecone_api_key=settings.pinecone_api_key
        )

        # For BM25, we need documents - load from vector store
        # In production, you'd store documents separately
        # For now, we'll create retriever without BM25
        self.pipeline.vector_store = vector_store

        # Create simple retriever
        from src.retrievers.hybrid_retriever import ContextualHybridRetriever

        # Note: For full hybrid search, you need to store original documents
        # This simplified version uses only vector search
        self.pipeline.retriever = vector_store.as_retriever(
            search_kwargs={"k": settings.top_k_retrieval}
        )

        # Create QA chain
        self.pipeline.qa_chain = create_qa_chain(
            retriever=self.pipeline.retriever,
            streaming=False,
            enable_memory=True
        )

        logger.info("✅ Connected successfully")

    async def query(self, question: str) -> dict:
        """
        Query the presentation.

        Args:
            question: User question

        Returns:
            Query result with answer and sources
        """
        if not self.pipeline or not self.pipeline.qa_chain:
            raise ValueError("Pipeline not initialized. Call initialize() first.")

        result = await self.pipeline.query(question, return_sources=True)
        return result

    def print_result(self, result: dict):
        """Print query result in nice format."""
        print("\n" + "="*60)
        print("💡 ANSWER")
        print("="*60)
        print(result['answer'])

        # Quality score
        if result.get('quality_score'):
            quality = result['quality_score']
            score = quality.get('score', 0)

            print("\n" + "-"*60)
            print(f"📊 Quality Score: {score:.0%}")

            if quality.get('has_sources'):
                print("  ✅ Has sources")
            if quality.get('cites_slides'):
                print("  ✅ Cites slides")
            if quality.get('not_uncertain'):
                print("  ✅ Confident answer")

        # Sources
        if self.show_sources and result.get('formatted_sources'):
            sources = result['formatted_sources']

            print("\n" + "="*60)
            print(f"📚 SOURCES ({len(sources)} documents)")
            print("="*60)

            for idx, source in enumerate(sources[:5], 1):
                print(f"\n[{idx}] Slide {source['slide_number']}: {source['slide_title']}")
                print(f"    Section: {source['section']}")
                print(f"    Content: {source['content'][:200]}...")

                if source.get('rrf_score'):
                    print(f"    RRF Score: {source['rrf_score']:.4f}")

        print("="*60 + "\n")

    async def interactive_loop(self):
        """Run interactive chat loop."""
        print("\n" + "="*60)
        print("🤖 PPT CHAT SESSION")
        print("="*60)
        print(f"Index: {self.index_name}")
        print("\nType your questions below. Commands:")
        print("  - 'quit' or 'exit' to end session")
        print("  - 'clear' to clear chat history")
        print("  - 'sources on/off' to toggle source display")
        print("="*60 + "\n")

        while True:
            try:
                # Get user input
                question = input("You: ").strip()

                if not question:
                    continue

                # Handle commands
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break

                if question.lower() == 'clear':
                    self.pipeline.clear_chat_history()
                    print("✅ Chat history cleared\n")
                    continue

                if question.lower().startswith('sources'):
                    if 'off' in question.lower():
                        self.show_sources = False
                        print("✅ Sources display: OFF\n")
                    else:
                        self.show_sources = True
                        print("✅ Sources display: ON\n")
                    continue

                # Query
                print("\n🤔 Thinking...\n")
                result = await self.query(question)
                self.print_result(result)

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
                logger.error(f"Query failed: {e}")


async def single_query(
    question: str,
    show_sources: bool = True,
    json_output: bool = False
) -> dict:
    """
    Execute single query and exit.

    Args:
        question: Question to ask
        show_sources: Display sources
        json_output: Output as JSON

    Returns:
        Query result
    """
    session = ChatSession(show_sources=show_sources)
    await session.initialize()

    result = await session.query(question)

    if json_output:
        # Output as JSON
        output = {
            'question': question,
            'answer': result['answer'],
            'sources': result.get('formatted_sources', []),
            'quality_score': result.get('quality_score')
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        session.print_result(result)

    return result


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Query indexed PowerPoint presentations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive chat (uses PINECONE_INDEX_NAME from .env)
  python scripts/chat.py

  # Single query
  python scripts/chat.py --query "What is the revenue?"

  # Query with JSON output
  python scripts/chat.py --query "Summary?" --json

  # Disable sources
  python scripts/chat.py --no-sources

Note:
  - Index name is configured via PINECONE_INDEX_NAME in .env
  - All queries are executed against the index defined in .env
        """
    )

    parser.add_argument(
        '--query',
        '-q',
        type=str,
        default=None,
        help='Single query (non-interactive mode)'
    )

    parser.add_argument(
        '--interactive',
        '-i',
        action='store_true',
        help='Interactive chat mode (default if no --query)'
    )

    parser.add_argument(
        '--no-sources',
        action='store_true',
        help='Hide source documents'
    )

    parser.add_argument(
        '--no-reranking',
        action='store_true',
        help='Disable reranking'
    )

    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON (only for single query)'
    )

    args = parser.parse_args()

    # Determine mode
    if args.query:
        # Single query mode
        try:
            asyncio.run(single_query(
                question=args.query,
                show_sources=not args.no_sources,
                json_output=args.json
            ))
        except Exception as e:
            logger.error(f"Query failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    else:
        # Interactive mode
        try:
            session = ChatSession(
                use_reranking=not args.no_reranking,
                show_sources=not args.no_sources
            )
            asyncio.run(session.initialize())
            asyncio.run(session.interactive_loop())

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
        except Exception as e:
            logger.error(f"Chat session failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()
