"""
Contextual Text Splitter with LLM-generated context.

Implements the core Contextual Retrieval approach from Anthropic.
"""
from langchain.text_splitter import TextSplitter
from langchain.schema import Document
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from typing import List, Optional, Union
import asyncio
from loguru import logger

from src.config import settings
from src.utils.rate_limiter import rate_limiter, with_retry
from src.utils.caching import get_cached_llm


class ContextualTextSplitter(TextSplitter):
    """
    Text splitter that adds LLM-generated context to each chunk.

    This is THE key component that makes contextual retrieval work!
    """

    def __init__(
        self,
        llm: Optional[Union[ChatAnthropic, ChatOpenAI]] = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        add_context: bool = True,
        batch_size: int = 5,
        **kwargs
    ):
        """
        Initialize contextual splitter.

        Args:
            llm: Language model for context generation (default: from settings)
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks
            add_context: Whether to add context (set False to skip)
            batch_size: Number of chunks to process in parallel
        """
        super().__init__(**kwargs)

        self.chunk_size = chunk_size or settings.max_chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.add_context = add_context
        self.batch_size = batch_size

        # Initialize LLM for context generation (OpenAI or Anthropic)
        if llm:
            self.llm = llm
        else:
            if settings.context_generation_provider == "openai":
                # Use cached OpenAI LLM
                self.llm = get_cached_llm(
                    model=settings.context_generation_model,
                    max_tokens=150,  # Context should be 50-100 tokens
                    temperature=0.0  # Deterministic
                )
                self.provider = "openai"
            else:
                # Use Anthropic
                self.llm = ChatAnthropic(
                    model=settings.context_generation_model,
                    api_key=settings.anthropic_api_key,
                    max_tokens=150,
                    temperature=0.0
                )
                self.provider = "anthropic"

        # Context generation prompt (Optimized for OpenAI caching)
        # Static instructions first (will be cached), dynamic content last
        self.context_prompt = PromptTemplate(
            input_variables=[
                "presentation_title",
                "slide_number",
                "total_slides",
                "section",
                "slide_title",
                "chunk_content",
                "prev_slide_title",
                "next_slide_title"
            ],
            template="""<instructions>
You are a contextual retrieval assistant. Your task is to generate a concise context (50-100 tokens) for text chunks from PowerPoint presentations.
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else. 

The context should:
1. Situate the chunk within the presentation structure
2. Reference slide position and section
3. Explain how this content relates to the broader presentation narrative
4. Mention relationships to previous/next slides if relevant

Output only the context, nothing else.
</instructions>

<document>
Presentation: {presentation_title}
Total Slides: {total_slides}
All slides content:
{all_slides_content}
</document>

<current_chunk>
Slide: {slide_number}
Section: {section}
Slide Title: {slide_title}
Previous Slide: {prev_slide_title}
Next Slide: {next_slide_title}

Content:
{chunk_content}
</current_chunk>

Context:"""
        )

        provider_name = getattr(self, 'provider', 'custom')
        model_name = getattr(self.llm, 'model_name', getattr(self.llm, 'model', 'unknown'))
        
        logger.info(
            f"ContextualTextSplitter initialized: "
            f"provider={provider_name}, model={model_name}, "
            f"chunk_size={self.chunk_size}, add_context={self.add_context}, "
            f"caching={'enabled' if settings.enable_llm_cache else 'disabled'}"
        )

    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks.

        This is the abstract method required by TextSplitter.
        Returns list of text chunks without context.
        """
        return self._split_text(text)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks and add context.

        This is a synchronous wrapper around async implementation.
        """
        return asyncio.run(self.asplit_documents(documents))

    async def asplit_documents(self, all_doc_text: str, documents: List[Document]) -> List[Document]:
        """
        Async version: Split documents into contextual chunks.
        """
        all_chunks = []

        for doc_idx, doc in enumerate(documents):
            logger.info(
                f"Processing document {doc_idx + 1}/{len(documents)}: "
                f"Slide {doc.metadata.get('slide_number', '?')}"
            )

            # Split document into chunks
            text_chunks = self._split_text(doc.page_content)

            # Create chunk documents
            chunks = []
            for chunk_idx, chunk_text in enumerate(text_chunks):
                chunk_metadata = {
                    **doc.metadata,
                    "chunk_index": chunk_idx,
                    "chunk_id": (
                        f"{doc.metadata.get('presentation_id', 'unknown')}_"
                        f"{doc.metadata.get('slide_number', 0)}_"
                        f"{chunk_idx}"
                    ),
                    "chunk_size": len(chunk_text.split())
                }

                chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata,
                    "doc": doc
                })

            # Generate contexts in batches
            if self.add_context and chunks:
                chunks = await self._add_contexts_batch(all_doc_text,chunks)

            # Create final documents
            for chunk in chunks:
                if self.add_context and "context" in chunk:
                    # Prepend context to chunk for embedding
                    content = f"{chunk['context']}\n\n{chunk['text']}"
                    with open(f"CONTEXT/{chunk["metadata"]["slide_number"]}.txt", "w", encoding="utf-8") as f:
                        f.write(f"{chunk['context']}\n\n{chunk['text']}\n\n")
                    # chunk["metadata"]["context"] = chunk["context"]
                    # chunk["metadata"]["original_text"] = chunk["text"]
                else:
                    content = chunk["text"]
                chunk["metadata"].pop("text", None)
                all_chunks.append(
                    Document(
                        page_content=content,
                        metadata=chunk["metadata"]
                    )
                )

        logger.info(f"Created {len(all_chunks)} contextual chunks from {len(documents)} documents")
        return all_chunks

    async def _add_contexts_batch(self, all_doc_text: str, chunks: List[dict]) -> List[dict]:
        """
        Add context to chunks in batches for efficiency.
        """
        total = len(chunks)
        logger.info(f"Generating contexts for {total} chunks...")

        # Process in batches
        for i in range(0, total, self.batch_size):
            batch = chunks[i:i + self.batch_size]

            # Generate contexts in parallel
            tasks = [
                self._generate_context_async(all_doc_text, chunk)
                for chunk in batch
            ]

            contexts = await asyncio.gather(*tasks, return_exceptions=True)

            # Add contexts to chunks
            for chunk, context in zip(batch, contexts):
                if isinstance(context, Exception):
                    logger.error(f"Context generation failed: {context}")
                    chunk["context"] = ""  # Empty context on failure
                else:
                    chunk["context"] = context

            logger.debug(f"Processed {min(i + self.batch_size, total)}/{total} chunks")

        return chunks

    @with_retry(max_attempts=3)
    async def _generate_context_async(self, all_doc_text: str, chunk: dict) -> str:
        """
        Generate context for a single chunk (async).
        """
        # Rate limiting
        await rate_limiter.wait_if_needed(
            key="anthropic",
            estimated_tokens=rate_limiter.count_tokens(chunk["text"])
        )

        try:
            metadata = chunk["metadata"]
            doc = chunk["doc"]

            # Get prev/next slide titles
            prev_slide = ""
            next_slide = ""

            # Build prompt
            prompt_text = self.context_prompt.format(
                presentation_title=metadata.get("presentation_title", "Unknown"),
                slide_number=metadata.get("slide_number", "?"),
                total_slides=metadata.get("total_slides", "?"),
                all_slides_content=all_doc_text,
                section=metadata.get("section", "Unknown"),
                slide_title=metadata.get("slide_title", ""),
                chunk_content=chunk["text"][:500],  # Limit input size
                prev_slide_title=prev_slide,
                next_slide_title=next_slide
            )

            # Call LLM
            response = await self.llm.ainvoke(prompt_text)

            context = response.content.strip()

            # Validate length (should be 50-100 tokens)
            token_count = len(context.split())
            if token_count > 150:
                logger.warning(f"Context too long: {token_count} tokens, truncating")
                words = context.split()[:100]
                context = " ".join(words)

            return context

        except Exception as e:
            logger.error(f"Failed to generate context: {e}")
            return ""

    def _split_text(self, text: str) -> List[str]:
        """
        Split text into chunks based on size.

        Uses sentence-based splitting with overlap.
        """
        import re

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence.split())

            # Check if adding this sentence would exceed chunk size
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(" ".join(current_chunk))

                # Start new chunk with overlap
                overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_size = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_size += sentence_size

        # Add final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks


# Helper function for standalone usage
async def create_contextual_chunks(
    documents: List[Document],
    add_context: bool = True
) -> List[Document]:
    """
    Convenience function to create contextual chunks from documents.

    Args:
        documents: List of LangChain documents
        add_context: Whether to add LLM-generated context

    Returns:
        List of contextual chunks
    """
    splitter = ContextualTextSplitter(add_context=add_context)
    chunks = await splitter.asplit_documents(documents)
    return chunks
