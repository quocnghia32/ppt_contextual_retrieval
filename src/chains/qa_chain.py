"""
QA Chain for answering questions based on retrieved context.

Uses Claude for answer generation with quality checking.
"""
from typing import List, Dict, Optional, Any
from langchain.memory import ConversationBufferMemory
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_xai import ChatXAI
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from typing import Union
from loguru import logger
from langchain_core.output_parsers import StrOutputParser

from src.config import settings
from src.utils.rate_limiter import rate_limiter, with_retry
from src.utils.caching import get_cached_llm
from src.utils.caching_azure import get_cached_llm_azure



# Answer generation prompt (Optimized for OpenAI caching)
# Static instructions first (cached), dynamic content last
ANSWER_PROMPT = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""<instructions>
You are an AI assistant analyzing PowerPoint presentations. Your role is to answer questions based ONLY on the provided context from the presentation.

Guidelines:
1. **Cite Sources**: Always reference specific slides (e.g., "According to Slide 5...")
2. **Be Specific**: Use exact data points, charts, and figures from the context
3. **Acknowledge Uncertainty**: If the context doesn't contain the answer, clearly state: "I don't have enough information in the presentation to answer this question accurately."
4. **Structure**: Use bullet points or numbered lists for clarity when presenting multiple points
5. **Context-Aware**: Consider the presentation flow and narrative structure
6. **Accuracy**: Never make up information not present in the context

Format your response with:
- Direct answer first
- Supporting evidence with slide citations
- Data points and figures when relevant
</instructions>

<presentation_context>
{context}
</presentation_context>

<conversation_history>
{chat_history}
</conversation_history>

<user_question>
{question}
</user_question>

Answer:"""
)

# Condense question prompt for follow-up questions
CONDENSE_QUESTION_PROMPT = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question that can be answered without the chat history.

Chat History:
{chat_history}

Follow-Up Question: {question}

Standalone Question:"""
)
def get_buffer_string(chat_memory):
    """Minimal reimplementation for older LangChain versions."""
    messages = getattr(chat_memory, "messages", [])
    lines = []
    for m in messages:
        role = getattr(m, "type", getattr(m, "role", ""))
        content = getattr(m, "content", str(m))
        if role in ("human", "user"):
            lines.append(f"Human: {content}")
        elif role in ("ai", "assistant"):
            lines.append(f"AI: {content}")
        else:
            lines.append(content)
    return "\n".join(lines)


class PPTQAChain:
    """
    Question-Answering chain for PowerPoint presentations.

    Handles retrieval, answer generation, and conversation memory.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: Optional[Any] = None,
        enable_streaming: bool = False,
        enable_memory: bool = True,
        quality_check: bool = True
    ):
        """
        Initialize QA chain.

        Args:
            retriever: Document retriever (hybrid retriever)
            llm: Language model for answer generation (OpenAI)
            enable_streaming: Enable token-by-token streaming
            enable_memory: Enable conversation memory
            quality_check: Enable answer quality checking
        """
        self.retriever = retriever
        self.enable_streaming = enable_streaming
        self.enable_memory = enable_memory
        self.quality_check = quality_check

        # Initialize LLM (OpenAI)
        callbacks = [StreamingStdOutCallbackHandler()] if enable_streaming else None

        if llm:
            self.llm = llm
            self.provider = "custom"
        else:
            if settings.answer_generation_provider == "openai":
                # Use cached OpenAI LLM
                self.llm = get_cached_llm(
                    model=settings.answer_generation_model,
                    max_tokens=2000,
                    temperature=0.0,
                    streaming=enable_streaming,
                    callbacks=callbacks
                )
                self.provider = "openai"
            elif settings.answer_generation_provider == "azure":
                # Use Azure OpenAI LLM
                self.llm = AzureChatOpenAI(
                    api_key=settings.azure_openai_api_key,
                    azure_endpoint=settings.azure_openai_endpoint,
                    api_version=settings.azure_openai_api_version_chat,
                    azure_deployment=settings.azure_openai_chat_deployment,
                    temperature=0.0,
                    max_tokens=200,
                )

                self.provider = "azure"
            else:
                # Use Grok
                self.llm = ChatXAI(
                    model = "grok-4-fast-reasoning",
                    api_key=settings.xai_api_key,
                    max_tokens=2000,
                    temperature=0.0,
                    streaming=enable_streaming,
                    callbacks=callbacks
                )
                self.provider = "xai"


        # Initialize memory
        self.memory = None
        if enable_memory:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )

        # # Create chain
        # self.chain = ConversationalRetrievalChain.from_llm(
        #     llm=self.llm,
        #     retriever=self.retriever,
        #     memory=self.memory,
        #     combine_docs_chain_kwargs={"prompt": ANSWER_PROMPT},
        #     condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        #     return_source_documents=True,
        #     verbose=settings.verbose_logging,
        # )
        self._condense_prompt = CONDENSE_QUESTION_PROMPT
        self._answer_prompt = ANSWER_PROMPT


        model_name = getattr(self.llm, 'model_name', getattr(self.llm, 'model', 'unknown'))

        logger.info(
            f"PPTQAChain initialized: "
            f"provider={self.provider}, model={model_name}, "
            f"streaming={enable_streaming}, memory={enable_memory}, "
            f"quality_check={quality_check}, "
            f"caching={'enabled' if settings.enable_llm_cache else 'disabled'}"
        )
    def _get_chat_history(self) -> str:
        if not self.memory or not hasattr(self.memory, "chat_memory"):
            return ""
        try:
            return get_buffer_string(self.memory.chat_memory)
        except Exception:
            # Fail-safe: return empty if memory backend differs
            return ""

    def _condense_question(self, user_input: str) -> str:
        """Produce a standalone query from chat history + user question."""
        chat_history = self._get_chat_history()
        logger.warning(f"Chat history: {len(chat_history)} tokens")
        msg = self._condense_prompt.format(
            chat_history=chat_history,
            question=user_input,
        )
        condensed = self.llm.invoke(msg)
        return condensed.content

    def _retrieve(self, standalone_query: str) -> List[Document]:
        """Fetch relevant documents via the configured retriever."""
        return self.retriever.get_relevant_documents(standalone_query)
    
    def _format_docs_as_context(self, docs: List[Document]) -> str:
        """
        Flatten retrieved docs into a single string with slide refs
        for ANSWER_PROMPT's {context}. Prefer metadata['slide'],
        otherwise fallback to 'page', 'source', then '?'.
        """
        parts: List[str] = []
        for d in docs:
            meta = getattr(d, "metadata", {}) or {}
            slide = meta.get("slide") or meta.get("page") or meta.get("source") or "?"
            pc = getattr(d, "page_content", "") or ""
            parts.append(f"[Slide {slide}] {pc}")
        return "\n\n".join(parts)

    def _answer(self, standalone_query: str, docs: List[Document]) -> str:
        """
        Run ANSWER_PROMPT with required inputs:
        - context: flattened string of retrieved docs (with slide refs)
        - question: the standalone question
        - chat_history: serialized chat history string
        """
        context = self._format_docs_as_context(docs)
        chat_history = self._get_chat_history()
        prompt_text = self._answer_prompt.format(
            context=context,
            question=standalone_query,
            chat_history=chat_history,
        )

        out = self.llm.invoke(prompt_text)
        return out.content


    @with_retry(max_attempts=3)
    async def aquery(
        self,
        question: str,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Async query the QA chain.

        Args:
            question: User question
            return_sources: Whether to return source documents

        Returns:
            Dict with answer and optional sources
        """
        # Rate limiting
        estimated_tokens = rate_limiter.count_tokens(question) + 1000
        await rate_limiter.wait_if_needed(
            key="openai",
            estimated_tokens=estimated_tokens
        )

        try:
            # Run chain
            logger.warning(f"[Question] : {question}")
            standalone = self._condense_question(question)

            logger.debug(f"[Standalone Question] : {standalone}")

            source_docs = self._retrieve(standalone)
            if not source_docs:
                source_docs = []
    
            logger.info(f"[Retrieved {len(source_docs)} docs]")
            for i, d in enumerate(source_docs, 1):
                logger.debug(f"Doc {i}: {d.page_content[:100]}")

            answer = self._answer(standalone, source_docs)

            if self.memory:
                self.memory.chat_memory.add_user_message(question)
                self.memory.chat_memory.add_ai_message(answer)

            # Quality check
            if self.quality_check:
                quality_score = self._check_answer_quality(
                    question,
                    answer,
                    source_docs
                )
            else:
                quality_score = None

            response = {
                "answer": answer,
                "question": question,
                "source_documents": source_docs,
                "quality_score": quality_score
            }

            logger.info(f"Query answered: {len(answer)} chars, {len(source_docs)} sources")
            return response

        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise

    def query(
        self,
        question: str,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Synchronous version of query.
        """
        import asyncio
        return asyncio.run(self.aquery(question, return_sources))

    def _check_answer_quality(
        self,
        question: str,
        answer: str,
        sources: List[Document]
    ) -> Dict[str, Any]:
        """
        Check answer quality (simple heuristics).

        In production, could use LLM-based quality checking.
        """
        quality = {
            "has_answer": len(answer) > 50,
            "has_sources": len(sources) > 0,
            "cites_slides": "slide" in answer.lower() or "slide" in answer.lower(),
            "not_uncertain": "don't have enough information" not in answer.lower(),
            "score": 0.0
        }

        # Calculate score
        score = 0.0
        if quality["has_answer"]:
            score += 0.3
        if quality["has_sources"]:
            score += 0.3
        if quality["cites_slides"]:
            score += 0.2
        if quality["not_uncertain"]:
            score += 0.2

        quality["score"] = score

        return quality

    def clear_memory(self):
        """Clear conversation memory."""
        if self.memory:
            self.memory.clear()
            logger.info("Conversation memory cleared")

    def get_chat_history(self) -> List[Dict]:
        """Get conversation history."""
        if self.memory:
            return self.memory.load_memory_variables({}).get("chat_history", [])
        return []


class StreamingPPTQAChain(PPTQAChain):
    """
    QA Chain with streaming support for real-time responses.
    """

    def __init__(self, retriever: BaseRetriever, **kwargs):
        """Initialize with streaming enabled."""
        super().__init__(
            retriever,
            enable_streaming=True,
            **kwargs
        )

    async def aquery_stream(
        self,
        question: str,
        callback = None
    ):
        """
        Stream answer tokens in real-time.

        Args:
            question: User question
            callback: Callback function for each token
        """
        # Rate limiting
        estimated_tokens = rate_limiter.count_tokens(question) + 1000
        await rate_limiter.wait_if_needed(
            key="openai",
            estimated_tokens=estimated_tokens
        )

        try:
            # Stream response
            full_answer = ""
            async for chunk in self.chain.astream({"question": question}):
                if "answer" in chunk:
                    token = chunk["answer"]
                    full_answer += token

                    if callback:
                        callback(token)

            # Get source documents
            result = await self.chain.ainvoke({"question": question})
            source_docs = result.get("source_documents", [])

            return {
                "answer": full_answer,
                "question": question,
                "source_documents": source_docs
            }

        except Exception as e:
            logger.error(f"Streaming query failed: {e}")
            raise


# Helper function to create QA chain
def create_qa_chain(
    retriever: BaseRetriever,
    streaming: bool = False,
    **kwargs
) -> PPTQAChain:
    """
    Create QA chain for PowerPoint question answering.

    Args:
        retriever: Document retriever
        streaming: Enable streaming responses
        **kwargs: Additional chain arguments

    Returns:
        Configured QA chain
    """
    if streaming:
        chain = StreamingPPTQAChain(retriever, **kwargs)
    else:
        chain = PPTQAChain(retriever, **kwargs)

    logger.info(f"QA chain created: streaming={streaming}")
    return chain