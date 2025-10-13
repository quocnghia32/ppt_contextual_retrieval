# LangChain Implementation - PPT Contextual Retrieval

**Version**: 1.0
**Date**: 2025-10-10
**Purpose**: Complete LangChain-based implementation architecture

---

## 📋 Table of Contents
1. [Overview](#overview)
2. [LangChain Architecture](#langchain-architecture)
3. [Component Mapping](#component-mapping)
4. [Indexing Pipeline](#indexing-pipeline)
5. [Query Pipeline](#query-pipeline)
6. [Custom Components](#custom-components)
7. [Implementation Guide](#implementation-guide)
8. [Best Practices](#best-practices)

---

## 🎯 Overview

### Why LangChain?

**Benefits**:
- ✅ **Standardized Components**: Pre-built chains, retrievers, tools
- ✅ **Prompt Management**: PromptTemplate với variable substitution
- ✅ **LLM Abstraction**: Easy switching between models
- ✅ **Vector Store Integration**: Built-in support for Pinecone, Qdrant, ChromaDB
- ✅ **Memory Management**: Conversation history, context windows
- ✅ **Observability**: LangSmith integration for debugging
- ✅ **Production Ready**: Battle-tested framework

**LangChain Version**: 0.1.0+ (latest stable)

**Core Dependencies**:
```bash
pip install langchain==0.1.0
pip install langchain-anthropic  # Claude integration
pip install langchain-openai     # OpenAI embeddings
pip install langchain-pinecone   # Pinecone vector store
pip install langsmith            # Observability (optional)
```

---

## 🏗️ LangChain Architecture

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    INDEXING PIPELINE                          │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  PPT File → PPTLoader → Documents                            │
│              (Custom)                                         │
│                  ↓                                            │
│          ContextualTextSplitter → Chunks                     │
│              (Custom)                                         │
│                  ↓                                            │
│          ContextEnricher → Chunks + Context                  │
│         (LLMChain + Custom)                                  │
│                  ↓                                            │
│          EmbeddingFunction → Embeddings                      │
│         (OpenAIEmbeddings)                                   │
│                  ↓                                            │
│          VectorStore.from_documents()                        │
│              (Pinecone)                                       │
│                                                               │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│                    QUERY PIPELINE                             │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  User Query                                                   │
│      ↓                                                        │
│  QueryAnalysisChain → Intent + Reformulated Query           │
│  (Custom LLMChain)                                           │
│      ↓                                                        │
│  HybridRetriever → Retrieved Docs                           │
│  (Custom Retriever: Vector + BM25)                          │
│      ↓                                                        │
│  ContextualCompressionRetriever → Reranked Docs             │
│  (Built-in + Custom Reranker)                               │
│      ↓                                                        │
│  ConversationalRetrievalChain → Answer                      │
│  (Built-in, customized)                                     │
│      ↓                                                        │
│  QualityCheckChain → Validated Answer                       │
│  (Custom LLMChain)                                           │
│      ↓                                                        │
│  Response + Sources + Follow-ups                            │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## 🗺️ Component Mapping

### Original Design → LangChain Components

| Original Component | LangChain Component | Type |
|-------------------|---------------------|------|
| **PPT Parser** | Custom `PPTLoader` | Document Loader |
| **Chunking** | Custom `ContextualTextSplitter` | Text Splitter |
| **Context Generation** | `LLMChain` + Custom Prompt | Chain |
| **Vector Store Abstraction** | `VectorStore` (abstract class) | Built-in |
| **Pinecone Adapter** | `Pinecone` | Vector Store |
| **Embeddings** | `OpenAIEmbeddings` or Custom | Embeddings |
| **BM25 Index** | Custom `BM25Retriever` | Retriever |
| **Hybrid Retrieval** | Custom `HybridRetriever` | Retriever |
| **Reranking** | `ContextualCompressionRetriever` | Retriever |
| **Query Classification** | `LLMChain` with Router | Chain |
| **Answer Generation** | `ConversationalRetrievalChain` | Chain |
| **Memory** | `ConversationBufferMemory` | Memory |
| **Quality Check** | Custom `LLMChain` | Chain |

---

## 📥 Indexing Pipeline

### 1. Custom PPT Document Loader

```python
from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document
from typing import List
import hashlib
from pptx import Presentation

class PPTLoader(BaseLoader):
    """
    Custom LangChain Document Loader for PowerPoint files.

    Extends BaseLoader to parse .pptx files and extract slides
    with proper metadata.
    """

    def __init__(
        self,
        file_path: str,
        extract_images: bool = True,
        include_speaker_notes: bool = True
    ):
        self.file_path = file_path
        self.extract_images = extract_images
        self.include_speaker_notes = include_speaker_notes

    def load(self) -> List[Document]:
        """
        Load PPT and return list of LangChain Documents.

        Each slide becomes a Document with metadata.
        """
        prs = Presentation(self.file_path)
        documents = []

        # Extract presentation metadata
        presentation_id = hashlib.md5(self.file_path.encode()).hexdigest()
        title = self._extract_title(prs)

        for slide_idx, slide in enumerate(prs.slides):
            # Extract slide content
            slide_text = self._extract_slide_text(slide)

            # Extract speaker notes
            speaker_notes = ""
            if self.include_speaker_notes and slide.has_notes_slide:
                speaker_notes = slide.notes_slide.notes_text_frame.text

            # Build metadata
            metadata = {
                "source": self.file_path,
                "presentation_id": presentation_id,
                "presentation_title": title,
                "slide_number": slide_idx + 1,
                "total_slides": len(prs.slides),
                "slide_title": self._extract_slide_title(slide),
                "speaker_notes": speaker_notes,
                "has_images": len(slide.shapes) > 0,
                "type": "slide"
            }

            # Create Document
            doc = Document(
                page_content=slide_text,
                metadata=metadata
            )
            documents.append(doc)

        return documents

    def _extract_title(self, prs: Presentation) -> str:
        """Extract presentation title from first slide"""
        if len(prs.slides) > 0:
            first_slide = prs.slides[0]
            if first_slide.shapes.title:
                return first_slide.shapes.title.text
        return "Untitled Presentation"

    def _extract_slide_title(self, slide) -> str:
        """Extract slide title"""
        if slide.shapes.title:
            return slide.shapes.title.text
        return ""

    def _extract_slide_text(self, slide) -> str:
        """Extract all text from slide"""
        text_parts = []
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text_parts.append(shape.text)
        return "\n".join(text_parts)


# Usage
loader = PPTLoader("presentation.pptx")
documents = loader.load()

print(f"Loaded {len(documents)} slides")
print(f"First slide: {documents[0].page_content[:100]}")
print(f"Metadata: {documents[0].metadata}")
```

---

### 2. Contextual Text Splitter

```python
from langchain.text_splitter import TextSplitter
from langchain.schema import Document
from typing import List, Optional
from langchain.llms import Anthropic

class ContextualTextSplitter(TextSplitter):
    """
    Custom Text Splitter that adds contextual information
    to each chunk using LLM.

    Implements the Contextual Retrieval approach.
    """

    def __init__(
        self,
        llm: Anthropic,
        chunk_size: int = 400,
        chunk_overlap: int = 50,
        add_context: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.llm = llm
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.add_context = add_context

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks and add context.
        """
        chunks = []

        for doc in documents:
            # Split into smaller chunks
            doc_chunks = self._split_text(doc.page_content)

            # Generate context for each chunk
            for idx, chunk_text in enumerate(doc_chunks):
                # Create chunk metadata
                chunk_metadata = {
                    **doc.metadata,
                    "chunk_index": idx,
                    "chunk_id": f"{doc.metadata.get('presentation_id', 'unknown')}_{doc.metadata.get('slide_number', 0)}_{idx}"
                }

                # Generate contextual description
                if self.add_context:
                    context = self._generate_context(
                        chunk_text=chunk_text,
                        document=doc,
                        chunk_index=idx
                    )
                    chunk_metadata["context"] = context

                    # Prepend context to chunk for embedding
                    chunk_content = f"{context}\n\n{chunk_text}"
                else:
                    chunk_content = chunk_text

                chunks.append(
                    Document(
                        page_content=chunk_content,
                        metadata=chunk_metadata
                    )
                )

        return chunks

    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks based on size"""
        # Simple sentence-based splitting
        # In production, use more sophisticated logic
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence.split())

            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Keep overlap
                overlap_sentences = current_chunk[-2:]
                current_chunk = overlap_sentences + [sentence]
                current_size = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_size += sentence_size

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _generate_context(
        self,
        chunk_text: str,
        document: Document,
        chunk_index: int
    ) -> str:
        """
        Generate contextual description using LLM.

        Uses prompt template from prompts.md (1.1)
        """
        from langchain.prompts import PromptTemplate

        # Context generation prompt
        prompt = PromptTemplate(
            input_variables=[
                "presentation_title",
                "slide_number",
                "total_slides",
                "slide_title",
                "chunk_content"
            ],
            template="""
<document>
Presentation: {presentation_title}
Slide {slide_number} of {total_slides}: {slide_title}
</document>

<chunk>
{chunk_content}
</chunk>

Generate a concise context (50-100 tokens) to situate this chunk within the presentation.
Include the slide's position and role in the overall narrative.
Output only the context, nothing else.

Context:"""
        )

        # Generate context
        context = self.llm(
            prompt.format(
                presentation_title=document.metadata.get("presentation_title", ""),
                slide_number=document.metadata.get("slide_number", ""),
                total_slides=document.metadata.get("total_slides", ""),
                slide_title=document.metadata.get("slide_title", ""),
                chunk_content=chunk_text
            )
        )

        return context.strip()


# Usage
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)

splitter = ContextualTextSplitter(
    llm=llm,
    chunk_size=400,
    chunk_overlap=50,
    add_context=True
)

# Split documents with context
contextual_chunks = splitter.split_documents(documents)

print(f"Created {len(contextual_chunks)} contextual chunks")
print(f"Sample chunk: {contextual_chunks[0].page_content}")
print(f"Context: {contextual_chunks[0].metadata.get('context')}")
```

---

### 3. Vector Store Setup with Pinecone

```python
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
import os

class PPTVectorStoreBuilder:
    """
    Builder for creating and populating vector store.
    """

    def __init__(
        self,
        pinecone_api_key: str,
        pinecone_index_name: str,
        embedding_model: str = "text-embedding-3-small"
    ):
        self.pinecone_api_key = pinecone_api_key
        self.index_name = pinecone_index_name

        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

    def create_index(self, dimension: int = 1536):
        """Create Pinecone index if not exists"""
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print(f"Created index: {self.index_name}")

    def build_vector_store(
        self,
        documents: List[Document],
        namespace: Optional[str] = None
    ) -> PineconeVectorStore:
        """
        Build vector store from documents.

        Returns LangChain PineconeVectorStore instance.
        """
        # Ensure index exists
        self.create_index()

        # Create vector store
        vector_store = PineconeVectorStore.from_documents(
            documents=documents,
            embedding=self.embeddings,
            index_name=self.index_name,
            namespace=namespace or ""
        )

        print(f"Indexed {len(documents)} documents")
        return vector_store


# Usage - Complete Indexing Pipeline
def index_presentation(ppt_file_path: str):
    """
    Complete indexing pipeline using LangChain.
    """
    # 1. Load PPT
    loader = PPTLoader(ppt_file_path)
    documents = loader.load()
    print(f"✓ Loaded {len(documents)} slides")

    # 2. Split with context
    llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)
    splitter = ContextualTextSplitter(llm=llm, chunk_size=400)
    chunks = splitter.split_documents(documents)
    print(f"✓ Created {len(chunks)} contextual chunks")

    # 3. Build vector store
    builder = PPTVectorStoreBuilder(
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        pinecone_index_name="ppt-contextual-retrieval"
    )
    vector_store = builder.build_vector_store(chunks)
    print(f"✓ Indexed to Pinecone")

    return vector_store


# Run indexing
vector_store = index_presentation("presentation.pptx")
```

---

## 🔍 Query Pipeline

### 1. Custom Hybrid Retriever

```python
from langchain.schema import BaseRetriever, Document
from typing import List
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever(BaseRetriever):
    """
    Custom retriever combining vector search and BM25.

    Implements Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        vector_store: PineconeVectorStore,
        bm25_index: BM25Okapi,
        documents: List[Document],
        top_k: int = 20,
        rrf_k: int = 60
    ):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.documents = documents
        self.top_k = top_k
        self.rrf_k = rrf_k

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve documents using hybrid search.
        """
        # 1. Vector similarity search
        vector_docs = self.vector_store.similarity_search(
            query,
            k=self.top_k
        )

        # 2. BM25 search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:self.top_k]
        bm25_docs = [self.documents[i] for i in bm25_top_indices]

        # 3. Reciprocal Rank Fusion
        fused_docs = self._reciprocal_rank_fusion(
            [vector_docs, bm25_docs]
        )

        return fused_docs[:self.top_k]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version"""
        return self._get_relevant_documents(query)

    def _reciprocal_rank_fusion(
        self,
        doc_lists: List[List[Document]]
    ) -> List[Document]:
        """
        Fuse multiple ranked lists using RRF.

        Score(d) = Σ 1/(k + rank_i(d))
        """
        doc_scores = {}

        for doc_list in doc_lists:
            for rank, doc in enumerate(doc_list):
                doc_id = doc.metadata.get("chunk_id", id(doc))

                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        "doc": doc,
                        "score": 0.0
                    }

                doc_scores[doc_id]["score"] += 1.0 / (self.rrf_k + rank + 1)

        # Sort by score
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        return [item["doc"] for item in sorted_docs]


# Build BM25 index
def build_bm25_index(documents: List[Document]) -> BM25Okapi:
    """Build BM25 index from documents"""
    # Extract contextualized text
    corpus = [doc.page_content for doc in documents]

    # Tokenize
    tokenized_corpus = [doc.lower().split() for doc in corpus]

    # Create BM25 index
    bm25 = BM25Okapi(tokenized_corpus)

    return bm25


# Usage
bm25_index = build_bm25_index(chunks)

hybrid_retriever = HybridRetriever(
    vector_store=vector_store,
    bm25_index=bm25_index,
    documents=chunks,
    top_k=20
)

# Test retrieval
results = hybrid_retriever.get_relevant_documents("revenue growth 2024")
print(f"Retrieved {len(results)} documents")
```

---

### 2. Reranking with Contextual Compression

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

# Setup reranker
reranker = CohereRerank(
    cohere_api_key=os.getenv("COHERE_API_KEY"),
    top_n=5,
    model="rerank-english-v2.0"
)

# Wrap hybrid retriever with compression
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=hybrid_retriever
)

# Retrieve and rerank
final_docs = compression_retriever.get_relevant_documents("revenue growth")
print(f"Top {len(final_docs)} reranked documents")
```

---

### 3. Conversational Retrieval Chain

```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate

# Initialize LLM
llm = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    temperature=0.3,
    max_tokens=500
)

# Memory for conversation
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# Custom prompt template (from prompts.md - 4.1)
qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
<context>
You are an AI assistant helping users understand presentation content.
Answer questions accurately based ONLY on the provided context.
</context>

<retrieved_context>
{context}
</retrieved_context>

<user_query>
{question}
</user_query>

<instructions>
Answer the user's question based on the retrieved context.

Requirements:
1. Answer directly and concisely
2. Cite slide numbers when referencing information
3. If context is insufficient, say "Based on available slides, I cannot find information about [topic]"
4. Do NOT hallucinate - only use provided context
5. Structure: Direct answer first, then supporting details

Format:
[Direct answer in 1-2 sentences]

Details:
- [Supporting point from slide X]
- [Supporting point from slide Y]
</instructions>

Answer:"""
)

# Create conversational retrieval chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=compression_retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": qa_prompt},
    return_source_documents=True,
    verbose=True
)

# Query
response = qa_chain({"question": "What was the revenue growth in 2024?"})

print("Answer:", response["answer"])
print("\nSources:")
for doc in response["source_documents"]:
    print(f"  - Slide {doc.metadata.get('slide_number')}: {doc.page_content[:100]}")
```

---

### 4. Custom Query Analysis Chain

```python
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate

class QueryAnalysisChain:
    """
    Chain to analyze query intent and reformulate if needed.
    """

    def __init__(self, llm: ChatAnthropic):
        self.llm = llm
        self.intent_chain = self._create_intent_chain()
        self.reformulation_chain = self._create_reformulation_chain()

    def _create_intent_chain(self) -> LLMChain:
        """Create chain for intent classification"""
        prompt = ChatPromptTemplate.from_template("""
Query: "{query}"

Classify this query's intent. Choose ONE:

1. FACT_FINDING - Seeking specific data, numbers, facts
2. TREND_ANALYSIS - Looking for patterns, trends, changes
3. COMPARISON - Comparing entities, periods, options
4. EXPLANATION - Seeking to understand why/how
5. SUMMARY - Wanting overview or summary
6. LOCATION - Finding where information is

Output format:
Intent: [INTENT_TYPE]
Confidence: [high/medium/low]
""")

        return LLMChain(llm=self.llm, prompt=prompt)

    def _create_reformulation_chain(self) -> LLMChain:
        """Create chain for query reformulation"""
        prompt = ChatPromptTemplate.from_template("""
<presentation_context>
Title: {presentation_title}
Sections: {sections}
</presentation_context>

User Query: "{query}"

This query is vague. Reformulate it into a clear, specific query.

Output only the reformulated query, nothing else.
""")

        return LLMChain(llm=self.llm, prompt=prompt)

    def analyze(self, query: str, presentation_metadata: dict) -> dict:
        """
        Analyze query and return intent + reformulated query if needed.
        """
        # Classify intent
        intent_result = self.intent_chain.run(query=query)

        # Parse intent (simple parsing, in production use structured output)
        if "Confidence: low" in intent_result:
            # Reformulate
            reformulated = self.reformulation_chain.run(
                query=query,
                presentation_title=presentation_metadata.get("title", ""),
                sections=presentation_metadata.get("sections", "")
            )
            return {
                "original_query": query,
                "reformulated_query": reformulated.strip(),
                "intent": intent_result,
                "confidence": "low"
            }
        else:
            return {
                "original_query": query,
                "reformulated_query": query,
                "intent": intent_result,
                "confidence": "high"
            }


# Usage
query_analyzer = QueryAnalysisChain(llm)

analysis = query_analyzer.analyze(
    query="revenue",
    presentation_metadata={
        "title": "Q4 Business Review",
        "sections": "Introduction, Market Analysis, Financial Results"
    }
)

print(f"Original: {analysis['original_query']}")
print(f"Reformulated: {analysis['reformulated_query']}")
print(f"Intent: {analysis['intent']}")
```

---

### 5. Complete Query Pipeline

```python
class PPTRetrievalSystem:
    """
    Complete LangChain-based PPT retrieval system.
    """

    def __init__(
        self,
        vector_store: PineconeVectorStore,
        documents: List[Document],
        llm: ChatAnthropic
    ):
        self.vector_store = vector_store
        self.documents = documents
        self.llm = llm

        # Build components
        self.bm25_index = build_bm25_index(documents)
        self.hybrid_retriever = HybridRetriever(
            vector_store=vector_store,
            bm25_index=self.bm25_index,
            documents=documents
        )
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=CohereRerank(top_n=5),
            base_retriever=self.hybrid_retriever
        )
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.compression_retriever,
            memory=ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            ),
            return_source_documents=True
        )
        self.query_analyzer = QueryAnalysisChain(llm)

    def query(
        self,
        user_query: str,
        presentation_metadata: dict
    ) -> dict:
        """
        Process query through complete pipeline.
        """
        # 1. Analyze query
        analysis = self.query_analyzer.analyze(user_query, presentation_metadata)

        # 2. Use reformulated query if available
        query = analysis["reformulated_query"]

        # 3. Run QA chain
        response = self.qa_chain({"question": query})

        # 4. Format response
        return {
            "answer": response["answer"],
            "sources": [
                {
                    "slide_number": doc.metadata.get("slide_number"),
                    "slide_title": doc.metadata.get("slide_title"),
                    "content": doc.page_content[:200]
                }
                for doc in response["source_documents"]
            ],
            "query_analysis": analysis
        }


# Usage
system = PPTRetrievalSystem(
    vector_store=vector_store,
    documents=chunks,
    llm=ChatAnthropic(model="claude-3-sonnet-20240229")
)

# Query
result = system.query(
    user_query="What was revenue growth?",
    presentation_metadata={
        "title": "Q4 Business Review",
        "sections": "Intro, Market, Financial, Strategy"
    }
)

print("Answer:", result["answer"])
print("\nSources:")
for source in result["sources"]:
    print(f"  Slide {source['slide_number']}: {source['slide_title']}")
```

---

## 🛠️ Custom Components

### Custom Chain for Quality Check

```python
from langchain.chains.base import Chain
from typing import Dict, Any

class QualityCheckChain(Chain):
    """
    Custom chain to validate answer quality.
    """

    llm: ChatAnthropic
    check_prompt: PromptTemplate

    @property
    def input_keys(self) -> List[str]:
        return ["query", "answer", "source_docs"]

    @property
    def output_keys(self) -> List[str]:
        return ["validated_answer", "quality_score", "issues"]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate answer quality.
        """
        # Build prompt
        prompt_input = {
            "query": inputs["query"],
            "answer": inputs["answer"],
            "sources": "\n".join([
                doc.page_content for doc in inputs["source_docs"]
            ])
        }

        # Check quality
        quality_result = self.llm(self.check_prompt.format(**prompt_input))

        # Parse result (in production, use structured output)
        if "FAIL" in quality_result:
            return {
                "validated_answer": inputs["answer"],
                "quality_score": "fail",
                "issues": quality_result
            }
        else:
            return {
                "validated_answer": inputs["answer"],
                "quality_score": "pass",
                "issues": None
            }


# Usage
quality_chain = QualityCheckChain(
    llm=ChatAnthropic(model="claude-3-haiku-20240307"),
    check_prompt=PromptTemplate(
        input_variables=["query", "answer", "sources"],
        template="""..."""  # From prompts.md 7.1
    )
)

quality_result = quality_chain({
    "query": "revenue growth",
    "answer": "Revenue grew 75%",
    "source_docs": retrieved_docs
})
```

---

## 📚 Implementation Guide

### Step-by-Step Implementation

#### Phase 1: Setup

```python
# 1. Install dependencies
"""
pip install langchain==0.1.0
pip install langchain-anthropic
pip install langchain-openai
pip install langchain-pinecone
pip install langsmith
pip install python-pptx
pip install rank-bm25
pip install cohere
"""

# 2. Environment variables
"""
ANTHROPIC_API_KEY=your_key
OPENAI_API_KEY=your_key
PINECONE_API_KEY=your_key
COHERE_API_KEY=your_key
LANGCHAIN_API_KEY=your_key  # For LangSmith
LANGCHAIN_TRACING_V2=true   # Enable tracing
"""

# 3. Project structure
"""
src/
├── loaders/
│   └── ppt_loader.py          # PPTLoader
├── splitters/
│   └── contextual_splitter.py # ContextualTextSplitter
├── retrievers/
│   ├── hybrid_retriever.py    # HybridRetriever
│   └── bm25_utils.py
├── chains/
│   ├── qa_chain.py            # Main QA chain
│   ├── query_analysis.py      # QueryAnalysisChain
│   └── quality_check.py       # QualityCheckChain
├── prompts/
│   ├── context_generation.py  # Prompt templates
│   ├── answer_generation.py
│   └── quality_check.py
└── main.py                    # Main application
"""
```

#### Phase 2: Indexing

```python
# main.py - Indexing script

from src.loaders.ppt_loader import PPTLoader
from src.splitters.contextual_splitter import ContextualTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
import os

def index_presentation(ppt_path: str):
    """Index a presentation"""

    # 1. Load
    print("Loading presentation...")
    loader = PPTLoader(ppt_path)
    docs = loader.load()

    # 2. Split with context
    print("Creating contextual chunks...")
    llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)
    splitter = ContextualTextSplitter(llm=llm)
    chunks = splitter.split_documents(docs)

    # 3. Embed and index
    print("Indexing to Pinecone...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name="ppt-contextual-retrieval",
        namespace=f"ppt_{hash(ppt_path)}"
    )

    print(f"✓ Indexed {len(chunks)} chunks")
    return vector_store

if __name__ == "__main__":
    vector_store = index_presentation("presentation.pptx")
```

#### Phase 3: Query API

```python
# main.py - Query API

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="PPT Retrieval API")

# Global system instance
retrieval_system = None

class QueryRequest(BaseModel):
    query: str
    presentation_id: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    conversation_id: str

@app.on_event("startup")
async def startup():
    """Initialize system on startup"""
    global retrieval_system

    # Load vector store
    vector_store = PineconeVectorStore(
        index_name="ppt-contextual-retrieval",
        embedding=OpenAIEmbeddings()
    )

    # Initialize system
    retrieval_system = PPTRetrievalSystem(
        vector_store=vector_store,
        documents=[],  # Load from cache/db
        llm=ChatAnthropic(model="claude-3-sonnet-20240229")
    )

@app.post("/query", response_model=QueryResponse)
async def query_presentation(request: QueryRequest):
    """Query a presentation"""
    try:
        result = retrieval_system.query(
            user_query=request.query,
            presentation_metadata={}  # Load from DB
        )

        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            conversation_id="..."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run: uvicorn main:app --reload
```

---

## 🎯 Best Practices

### 1. LangSmith Integration

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ppt-contextual-retrieval"

# Traces will automatically appear in LangSmith dashboard
# https://smith.langchain.com
```

### 2. Caching

```python
from langchain.cache import InMemoryCache, RedisCache
from langchain.globals import set_llm_cache

# In-memory cache (development)
set_llm_cache(InMemoryCache())

# Redis cache (production)
from redis import Redis
set_llm_cache(RedisCache(redis_=Redis.from_url("redis://localhost:6379")))
```

### 3. Streaming Responses

```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# Answers will stream token-by-token
```

### 4. Error Handling

```python
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    try:
        result = qa_chain({"question": query})
        print(f"Tokens used: {cb.total_tokens}")
        print(f"Cost: ${cb.total_cost}")
    except Exception as e:
        logger.error(f"Query failed: {e}")
        # Fallback logic
```

### 5. Testing

```python
import pytest
from langchain.llms.fake import FakeListLLM

def test_contextual_splitter():
    """Test splitter with fake LLM"""
    fake_llm = FakeListLLM(
        responses=["Test context 1", "Test context 2"]
    )

    splitter = ContextualTextSplitter(llm=fake_llm)
    docs = [Document(page_content="Test", metadata={})]

    chunks = splitter.split_documents(docs)

    assert len(chunks) > 0
    assert "Test context" in chunks[0].metadata["context"]
```

---

## 📊 Comparison: LangChain vs Custom Implementation

| Aspect | Custom | LangChain |
|--------|--------|-----------|
| **Setup Time** | 2-3 weeks | 3-5 days |
| **Code Lines** | ~3000 | ~1000 |
| **Flexibility** | High | Medium |
| **Maintenance** | High effort | Low effort |
| **Community Support** | None | Large |
| **Debugging** | Manual | LangSmith |
| **Updates** | Manual | Auto via pip |
| **Best For** | Custom requirements | Standard RAG |

**Recommendation**: Use LangChain unless you have very specific custom requirements.

---

## 🚀 Production Deployment

### Docker Setup

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### requirements.txt

```txt
langchain==0.1.0
langchain-anthropic==0.1.0
langchain-openai==0.1.0
langchain-pinecone==0.1.0
fastapi==0.109.0
uvicorn==0.27.0
python-pptx==0.6.23
rank-bm25==0.2.2
cohere==4.47
redis==5.0.1
prometheus-client==0.19.0
```

---

**End of LangChain Implementation Guide**
