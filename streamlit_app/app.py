"""
Streamlit Frontend for PPT Contextual Retrieval System.
"""
import streamlit as st
import sys
import os
from pathlib import Path
import asyncio
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.pipeline import PPTContextualRetrievalPipeline
from src.utils.rate_limiter import rate_limiter
from loguru import logger

# Configure logger
logger.add(
    "logs/app_{time}.log",
    rotation="1 day",
    retention="7 days",
    level="INFO"
)

# Page configuration
st.set_page_config(
    page_title="PPT Contextual Retrieval",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1E88E5;
    margin-bottom: 1rem;
}
.sub-header {
    font-size: 1.2rem;
    color: #666;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1E88E5;
}
.success-box {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
.error-box {
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    color: #721c24;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'indexed_presentations' not in st.session_state:
        st.session_state.indexed_presentations = {}

    if 'current_presentation' not in st.session_state:
        st.session_state.current_presentation = None

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'pipelines' not in st.session_state:
        st.session_state.pipelines = {}


def main():
    """Main application."""
    init_session_state()

    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50?text=PPT+RAG", use_column_width=True)

        st.markdown("## 🎯 Navigation")
        page = st.radio(
            "Select Page",
            ["📤 Upload & Index", "💬 Query", "📊 Stats", "⚙️ Settings"],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Rate limit stats
        st.markdown("### 📈 Rate Limits")
        stats = rate_limiter.get_stats("anthropic")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Requests", f"{stats['requests_in_last_minute']}/50")
        with col2:
            st.metric("Tokens", f"{stats['tokens_in_last_minute']:,}")

        st.markdown("---")
        st.markdown("### 📚 Indexed Files")
        if st.session_state.indexed_presentations:
            for ppt_id, info in st.session_state.indexed_presentations.items():
                st.text(f"✓ {info['title'][:30]}...")
        else:
            st.info("No presentations indexed yet")

    # Main content
    if page == "📤 Upload & Index":
        show_upload_page()
    elif page == "💬 Query":
        show_query_page()
    elif page == "📊 Stats":
        show_stats_page()
    elif page == "⚙️ Settings":
        show_settings_page()


def show_upload_page():
    """Upload and indexing page."""
    st.markdown('<div class="main-header">📤 Upload & Index Presentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload a PowerPoint file to create a searchable knowledge base</div>', unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a PowerPoint file (.pptx)",
        type=['pptx'],
        help="Upload a .pptx file to index it for retrieval"
    )

    if uploaded_file:
        # Save uploaded file
        upload_path = Path(settings.upload_dir) / uploaded_file.name
        with open(upload_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"✓ File uploaded: {uploaded_file.name}")

        # Indexing options
        col1, col2, col3 = st.columns(3)
        with col1:
            extract_images = st.checkbox("Extract Images", value=True)
        with col2:
            include_notes = st.checkbox("Include Speaker Notes", value=True)
        with col3:
            add_context = st.checkbox("Add Contextual Descriptions", value=True)

        # Index button
        if st.button("🚀 Start Indexing", type="primary", use_container_width=True):
            with st.spinner("Indexing presentation... This may take a few minutes."):
                try:
                    # Generate index name
                    ppt_id = uploaded_file.name.replace('.pptx', '').replace(' ', '-').lower()
                    index_name = f"ppt-{ppt_id}"[:50]

                    # Create pipeline
                    progress_bar = st.progress(0, text="Initializing pipeline...")
                    pipeline = PPTContextualRetrievalPipeline(
                        index_name=index_name,
                        use_contextual=add_context,
                        use_vision=extract_images,
                        use_reranking=True
                    )

                    progress_bar.progress(20, text="Loading and analyzing presentation...")

                    # Index presentation
                    stats = asyncio.run(pipeline.index_presentation(
                        str(upload_path),
                        extract_images=extract_images,
                        include_notes=include_notes,
                        analyze_images=extract_images
                    ))

                    progress_bar.progress(100, text="Indexing complete!")

                    # Store pipeline and metadata
                    st.session_state.pipelines[ppt_id] = pipeline
                    st.session_state.indexed_presentations[ppt_id] = {
                        'title': stats.get('presentation', uploaded_file.name),
                        'slides': stats['slides'],
                        'chunks': stats['chunks'],
                        'timestamp': datetime.now().isoformat(),
                        'index_name': index_name,
                        'contextual': stats['contextual'],
                        'vision_analyzed': stats['vision_analyzed']
                    }

                    st.session_state.current_presentation = ppt_id

                    # Success message
                    st.markdown(f"""
                    <div class="success-box">
                        <h3>✅ Indexing Complete!</h3>
                        <p><strong>Presentation:</strong> {stats['presentation']}</p>
                        <p><strong>Slides:</strong> {stats['slides']}</p>
                        <p><strong>Chunks:</strong> {stats['chunks']}</p>
                        <p><strong>Contextual:</strong> {'Yes' if stats['contextual'] else 'No'}</p>
                        <p><strong>Vision Analysis:</strong> {'Yes' if stats['vision_analyzed'] else 'No'}</p>
                        <p><strong>Vector Store:</strong> Pinecone ({index_name})</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.success("🎉 Presentation indexed successfully! You can now query it.")

                except Exception as e:
                    st.markdown(f"""
                    <div class="error-box">
                        <h3>❌ Indexing Failed</h3>
                        <p>{str(e)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    logger.error(f"Indexing failed: {e}")
                    import traceback
                    logger.error(traceback.format_exc())


def show_query_page():
    """Query interface page."""
    st.markdown('<div class="main-header">💬 Query Presentation</div>', unsafe_allow_html=True)

    if not st.session_state.indexed_presentations:
        st.warning("⚠️ No presentations indexed yet. Please upload and index a presentation first.")
        return

    # Select presentation
    ppt_options = {
        ppt_id: info['title']
        for ppt_id, info in st.session_state.indexed_presentations.items()
    }

    selected_ppt = st.selectbox(
        "Select Presentation",
        options=list(ppt_options.keys()),
        format_func=lambda x: ppt_options[x]
    )

    if selected_ppt:
        ppt_info = st.session_state.indexed_presentations[selected_ppt]

        # Show presentation info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Slides", ppt_info['slides'])
        with col2:
            st.metric("Chunks", ppt_info['chunks'])
        with col3:
            st.metric("Indexed", ppt_info['timestamp'].split('T')[0])

        st.markdown("---")

        # Query input
        query = st.text_input(
            "Ask a question about the presentation",
            placeholder="e.g., What was the revenue growth in Q4?",
            key="query_input"
        )

        # Clear history button
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🗑️ Clear History"):
                if selected_ppt in st.session_state.pipelines:
                    st.session_state.pipelines[selected_ppt].clear_chat_history()
                    st.success("Chat history cleared!")
                    st.rerun()

        if query:
            # Get pipeline
            if selected_ppt not in st.session_state.pipelines:
                st.error("⚠️ Pipeline not found. Please re-index the presentation.")
                return

            pipeline = st.session_state.pipelines[selected_ppt]

            with st.spinner("🤔 Thinking..."):
                try:
                    # Query the pipeline
                    result = asyncio.run(pipeline.query(query, return_sources=True))

                    # Display answer
                    st.markdown("### 💡 Answer")
                    st.markdown(result['answer'])

                    # Display quality score
                    if result.get('quality_score'):
                        quality = result['quality_score']
                        score = quality.get('score', 0)

                        st.markdown("---")
                        st.markdown("### 📊 Answer Quality")

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Score", f"{score:.0%}")
                        with col2:
                            st.metric("Sources", "✅" if quality.get('has_sources') else "❌")
                        with col3:
                            st.metric("Citations", "✅" if quality.get('cites_slides') else "❌")
                        with col4:
                            st.metric("Confident", "✅" if quality.get('not_uncertain') else "❌")

                    # Display sources
                    if result.get('formatted_sources'):
                        st.markdown("---")
                        st.markdown("### 📚 Sources")

                        sources = result['formatted_sources']
                        for idx, source in enumerate(sources[:5]):
                            with st.expander(
                                f"📄 Slide {source['slide_number']}: {source['slide_title']}",
                                expanded=(idx == 0)
                            ):
                                st.markdown(f"**Section:** {source['section']}")
                                st.markdown(f"**Content:**")
                                st.markdown(source['content'])

                                # Show ranking scores
                                if source.get('rrf_score'):
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("RRF Score", f"{source['rrf_score']:.4f}")
                                    with col2:
                                        st.metric("Vector Rank", source.get('vector_rank', 'N/A'))
                                    with col3:
                                        st.metric("BM25 Rank", source.get('bm25_rank', 'N/A'))

                    # Add to chat history
                    if 'chat_messages' not in st.session_state:
                        st.session_state.chat_messages = []

                    st.session_state.chat_messages.append({
                        "question": query,
                        "answer": result['answer']
                    })

                except Exception as e:
                    st.error(f"❌ Query failed: {str(e)}")
                    logger.error(f"Query failed: {e}")
                    import traceback
                    logger.error(traceback.format_exc())




def show_stats_page():
    """Statistics page."""
    st.markdown('<div class="main-header">📊 System Statistics</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Indexed Presentations",
            len(st.session_state.indexed_presentations)
        )

    with col2:
        total_chunks = sum(
            info.get('chunks', 0)
            for info in st.session_state.indexed_presentations.values()
        )
        st.metric("Total Chunks", total_chunks)

    with col3:
        total_slides = sum(
            info.get('slides', 0)
            for info in st.session_state.indexed_presentations.values()
        )
        st.metric("Total Slides", total_slides)

    with col4:
        st.metric("Queries", len(st.session_state.chat_history))

    st.markdown("---")

    # Rate limit details
    st.markdown("### 📈 Rate Limit Details")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Anthropic (Claude)")
        stats = rate_limiter.get_stats("anthropic")
        st.json(stats)

    with col2:
        st.markdown("#### OpenAI")
        stats = rate_limiter.get_stats("openai_vision")
        st.json(stats)


def show_settings_page():
    """Settings page."""
    st.markdown('<div class="main-header">⚙️ Settings</div>', unsafe_allow_html=True)

    st.markdown("### 🤖 Model Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.text_input("Context Model", value=settings.context_generation_model, disabled=True)
        st.text_input("Answer Model", value=settings.answer_generation_model, disabled=True)

    with col2:
        st.text_input("Embedding Model", value=settings.embedding_model, disabled=True)
        st.text_input("Vision Model", value=settings.vision_model, disabled=True)

    st.markdown("---")

    st.markdown("### 🔧 Chunking Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.number_input("Chunk Size", value=settings.max_chunk_size, disabled=True)

    with col2:
        st.number_input("Chunk Overlap", value=settings.chunk_overlap, disabled=True)

    st.markdown("---")

    st.markdown("### 📊 Retrieval Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.number_input("Top K Retrieval", value=settings.top_k_retrieval, disabled=True)

    with col2:
        st.number_input("Top N Rerank", value=settings.top_n_rerank, disabled=True)


if __name__ == "__main__":
    main()
