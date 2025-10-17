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

# Import comprehensive UI utilities
from streamlit_app.ui_utils import (
    get_presentation_manager,
    list_presentations_sync,
    upload_and_index_sync,
    query_presentation_sync,
    delete_presentation_sync,
    get_statistics_sync
)

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
            ["📚 Presentations", "📤 Upload & Index", "💬 Query", "📊 Stats", "⚙️ Settings"],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Rate limit stats
        st.markdown("### 📈 Rate Limits")
        stats = rate_limiter.get_stats("openai_vision")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Requests", f"{stats['requests_in_last_minute']}/50")
        with col2:
            st.metric("Tokens", f"{stats['tokens_in_last_minute']:,}")

        st.markdown("---")
        st.markdown("### 📚 Indexed Files")
        try:
            presentations = list_presentations_sync()
            if presentations:
                for pres in presentations[:5]:  # Show max 5
                    st.text(f"✓ {pres['name'][:30]}...")
                if len(presentations) > 5:
                    st.text(f"... and {len(presentations) - 5} more")
            else:
                st.info("No presentations indexed")
        except:
            st.info("Loading...")

    # Main content
    if page == "📚 Presentations":
        show_presentations_page()
    elif page == "📤 Upload & Index":
        show_upload_page()
    elif page == "💬 Query":
        show_query_page()
    elif page == "📊 Stats":
        show_stats_page()
    elif page == "⚙️ Settings":
        show_settings_page()


def show_presentations_page():
    """Presentations management page."""
    st.markdown('<div class="main-header">📚 Manage Presentations</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Browse, search, and manage indexed presentations</div>', unsafe_allow_html=True)

    # Load presentations from BM25Store
    try:
        presentations = list_presentations_sync()

        if not presentations:
            st.info("📭 No presentations indexed yet. Upload a presentation to get started!")
            return

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Presentations", len(presentations))
        with col2:
            total_slides = sum(p['total_slides'] for p in presentations)
            st.metric("Total Slides", total_slides)
        with col3:
            total_chunks = sum(p['total_chunks'] for p in presentations)
            st.metric("Total Chunks", total_chunks)

        st.markdown("---")

        # Search box
        search = st.text_input("🔍 Search presentations", placeholder="Type to filter...")

        # Filter presentations
        if search:
            filtered = [
                p for p in presentations
                if search.lower() in p['name'].lower()
                or search.lower() in p.get('title', '').lower()
            ]
        else:
            filtered = presentations

        st.markdown(f"### 📋 Presentations ({len(filtered)})")

        # Display presentations
        for idx, pres in enumerate(filtered):
            with st.expander(
                f"📄 {pres['name']} ({pres['total_slides']} slides, {pres['total_chunks']} chunks)",
                expanded=(idx == 0)
            ):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(f"**ID:** `{pres['presentation_id']}`")
                    st.markdown(f"**Title:** {pres.get('title', 'N/A')}")
                    st.markdown(f"**Slides:** {pres['total_slides']}")
                    st.markdown(f"**Chunks:** {pres['total_chunks']}")
                    st.markdown(f"**Indexed:** {pres['indexed_at'].split('T')[0]}")
                    st.markdown(f"**Pinecone Index:** {pres['pinecone_index_name']}")

                with col2:
                    # Delete button
                    if st.button(
                        "🗑️ Delete",
                        key=f"delete_{pres['presentation_id']}",
                        type="secondary",
                        use_container_width=True
                    ):
                        # Confirm deletion
                        st.session_state[f'confirm_delete_{pres["presentation_id"]}'] = True

                    # Show confirmation if requested
                    if st.session_state.get(f'confirm_delete_{pres["presentation_id"]}'):
                        st.warning("⚠️ Confirm deletion?")

                        col_a, col_b = st.columns(2)
                        with col_a:
                            if st.button(
                                "Yes",
                                key=f"confirm_yes_{pres['presentation_id']}",
                                type="primary",
                                use_container_width=True
                            ):
                                with st.spinner("Deleting..."):
                                    result = delete_presentation_sync(
                                        pres['presentation_id'],
                                        delete_pinecone=True,
                                        delete_images=True
                                    )

                                    if result['deleted_from_bm25']:
                                        st.success(f"✅ Deleted {pres['name']}")
                                        # Clear confirmation state
                                        del st.session_state[f'confirm_delete_{pres["presentation_id"]}']
                                        st.rerun()
                                    else:
                                        st.error("❌ Deletion failed")
                                        if result['errors']:
                                            for err in result['errors']:
                                                st.error(err)

                        with col_b:
                            if st.button(
                                "No",
                                key=f"confirm_no_{pres['presentation_id']}",
                                use_container_width=True
                            ):
                                del st.session_state[f'confirm_delete_{pres["presentation_id"]}']
                                st.rerun()

    except Exception as e:
        st.error(f"❌ Failed to load presentations: {str(e)}")
        logger.error(f"Failed to load presentations: {e}")
        import traceback
        logger.error(traceback.format_exc())


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
            try:
                # Progress bar
                progress_bar = st.progress(0, text="Initializing...")

                def update_progress(pct, msg):
                    progress_bar.progress(pct, text=msg)

                # Call comprehensive upload_and_index
                stats = upload_and_index_sync(
                    file_path=str(upload_path),
                    use_contextual=add_context,
                    use_vision=extract_images,
                    extract_images=extract_images,
                    include_notes=include_notes,
                    progress_callback=update_progress
                )

                # Success message
                st.markdown(f"""
                <div class="success-box">
                    <h3>✅ Indexing Complete!</h3>
                    <p><strong>Presentation ID:</strong> {stats['presentation_id']}</p>
                    <p><strong>Title:</strong> {stats['presentation']}</p>
                    <p><strong>Slides:</strong> {stats['slides']}</p>
                    <p><strong>Chunks:</strong> {stats['chunks']}</p>
                    <p><strong>Contextual:</strong> {'Yes' if stats['contextual'] else 'No'}</p>
                    <p><strong>Vision Analysis:</strong> {'Yes' if stats['vision_analyzed'] else 'No'}</p>
                    <p><strong>Vector Store:</strong> Pinecone ({settings.pinecone_index_name})</p>
                    <p><strong>Text Search:</strong> BM25 (SQLite + Serialized Index)</p>
                    <p><strong>Indexed At:</strong> {stats['indexed_at'].split('T')[0]}</p>
                </div>
                """, unsafe_allow_html=True)

                st.success("🎉 Presentation indexed successfully! Go to 'Presentations' or 'Query' page.")
                st.balloons()

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

    # Load presentations from BM25Store
    try:
        presentations = list_presentations_sync()

        if not presentations:
            st.warning("⚠️ No presentations indexed yet. Please upload and index a presentation first.")
            return

        # Create options: Add "All Presentations" option
        ppt_options = {"__all__": "🌐 All Presentations (Cross-Document Search)"}
        ppt_options.update({
            p['presentation_id']: f"{p['name']} ({p['total_chunks']} chunks)"
            for p in presentations
        })

        selected_ppt = st.selectbox(
            "Select Presentation",
            options=list(ppt_options.keys()),
            format_func=lambda x: ppt_options[x]
        )

    except Exception as e:
        st.error(f"❌ Failed to load presentations: {str(e)}")
        logger.error(f"Failed to load presentations: {e}")
        return

    if selected_ppt:
        # Show presentation info (if not "all")
        if selected_ppt != "__all__":
            ppt_info = next((p for p in presentations if p['presentation_id'] == selected_ppt), None)
            if ppt_info:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Slides", ppt_info['total_slides'])
                with col2:
                    st.metric("Chunks", ppt_info['total_chunks'])
                with col3:
                    st.metric("Indexed", ppt_info['indexed_at'].split('T')[0])
        else:
            # Show aggregate stats for all
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Presentations", len(presentations))
            with col2:
                total_slides = sum(p['total_slides'] for p in presentations)
                st.metric("Total Slides", total_slides)
            with col3:
                total_chunks = sum(p['total_chunks'] for p in presentations)
                st.metric("Total Chunks", total_chunks)

        st.markdown("---")

        # Clear history button
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🗑️ Clear History"):
                manager = get_presentation_manager()
                asyncio.run(manager.clear_chat_history())
                # Clear query input by removing from session state
                if "query_input" in st.session_state:
                    del st.session_state["query_input"]
                st.success("Chat history cleared!")
                st.rerun()

        # Query input
        query = st.text_input(
            "Ask a question",
            placeholder="e.g., What was the revenue growth?" if selected_ppt != "__all__" else "e.g., Compare revenue across all reports",
            key="query_input"
        )

        if query:
            with st.spinner("🤔 Thinking..."):
                try:
                    # Determine presentation_id for query
                    query_ppt_id = None if selected_ppt == "__all__" else selected_ppt

                    # Query using comprehensive method
                    result = query_presentation_sync(
                        query=query,
                        presentation_id=query_ppt_id,
                        return_sources=True
                    )

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

    try:
        # Get comprehensive statistics
        stats = get_statistics_sync()

        # Main metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Indexed Presentations", stats['total_presentations'])

        with col2:
            st.metric("Total Documents", f"{stats['total_documents']:,}")

        with col3:
            st.metric("Total Slides", stats['total_slides'])

        with col4:
            st.metric("Storage Size", f"{stats['total_size_mb']:.2f} MB")

        st.markdown("---")

        # Backend details
        st.markdown("### 🔧 Backend Configuration")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Text Search")
            st.write(f"**Backend:** {stats['backend_type']}")
            st.write(f"**SQLite:** {stats['sqlite_size_mb']:.2f} MB")
            st.write(f"**Index:** {stats['index_size_mb']:.2f} MB")
            st.write(f"**Loaded:** {'✅' if stats['index_loaded'] else '❌'}")

        with col2:
            st.markdown("#### Vector Store")
            st.write(f"**Provider:** Pinecone")
            st.write(f"**Index:** {stats['pinecone_index']}")
            st.write(f"**Dimension:** 3072 (text-embedding-3-large)")
            st.write(f"**Metric:** cosine")

        with col3:
            st.markdown("#### Models")
            st.write(f"**Embedding:** {settings.embedding_model}")
            st.write(f"**Context:** {settings.context_generation_model}")
            st.write(f"**Answer:** {settings.answer_generation_model}")
            st.write(f"**Vision:** {settings.vision_model}")

        st.markdown("---")

        # Presentations breakdown
        st.markdown("### 📚 Presentations Breakdown")

        if stats['presentations']:
            for pres in stats['presentations']:
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

                with col1:
                    st.write(f"**{pres['name']}**")
                with col2:
                    st.write(f"{pres['total_slides']} slides")
                with col3:
                    st.write(f"{pres['total_chunks']} chunks")
                with col4:
                    st.write(pres['indexed_at'].split('T')[0])
        else:
            st.info("No presentations indexed yet")

        st.markdown("---")

        # Rate limit details
        st.markdown("### 📈 Rate Limit Details")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### OpenAI")
            openai_stats = rate_limiter.get_stats("openai_vision")
            st.json(openai_stats)

        with col2:
            st.markdown("#### Anthropic")
            anthropic_stats = rate_limiter.get_stats("anthropic")
            st.json(anthropic_stats)

    except Exception as e:
        st.error(f"❌ Failed to load statistics: {str(e)}")
        logger.error(f"Failed to load statistics: {e}")
        import traceback
        logger.error(traceback.format_exc())


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
