import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
import tempfile
import os
from dynamic_dataset_engine import (
    load_any_dataset,
    auto_profile,
    generate_chunks_from_profile,
    build_session_store,
    session_retrieve,
)
from rag_core import build_prompt, call_llm


# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="DataLens AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# CUSTOM CSS FOR PROFESSIONAL DARK THEME
# ============================================================================
def apply_custom_css():
    css = """
    <style>
    /* Main background */
    body, [data-testid="stAppViewContainer"] {
        background-color: #0f1117 !important;
        color: #e0e0e0 !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1d27 !important;
        border-right: 1px solid #2e3250;
    }
    
    /* Cards and containers */
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
        background-color: #1e2130;
        border: 1px solid #2e3250;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 16px;
    }
    
    /* Titles and headings */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    /* Main title */
    [data-testid="stMarkdownContainer"] > h1:first-of-type {
        font-size: 2.8rem !important;
        color: white !important;
    }
    
    /* Subtitle and text */
    [data-testid="stMarkdownContainer"] p {
        color: #c0c0c0;
        line-height: 1.8;
    }
    
    /* Buttons */
    button {
        background-color: #6366f1 !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        border: none !important;
    }
    
    button:hover {
        background-color: #4f46e5 !important;
    }
    
    /* Input fields */
    input, textarea, [data-testid="stTextInput"], [data-testid="stTextArea"] {
        background-color: #1e2130 !important;
        color: #e0e0e0 !important;
        border: 1px solid #2e3250 !important;
        border-radius: 8px !important;
    }
    
    /* Select/radio */
    [data-testid="stRadio"] label {
        color: #e0e0e0 !important;
    }
    
    /* Badge pills for output types */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 4px 4px 4px 0;
    }
    
    .badge-summary {
        background-color: #3b82f6;
        color: white;
    }
    
    .badge-features {
        background-color: #10b981;
        color: white;
    }
    
    .badge-insights {
        background-color: #f59e0b;
        color: white;
    }
    
    /* Spinner color */
    [data-testid="stSpinner"] {
        color: #6366f1 !important;
    }
    
    /* Info/warning/error boxes */
    [data-testid="stAlert"] {
        background-color: #1e2130 !important;
        border: 1px solid #2e3250 !important;
        border-radius: 8px !important;
        color: #e0e0e0 !important;
    }
    
    /* Expander */
    [data-testid="stExpander"] {
        background-color: #1e2130 !important;
        border: 1px solid #2e3250 !important;
        border-radius: 8px !important;
    }
    
    /* Dataframe */
    [data-testid="stDataFrame"] {
        background-color: #1e2130 !important;
    }
    
    /* Divider */
    hr {
        border-color: #2e3250 !important;
    }
    
    /* Text in muted style */
    .muted {
        color: #8b8b8b;
        font-size: 0.9rem;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


apply_custom_css()


# ============================================================================
# CACHED EMBEDDER (for efficient reuse)
# ============================================================================
@st.cache_resource
def get_embedder():
    """Load embedding model once and cache it."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-mpnet-base-v2")


# ============================================================================
# SIDEBAR LAYOUT
# ============================================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("### 🔍 DataLens AI")
        st.markdown(
            "<p style='color: #888; font-size: 0.9rem; margin-top: -10px;'>"
            "Turn raw data into business intelligence"
            "</p>",
            unsafe_allow_html=True
        )
        
        st.divider()
        
        st.markdown("#### How it works")
        st.markdown(
            """
            1. **Upload** your dataset
            2. **Choose** what you want to know
            3. **Write** or select your query
            4. **Get** AI-generated insights
            """,
            unsafe_allow_html=False
        )
        
        st.divider()
        
        st.markdown("#### Supported formats")
        cols = st.columns(4)
        for i, fmt in enumerate(["CSV", "XLSX", "Parquet", "JSON"]):
            with cols[i]:
                st.markdown(
                    f"<div style='text-align: center; padding: 8px; "
                    f"background: #2e3250; border-radius: 6px; "
                    f"color: #6366f1; font-weight: 600;'>{fmt}</div>",
                    unsafe_allow_html=True
                )
        
        st.divider()
        
        st.markdown("#### Output types")
        st.markdown(
            """
            **📋 Summary**  
            Overview of the dataset
            
            **💡 Feature Suggestions**  
            New columns to engineer
            
            **📊 Business Insights**  
            Findings for stakeholders
            """,
            unsafe_allow_html=False
        )
        
        st.divider()
        
        st.markdown(
            "<p class='muted' style='text-align: center; margin-top: 40px;'>"
            "Powered by Groq LLaMA 3.1 · FAISS · sentence-transformers"
            "</p>",
            unsafe_allow_html=True
        )


# ============================================================================
# MAIN AREA LAYOUT
# ============================================================================
def render_main():
    # Header
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 0;'>DataLens AI</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center; color: #888; font-size: 1.1rem; margin-top: -10px;'>"
        "Upload any dataset. Ask any question. Get instant business-friendly insights."
        "</p>",
        unsafe_allow_html=True
    )
    st.divider()
    
    # Input panel
    col_left, col_right = st.columns([1.2, 1])
    
    # LEFT COLUMN — Dataset upload + query input
    with col_left:
        st.subheader("1️⃣ Upload your dataset")
        uploaded_file = st.file_uploader(
            label="Select file",
            type=["csv", "xlsx", "parquet", "json"],
            label_visibility="collapsed",
            help="Max recommended size: 50MB"
        )
        
        if uploaded_file:
            size_kb = uploaded_file.size / 1024
            st.success(f"✅ {uploaded_file.name} uploaded ({size_kb:.1f} KB)")
        
        st.subheader("2️⃣ Enter your query")
        
        tab1, tab2 = st.tabs(["Custom Query", "Preset Queries"])
        
        query_input = None
        with tab1:
            query_input = st.text_area(
                label="Enter your question",
                placeholder="e.g. What are the key business insights from this dataset that a product manager should act on immediately?",
                height=120,
                label_visibility="collapsed",
                key="custom_query"
            )
        
        with tab2:
            presets = [
                "Give me a complete overview of this dataset",
                "What are the most important business insights?",
                "What new features should I engineer from this data?",
                "What data quality issues should I be aware of?",
                "Which columns are most useful for predictive modeling?"
            ]
            selected_preset = st.radio(
                label="Select preset",
                options=presets,
                label_visibility="collapsed",
                key="preset_query"
            )
            query_input = selected_preset
    
    # RIGHT COLUMN — Output type selector + generate button
    with col_right:
        st.subheader("3️⃣ Choose output type")
        
        output_type_options = {
            "📋 Dataset Summary": "summary",
            "💡 Feature Suggestions": "feature_suggestions",
            "📊 Business Insights": "business_insights"
        }
        
        selected_option = st.radio(
            label="Select output type",
            options=list(output_type_options.keys()),
            label_visibility="collapsed",
            key="output_type_selector"
        )
        output_type = output_type_options[selected_option]
        
        descriptions = {
            "summary": "A structured overview of schema, scale, quality, and key metrics.",
            "feature_suggestions": "5 engineered features with formula and business value.",
            "business_insights": "3 findings with impact and recommended actions."
        }
        st.markdown(
            f"<p style='color: #888; font-size: 0.85rem; margin-top: -10px;'>"
            f"{descriptions[output_type]}"
            f"</p>",
            unsafe_allow_html=True
        )
        
        st.subheader("4️⃣ Generate")
        generate_button = st.button(
            "🚀 Generate Insights",
            use_container_width=True,
            type="primary"
        )
        
        st.markdown(
            "<p class='muted'>"
            "ℹ️ First run loads the embedding model (~30 seconds). "
            "Subsequent queries on the same dataset are instant."
            "</p>",
            unsafe_allow_html=True
        )
    
    return uploaded_file, query_input, output_type, generate_button


# ============================================================================
# PROCESSING AND OUTPUT
# ============================================================================
def process_and_display(uploaded_file, query, output_type):
    """Main processing pipeline."""
    
    # Validations
    if not uploaded_file:
        st.error("❌ Please upload a dataset to continue.")
        return
    
    if not query or query.strip() == "":
        st.error("❌ Please enter a query or select a preset.")
        return
    
    # Create unique cache key for this file
    file_key = f"{uploaded_file.name}_{uploaded_file.size}"
    
    try:
        with st.spinner("🔍 Analyzing your dataset..."):
            # Save temp file
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=Path(uploaded_file.name).suffix
            ) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_path = tmp_file.name
            
            try:
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Load dataset
                status_text.text("20% — Profiling dataset schema and statistics...")
                progress_bar.progress(20)
                df = load_any_dataset(temp_path)
                dataset_name = Path(uploaded_file.name).stem
                
                # Check cache
                store_already_built = (
                    "session_store" in st.session_state and
                    st.session_state.get("loaded_file_key") == file_key
                )
                
                if store_already_built:
                    status_text.text("60% — Using cached vector store...")
                    progress_bar.progress(60)
                    index, chunks, embedder = st.session_state["session_store"]
                else:
                    # Profile
                    status_text.text("25% — Profiling dataset schema and statistics...")
                    progress_bar.progress(25)
                    profile = auto_profile(df, dataset_name)
                    
                    # Generate chunks
                    status_text.text("40% — Generating retrieval context chunks...")
                    progress_bar.progress(40)
                    chunks = generate_chunks_from_profile(profile)
                    
                    # Build store with cached embedder
                    status_text.text("60% — Building session vector store...")
                    progress_bar.progress(60)
                    embedder = get_embedder()
                    index, chunks, embedder = build_session_store(chunks, embedder=embedder)
                    
                    # Cache the store
                    st.session_state["session_store"] = (index, chunks, embedder)
                    st.session_state["loaded_file_key"] = file_key
                    st.session_state["profile"] = profile
                
                # Retrieve
                status_text.text("80% — Retrieving relevant context...")
                progress_bar.progress(80)
                retrieved_chunks = session_retrieve(query, index, chunks, embedder)
                
                if not retrieved_chunks:
                    st.warning(
                        "⚠️ Low relevance context found. Try rephrasing your query."
                    )
                    progress_bar.empty()
                    status_text.empty()
                    return
                
                # Generate
                status_text.text("90% — Generating insights with LLM...")
                progress_bar.progress(90)
                prompt = build_prompt(query, retrieved_chunks, output_type)
                llm_response = call_llm(prompt)
                
                progress_bar.progress(100)
                status_text.text("✅ Done!")
                progress_bar.empty()
                status_text.empty()
                
                if not llm_response:
                    st.error("❌ LLM returned an empty response. Try again.")
                    return
                
                # Store result
                st.session_state["last_result"] = {
                    "response": llm_response,
                    "query": query,
                    "output_type": output_type,
                    "dataset_name": dataset_name,
                    "chunks_used": [c["chunk_id"] for c in retrieved_chunks],
                    "retrieved_chunks": retrieved_chunks,
                }
                
                st.rerun()
            
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
    
    except FileNotFoundError as e:
        st.error(f"❌ File not found: {str(e)}")
    except ValueError as e:
        st.error(f"❌ {str(e)}")
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        st.write("**Debug info:**")
        st.write(str(e))


# ============================================================================
# RESULT DISPLAY
# ============================================================================
def display_result():
    """Display the stored result from session state with enhanced UI."""
    if "last_result" not in st.session_state:
        return
    
    result = st.session_state["last_result"]
    
    # ========== HEADER SECTION ==========
    st.markdown(
        "<h2 style='color: #ffffff; margin-bottom: 8px;'>✨ Your Insights</h2>",
        unsafe_allow_html=True
    )
    
    header_col1, header_col2, header_col3 = st.columns([2, 1.5, 1])
    
    with header_col1:
        st.markdown(
            f"<p style='color: #6366f1; font-weight: 600; margin: 0; font-size: 1.1rem;'>"
            f"📋 Dataset: <span style='color: #e0e0e0;'>{result['dataset_name']}</span>"
            f"</p>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<p style='color: #888; margin: 4px 0 0 0; font-size: 0.95rem;'>"
            f"{result['query']}"
            f"</p>",
            unsafe_allow_html=True
        )
    
    with header_col2:
        output_type = result["output_type"]
        badge_colors = {
            "summary": "#3b82f6",
            "feature_suggestions": "#10b981",
            "business_insights": "#f59e0b"
        }
        badge_labels = {
            "summary": "📊 Summary",
            "feature_suggestions": "💡 Features",
            "business_insights": "📈 Insights"
        }
        st.markdown(
            f"<div style='background: {badge_colors[output_type]}; "
            f"color: white; padding: 8px 16px; border-radius: 6px; "
            f"font-weight: 600; text-align: center; display: inline-block;'>"
            f"{badge_labels[output_type]}</div>",
            unsafe_allow_html=True
        )
    
    with header_col3:
        st.markdown(
            f"<p style='color: #888; font-size: 0.85rem; text-align: right;'>"
            f"Context: {len(result['chunks_used'])} chunks"
            f"</p>",
            unsafe_allow_html=True
        )
    
    st.divider()
    
    # ========== MAIN INSIGHTS CARD ==========
    st.markdown(
        "<h3 style='color: #ffffff; margin-bottom: 16px;'>📝 Generated Insights</h3>",
        unsafe_allow_html=True
    )
    
    # Enhanced insight display with better visual hierarchy
    insight_html = result['response'].replace('\n', '<br>')
    st.markdown(
        f"<div style='background: linear-gradient(135deg, #1e2130 0%, #16192b 100%); "
        f"border-left: 4px solid #6366f1; border-radius: 8px; padding: 24px; "
        f"line-height: 1.9; color: #e0e0e0; font-size: 1rem;'>"
        f"{insight_html}"
        f"</div>",
        unsafe_allow_html=True
    )
    
    st.markdown("")  # spacer
    
    # ========== METADATA SECTION ==========
    st.markdown(
        "<h3 style='color: #ffffff; margin-top: 32px; margin-bottom: 16px;'>📊 Data & Context</h3>",
        unsafe_allow_html=True
    )
    
    meta_col1, meta_col2 = st.columns(2)
    
    # Profile stats
    if "profile" in st.session_state:
        profile = st.session_state["profile"]
        summary = profile["summary_stats"]
        
        with meta_col1:
            with st.expander("📁 Dataset Overview", expanded=False):
                stats_html = ""
                stats = {
                    "Total Rows": f"<strong>{profile['shape'][0]:,}</strong>",
                    "Total Columns": f"<strong>{profile['shape'][1]}</strong>",
                    "Null Rate": f"<span style='color: #f59e0b;'><strong>{summary['total_null_rate']*100:.2f}%</strong></span>",
                    "Numeric": f"<strong>{summary['n_numeric_cols']}</strong>",
                    "Categorical": f"<strong>{summary['n_categorical_cols']}</strong>",
                    "Duplicates": f"<strong>{summary['duplicate_row_count']:,}</strong>",
                    "Memory": f"<strong>{summary['memory_usage_mb']:.2f} MB</strong>"
                }
                for key, value in stats.items():
                    stats_html += f"<div style='margin: 8px 0; color: #c0c0c0;'>{key}: {value}</div>"
                
                st.markdown(stats_html, unsafe_allow_html=True)
        
        # Column profile
        with meta_col2:
            with st.expander("🔎 Column Details", expanded=False):
                cols_data = []
                for col in profile["columns"]:
                    if col.get("is_id_column") or col.get("is_constant"):
                        continue
                    cols_data.append({
                        "Column": col["column"],
                        "Type": col["dtype"].upper(),
                        "Null %": f"{col['null_rate']*100:.1f}%",
                        "Unique": col["n_unique"]
                    })
                if cols_data:
                    import pandas as pd
                    df_profile = pd.DataFrame(cols_data)
                    st.dataframe(
                        df_profile,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Null %": st.column_config.ProgressColumn(
                                "Null %",
                                min_value=0,
                                max_value=100,
                            ),
                        }
                    )
    
    st.markdown("")  # spacer
    
    # ========== RETRIEVAL TRANSPARENCY ==========
    with st.expander("🔍 Retrieval Context (What the AI used)", expanded=False):
        retrieval_data = []
        for chunk in result["retrieved_chunks"]:
            retrieval_data.append({
                "Source": chunk.get("chunk_type", "—").replace("_", " ").title(),
                "ID": chunk.get("chunk_id", "—"),
                "Relevance": f"{min(chunk.get('score', 0), 1):.2%}"
            })
        if retrieval_data:
            import pandas as pd
            df_retrieval = pd.DataFrame(retrieval_data)
            st.dataframe(
                df_retrieval,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Relevance": st.column_config.ProgressColumn(
                        "Relevance",
                        min_value=0,
                        max_value=1,
                    ),
                }
            )
    
    st.markdown("")  # spacer
    st.divider()
    
    # ========== DOWNLOAD & ACTIONS ==========
    st.markdown(
        "<h3 style='color: #ffffff; margin-bottom: 12px;'>💾 Export</h3>",
        unsafe_allow_html=True
    )
    
    download_text = (
        f"{'='*60}\n"
        f"INSIGHTS REPORT\n"
        f"{'='*60}\n\n"
        f"Dataset: {result['dataset_name']}\n"
        f"Output Type: {result['output_type'].replace('_', ' ').title()}\n"
        f"Query: {result['query']}\n"
        f"{'='*60}\n\n"
        f"{result['response']}\n\n"
        f"{'='*60}\n"
        f"Generated by DataLens AI\n"
    )
    
    col_download, col_regenerate = st.columns([1, 1])
    with col_download:
        st.download_button(
            label="⬇️ Download as Text",
            data=download_text,
            file_name=f"{result['dataset_name']}_insights.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col_regenerate:
        if st.button("🔄 Generate New Insights", use_container_width=True):
            if "last_result" in st.session_state:
                del st.session_state["last_result"]
            st.rerun()


# ============================================================================
# MAIN APP FLOW
# ============================================================================
def main():
    render_sidebar()
    
    uploaded_file, query_input, output_type, generate_button = render_main()
    
    if generate_button:
        process_and_display(uploaded_file, query_input, output_type)
    
    if "last_result" in st.session_state:
        st.divider()
        st.markdown("---")
        display_result()


if __name__ == "__main__":
    main()
