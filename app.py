"""
HireSense v2 — Agentic Resume Search Engine
Streamlit UI  |  LLM: Llama-3.1-8B-Instruct  |  Embeddings: sentence-transformers
"""
import streamlit as st

from src.config import CATEGORIES, LANGSMITH_API_KEY, LANGSMITH_PROJECT
from src.query_pipeline import answer, retrieve

st.set_page_config(page_title="HireSense", page_icon="🔍", layout="wide")

st.title("🔍 HireSense v2")
st.caption(
    "**Resume Search Engine** · "
    "Llama-3.1-8B-Instruct (HuggingFace) · "
    "sentence-transformers · Pinecone"
)

if LANGSMITH_API_KEY:
    project_url = f"https://smith.langchain.com/o/~/projects/p/{LANGSMITH_PROJECT}"
    st.markdown(
        f'<a href="{project_url}" target="_blank">'
        f'<img src="https://img.shields.io/badge/LangSmith-traced-brightgreen?logo=langchain" '
        f'alt="LangSmith traced" style="height:20px"/></a>',
        unsafe_allow_html=True,
    )

st.divider()

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Results to fetch", min_value=1, max_value=10, value=5)
    st.markdown("---")
    st.subheader("Supported categories")
    for cat in sorted(CATEGORIES):
        st.caption(cat)

user_query = st.text_input(
    "Enter your hiring query:",
    placeholder="e.g. Senior Python engineer with AWS and machine learning experience",
)

if st.button("Search", type="primary") and user_query.strip():

    with st.spinner("Classifying query, searching resumes, generating answer…"):
        try:
            retrieved = retrieve(user_query, top_k=top_k)
            result    = answer(user_query, retrieved)
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()

    st.markdown(
        f"**Predicted category:** &nbsp;"
        f"<span style='background:#1f77b4;color:white;padding:3px 10px;"
        f"border-radius:12px;font-size:0.85rem'>{retrieved['category']}</span>",
        unsafe_allow_html=True,
    )
    st.divider()

    col_answer, col_docs = st.columns([1, 1], gap="large")

    with col_answer:
        st.subheader("💬 Generated Answer")
        st.write(result["answer"])
        if LANGSMITH_API_KEY:
            st.info("This run is traced in LangSmith — check your project dashboard.", icon="🔗")

    with col_docs:
        st.subheader(f"📄 Top {len(retrieved['docs'])} Matching Resumes")
        if not retrieved["docs"]:
            st.warning("No documents found. Try a different query.")
        for doc in retrieved["docs"]:
            with st.expander(f"**{doc['id']}** — score: {doc['score']:.4f}"):
                st.text(doc["text"][:600] + ("…" if len(doc["text"]) > 600 else ""))

    st.divider()
    with st.expander("🔎 Processing Trace", expanded=True):
        for step in retrieved["trace"] + [result["trace"]]:
            st.json(step)

elif st.session_state.get("_searched") and not user_query.strip():
    st.warning("Please enter a query before searching.")
