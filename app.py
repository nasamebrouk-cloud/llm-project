import os

import streamlit as st
from dotenv import load_dotenv

from config import *
from src.pdf_loader import PDFLoader
from src.text_splitter import DocumentSplitter
from src.embedding_manager import EmbeddingManager
from src.vector_store import VectorStoreManager
from src.llm_manager import LLMManager
from src.rag_pipeline import RAGPipeline

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")


# ==============================
# INIT RAG
# ==============================

@st.cache_resource
def init_rag():
    pdf_loader = PDFLoader(PDF_PATH)
    documents = pdf_loader.load_pdf()

    splitter = DocumentSplitter(CHUNK_SIZE, CHUNK_OVERLAP)
    split_docs = splitter.split_documents(documents)

    embedding_manager = EmbeddingManager(EMBEDDING_MODEL)
    embeddings = embedding_manager.load_embeddings()

    vector_manager = VectorStoreManager(split_docs, embeddings)
    vector_store = vector_manager.create_vector_store()

    llm_manager = LLMManager(OPENAI_MODEL)
    llm = llm_manager.load_llm()

    rag = RAGPipeline(vector_store, llm, TOP_K_RESULTS)
    return rag


# ==============================
# SESSION
# ==============================

if "history" not in st.session_state:
    st.session_state.history = []


# ==============================
# UI CONFIG
# ==============================

st.set_page_config(
    page_title="RAG Assistant",
    layout="wide"
)

st.title("📄 RAG PDF Assistant")
st.caption("Ask questions about your document")

if not OPENAI_API_KEY:
    st.error("OpenAI API key is missing. Set OPENAI_API_KEY in your environment or Streamlit secrets.")
    st.stop()

rag = init_rag()


# ==============================
# SIDEBAR (simplifiée)
# ==============================

with st.sidebar:
    st.header("📁 Document")
    st.success("PDF loaded")

    st.markdown("---")

    if OPENAI_API_KEY:
        st.success("🟢 OpenAI API connected")
    else:
        st.warning("🔴 Set OPENAI_API_KEY in your environment or Streamlit secrets")

    if st.button("🗑️ Clear history"):
        st.session_state.history = []


# ==============================
# LAYOUT
# ==============================

col1, col2 = st.columns([3, 1])


# ==============================
# MAIN
# ==============================

with col1:
    question = st.text_input("Ask your question")

    if st.button("Search"):
        if question.strip():
            with st.spinner("Processing..."):
                response = rag.ask_question(question)

                sources = rag.vector_store.similarity_search(
                    question, k=TOP_K_RESULTS
                )

                st.session_state.history.append(
                    (question, response, sources)
                )

    if st.session_state.history:
        q, r, sources = st.session_state.history[-1]

        st.subheader("Answer")
        st.write(r)

        st.subheader("Sources")

        for i, doc in enumerate(sources):
            st.markdown(f"**Chunk {i+1}**")
            st.write(doc.page_content[:300] + "...")
            st.divider()


# ==============================
# HISTORY
# ==============================

with col2:
    st.subheader("History")

    for q, _, _ in reversed(st.session_state.history):
        st.write(f"• {q}")