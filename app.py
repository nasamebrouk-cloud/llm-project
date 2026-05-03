import streamlit as st

from config import *
from src.pdf_loader import PDFLoader
from src.text_splitter import DocumentSplitter
from src.embedding_manager import EmbeddingManager
from src.vector_store import VectorStoreManager
from src.llm_manager import LLMManager
from src.rag_pipeline import RAGPipeline


@st.cache_resource
def initialize_rag():
    pdf_loader = PDFLoader(PDF_PATH)
    documents = pdf_loader.load_pdf()

    splitter = DocumentSplitter(
        CHUNK_SIZE,
        CHUNK_OVERLAP
    )
    split_docs = splitter.split_documents(documents)

    embedding_manager = EmbeddingManager(
        EMBEDDING_MODEL
    )
    embeddings = embedding_manager.load_embeddings()

    vector_manager = VectorStoreManager(
        split_docs,
        embeddings
    )
    vector_store = vector_manager.create_vector_store()

    llm_manager = LLMManager(
        OLLAMA_MODEL
    )
    llm = llm_manager.load_llm()

    rag = RAGPipeline(
        vector_store,
        llm,
        TOP_K_RESULTS
    )
    return rag


def main():
    st.title("RAG PDF Assistant")
    st.write("Ask a question about the loaded PDF document.")

    rag = initialize_rag()

    question = st.text_input("Question")
    if question:
        response = rag.ask_question(question)
        st.markdown("**Réponse :**")
        st.write(response)


if __name__ == "__main__":
    main()