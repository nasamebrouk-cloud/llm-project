from config import *

from src.pdf_loader import PDFLoader
from src.text_splitter import DocumentSplitter
from src.embedding_manager import EmbeddingManager
from src.vector_store import VectorStoreManager
from src.llm_manager import LLMManager
from src.rag_pipeline import RAGPipeline


def main():
    print("Chargement du PDF...")
    pdf_loader = PDFLoader(PDF_PATH)
    documents = pdf_loader.load_pdf()

    print("Découpage du document...")
    splitter = DocumentSplitter(
        CHUNK_SIZE,
        CHUNK_OVERLAP
    )
    split_docs = splitter.split_documents(documents)

    print("Chargement des embeddings...")
    embedding_manager = EmbeddingManager(
        EMBEDDING_MODEL
    )
    embeddings = embedding_manager.load_embeddings()

    print("Création de la base vectorielle...")
    vector_manager = VectorStoreManager(
        split_docs,
        embeddings
    )
    vector_store = vector_manager.create_vector_store()

    print("Connexion à Ollama + Llama 3...")
    llm_manager = LLMManager(
        OLLAMA_MODEL
    )
    llm = llm_manager.load_llm()

    rag = RAGPipeline(
        vector_store,
        llm,
        TOP_K_RESULTS
    )

    print("\n=== Système RAG prêt ===\n")

    while True:
        question = input("Pose ta question (ou 'exit') : ")

        if question.lower() == "exit":
            break

        response = rag.ask_question(question)

        print("\nRéponse :")
        print(response)
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()