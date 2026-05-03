"Créer la base vectorielle FAISS"

from langchain_community.vectorstores import FAISS


class VectorStoreManager:
    def __init__(self, documents, embeddings):
        self.documents = documents
        self.embeddings = embeddings

    def create_vector_store(self):
        vector_store = FAISS.from_documents(
            self.documents,
            self.embeddings
        )
        return vector_store