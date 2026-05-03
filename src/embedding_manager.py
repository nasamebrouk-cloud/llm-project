"Transformer le texte en embeddings."

from langchain_community.embeddings import HuggingFaceEmbeddings


class EmbeddingManager:
    def __init__(self, model_name):
        self.model_name = model_name

    def load_embeddings(self):
        embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name
        )
        return embeddings