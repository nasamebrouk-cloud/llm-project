"Découper le texte en petits morceaux."

from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentSplitter:
    def __init__(self, chunk_size, chunk_overlap):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        split_docs = splitter.split_documents(documents)
        return split_docs