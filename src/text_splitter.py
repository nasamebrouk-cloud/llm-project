"Découper le texte en petits morceaux."

from langchain_community.docstore.document import Document


class DocumentSplitter:
    def __init__(self, chunk_size, chunk_overlap):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_text(self, text):
        if not text:
            return []

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)

            if end >= text_length:
                break

            start = end - self.chunk_overlap
            if start < 0:
                start = 0

        return chunks

    def split_documents(self, documents):
        split_docs = []

        for document in documents:
            text = getattr(document, "page_content", None)
            if text is None:
                text = str(document)

            metadata = getattr(document, "metadata", {}) or {}
            text_chunks = self._split_text(text)

            for idx, chunk in enumerate(text_chunks):
                chunk_metadata = dict(metadata)
                chunk_metadata["chunk_index"] = idx
                split_docs.append(Document(page_content=chunk, metadata=chunk_metadata))

        return split_docs