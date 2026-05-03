"Faire le vrai pipeline RAG complet."

class RAGPipeline:
    def __init__(self, vector_store, llm, top_k):
        self.vector_store = vector_store
        self.llm = llm
        self.top_k = top_k

    def ask_question(self, question):
        results = self.vector_store.similarity_search(
            question,
            k=self.top_k
        )

        context = "\n\n".join([
            doc.page_content for doc in results
        ])

        prompt = f"""
Tu es un assistant intelligent.

Réponds à la question uniquement à partir du contexte suivant.
Contexte :
{context}

Question :
{question}

Réponse :
"""

        response = self.llm.invoke(prompt)
        return response