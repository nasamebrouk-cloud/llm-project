"la classe qui gère le modèle LLM"
from langchain_openai import ChatOpenAI


class LLMManager:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name

    def load_llm(self):
        llm = ChatOpenAI(
            model=self.model_name,
            temperature=0
        )
        return llm
