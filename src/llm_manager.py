"Connexion avec Ollama + Llama 3."
"ollama run llama3"
from langchain_ollama import OllamaLLM


class LLMManager:
    def __init__(self, model_name):
        self.model_name = model_name

    def load_llm(self):
        llm = OllamaLLM(model=self.model_name)
        return llm