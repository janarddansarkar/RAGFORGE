"""Models: the embedder and the LLM that config.toml asks for.

provider = "huggingface"  models run inside this Python process (no server, free, offline
                          once downloaded); names are Hugging Face ids, e.g. "BAAI/bge-m3"
provider = "ollama"       models are served by a running Ollama; names are Ollama tags

Everything else calls make_embeddings(config) / make_llm(config) and never names a provider.
An LLM has .model (its name) and .invoke(prompt).content (the reply), like ChatOllama.
"""

from functools import cached_property

from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings


class HuggingFaceEmbeddings(Embeddings):
    def __init__(self, model: str):
        self.model = model

    @cached_property
    def encoder(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.encoder.encode(texts, normalize_embeddings=True, batch_size=16)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class HuggingFaceChat:
    """A chat model from Hugging Face; greedy decoding, so answers are repeatable."""

    def __init__(self, model: str, max_new_tokens: int = 256):
        self.model = model
        self.max_new_tokens = max_new_tokens

    @cached_property
    def generator(self):
        import torch
        from transformers import pipeline

        if torch.cuda.is_available():
            device, dtype = "cuda", torch.bfloat16
        elif torch.backends.mps.is_available():
            device, dtype = "mps", torch.bfloat16
        else:
            device, dtype = "cpu", torch.float32
        return pipeline("text-generation", model=self.model, dtype=dtype, device=device)

    def invoke(self, prompt: str) -> AIMessage:
        reply = self.generator(
            [{"role": "user", "content": prompt}],
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )
        return AIMessage(content=str(reply[0]["generated_text"]).strip())


def make_embeddings(config: dict) -> Embeddings:
    if config["provider"] == "ollama":
        return OllamaEmbeddings(
            model=config["embedding_model"], base_url=config["ollama_base_url"]
        )
    return HuggingFaceEmbeddings(config["embedding_model"])


Llm = HuggingFaceChat | ChatOllama


def make_llm(config: dict) -> Llm:
    if config["provider"] == "ollama":
        return ChatOllama(
            model=config["llm_model"],
            base_url=config["ollama_base_url"],
            temperature=0,
            reasoning=False,  # no "thinking" output from qwen3
        )
    return HuggingFaceChat(config["llm_model"], config["max_new_tokens"])
