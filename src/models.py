"""Models: the embedder, the LLM and the audio model that config.toml asks for.

Every model is served by Ollama (https://ollama.com): free, and it runs on your own machine
or on a server you control (ollama_host in config.toml). Pull a model once with
`ollama pull <name>`; after that no internet is needed.

Everything else calls make_embeddings(config) / make_llm(config).
An LLM has .model (its name) and .invoke(prompt).content (the reply).
"""

import math
from functools import cached_property

import httpx
import ollama
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage

DEFAULT_HOST = "http://localhost:11434"


class OllamaModel:
    """What every Ollama model needs: a client, and clear errors when Ollama is not ready."""

    def __init__(self, model: str, host: str | None = None):
        self.model = model
        self.host = host or DEFAULT_HOST

    @cached_property
    def client(self) -> ollama.Client:
        return ollama.Client(host=self.host, timeout=600)

    def call(self, method: str, **kwargs):
        """Run a client method (chat, embed), turning Ollama's errors into plain advice."""
        try:
            return getattr(self.client, method)(model=self.model, **kwargs)
        except ollama.ResponseError as error:
            if error.status_code == 404:
                raise RuntimeError(
                    f"Ollama does not have the model '{self.model}'. "
                    f"Download it with: ollama pull {self.model}"
                ) from error
            raise
        except (httpx.ConnectError, ConnectionError) as error:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.host}. Start it with: ollama serve"
            ) from error

    def unload(self) -> None:
        """Free the model's memory now instead of after Ollama's idle timeout."""
        self.call("generate", prompt="", keep_alive=0)


class OllamaEmbeddings(OllamaModel, Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 32):  # a few at a time keeps requests small
            reply = self.call("embed", input=texts[start : start + 32])
            vectors += [list(vector) for vector in reply.embeddings]
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class OllamaChat(OllamaModel):
    """A chat model on Ollama; temperature 0, so answers are repeatable."""

    def __init__(
        self,
        model: str,
        max_new_tokens: int = 256,
        host: str | None = None,
        num_ctx: int = 8192,
    ):
        super().__init__(model, host)
        self.max_new_tokens = max_new_tokens
        self.num_ctx = (
            num_ctx  # Ollama's default window (4096) is too small for 5 chunks
        )

    def invoke(
        self, prompt: str, max_new_tokens: int | None = None, schema=None
    ) -> AIMessage:
        """schema (a pydantic class or a JSON schema dict) makes the reply valid JSON."""
        if schema is not None and hasattr(schema, "model_json_schema"):
            schema = schema.model_json_schema()
        reply = self.call(
            "chat",
            messages=[{"role": "user", "content": prompt}],
            format=schema,
            think=False,  # no hidden reasoning: replies stay short and parseable
            options={
                "temperature": 0,
                "num_predict": max_new_tokens or self.max_new_tokens,
                "num_ctx": self.num_ctx,
            },
        )
        return AIMessage(content=str(reply.message.content or "").strip())

    def digit_score(self, prompt: str) -> float:
        """Ask for one digit (0-9) and return it as 0-1: the average of the digits the model
        considered, weighted by their probability, so unsure answers fall between digits."""
        reply = self.call(
            "chat",
            messages=[{"role": "user", "content": prompt}],
            think=False,
            logprobs=True,
            top_logprobs=10,
            options={"temperature": 0, "num_predict": 1, "num_ctx": self.num_ctx},
        )
        weights: dict[int, float] = {}
        for choice in reply.logprobs[0].top_logprobs if reply.logprobs else []:
            token = choice.token.strip()
            if len(token) == 1 and token in "0123456789":  # not look-alikes such as '₀'
                weights[int(token)] = weights.get(int(token), 0.0) + math.exp(
                    choice.logprob
                )
        total = sum(weights.values())
        if not total:
            return 0.0
        return sum(digit * weight for digit, weight in weights.items()) / total / 9


def make_embeddings(config: dict) -> OllamaEmbeddings:
    return OllamaEmbeddings(config["embedding_model"], config.get("ollama_host"))


def make_llm(config: dict) -> OllamaChat:
    return OllamaChat(
        config["llm_model"],
        config["max_new_tokens"],
        config.get("ollama_host"),
        config["ollama_num_ctx"],
    )
