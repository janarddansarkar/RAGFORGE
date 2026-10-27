"""Models: the embedder and the LLM that the config asks for.

Two kinds of provider:
  - ollama (https://ollama.com): free, runs on your own machine or a server you control
    (ollama_host in config.toml). Pull a model once with `ollama pull <name>`. The default,
    used by the command line scripts and the eval.
  - a cloud provider (openai, gemini, anthropic): the user brings their own API key. All three
    are reached with the `openai` SDK through their OpenAI-compatible endpoints (the
    [providers.*] tables in config.toml). The key lives in memory only (see use_providers).

Everything else calls make_embeddings(config) / make_llm(config).
An LLM has .model (its name), .invoke(prompt).content (the reply) and .stream(prompt).
"""

import json
import math
import re
from contextlib import contextmanager
from functools import cached_property

import httpx
import ollama
import openai
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage

DEFAULT_HOST = "http://localhost:11434"


class OllamaModel:
    """What every Ollama model needs: a client, and clear errors when Ollama is not ready."""

    def __init__(
        self, model: str, host: str | None = None, keep_alive: str | None = None
    ):
        self.model = model
        self.host = host or DEFAULT_HOST
        self.keep_alive = (
            keep_alive  # how long Ollama keeps the model loaded after a call
        )

    @cached_property
    def client(self) -> ollama.Client:
        return ollama.Client(host=self.host, timeout=600)

    @contextmanager
    def friendly_errors(self):
        """Turn Ollama's errors into plain advice."""
        try:
            yield
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

    def call(self, method: str, **kwargs):
        """Run a client method (chat, embed), turning Ollama's errors into plain advice."""
        kwargs = {"keep_alive": self.keep_alive, **kwargs}
        with self.friendly_errors():
            return getattr(self.client, method)(model=self.model, **kwargs)

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
        keep_alive: str | None = None,
    ):
        super().__init__(model, host, keep_alive)
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

    def stream(self, prompt: str, max_new_tokens: int | None = None):
        """Like invoke(), but yields the reply piece by piece as the model writes it."""
        with self.friendly_errors():
            for part in self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                think=False,
                keep_alive=self.keep_alive,
                options={
                    "temperature": 0,
                    "num_predict": max_new_tokens or self.max_new_tokens,
                    "num_ctx": self.num_ctx,
                },
            ):
                if part.message.content:
                    yield part.message.content

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


class CloudModel:
    """What every cloud model needs: a client for the provider's OpenAI-compatible endpoint,
    and plain errors for the things a user will meet (a wrong key, a rate limit)."""

    def __init__(self, provider: str, model: str, settings: dict, api_key: str | None):
        if not api_key:
            raise RuntimeError(f"Enter your {provider} API key first.")
        self.provider = provider
        self.model = model
        self.settings = settings  # the [providers.<name>] table of config.toml
        self.api_key = api_key

    @cached_property
    def client(self) -> openai.OpenAI:
        return openai.OpenAI(
            base_url=self.settings["base_url"], api_key=self.api_key, timeout=120
        )

    @contextmanager
    def friendly_errors(self):
        try:
            yield
        except openai.AuthenticationError as error:
            raise RuntimeError(f"{self.provider} rejected the API key.") from error
        except openai.PermissionDeniedError as error:
            raise RuntimeError(
                f"{self.provider} says this key may not use '{self.model}'."
            ) from error
        except openai.NotFoundError as error:
            raise RuntimeError(
                f"{self.provider} has no model '{self.model}'. "
                "Change it in config.toml ([providers.*])."
            ) from error
        except openai.RateLimitError as error:
            raise RuntimeError(
                f"{self.provider} rate limit or quota reached: {error.message}"
            ) from error
        except openai.APIConnectionError as error:
            raise RuntimeError(f"Cannot reach {self.provider}.") from error

    def unload(self) -> None:
        """Nothing to free: the provider holds the model."""


class CloudEmbeddings(CloudModel, Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 32):
            with self.friendly_errors():
                reply = self.client.embeddings.create(
                    model=self.model, input=texts[start : start + 32]
                )
            vectors += [list(item.embedding) for item in reply.data]
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class CloudChat(CloudModel):
    """A chat model at a cloud provider; temperature 0, so answers are repeatable."""

    def __init__(
        self,
        provider: str,
        model: str,
        settings: dict,
        api_key: str | None,
        max_new_tokens: int = 256,
    ):
        super().__init__(provider, model, settings, api_key)
        self.max_new_tokens = max_new_tokens

    def request(self, prompt: str, max_new_tokens: int | None, **extra) -> dict:
        """The arguments of one chat call. Providers name the length limit differently."""
        limit = {self.settings["token_param"]: max_new_tokens or self.max_new_tokens}
        messages = [{"role": "user", "content": prompt}]
        return {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            **limit,
            **extra,
        }

    def invoke(
        self, prompt: str, max_new_tokens: int | None = None, schema=None
    ) -> AIMessage:
        """schema (a pydantic class or a JSON schema dict) asks for a reply in that JSON shape."""
        extra = {}
        if schema is not None:
            if hasattr(schema, "model_json_schema"):
                schema = schema.model_json_schema()
            # Said in the prompt for every provider; also enforced as JSON where it is supported.
            prompt += (
                "\n\nReply with one JSON object only, matching this JSON schema:\n"
                + json.dumps(schema)
            )
            if self.settings["json_mode"]:
                extra["response_format"] = {"type": "json_object"}
        with self.friendly_errors():
            reply = self.client.chat.completions.create(
                **self.request(prompt, max_new_tokens, **extra)
            )
        return AIMessage(content=str(reply.choices[0].message.content or "").strip())

    def stream(self, prompt: str, max_new_tokens: int | None = None):
        """Like invoke(), but yields the reply piece by piece as the model writes it."""
        with self.friendly_errors():
            for part in self.client.chat.completions.create(
                **self.request(prompt, max_new_tokens, stream=True)
            ):
                if part.choices and part.choices[0].delta.content:
                    yield part.choices[0].delta.content

    def digit_score(self, prompt: str) -> float:
        """Ask for one digit (0-9) and return it as 0-1. Where the provider returns
        probabilities this is the probability-weighted average (as OllamaChat does);
        otherwise it is just the digit the model wrote."""
        if not self.settings["logprobs"]:
            reply = str(self.invoke(prompt, max_new_tokens=5).content)
            digit = re.search(r"\d", reply)
            return int(digit.group()) / 9 if digit else 0.0
        with self.friendly_errors():
            reply = self.client.chat.completions.create(
                **self.request(prompt, 1, logprobs=True, top_logprobs=10)
            )
        content = reply.choices[0].logprobs.content if reply.choices[0].logprobs else []
        weights: dict[int, float] = {}
        for choice in content[0].top_logprobs if content else []:
            token = choice.token.strip()
            if len(token) == 1 and token in "0123456789":
                weights[int(token)] = weights.get(int(token), 0.0) + math.exp(
                    choice.logprob
                )
        total = sum(weights.values())
        if not total:
            return 0.0
        return sum(digit * weight for digit, weight in weights.items()) / total / 9


def use_providers(
    config: dict,
    chat_provider: str,
    chat_key: str | None,
    embedding_provider: str,
    embedding_key: str | None,
) -> dict:
    """A copy of the config that sends every model call to these providers with these keys.
    The chat provider supplies the answer, reranker, guard and judge models; the embedding
    provider supplies the embedder. The keys stay in this dict (memory only)."""
    providers = config["providers"]
    if "embedding_model" not in providers[embedding_provider]:
        raise ValueError(f"{embedding_provider} has no embedding model; pick another.")
    chat = providers[chat_provider]
    return {
        **config,
        "chat_provider": chat_provider,
        "chat_api_key": chat_key,
        "embedding_provider": embedding_provider,
        "embedding_api_key": embedding_key,
        "embedding_model": providers[embedding_provider]["embedding_model"],
        **{
            name: chat[name]
            for name in ("llm_model", "reranker_model", "guard_model", "judge_model")
        },
    }


def make_embeddings(config: dict) -> OllamaEmbeddings | CloudEmbeddings:
    provider = config.get("embedding_provider", "ollama")
    if provider != "ollama":
        return CloudEmbeddings(
            provider,
            config["embedding_model"],
            config["providers"][provider],
            config.get("embedding_api_key"),
        )
    return OllamaEmbeddings(
        config["embedding_model"],
        config.get("ollama_host"),
        config.get("ollama_keep_alive"),
    )


def make_llm(config: dict) -> OllamaChat | CloudChat:
    provider = config.get("chat_provider", "ollama")
    if provider != "ollama":
        return CloudChat(
            provider,
            config["llm_model"],
            config["providers"][provider],
            config.get("chat_api_key"),
            config["max_new_tokens"],
        )
    return OllamaChat(
        config["llm_model"],
        config["max_new_tokens"],
        config.get("ollama_host"),
        config["ollama_num_ctx"],
        config.get("ollama_keep_alive"),
    )
