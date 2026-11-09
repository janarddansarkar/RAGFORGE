"""Models: the embedder and the LLM that the config asks for.

Every model is reached at a cloud provider (openai, gemini or anthropic) with the user's own
API key. All three are served by the `openai` SDK through their OpenAI-compatible endpoints
(the [providers.*] tables in providers.toml). The key lives in memory only (see use_providers).

Everything else calls make_embeddings(config) / make_llm(config).
An LLM has .model (its name), .invoke(prompt).content (the reply) and .stream(prompt).
"""

import json
import math
import re
from contextlib import contextmanager
from functools import cached_property

import openai
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage


class UsageTracker:
    """Counts the tokens every cloud call used, and what they cost at the prices in providers.toml.
    One tracker belongs to one conversation (it is made by use_providers). mark() / cost(since)
    give the cost of a stretch of calls, such as one question."""

    def __init__(self, providers: dict):
        self.providers = providers
        self.calls: list[
            tuple[str, str, int, int]
        ] = []  # provider, model, tokens in, out

    def add(self, provider: str, model: str, tokens_in: int, tokens_out: int) -> None:
        self.calls.append((provider, model, tokens_in, tokens_out))

    def mark(self) -> int:
        return len(self.calls)

    def cost(self, since: int = 0) -> dict:
        """{"dollars", "tokens_in", "tokens_out"} for the calls made after mark() returned `since`."""
        total = {"dollars": 0.0, "tokens_in": 0, "tokens_out": 0}
        for provider, model, tokens_in, tokens_out in self.calls[since:]:
            price = self.providers[provider].get("prices", {}).get(model, {})
            total["dollars"] += (
                tokens_in * price.get("input", 0) + tokens_out * price.get("output", 0)
            ) / 1_000_000
            total["tokens_in"] += tokens_in
            total["tokens_out"] += tokens_out
        return total

    @staticmethod
    def estimate(text: str) -> int:
        """Tokens in a text when the provider did not say: about 4 characters each."""
        return max(1, len(text) // 4)


class CloudModel:
    """What every cloud model needs: a client for the provider's OpenAI-compatible endpoint,
    and plain errors for the things a user will meet (a wrong key, a rate limit)."""

    def __init__(
        self,
        provider: str,
        model: str,
        settings: dict,
        api_key: str | None,
        usage: UsageTracker | None = None,
    ):
        if not api_key:
            raise RuntimeError(f"Enter your {provider} API key first.")
        self.provider = provider
        self.model = model
        self.settings = settings  # the [providers.<name>] table of config.toml
        self.api_key = api_key
        self.usage = usage

    def record(self, used, sent: str, received: str = "") -> None:
        """Count one call. `used` is the provider's usage report, or None: then it is estimated."""
        if self.usage is None:
            return
        if used is not None:
            tokens_in = used.prompt_tokens
            tokens_out = getattr(used, "completion_tokens", 0) or 0
        else:
            tokens_in = UsageTracker.estimate(sent)
            tokens_out = UsageTracker.estimate(received) if received else 0
        self.usage.add(self.provider, self.model, tokens_in, tokens_out)

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
                batch = texts[start : start + 32]
                reply = self.client.embeddings.create(model=self.model, input=batch)
            self.record(getattr(reply, "usage", None), " ".join(batch))
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
        usage: UsageTracker | None = None,
    ):
        super().__init__(provider, model, settings, api_key, usage)
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
        text = str(reply.choices[0].message.content or "").strip()
        self.record(getattr(reply, "usage", None), prompt, text)
        return AIMessage(content=text)

    def stream(self, prompt: str, max_new_tokens: int | None = None):
        """Like invoke(), but yields the reply piece by piece as the model writes it."""
        extra = {"stream": True}
        if self.settings["stream_usage"]:
            extra["stream_options"] = {
                "include_usage": True
            }  # a last chunk with the counts
        used, pieces = None, []
        with self.friendly_errors():
            for part in self.client.chat.completions.create(
                **self.request(prompt, max_new_tokens, **extra)
            ):
                used = getattr(part, "usage", None) or used
                if part.choices and part.choices[0].delta.content:
                    pieces.append(part.choices[0].delta.content)
                    yield part.choices[0].delta.content
        self.record(used, prompt, "".join(pieces))

    def digit_score(self, prompt: str) -> float:
        """Ask for one digit (0-9) and return it as 0-1. Where the provider returns
        probabilities this is the probability-weighted average;
        otherwise it is just the digit the model wrote."""
        if not self.settings["logprobs"]:
            reply = str(self.invoke(prompt, max_new_tokens=5).content)
            digit = re.search(r"\d", reply)
            return int(digit.group()) / 9 if digit else 0.0
        with self.friendly_errors():
            reply = self.client.chat.completions.create(
                **self.request(prompt, 1, logprobs=True, top_logprobs=10)
            )
        self.record(getattr(reply, "usage", None), prompt, "0")
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
        "usage": UsageTracker(providers),  # this conversation's token and cost count
        **{
            name: chat[name]
            for name in ("llm_model", "reranker_model", "guard_model", "judge_model")
        },
    }


def make_embeddings(config: dict) -> CloudEmbeddings:
    provider = config["embedding_provider"]
    return CloudEmbeddings(
        provider,
        config["embedding_model"],
        config["providers"][provider],
        config.get("embedding_api_key"),
        config.get("usage"),
    )


def make_llm(config: dict) -> CloudChat:
    provider = config["chat_provider"]
    return CloudChat(
        provider,
        config["llm_model"],
        config["providers"][provider],
        config.get("chat_api_key"),
        config["max_new_tokens"],
        config.get("usage"),
    )
