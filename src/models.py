"""Models: the embedder, the LLM and the reranker that config.toml asks for.

The LLMs are hosted models, run through LiteLLM (https://docs.litellm.ai/docs/providers): name
them with a provider prefix, e.g. "anthropic/claude-haiku-5-5". The API key comes from the
user (config["api_keys"], set by the web UI) or from the provider's environment variable
(ANTHROPIC_API_KEY, OPENAI_API_KEY, ...).

Embeddings and speech-to-text are hosted too, named the same way ("openai/text-embedding-3-small",
"openai/whisper-1"). Nothing runs on this machine. The reranker is a chat model that scores each
chunk.

Everything else calls make_embeddings(config) / make_llm(config).
An LLM has .model (its name) and .invoke(prompt).content (the reply).
"""

import math
import threading
import time

from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage

# A model name that starts with one of these + "/" is a hosted API model, run through LiteLLM.
API_PROVIDERS = (
    "anthropic", "openai", "gemini", "cohere", "voyage", "groq",
    "mistral", "deepseek", "together_ai", "azure", "bedrock", "vertex_ai",
)  # fmt: skip


def is_api_model(name: str | None) -> bool:
    """True for "anthropic/claude-haiku-5-5"."""
    return bool(name) and name.split("/")[0] in API_PROVIDERS


def api_key_for(config: dict, model: str) -> str | None:
    """The key the user gave for this model's provider (config["api_keys"], kept in memory
    only), or None: LiteLLM then reads the provider's environment variable."""
    return config.get("api_keys", {}).get(model.split("/")[0]) or None


def require_prefix(model: str) -> None:
    if not is_api_model(model):
        raise ValueError(
            f"'{model}' has no provider prefix. Write it like "
            "'anthropic/claude-haiku-5-5' or 'openai/gpt-4o' (see config.toml)."
        )


class LiteLLMEmbeddings(Embeddings):
    """An embedding model on a hosted API (through LiteLLM)."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        tracker=None,
        role: str = "embeddings",
    ):
        require_prefix(model)
        self.model = model
        self.api_key = api_key
        self.tracker = tracker  # counts the calls (see usage.py)
        self.role = role

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        import litellm

        vectors: list[list[float]] = []
        for start in range(0, len(texts), 32):
            reply = litellm.embedding(
                model=self.model, input=texts[start : start + 32], api_key=self.api_key
            )
            vectors += [list(item["embedding"]) for item in reply.data]
            if self.tracker:
                self.tracker.record(self.role, self.model, reply, "embedding")
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class LiteLLMChat:
    """A chat model on a hosted API (through LiteLLM); temperature 0, so answers are repeatable."""

    def __init__(
        self,
        model: str,
        max_new_tokens: int = 256,
        api_key: str | None = None,
        tracker=None,
        role: str = "answer",
    ):
        require_prefix(model)
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.api_key = api_key  # the user's own key; sent with every call, never stored
        self.tracker = tracker  # counts the calls (see usage.py)
        self.role = role  # what this model is used for, shown in the usage breakdown

    def _complete(self, prompt: str, max_new_tokens: int | None, **extra):
        import litellm

        litellm.drop_params = (
            True  # skip settings a provider does not support (e.g. logprobs)
        )
        if self.model.startswith("gemini/"):
            # Gemini "thinks" first by default, and the thinking uses up the small token limits
            # (a 256-token answer, a 1-token rerank score), leaving an empty reply.
            extra.setdefault("reasoning_effort", "disable")
        return litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_new_tokens or self.max_new_tokens,
            api_key=self.api_key,
            **extra,
        )

    def invoke(
        self, prompt: str, max_new_tokens: int | None = None, schema=None
    ) -> AIMessage:
        """schema (a pydantic class or a JSON schema dict) makes the reply valid JSON."""
        extra = {}
        if schema is not None:
            if not hasattr(schema, "model_json_schema"):  # a plain JSON schema dict
                schema = {
                    "type": "json_schema",
                    "json_schema": {"name": "reply", "schema": schema},
                }
            extra["response_format"] = schema
        reply = self._complete(prompt, max_new_tokens, **extra)
        if self.tracker:
            self.tracker.record(self.role, self.model, reply)
        return AIMessage(content=str(reply.choices[0].message.content or "").strip())

    def stream(self, prompt: str, max_new_tokens: int | None = None):
        """Like invoke(), but yields the reply piece by piece as the model writes it."""
        parts = []
        try:
            for part in self._complete(
                prompt,
                max_new_tokens,
                stream=True,
                stream_options={
                    "include_usage": True
                },  # the last piece carries the token counts
            ):
                parts.append(part)
                if part.choices and part.choices[0].delta.content:
                    yield part.choices[0].delta.content
        finally:
            self._record_stream(parts, prompt)

    def _record_stream(self, parts: list, prompt: str) -> None:
        """Count a streamed reply: LiteLLM joins the pieces into one response with its usage."""
        if not self.tracker or not parts:
            return
        import litellm

        try:
            whole = litellm.stream_chunk_builder(
                parts, messages=[{"role": "user", "content": prompt}]
            )
        except Exception:  # noqa: BLE001
            whole = None
        self.tracker.record(self.role, self.model, whole)

    def digit_score(self, prompt: str) -> float:
        """Ask for one digit (0-9) and return it as 0-1. With logprobs it is the average of the
        digits the model considered, weighted by probability; providers without logprobs
        (Anthropic) give the digit they wrote."""
        # Only OpenAI gives token probabilities: Gemini rejects the request and Anthropic ignores it.
        # Gemini returns nothing for a limit below about 8 tokens, so the others get a few spare.
        openai = self.model.startswith("openai/")
        asks = {"logprobs": True, "top_logprobs": 10} if openai else {}
        reply = self._complete(prompt, 1 if openai else 16, **asks)
        if self.tracker:
            self.tracker.record(self.role, self.model, reply)
        choice = reply.choices[0]
        weights: dict[int, float] = {}
        logprobs = getattr(choice, "logprobs", None)
        for top in logprobs.content[0].top_logprobs if logprobs else []:
            token = top.token.strip()
            if len(token) == 1 and token in "0123456789":  # not look-alikes such as '₀'
                weights[int(token)] = weights.get(int(token), 0.0) + math.exp(
                    top.logprob
                )
        if not weights:  # no logprobs: use the digit in the reply text
            text = (choice.message.content or "").strip()[:1]
            return int(text) / 9 if text in list("0123456789") else 0.0
        return sum(d * w for d, w in weights.items()) / sum(weights.values()) / 9


class LiteLLMReranker:
    """A cross-encoder behind a rerank API (e.g. cohere/rerank-v3.5): it reads the question and each
    chunk together and scores all of them in ONE call."""

    _lock = (
        threading.Lock()
    )  # shared by every reranker, so the pace holds across instances
    _last_call = 0.0

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        tracker=None,
        calls_per_minute: int = 0,
    ):
        require_prefix(model)
        self.model = model
        self.api_key = api_key
        self.tracker = tracker  # counts the calls (see usage.py)
        self.calls_per_minute = (
            calls_per_minute  # 0 = no limit; a Cohere trial key allows 10
        )

    def _wait_your_turn(self) -> None:
        """Space the calls out, so a rate-limited key (e.g. a trial key) is not rejected."""
        if not self.calls_per_minute:
            return
        gap = 60 / self.calls_per_minute
        with LiteLLMReranker._lock:
            pause = LiteLLMReranker._last_call + gap - time.monotonic()
            if pause > 0:
                time.sleep(pause)
            LiteLLMReranker._last_call = time.monotonic()

    def score(self, query: str, passages: list[str]) -> list[float]:
        """One relevance score (0 to 1) per passage, in the same order."""
        import litellm

        self._wait_your_turn()

        reply = litellm.rerank(
            model=self.model, query=query, documents=passages, api_key=self.api_key
        )
        if self.tracker:
            self.tracker.record("rerank", self.model, reply, "rerank")
        scores = [0.0] * len(passages)
        for result in reply.results:
            scores[result["index"]] = float(result["relevance_score"])
        return scores


def make_embeddings(config: dict, role: str = "embeddings") -> Embeddings:
    model = config["embedding_model"]
    return LiteLLMEmbeddings(
        model, api_key_for(config, model), config.get("usage_tracker"), role
    )


def make_llm(config: dict, role: str = "answer") -> LiteLLMChat:
    """The chat model in config["llm_model"]. role says what it is used for (answer, guard,
    judge, ...), so the usage breakdown can tell the jobs apart."""
    model = config["llm_model"]
    return LiteLLMChat(
        model,
        config["max_new_tokens"],
        api_key_for(config, model),
        config.get("usage_tracker"),
        role,
    )
