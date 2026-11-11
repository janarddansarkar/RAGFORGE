"""The hosted providers (OpenAI, Anthropic, Gemini) and the jobs a user can give each of them.

Every component that needs a model has its own choice of provider and model, so the answer
can come from Claude while the embeddings come from OpenAI. A provider only offers the jobs it
has a model for: Anthropic has no embeddings or speech-to-text.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Provider:
    name: str  # shown in the UI
    prefix: str  # the LiteLLM provider prefix
    chat: tuple[str, ...]  # chat models, best first
    embeddings: tuple[str, ...] = ()
    speech: tuple[str, ...] = ()

    def models(self, kind: str) -> tuple[str, ...]:
        return getattr(self, kind)


PROVIDERS = {
    p.name: p
    for p in (
        Provider(
            "OpenAI",
            "openai",
            chat=("gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"),
            embeddings=("text-embedding-3-small", "text-embedding-3-large"),
            speech=("whisper-1", "gpt-4o-transcribe"),
        ),
        Provider(
            "Anthropic",
            "anthropic",
            chat=("claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5"),
        ),
        Provider(
            "Google Gemini",
            "gemini",
            chat=(
                "gemini-2.0-flash",
                "gemini-2.0-flash-lite",
                "gemini-2.5-flash",
                "gemini-2.5-pro",
            ),
            embeddings=("text-embedding-004", "gemini-embedding-001"),
        ),
    )
}


@dataclass(frozen=True)
class Component:
    key: str  # the name of its setting in the choices dict
    label: str
    help: str
    kind: str  # "chat" | "embeddings" | "speech": which models of a provider it can use
    provider: str  # the default provider
    model: str  # the default model


COMPONENTS = (
    Component("answer", "Answers", "writes answers, retries and the chunk notes",
              "chat", "OpenAI", "gpt-4o"),
    Component("guard", "Input guard", "checks every question for prompt injection",
              "chat", "OpenAI", "gpt-4o-mini"),
    Component("judge", "Judge", "scores every answer (RAG triad)",
              "chat", "OpenAI", "gpt-4o-mini"),
    Component("reranker", "Reranker", "scores each retrieved chunk 0-10",
              "chat", "OpenAI", "gpt-4o-mini"),
    Component("embedding", "Embeddings", "turns text into vectors for the search",
              "embeddings", "OpenAI", "text-embedding-3-small"),
    Component("speech", "Voice input", "turns a spoken question into text (optional)",
              "speech", "OpenAI", "whisper-1"),
)  # fmt: skip

OFF = "Off"  # the choice for an optional component, e.g. voice input


def providers_for(kind: str) -> list[str]:
    """The providers that have a model for this kind of job."""
    return [name for name, p in PROVIDERS.items() if p.models(kind)]


def full_name(provider: str, model: str) -> str:
    return f"{PROVIDERS[provider].prefix}/{model}"


def needed_keys(choices: dict[str, tuple[str, str]]) -> list[str]:
    """The providers the user must give a key for: the ones their choices use."""
    names = [provider for provider, _ in choices.values() if provider != OFF]
    return list(dict.fromkeys(names))  # no repeats, in order


def settings_for(choices: dict[str, tuple[str, str]], keys: dict[str, str]) -> dict:
    """The config.toml overrides for these choices ({component key: (provider, model)}).
    keys maps a provider name to the user's API key; it is held in memory only."""
    chosen = {key: full_name(*pair) for key, pair in choices.items() if pair[0] != OFF}
    return {
        "llm_model": chosen["answer"],
        "retry_model": "",  # a second try uses the answer model
        "guard_model": chosen["guard"],
        "judge_model": chosen["judge"],
        "reranker_type": "llm",  # a chat model scores each chunk 0-10
        "reranker_model": chosen["reranker"],
        "embedding_model": chosen["embedding"],
        "stt_model": chosen.get("speech", ""),  # "" = no voice input
        "api_keys": {PROVIDERS[name].prefix: key for name, key in keys.items()},
    }
