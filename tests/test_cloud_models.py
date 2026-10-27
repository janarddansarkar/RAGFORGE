"""Cloud models with a fake OpenAI-style client: what is sent, the key handling, the errors."""

import math
from types import SimpleNamespace

import httpx
import openai
import pytest

from models import (
    CloudChat,
    CloudEmbeddings,
    OllamaChat,
    OllamaEmbeddings,
    make_embeddings,
    make_llm,
    use_providers,
)

OPENAI = {
    "base_url": "https://x/v1",
    "token_param": "max_completion_tokens",
    "json_mode": True,
    "logprobs": True,
    "embedding_model": "emb",
    "llm_model": "chat",
    "reranker_model": "rerank",
    "guard_model": "guard",
    "judge_model": "judge",
}
ANTHROPIC = {
    **{k: v for k, v in OPENAI.items() if k != "embedding_model"},
    "token_param": "max_tokens",
    "json_mode": False,
    "logprobs": False,
}
CONFIG = {
    "providers": {"openai": OPENAI, "anthropic": ANTHROPIC},
    "embedding_model": "local-emb",
    "llm_model": "local-chat",
    "max_new_tokens": 256,
    "ollama_num_ctx": 4096,
}


class FakeCompletions:
    def __init__(self, reply="  hi ", error=None, logprobs=None):
        self.calls, self.reply, self.error, self.logprobs = [], reply, error, logprobs

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        if kwargs.get("stream"):
            pieces = [
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=p))]
                )
                for p in ("a", "b")
            ]
            return [SimpleNamespace(choices=[]), *pieces]
        message = SimpleNamespace(content=self.reply)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, logprobs=self.logprobs)]
        )


def chat(settings=OPENAI, **kwargs):
    model = CloudChat("p", "chat", settings, "sk-1", 77)
    completions = FakeCompletions(**kwargs)
    model.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return model, completions


def test_chat_sends_model_limit_and_strips_reply():
    model, completions = chat()
    assert model.invoke("hello").content == "hi"
    sent = completions.calls[0]
    assert sent["model"] == "chat" and sent["temperature"] == 0
    assert sent["max_completion_tokens"] == 77 and "max_tokens" not in sent


def test_each_provider_names_the_limit_its_own_way():
    model, completions = chat(ANTHROPIC)
    model.invoke("x", max_new_tokens=5)
    assert completions.calls[0]["max_tokens"] == 5


def test_schema_goes_in_the_prompt_and_json_mode_only_where_supported():
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    model, completions = chat()
    model.invoke("x", schema=schema)
    assert "JSON schema" in completions.calls[0]["messages"][0]["content"]
    assert completions.calls[0]["response_format"] == {"type": "json_object"}
    model, completions = chat(ANTHROPIC)
    model.invoke("x", schema=schema)
    assert "response_format" not in completions.calls[0]
    assert "JSON schema" in completions.calls[0]["messages"][0]["content"]


def test_stream_yields_pieces_and_skips_empty_chunks():
    model, _ = chat()
    assert list(model.stream("x")) == ["a", "b"]


def test_digit_score_uses_probabilities_when_available():
    top = [
        SimpleNamespace(token="0", logprob=math.log(0.5)),
        SimpleNamespace(token="9", logprob=math.log(0.5)),
    ]
    logprobs = SimpleNamespace(content=[SimpleNamespace(top_logprobs=top)])
    assert chat(logprobs=logprobs)[0].digit_score("x") == pytest.approx(0.5)


def test_digit_score_reads_the_digit_without_probabilities():
    assert chat(ANTHROPIC, reply="Score: 9")[0].digit_score("x") == 1.0
    assert chat(ANTHROPIC, reply="no idea")[0].digit_score("x") == 0.0


def test_embeddings_are_sent_in_batches_of_32():
    calls = []

    def create(model, input):
        calls.append(len(input))
        return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0])] * len(input))

    embeddings = CloudEmbeddings("p", "emb", OPENAI, "sk-1")
    embeddings.client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    assert len(embeddings.embed_documents(["t"] * 70)) == 70
    assert calls == [32, 32, 6]


def response(status):
    return httpx.Response(status, request=httpx.Request("POST", "https://x"))


def test_errors_are_plain_advice():
    wrong_key = openai.AuthenticationError("no", response=response(401), body=None)
    with pytest.raises(RuntimeError, match="rejected the API key"):
        chat(error=wrong_key)[0].invoke("x")
    limit = openai.RateLimitError("slow down", response=response(429), body=None)
    with pytest.raises(RuntimeError, match="rate limit"):
        chat(error=limit)[0].invoke("x")
    offline = openai.APIConnectionError(request=httpx.Request("POST", "https://x"))
    with pytest.raises(RuntimeError, match="Cannot reach"):
        chat(error=offline)[0].invoke("x")


def test_no_key_is_refused_before_any_call():
    with pytest.raises(RuntimeError, match="API key"):
        CloudChat("openai", "chat", OPENAI, "")


def test_use_providers_swaps_models_and_keeps_keys_in_the_copy_only():
    result = use_providers(CONFIG, "anthropic", "ak", "openai", "ok")
    assert result["llm_model"] == "chat" and result["embedding_model"] == "emb"
    assert result["chat_api_key"] == "ak" and result["embedding_api_key"] == "ok"
    assert "chat_api_key" not in CONFIG
    with pytest.raises(ValueError, match="no embedding model"):
        use_providers(CONFIG, "openai", "k", "anthropic", "k")


def test_factories_pick_the_provider():
    assert isinstance(make_llm(CONFIG), OllamaChat)
    assert isinstance(make_embeddings(CONFIG), OllamaEmbeddings)
    cloud = use_providers(CONFIG, "openai", "k", "openai", "k")
    assert isinstance(make_llm(cloud), CloudChat)
    assert isinstance(make_embeddings(cloud), CloudEmbeddings)
