"""Model wrappers with fake LiteLLM / sentence-transformers: what is sent, and the retriever's scoring."""

import math
from types import SimpleNamespace

import pytest

from models import LiteLLMChat
from retriever import Retriever


def fake_reply(text="hi", logprobs=None):
    message = SimpleNamespace(content=text)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, logprobs=logprobs)]
    )


def test_chat_needs_a_provider_prefix():
    with pytest.raises(ValueError, match="provider prefix"):
        LiteLLMChat("qwen3:14b")


def test_chat_turns_a_pydantic_schema_into_a_response_format(monkeypatch):
    import litellm
    from pydantic import BaseModel

    class Verdict(BaseModel):
        ok: bool

    sent = {}
    monkeypatch.setattr(
        litellm, "completion", lambda **kw: sent.update(kw) or fake_reply()
    )
    LiteLLMChat("anthropic/m").invoke("x", schema=Verdict)
    assert sent["response_format"] is Verdict
    LiteLLMChat("anthropic/m").invoke("x", schema={"type": "object"})
    assert sent["response_format"]["json_schema"]["schema"] == {"type": "object"}


def test_embeddings_need_a_provider_prefix():
    from models import make_embeddings

    with pytest.raises(ValueError, match="provider prefix"):
        make_embeddings({"embedding_model": "all-minilm"})


def test_llm_rerank_scores_sort_chunks_and_ties_keep_their_order():
    from langchain_core.documents import Document

    scores = {"a": "3", "b": "9", "c": "9", "d": "no number"}
    retriever = Retriever.__new__(Retriever)
    retriever.config = {}  # no rerank_workers: one chunk at a time
    retriever._relevance = lambda query, passage: (
        0.0 if scores[passage] == "no number" else float(scores[passage]) / 10
    )
    docs = [Document(id=k, page_content=k) for k in "abcd"]
    ranked = retriever.rerank("q", docs, k=3)
    assert [d.id for d in ranked] == ["b", "c", "a"]  # b before c: stable on a tie
    assert ranked[0].metadata["rerank_score"] == 0.9


def test_relevance_asks_the_reranker_for_a_digit_score():
    retriever = Retriever.__new__(Retriever)
    asked = []
    retriever.__dict__["reranker"] = SimpleNamespace(
        digit_score=lambda prompt: asked.append(prompt) or 0.5
    )
    assert retriever._relevance("my question", "my passage") == 0.5
    assert "my question" in asked[0] and "my passage" in asked[0]


def digit_chat(top_logprobs, monkeypatch):
    """A LiteLLMChat whose model answers with these (token, probability) guesses."""
    import litellm

    entry = SimpleNamespace(
        top_logprobs=[
            SimpleNamespace(token=t, logprob=math.log(p)) for t, p in top_logprobs
        ]
    )
    reply = fake_reply("", SimpleNamespace(content=[entry]))
    monkeypatch.setattr(litellm, "completion", lambda **kwargs: reply)
    return LiteLLMChat("anthropic/m")


def test_digit_score_is_the_probability_weighted_digit(monkeypatch):
    assert digit_chat([("9", 1.0)], monkeypatch).digit_score("x") == 1.0
    assert digit_chat([("0", 1.0)], monkeypatch).digit_score("x") == 0.0
    # 50/50 between 0 and 9 -> halfway
    assert digit_chat([("0", 0.5), ("9", 0.5)], monkeypatch).digit_score(
        "x"
    ) == pytest.approx(0.5)


def test_digit_score_ignores_non_digit_tokens(monkeypatch):
    assert digit_chat([("The", 0.6), (" 9", 0.4)], monkeypatch).digit_score("x") == 1.0
    assert digit_chat([("The", 1.0)], monkeypatch).digit_score("x") == 0.0
    assert (
        digit_chat([("₀", 0.5), ("４", 0.3), ("9", 0.2)], monkeypatch).digit_score("x")
        == 1.0
    )


def test_rrf_keeps_the_scores_of_every_search_that_found_a_chunk():
    from langchain_core.documents import Document

    retriever = Retriever.__new__(Retriever)
    retriever.config = {"rrf_k": 60}
    both = Document(
        id="1", page_content="a", metadata={"page": 1, "dense_distance": 0.4}
    )
    only_bm25 = Document(
        id="2", page_content="b", metadata={"page": 2, "bm25_score": 3.0}
    )
    found_by_bm25 = Document(
        id="1", page_content="a", metadata={"page": 1, "bm25_score": 5.0}
    )
    ranked = retriever.rrf([[both], [found_by_bm25, only_bm25]])
    assert ranked[0].id == "1"
    assert ranked[0].metadata["dense_distance"] == 0.4
    assert ranked[0].metadata["bm25_score"] == 5.0
    assert ranked[0].metadata["rrf_score"] > ranked[1].metadata["rrf_score"]
    assert "rrf_score" not in both.metadata  # the original chunk is untouched


def test_parallel_rerank_gives_the_same_ranking_as_one_by_one():
    from langchain_core.documents import Document

    retriever = Retriever.__new__(Retriever)
    retriever.config = {"rerank_workers": 4}
    retriever._relevance = lambda query, passage: len(passage) / 100
    docs = [Document(id=str(n), page_content="x" * (n * 7 % 13 + 1)) for n in range(12)]
    ranked = retriever.rerank("q", docs, 5)
    expected = sorted(docs, key=lambda d: len(d.page_content), reverse=True)[:5]
    assert [d.id for d in ranked] == [d.id for d in expected]
    assert ranked[0].metadata["rerank_score"] == len(expected[0].page_content) / 100


def test_stream_yields_the_pieces_of_the_reply(monkeypatch):
    import litellm

    def piece(text):
        return SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=text))]
        )

    def fake_completion(**kwargs):
        assert kwargs["stream"] is True
        return [piece("Hel"), piece(""), piece("lo")]

    monkeypatch.setattr(litellm, "completion", fake_completion)
    assert list(LiteLLMChat("anthropic/m").stream("hi")) == ["Hel", "lo"]


# ---- API models through LiteLLM ----


def test_a_provider_prefix_means_an_api_model():
    from models import is_api_model

    assert is_api_model("anthropic/claude-haiku-5-5")
    assert is_api_model("cohere/rerank-v3.5")
    assert not is_api_model("qwen3:14b")  # no provider prefix
    assert not is_api_model("")


def test_make_llm_and_make_embeddings_build_the_api_classes():
    from models import LiteLLMChat, LiteLLMEmbeddings, make_embeddings, make_llm

    config = {
        "llm_model": "anthropic/x",
        "embedding_model": "openai/y",
        "max_new_tokens": 9,
    }
    assert isinstance(make_llm(config), LiteLLMChat)
    assert isinstance(make_embeddings(config), LiteLLMEmbeddings)


def test_litellm_chat_sends_the_prompt_and_returns_the_reply(monkeypatch):
    import litellm

    from models import LiteLLMChat

    sent = {}

    def fake_completion(**kwargs):
        sent.update(kwargs)
        message = SimpleNamespace(content=" hello ")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, logprobs=None)]
        )

    monkeypatch.setattr(litellm, "completion", fake_completion)
    chat = LiteLLMChat("mistral/x", max_new_tokens=50)
    assert chat.invoke("hi").content == "hello"
    assert sent["model"] == "mistral/x" and sent["max_tokens"] == 50
    assert chat.digit_score("hi") == 0.0  # "hello" is no digit


def test_litellm_digit_score_falls_back_to_the_written_digit(monkeypatch):
    import litellm

    from models import LiteLLMChat

    message = SimpleNamespace(content="9")
    reply = SimpleNamespace(choices=[SimpleNamespace(message=message, logprobs=None)])
    monkeypatch.setattr(litellm, "completion", lambda **kwargs: reply)
    assert LiteLLMChat("anthropic/x").digit_score("rate it") == 1.0


def test_litellm_embeddings(monkeypatch):
    import litellm

    from models import LiteLLMEmbeddings

    monkeypatch.setattr(
        litellm,
        "embedding",
        lambda model, input, api_key=None: SimpleNamespace(
            data=[{"embedding": [1.0, 2.0]}] * len(input)
        ),
    )
    assert LiteLLMEmbeddings("openai/y").embed_documents(["a", "b"]) == [[1.0, 2.0]] * 2


def test_gemini_calls_disable_thinking_so_small_token_limits_still_give_a_reply(
    monkeypatch,
):
    import litellm

    sent = {}
    monkeypatch.setattr(
        litellm, "completion", lambda **kw: sent.update(kw) or fake_reply()
    )
    LiteLLMChat("gemini/gemini-3.5-flash").invoke("x")
    assert sent["reasoning_effort"] == "disable"
    sent.clear()
    LiteLLMChat("openai/gpt-4o").invoke("x")
    assert "reasoning_effort" not in sent


# ---- the cross-encoder reranker (Cohere through LiteLLM) ----


def retriever_with(config: dict, scorer) -> Retriever:
    from langchain_core.documents import Document

    retriever = Retriever.__new__(Retriever)
    retriever.config = config
    retriever.__dict__["cross_encoder"] = scorer
    retriever.docs = [Document(id=k, page_content=k) for k in "abc"]
    return retriever


def test_cross_encoder_rerank_scores_all_chunks_in_one_call_and_sorts():
    calls = []
    scorer = SimpleNamespace(
        score=lambda query, passages: calls.append((query, passages)) or [0.2, 0.9, 0.5]
    )
    retriever = retriever_with({"reranker_type": "cross_encoder"}, scorer)
    ranked = retriever.rerank("q", retriever.docs, k=2)
    assert [d.id for d in ranked] == ["b", "c"]
    assert ranked[0].metadata["rerank_score"] == 0.9
    assert calls == [("q", ["a", "b", "c"])]  # one call, not one per chunk


def test_a_failed_cross_encoder_call_keeps_the_fused_order(caplog):
    def broken(query, passages):
        raise RuntimeError("rerank api down")

    retriever = retriever_with(
        {"reranker_type": "cross_encoder"}, SimpleNamespace(score=broken)
    )
    ranked = retriever.rerank("q", retriever.docs, k=3)
    assert [d.id for d in ranked] == ["a", "b", "c"]
    assert "keeping the fused search order" in caplog.text


def test_litellm_reranker_maps_scores_back_to_the_chunk_order_and_counts_the_call(
    monkeypatch,
):
    import litellm

    from models import LiteLLMReranker
    from usage import UsageTracker

    sent = {}
    monkeypatch.setattr(
        litellm,
        "rerank",
        lambda **kw: (
            sent.update(kw)
            or SimpleNamespace(
                results=[
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.1},
                ]
            )
        ),
    )
    monkeypatch.setattr(litellm, "completion_cost", lambda **kw: 0.002)
    tracker = UsageTracker()
    scores = LiteLLMReranker("cohere/rerank-v3.5", "key-c", tracker).score(
        "q", ["a", "b"]
    )
    assert scores == [0.1, 0.9]
    assert sent["model"] == "cohere/rerank-v3.5" and sent["api_key"] == "key-c"
    assert [(c.role, c.cost) for c in tracker.calls] == [("rerank", 0.002)]


def test_the_reranker_spaces_its_calls_for_a_rate_limited_key(monkeypatch):
    import litellm

    import models
    from models import LiteLLMReranker

    clock = {"now": 100.0, "slept": []}
    monkeypatch.setattr(models.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        models.time,
        "sleep",
        lambda s: (clock["slept"].append(s), clock.update(now=clock["now"] + s)),
    )
    monkeypatch.setattr(
        litellm,
        "rerank",
        lambda **kw: SimpleNamespace(results=[{"index": 0, "relevance_score": 1.0}]),
    )
    LiteLLMReranker._last_call = 0.0
    fast = LiteLLMReranker(
        "cohere/rerank-v3.5", calls_per_minute=10
    )  # one call every 6 s
    fast.score("q", ["a"])
    fast.score("q", ["a"])
    fast.score("q", ["a"])
    assert clock["slept"] == [6.0, 6.0]  # the first call goes straight through
    clock["slept"].clear()
    LiteLLMReranker("cohere/rerank-v3.5").score("q", ["a"])  # no limit: no waiting
    assert clock["slept"] == []


# ---- request timeout ----


def test_every_model_request_gets_a_timeout_and_retries_from_the_config(monkeypatch):
    import litellm

    from models import DEFAULT_RETRIES, DEFAULT_TIMEOUT, make_llm, use_timeouts

    monkeypatch.setattr(litellm, "request_timeout", 6000, raising=False)
    monkeypatch.setattr(litellm, "num_retries", 0, raising=False)
    use_timeouts({})  # no setting: the defaults, never "wait for ever"
    assert (litellm.request_timeout, litellm.num_retries) == (
        DEFAULT_TIMEOUT,
        DEFAULT_RETRIES,
    )
    assert DEFAULT_TIMEOUT <= 120
    make_llm(
        {
            "llm_model": "openai/m",
            "max_new_tokens": 5,
            "request_timeout": 15,
            "request_retries": 1,
        }
    )
    assert (litellm.request_timeout, litellm.num_retries) == (15, 1)  # config.toml wins


def test_claude_gets_minimal_thinking_and_room_in_the_token_limit(monkeypatch):
    import litellm

    from models import CLAUDE_THINKING_ROOM, LiteLLMChat

    sent = {}
    monkeypatch.setattr(
        litellm, "completion", lambda **kw: sent.update(kw) or fake_reply()
    )
    LiteLLMChat("anthropic/claude-sonnet-5-5", max_new_tokens=256).invoke("x")
    # Claude's thinking counts against max_tokens: without room, a 256-token limit gave empty replies
    assert sent["reasoning_effort"] == "minimal"
    assert sent["max_tokens"] == 256 + CLAUDE_THINKING_ROOM
    sent.clear()
    LiteLLMChat("openai/gpt-4o", max_new_tokens=256).invoke("x")
    assert (
        sent["max_tokens"] == 256 and "reasoning_effort" not in sent
    )  # other providers unchanged
