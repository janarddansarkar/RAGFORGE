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
    chat = LiteLLMChat("anthropic/x", max_new_tokens=50)
    assert chat.invoke("hi").content == "hello"
    assert sent["model"] == "anthropic/x" and sent["max_tokens"] == 50
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
