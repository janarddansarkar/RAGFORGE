"""Ollama model wrappers with a fake client: what is sent, and the errors people will meet."""

import math
from types import SimpleNamespace

import httpx
import ollama
import pytest

from models import OllamaChat, OllamaEmbeddings
from retriever import Retriever


class FakeClient:
    def __init__(self, error=None):
        self.error = error
        self.calls: list[tuple[str, dict]] = []

    def _record(self, name, kwargs):
        self.calls.append((name, kwargs))
        if self.error:
            raise self.error

    def chat(self, **kwargs):
        self._record("chat", kwargs)
        return SimpleNamespace(message=SimpleNamespace(content="  hi there "))

    def embed(self, **kwargs):
        self._record("embed", kwargs)
        return SimpleNamespace(embeddings=[[1.0, 0.0]] * len(kwargs["input"]))


def with_client(model, client):
    model.client = client  # replaces the cached client
    return model


def test_chat_sends_the_model_options_and_strips_the_reply():
    client = FakeClient()
    chat = with_client(OllamaChat("qwen2.5:3b", 77, num_ctx=4096), client)
    assert chat.invoke("hello").content == "hi there"
    _, sent = client.calls[0]
    assert sent["model"] == "qwen2.5:3b"
    assert sent["options"] == {"temperature": 0, "num_predict": 77, "num_ctx": 4096}
    assert sent["think"] is False and sent["format"] is None


def test_chat_turns_a_pydantic_schema_into_json_schema():
    from pydantic import BaseModel

    class Verdict(BaseModel):
        ok: bool

    client = FakeClient()
    with_client(OllamaChat("m"), client).invoke("x", schema=Verdict)
    assert client.calls[0][1]["format"]["properties"]["ok"]["type"] == "boolean"


def test_embeddings_are_sent_in_batches_of_32():
    client = FakeClient()
    vectors = with_client(OllamaEmbeddings("bge-m3"), client).embed_documents(
        [f"t{i}" for i in range(70)]
    )
    assert len(vectors) == 70
    assert [len(kwargs["input"]) for _, kwargs in client.calls] == [32, 32, 6]


def test_a_missing_model_says_how_to_pull_it():
    error = ollama.ResponseError("not found", 404)
    with pytest.raises(RuntimeError, match="ollama pull nope"):
        with_client(OllamaChat("nope"), FakeClient(error)).invoke("x")


def test_an_unreachable_server_says_how_to_start_it():
    with pytest.raises(RuntimeError, match="ollama serve"):
        with_client(OllamaChat("m"), FakeClient(httpx.ConnectError("refused"))).invoke(
            "x"
        )


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


def digit_chat(top_logprobs):
    """An OllamaChat whose model answers with these (token, probability) guesses."""
    entry = SimpleNamespace(
        top_logprobs=[
            SimpleNamespace(token=t, logprob=math.log(p)) for t, p in top_logprobs
        ]
    )
    client = SimpleNamespace(chat=lambda **kwargs: SimpleNamespace(logprobs=[entry]))
    return with_client(OllamaChat("m"), client)


def test_digit_score_is_the_probability_weighted_digit():
    assert digit_chat([("9", 1.0)]).digit_score("x") == 1.0
    assert digit_chat([("0", 1.0)]).digit_score("x") == 0.0
    # 50/50 between 0 and 9 -> halfway
    assert digit_chat([("0", 0.5), ("9", 0.5)]).digit_score("x") == pytest.approx(0.5)


def test_digit_score_ignores_non_digit_tokens():
    assert digit_chat([("The", 0.6), (" 9", 0.4)]).digit_score("x") == 1.0
    assert digit_chat([("The", 1.0)]).digit_score("x") == 0.0
    assert digit_chat([("₀", 0.5), ("４", 0.3), ("9", 0.2)]).digit_score("x") == 1.0


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


def test_stream_yields_the_pieces_of_the_reply():
    from types import SimpleNamespace

    from models import OllamaChat

    class FakeClient:
        def chat(self, **kwargs):
            assert kwargs["stream"] is True
            for text in ("Hel", "", "lo"):
                yield SimpleNamespace(message=SimpleNamespace(content=text))

    chat = OllamaChat("any")
    chat.__dict__["client"] = FakeClient()  # cached_property: put the fake in place
    assert list(chat.stream("hi")) == ["Hel", "lo"]
