"""Retriever scoring: LLM and cross-encoder reranking, parallel reranking and RRF fusion."""

from types import SimpleNamespace

from retriever import Retriever


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


def test_cross_encoder_rerank_scores_all_pairs_at_once_and_sorts():
    from langchain_core.documents import Document

    retriever = Retriever.__new__(Retriever)
    retriever.config = {"reranker_type": "cross_encoder"}
    seen = []
    retriever.__dict__["cross_encoder"] = SimpleNamespace(
        score=lambda pairs: seen.append(pairs) or [0.2, 0.9, 0.5]
    )
    docs = [Document(id=k, page_content=k) for k in "abc"]
    ranked = retriever.rerank("q", docs, k=2)
    assert [d.id for d in ranked] == ["b", "c"]
    assert ranked[0].metadata["rerank_score"] == 0.9
    assert seen == [
        [("q", "a"), ("q", "b"), ("q", "c")]
    ]  # one batch, not one call each
