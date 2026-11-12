"""Retriever: search the vector store with dense (vectors) and BM25 (keywords) search, fuse them
with RRF, and rerank with an LLM (a chat model scores each chunk). The retry's wider search also
asks the LLM for other phrasings of the question (multi-query).
"""

import re
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from models import LiteLLMChat, make_llm
from vectorstore import VectorStore, load_config

MULTI_QUERY_PROMPT = """Write {n} different phrasings of the question below, \
to help search research papers. One per line, no numbering, nothing else.

Question: {query}"""


RERANK_PROMPT = """Rate how well the passage answers the question, from 0 (unrelated) \
to 9 (fully answers it). Reply with the single digit only.

Question: {query}

Passage:
{passage}"""


class Retriever:
    # The retrieval pipelines that can be asked for by name.
    METHODS = (
        "dense",
        "bm25",
        "hybrid",
        "hybrid_rerank",
        "multi_query_hybrid_rerank",
    )

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()

        # Open the store built by vectorstore.py (same embedding model as the config).
        vector_store = VectorStore(self.config)
        if vector_store.count() == 0:
            raise RuntimeError(
                "Vector store is empty: build it first (VectorStore.build)."
            )
        # Queries must be embedded with the same model as the stored chunks.
        built_with = vector_store.built_with()
        if built_with != self.config["embedding_model"]:
            raise RuntimeError(
                f"Store was built with '{built_with}' but the embedding model is "
                f"'{self.config['embedding_model']}'. Delete {self.config['vs_dir']} and build it again."
            )
        self.store = vector_store.store

        # BM25 keeps its own in-memory index over the same chunks.
        data = self.store.get(include=["documents", "metadatas"])
        self.chunks = [
            Document(id=id_, page_content=text, metadata=meta or {})
            for id_, text, meta in zip(
                data["ids"], data["documents"], data["metadatas"]
            )
        ]
        self.bm25_index = BM25Okapi(
            [self._tokenize(c.page_content) for c in self.chunks]
        )

    # The LLM and reranker are loaded only the first time a pipeline needs them.
    @cached_property
    def llm(self) -> LiteLLMChat:
        return make_llm(self.config)

    @cached_property
    def reranker(self) -> LiteLLMChat:
        """The model that scores chunks: it only ever writes a number, so replies are tiny."""
        return make_llm(
            {
                **self.config,
                "llm_model": self.config["reranker_model"],
                "max_new_tokens": 4,
            },
            role="rerank",
        )

    # 1. SEARCH ---- the two basic ways to find chunks
    def dense(self, query: str, k: int) -> list[Document]:
        """Vector search: finds chunks with similar meaning."""
        found = self.store.similarity_search_with_score(query, k=k)
        # The distance (lower = closer) goes on the chunk so the UI can show it.
        return [self.with_scores(doc, dense_distance=float(d)) for doc, d in found]

    def bm25(self, query: str, k: int) -> list[Document]:
        """Keyword search: finds chunks sharing the query's (rare) words."""
        scores = self.bm25_index.get_scores(self._tokenize(query))
        best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            self.with_scores(self.chunks[i], bm25_score=float(scores[i]))
            for i in best
            if scores[i] > 0
        ]

    @staticmethod
    def with_scores(doc: Document, **scores: float) -> Document:
        """A copy of the chunk with scores added to its metadata (the shared chunk is untouched)."""
        return Document(
            id=doc.id,
            page_content=doc.page_content,
            metadata={**doc.metadata, **scores},
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    # 2. COMBINE ---- merge several result lists, or re-order one
    def rrf(self, result_lists: list[list[Document]]) -> list[Document]:
        """Reciprocal Rank Fusion: a chunk ranked high in any list rises to the top."""
        scores: dict[str, float] = {}
        docs: dict[str, Document] = {}
        for results in result_lists:
            for rank, doc in enumerate(results, start=1):
                key = str(doc.id)
                scores[key] = scores.get(key, 0.0) + 1 / (self.config["rrf_k"] + rank)
                if key in docs:  # found by both searches: keep the scores from both
                    doc = self.with_scores(doc, **docs[key].metadata)
                docs[key] = doc
        ranked = sorted(docs, key=lambda key: scores[key], reverse=True)
        return [self.with_scores(docs[key], rrf_score=scores[key]) for key in ranked]

    def rerank(self, query: str, docs: list[Document], k: int) -> list[Document]:
        """Score every (question, chunk) pair from 0 to 1 and keep the best k. The scorer is an
        LLM asked for a number. Equal scores keep their incoming order (sorted() is stable), so
        ties fall back to the fusion ranking."""
        if not docs:
            return []
        # Each chunk is scored on its own, so several can be asked at once (map keeps the order).
        with ThreadPoolExecutor(self.config.get("rerank_workers", 1)) as pool:
            scores = list(
                pool.map(lambda d: self._relevance(query, d.page_content), docs)
            )
        ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
        # New Documents, so the score doesn't leak into the shared BM25 chunks.
        return [
            Document(
                id=doc.id,
                page_content=doc.page_content,
                metadata={**doc.metadata, "rerank_score": float(score)},
            )
            for doc, score in ranked[:k]
        ]

    def _relevance(self, query: str, passage: str) -> float:
        return self.reranker.digit_score(
            RERANK_PROMPT.format(query=query, passage=passage)
        )

    # 3. QUERY TRANSFORM ---- let the LLM improve the query before searching
    def _ask(self, prompt: str) -> str:
        return str(self.llm.invoke(prompt).content).strip()

    def multi_queries(self, query: str) -> list[str]:
        reply = self._ask(
            MULTI_QUERY_PROMPT.format(n=self.config["num_queries"], query=query)
        )
        # Drop any numbering or bullets the model adds anyway: "1. ", "- ", "* "
        lines = [
            re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line) for line in reply.splitlines()
        ]
        return [line for line in lines if line][: self.config["num_queries"]]

    # 4. PIPELINES ---- the named methods in METHODS; all take (query, k)
    def hybrid(self, query: str, k: int) -> list[Document]:
        n = self.config["fetch_k"]
        return self.rrf([self.dense(query, n), self.bm25(query, n)])[:k]

    def hybrid_rerank(self, query: str, k: int) -> list[Document]:
        return self.rerank(query, self.hybrid(query, self.config["fetch_k"]), k)

    # Rerankers always score against the user's original query.
    def multi_query_hybrid_rerank(self, query: str, k: int) -> list[Document]:
        queries = [query, *self.multi_queries(query)]
        n = self.config["fetch_k"]
        candidates = self.rrf([self.hybrid(q, n) for q in queries])[:n]
        return self.rerank(query, candidates, k)

    def retrieve_wider(self, query: str, k: int) -> list[Document]:
        """A second try after a bad answer: the question is also searched in other words
        (multi-query) and twice as many candidates go to the reranker."""
        normal = self.config
        self.config = {**normal, "fetch_k": normal["fetch_k"] * 2}
        try:
            return self.multi_query_hybrid_rerank(query, k)
        finally:
            self.config = normal

    # ENTRY POINT ---- retrieve(query, method="hybrid_rerank")
    def retrieve(
        self, query: str, method: str = "hybrid_rerank", k: int | None = None
    ) -> list[Document]:
        if method not in self.METHODS:
            raise ValueError(f"Unknown method '{method}'. Choose from: {self.METHODS}")
        return getattr(self, method)(query, k or self.config["top_k"])
