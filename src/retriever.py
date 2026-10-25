"""Retriever: search the vector store (./vs) with several techniques.

Techniques: dense (vectors), BM25 (keywords), RRF fusion, cross-encoder reranking,
and LLM query transforms (rewrite, multi-query, HyDE) - on their own and combined.

Build the store first:  uv run python src/vectorstore.py
"""

import re
import sys
import time
from functools import cached_property

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from models import HuggingFaceChat, make_llm
from vectorstore import VectorStore, load_config

REWRITE_PROMPT = """Rewrite the question below into a clear, specific search query \
for finding passages in research papers. Reply with the query only.

Question: {query}"""

MULTI_QUERY_PROMPT = """Write {n} different phrasings of the question below, \
to help search research papers. One per line, no numbering, nothing else.

Question: {query}"""

HYDE_PROMPT = """Write a short passage (3-4 sentences) from a research paper \
that answers the question below. Reply with the passage only.

Question: {query}"""


class Retriever:
    # Every retrieval pipeline, by name. The eval runs all of them.
    METHODS = (
        # single techniques
        "dense",
        "bm25",
        "rewrite",
        "multi_query",
        "hyde",
        # combinations
        "hybrid",
        "dense_rerank",
        "hybrid_rerank",
        "rewrite_hybrid_rerank",
        "multi_query_hybrid_rerank",
        "hyde_hybrid_rerank",
    )

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()

        # Open the store built by vectorstore.py (same embedding model from config.toml).
        vector_store = VectorStore(self.config)
        if vector_store.count() == 0:
            raise RuntimeError(
                "Vector store is empty. Build it first: "
                "uv run python src/vectorstore.py"
            )
        # Queries must be embedded with the same model as the stored chunks.
        built_with = vector_store.built_with()
        if built_with != self.config["embedding_model"]:
            raise RuntimeError(
                f"Store was built with '{built_with}' but config.toml says "
                f"'{self.config['embedding_model']}'. Rebuild it: "
                "uv run python src/vectorstore.py --rebuild"
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
    def llm(self) -> HuggingFaceChat:
        return make_llm(self.config)

    @cached_property
    def reranker(self) -> CrossEncoder:
        return CrossEncoder(self.config["reranker_model"])

    # 1. SEARCH ---- the two basic ways to find chunks
    def dense(self, query: str, k: int) -> list[Document]:
        """Vector search: finds chunks with similar meaning."""
        return self.store.similarity_search(query, k=k)

    def bm25(self, query: str, k: int) -> list[Document]:
        """Keyword search: finds chunks sharing the query's (rare) words."""
        scores = self.bm25_index.get_scores(self._tokenize(query))
        best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.chunks[i] for i in best if scores[i] > 0]

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
                docs[key] = doc
        return sorted(docs.values(), key=lambda d: scores[str(d.id)], reverse=True)

    def rerank(self, query: str, docs: list[Document], k: int) -> list[Document]:
        """Cross-encoder reads (query, chunk) together and scores the match."""
        if not docs:
            return []
        scores = self.reranker.predict([(query, d.page_content) for d in docs])
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

    # 3. QUERY TRANSFORM ---- let the LLM improve the query before searching
    def _ask(self, prompt: str) -> str:
        return str(self.llm.invoke(prompt).content).strip()

    def rewrite_query(self, query: str) -> str:
        return self._ask(REWRITE_PROMPT.format(query=query))

    def multi_queries(self, query: str) -> list[str]:
        reply = self._ask(
            MULTI_QUERY_PROMPT.format(n=self.config["num_queries"], query=query)
        )
        # Drop any numbering or bullets the model adds anyway: "1. ", "- ", "* "
        lines = [
            re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line) for line in reply.splitlines()
        ]
        return [line for line in lines if line][: self.config["num_queries"]]

    def hyde_passage(self, query: str) -> str:
        """HyDE: a made-up answer is often closer to the real passage than the question."""
        return self._ask(HYDE_PROMPT.format(query=query))

    # 4. PIPELINES ---- the named methods in METHODS; all take (query, k)
    def hybrid(self, query: str, k: int) -> list[Document]:
        n = self.config["fetch_k"]
        return self.rrf([self.dense(query, n), self.bm25(query, n)])[:k]

    def dense_rerank(self, query: str, k: int) -> list[Document]:
        return self.rerank(query, self.dense(query, self.config["fetch_k"]), k)

    def hybrid_rerank(self, query: str, k: int) -> list[Document]:
        return self.rerank(query, self.hybrid(query, self.config["fetch_k"]), k)

    def rewrite(self, query: str, k: int) -> list[Document]:
        return self.dense(self.rewrite_query(query), k)

    def multi_query(self, query: str, k: int) -> list[Document]:
        queries = [query, *self.multi_queries(query)]
        n = self.config["fetch_k"]
        return self.rrf([self.dense(q, n) for q in queries])[:k]

    def hyde(self, query: str, k: int) -> list[Document]:
        return self.dense(self.hyde_passage(query), k)

    # Rerankers always score against the user's original query.
    def rewrite_hybrid_rerank(self, query: str, k: int) -> list[Document]:
        candidates = self.hybrid(self.rewrite_query(query), self.config["fetch_k"])
        return self.rerank(query, candidates, k)

    def multi_query_hybrid_rerank(self, query: str, k: int) -> list[Document]:
        queries = [query, *self.multi_queries(query)]
        n = self.config["fetch_k"]
        candidates = self.rrf([self.hybrid(q, n) for q in queries])[:n]
        return self.rerank(query, candidates, k)

    def hyde_hybrid_rerank(self, query: str, k: int) -> list[Document]:
        n = self.config["fetch_k"]
        # HyDE helps the vector search; BM25 keeps the original words.
        candidates = self.rrf(
            [self.dense(self.hyde_passage(query), n), self.bm25(query, n)]
        )
        return self.rerank(query, candidates[:n], k)

    # ENTRY POINT ---- retrieve(query, method="hybrid_rerank")
    def retrieve(
        self, query: str, method: str = "hybrid_rerank", k: int | None = None
    ) -> list[Document]:
        if method not in self.METHODS:
            raise ValueError(f"Unknown method '{method}'. Choose from: {self.METHODS}")
        return getattr(self, method)(query, k or self.config["top_k"])


# TRY IT ---- python src/retriever.py "your question"
if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What rank r does LoRA use?"
    retriever = Retriever()
    print(f"Q: {question}\n")

    for method in Retriever.METHODS:
        start = time.perf_counter()
        docs = retriever.retrieve(question, method=method, k=3)
        ms = (time.perf_counter() - start) * 1000
        hits = ", ".join(
            f"{d.metadata['source'].removesuffix('.pdf')} p.{d.metadata['page']}"
            for d in docs
        )
        print(f"{method:28} {ms:8.0f} ms   {hits}")
