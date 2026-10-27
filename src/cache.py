"""Session cache: a question that is similar to one already answered gets the saved answer.

No answer-model call and no search: the question is embedded (a small, quick call to the
embedding model), compared with the earlier questions, and a close enough match returns
the saved answer, chunks and (if there was one) the RAG-triad scores.

Each SessionCache is private to one set of uploaded documents: it writes its own file in
its own temporary folder, so no two caches (other users, or the same user's earlier
documents) ever share or wipe each other. The app calls destroy() when the documents are
replaced; the file is also removed when the program exits.

Try it:  uv run python src/cache.py     (a small demo with no models)
"""

import atexit
import json
import math
import shutil
import tempfile
from pathlib import Path

from langchain_core.documents import Document

from models import make_embeddings
from vectorstore import load_config


class SessionCache:
    def __init__(self, config: dict | None = None, file: Path | None = None):
        self.config = config or load_config()
        self.threshold = self.config["cache_threshold"]
        self.embeddings = make_embeddings(self.config)
        self.file = (
            file or Path(tempfile.mkdtemp(prefix="ragforge_cache_")) / "cache.json"
        )
        self.entries: list[dict] = []
        atexit.register(self.destroy)

    @staticmethod
    def similarity(a: list[float], b: list[float]) -> float:
        """Cosine similarity: 1 = same direction (same meaning), 0 = unrelated."""
        dot = sum(x * y for x, y in zip(a, b))
        return dot / (math.hypot(*a) * math.hypot(*b) or 1)

    def find(self, question: str) -> dict | None:
        """The saved entry for the most similar earlier question, if it is close enough."""
        if not self.entries:
            return None
        vector = self.embeddings.embed_query(question)
        best = max(self.entries, key=lambda e: self.similarity(vector, e["vector"]))
        if self.similarity(vector, best["vector"]) >= self.threshold:
            return best
        return None

    def add(self, question: str, answer: str, chunks: list[Document]) -> dict:
        entry = {
            "question": question,
            "vector": self.embeddings.embed_query(question),
            "answer": answer,
            "chunks": [
                {"text": c.page_content, "metadata": c.metadata} for c in chunks
            ],
            "triad": None,  # filled in by the UI once the answer has been scored
        }
        self.entries.append(entry)
        self.save()
        return entry

    @staticmethod
    def chunks_of(entry: dict) -> list[Document]:
        return [
            Document(page_content=c["text"], metadata=c["metadata"])
            for c in entry["chunks"]
        ]

    def save(self) -> None:
        self.file.parent.mkdir(parents=True, exist_ok=True)
        # The vectors stay in memory only; the file is just the readable part.
        readable = [{k: v for k, v in e.items() if k != "vector"} for e in self.entries]
        self.file.write_text(json.dumps(readable, indent=1))

    def destroy(self) -> None:
        """Delete this cache's file (and its private folder) and forget the entries."""
        self.entries = []
        self.file.unlink(missing_ok=True)
        if self.file.parent.name.startswith("ragforge_cache_"):
            shutil.rmtree(self.file.parent, ignore_errors=True)


# TRY IT ---- python src/cache.py
if __name__ == "__main__":

    class WordEmbeddings:  # a toy embedder: a vector of letter counts
        def embed_query(self, text: str) -> list[float]:
            return [
                text.lower().count(letter) for letter in "abcdefghijklmnopqrstuvwxyz"
            ]

    cache = SessionCache.__new__(SessionCache)
    cache.threshold, cache.embeddings = 0.95, WordEmbeddings()
    cache.file, cache.entries = Path(tempfile.mkdtemp()) / "demo.json", []
    cache.add("What rank r does LoRA use?", "r = 8", [])
    for question in ("what rank r does lora use", "What is the capital of France?"):
        hit = cache.find(question)
        print(f"{question!r}: {'cached -> ' + hit['answer'] if hit else 'miss'}")
    cache.destroy()
