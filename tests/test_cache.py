"""Session cache with a toy embedder: similar questions hit, others miss, the file goes away."""

import json

from langchain_core.documents import Document

from cache import SessionCache


class LetterEmbeddings:
    def embed_query(self, text: str) -> list[float]:
        return [text.lower().count(letter) for letter in "abcdefghijklmnopqrstuvwxyz"]


def make_cache(tmp_path) -> SessionCache:
    cache = SessionCache.__new__(SessionCache)  # skip loading config and a model
    cache.threshold, cache.embeddings = 0.95, LetterEmbeddings()
    cache.file, cache.entries = tmp_path / "cache.json", []
    return cache


def test_similar_question_hits_and_other_misses(tmp_path):
    cache = make_cache(tmp_path)
    chunk = Document(page_content="text", metadata={"source": "a.pdf", "page": 1})
    cache.add("What rank r does LoRA use?", "r = 8", [chunk])
    hit = cache.find("what rank r does lora use")
    assert hit["answer"] == "r = 8"
    assert cache.chunks_of(hit)[0].metadata["page"] == 1
    assert cache.find("What is the capital of France?") is None


def test_empty_cache_misses_and_file_is_json_then_destroyed(tmp_path):
    cache = make_cache(tmp_path)
    assert cache.find("anything") is None
    cache.add("q", "a", [])
    assert json.loads(cache.file.read_text())[0]["answer"] == "a"
    cache.destroy()
    assert not cache.file.exists()
