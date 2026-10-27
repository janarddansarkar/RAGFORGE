"""Context notes help the search but must not reach the answer: chunks keep their own text apart."""

from types import SimpleNamespace

from langchain_core.documents import Document

from chunking import (
    ChunkingPipeline,
    ContextualChunker,
    RecursiveChunker,
    chunk_body,
    chunk_note,
)
from generator import Generator

NOTE = "This document is a tax form."
TEXT = "Backup withholding is 24% of reportable payments."
PAGE = Document(page_content=TEXT, metadata={"source": "w9.pdf", "page": 2})


class FakeLLM:
    model = "fake"

    def invoke(self, prompt):
        return SimpleNamespace(content=NOTE)


def contextual(tmp_path) -> ContextualChunker:
    return ContextualChunker(RecursiveChunker(1000, 0), FakeLLM(), tmp_path / "c.json")


def test_a_contextual_chunk_is_split_into_note_and_text(tmp_path):
    assert contextual(tmp_path).split_parts(PAGE) == [(NOTE, TEXT)]


def test_split_page_still_gives_the_note_in_front_of_the_text(tmp_path):
    assert contextual(tmp_path).split_page(PAGE) == [f"{NOTE}\n\n{TEXT}"]


def test_a_plain_chunker_has_no_note():
    assert RecursiveChunker(1000, 0).split_parts(PAGE) == [("", TEXT)]


def test_the_stored_chunk_keeps_the_note_for_search_and_the_text_for_answers(tmp_path):
    pipeline = ChunkingPipeline(contextual(tmp_path), tmp_path)
    pipeline.load_pages = lambda: [PAGE]
    (chunk,) = pipeline.chunks()
    assert chunk.page_content == f"{NOTE}\n\n{TEXT}"  # what is embedded and searched
    assert chunk_body(chunk) == TEXT
    assert chunk_note(chunk) == NOTE


def test_a_chunk_without_a_note_is_its_own_body(tmp_path):
    pipeline = ChunkingPipeline(RecursiveChunker(1000, 0), tmp_path)
    pipeline.load_pages = lambda: [PAGE]
    (chunk,) = pipeline.chunks()
    assert "chunk_text" not in chunk.metadata
    assert chunk_body(chunk) == TEXT and chunk_note(chunk) == ""


def test_a_store_built_before_this_change_still_works():
    old = Document(page_content=f"{NOTE}\n\n{TEXT}", metadata={"source": "w9.pdf"})
    assert chunk_body(old) == old.page_content  # no separate text: nothing to strip
    assert chunk_note(old) == ""


def test_contextual_stores_get_their_own_name_so_old_ones_are_rebuilt(tmp_path):
    assert contextual(tmp_path).name == "contextual_body_recursive_1000_0"


def test_the_answer_prompt_uses_the_text_and_not_the_note():
    chunk = Document(
        page_content=f"{NOTE}\n\n{TEXT}",
        metadata={"source": "w9.pdf", "page": 2, "chunk_text": TEXT},
    )
    prompt = Generator.__new__(Generator).build_prompt("What is the rate?", [chunk])
    assert TEXT in prompt and NOTE not in prompt
