"""GoldenQuery: reads "Doc — pp.4–5" and decides which chunks are relevant."""

import pytest
from langchain_core.documents import Document

from eval.golden import GoldenQuery


def golden(source: str) -> GoldenQuery:
    return GoldenQuery({"id": "X1", "query": "q?", "source": source})


@pytest.mark.parametrize(
    ("source", "pages"),
    [
        ("LoRA — p.4", {4}),
        ("LoRA — pp.4–5", {4, 5}),
        ("LoRA — pp.2,5", {2, 5}),
        ("LoRA — pp.1-3", {1, 2, 3}),
    ],
)
def test_pages_are_read_from_the_source(source, pages):
    assert golden(source).pages == pages


def test_document_name_is_lowercase():
    assert golden("Attention is all you need — p.3").doc == "attention is all you need"


def test_unreadable_pages_raise():
    with pytest.raises(ValueError):
        golden("LoRA — somewhere")


def chunk(source: str, page: int) -> Document:
    return Document(page_content="text", metadata={"source": source, "page": page})


def test_a_chunk_is_relevant_only_on_the_right_document_and_page():
    query = golden("LoRA — pp.4–5")
    assert query.match(chunk("LoRA.pdf", 4)) == 4
    assert query.match(chunk("LoRA.pdf", 6)) is None
    assert query.match(chunk("QLoRA.pdf", 4)) is None


def test_metrics_use_the_number_of_golden_pages():
    query = golden("LoRA — pp.4–5")
    metrics = query.metrics([chunk("LoRA.pdf", 4), chunk("QLoRA.pdf", 5)])
    assert metrics.total_relevant == 2
    assert metrics.recall(2) == 0.5
