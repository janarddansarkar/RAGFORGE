"""Citations list the pages the answer cites, not every retrieved page."""

from langchain_core.documents import Document

from rag import Report


def chunk(source, page):
    return Document(page_content="x", metadata={"source": source, "page": page})


CHUNKS = [chunk("LoRA.pdf", 4), chunk("LoRA.pdf", 10), chunk("LoRA.pdf", 1)]


def test_only_cited_pages_are_listed():
    assert Report.cited("r = 4 [LoRA p.10]", CHUNKS) == ["LoRA.pdf p.10"]


def test_p1_is_not_matched_by_p10():
    assert Report.cited("see [LoRA p.10]", CHUNKS) == ["LoRA.pdf p.10"]


def test_several_cited_pages_keep_rank_order():
    answer = "a [LoRA p.1] b [LoRA p.4]"
    assert Report.cited(answer, CHUNKS) == ["LoRA.pdf p.4", "LoRA.pdf p.1"]


def test_no_matching_tag_falls_back_to_every_retrieved_page():
    assert Report.cited("no tags here", CHUNKS) == [
        "LoRA.pdf p.4",
        "LoRA.pdf p.10",
        "LoRA.pdf p.1",
    ]
