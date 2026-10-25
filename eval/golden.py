"""A golden question and the document pages that answer it."""

import re

from langchain_core.documents import Document

from metrics.retriever_metrics import RetrieverMetrics


class GoldenQuery:
    """One golden question and the document pages that answer it."""

    def __init__(self, item: dict):
        self.id = item["id"]
        self.query = item["query"]
        # "LoRA — §4.1, pp.4–5"  ->  document "lora", pages {4, 5}
        doc, _, where = item["source"].partition(" — ")
        self.doc = doc.strip().lower()
        self.pages = self._parse_pages(where)

    @staticmethod
    def _parse_pages(where: str) -> set[int]:
        """'p.4' -> {4}   'pp.4–5' -> {4, 5}   'pp.2,5' -> {2, 5}"""
        text = re.search(r"pp?\.\s*([\d,–-]+)\s*$", where)
        if not text:
            raise ValueError(f"Can't read the pages from '{where}'")
        pages: set[int] = set()
        for part in text.group(1).split(","):
            first, _, last = part.replace("–", "-").partition("-")
            pages.update(range(int(first), int(last or first) + 1))
        return pages

    def match(self, chunk: Document) -> int | None:
        """The golden page this chunk is on, or None if it is not relevant."""
        source = chunk.metadata["source"].removesuffix(".pdf").strip().lower()
        page = chunk.metadata["page"]
        return page if source == self.doc and page in self.pages else None

    def metrics(self, chunks: list[Document]) -> RetrieverMetrics:
        return RetrieverMetrics([self.match(c) for c in chunks], len(self.pages))
