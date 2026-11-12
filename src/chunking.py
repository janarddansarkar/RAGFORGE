"""Chunking pipeline: PDFs -> pages -> chunks with stable IDs.

One pipeline is shared by everything that needs chunks:
  - vectorstore.py builds the Chroma store from pipeline.chunks()
  - the Retriever opens the store of the same pipeline (the name picks the collection)

Chunks are split recursively (config.toml). With contextual = true, the chunker is wrapped so an
LLM adds a short context note to every chunk.
"""

import hashlib
import json
import re
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models import LiteLLMChat, make_llm


def chunk_body(chunk: Document) -> str:
    """The text the document really says. With contextual chunking a stored chunk is an LLM
    note followed by this text; the note helps the search, but answers must come from the text."""
    return chunk.metadata.get("chunk_text", chunk.page_content)


def chunk_note(chunk: Document) -> str:
    """The context note in front of a chunk's text ("" when it has none)."""
    body = chunk_body(chunk)
    return chunk.page_content[: -len(body)].strip() if body else ""


class Chunker(ABC):
    """Splits the text of ONE page into chunk strings."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifies the strategy and its settings, e.g. 'recursive_1000_150'."""

    @abstractmethod
    def split_page(self, page: Document) -> list[str]: ...

    def split_parts(self, page: Document) -> list[tuple[str, str]]:
        """The page's chunks as (context note, text) pairs; the note is "" unless an LLM wrote one."""
        return [("", text) for text in self.split_page(page)]


class RecursiveChunker(Chunker):
    """Split on paragraphs, then lines, then words, until pieces fit the size."""

    def __init__(self, size: int, overlap: int):
        self.size = size
        self.overlap = overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=size, chunk_overlap=overlap
        )

    @property
    def name(self) -> str:
        return f"recursive_{self.size}_{self.overlap}"

    def split_page(self, page: Document) -> list[str]:
        return self.splitter.split_text(page.page_content)


CONTEXT_PROMPT = """<page>
Document: {source}, page {page_number}
{page}
</page>

Here is a chunk taken from that page:
<chunk>
{chunk}
</chunk>

Write one or two short sentences that place this chunk in its document: what the \
document is, and what topic or section of it the chunk belongs to. The sentences \
will be added in front of the chunk to improve search. Use only what the page and the \
document name say: do not invent titles, names or expansions of abbreviations. \
Reply with those sentences only."""


class ContextualChunker(Chunker):
    """Wraps another chunker and puts an LLM-written context note in front of each chunk."""

    def __init__(self, base: Chunker, llm: LiteLLMChat, cache_file: Path):
        self.base = base
        self.llm = llm
        self.cache_file = cache_file
        # The LLM calls are slow, so notes are saved and reused on the next build.
        self.cache: dict[str, str] = (
            json.loads(cache_file.read_text()) if cache_file.exists() else {}
        )

    @property
    def name(self) -> str:
        # "body": a chunk also keeps its own text apart from the note (see chunk_body), so
        # stores built before that change have another name and are rebuilt (notes are reused).
        return f"contextual_body_{self.base.name}"

    def split_parts(self, page: Document) -> list[tuple[str, str]]:
        parts = [
            (self._context_note(page, chunk), chunk)
            for chunk in self.base.split_page(page)
        ]
        self.cache_file.write_text(json.dumps(self.cache))
        return parts

    def split_page(self, page: Document) -> list[str]:
        return [f"{note}\n\n{chunk}" for note, chunk in self.split_parts(page)]

    def _context_note(self, page: Document, chunk: str) -> str:
        key = hashlib.sha1(
            f"{self.llm.model}|{page.page_content}|{chunk}".encode()
        ).hexdigest()
        if key not in self.cache:
            prompt = CONTEXT_PROMPT.format(
                source=page.metadata["source"],
                page_number=page.metadata["page"],
                page=page.page_content,
                chunk=chunk,
            )
            self.cache[key] = str(self.llm.invoke(prompt).content).strip()
        return self.cache[key]


class ChunkingPipeline:
    def __init__(self, chunker: Chunker, data_dir: Path):
        self.chunker = chunker
        self.data_dir = data_dir

    @property
    def name(self) -> str:
        return self.chunker.name

    @classmethod
    def from_config(cls, config: dict) -> "ChunkingPipeline":
        """Build the pipeline that config.toml describes."""
        chunker: Chunker = RecursiveChunker(
            config["chunk_size"], config["chunk_overlap"]
        )

        if config["contextual"]:
            llm = make_llm(config, role="notes")
            cache_file = config["vs_dir"] / "context_cache.json"
            chunker = ContextualChunker(chunker, llm, cache_file)
        return cls(chunker, config["data_dir"])

    # 1. LOAD ---- one Document per PDF page
    def load_pages(self) -> list[Document]:
        pages = []
        for path in sorted(self.data_dir.iterdir()):
            if path.suffix.lower() != ".pdf":
                continue
            for page_num, text in enumerate(self._pdf_texts(path), start=1):
                text = self._clean(text)
                if text:  # skip blank pages
                    pages.append(
                        Document(
                            page_content=text,
                            metadata={"source": path.name, "page": page_num},
                        )
                    )
        return pages

    def _pdf_texts(self, path: Path) -> list[str]:
        """The text of every page; a page with no text layer (a scan) is read with OCR."""
        texts = []
        for page in pymupdf.open(path).pages():
            text = str(page.get_text())  # str() for the type checker
            texts.append(text if text.strip() else self._ocr(page))
        return texts

    @staticmethod
    def _ocr(page: pymupdf.Page) -> str:
        """Text read from the page image with Tesseract (brew install tesseract)."""
        try:
            textpage = page.get_textpage_ocr(language="eng", dpi=300, full=True)
        except RuntimeError as error:  # no Tesseract installed: say so, do not crash
            print(f"OCR failed: {error}", file=sys.stderr)
            return ""
        return str(page.get_text(textpage=textpage))

    @staticmethod
    def _clean(text: str) -> str:
        # Re-join words split across lines: "LAN-\nGUAGE" -> "LANGUAGE"
        return re.sub(r"(\w)-\n(\w)", r"\1\2", text).strip()

    # 2. CHUNK ---- every chunk stays inside one page and gets a stable ID
    def chunks(self) -> list[Document]:
        chunks = []
        for page in self.load_pages():
            source = Path(page.metadata["source"])
            stem = source.stem.strip().replace(" ", "_")
            for i, (note, text) in enumerate(self.chunker.split_parts(page)):
                chunk_id = f"{stem}_p{page.metadata['page']}_{i}"
                metadata = {**page.metadata, "chunk_index": i, "chunk_id": chunk_id}
                if note:
                    metadata["chunk_text"] = text  # the note stays out of the answers
                chunks.append(
                    Document(
                        id=chunk_id,
                        page_content=f"{note}\n\n{text}" if note else text,
                        metadata=metadata,
                    )
                )
        return chunks
