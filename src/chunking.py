"""Chunking pipeline: PDFs -> pages -> chunks with stable IDs.

One pipeline is shared by everything that needs chunks:
  - vectorstore.py builds the Chroma store from pipeline.chunks()
  - the Retriever opens the store of the same pipeline (the name picks the collection)

Strategies (config.toml): recursive | semantic. With contextual = true,
any of them is wrapped so an LLM adds a short context note to every chunk.

Look at the chunks without building the store:
    uv run python src/chunking.py
"""

import hashlib
import json
import re
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pymupdf
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models import OllamaChat, make_embeddings, make_llm


class Chunker(ABC):
    """Splits the text of ONE page into chunk strings."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifies the strategy and its settings, e.g. 'recursive_1000_150'."""

    @abstractmethod
    def split_page(self, page: Document) -> list[str]: ...


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


class SemanticChunker(Chunker):
    """Start a new chunk where the meaning changes between neighbouring sentences."""

    def __init__(self, embeddings: Embeddings, percentile: int, max_chars: int):
        self.embeddings = embeddings
        self.percentile = percentile
        self.max_chars = max_chars

    @property
    def name(self) -> str:
        return f"semantic_p{self.percentile}"

    def split_page(self, page: Document) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", page.page_content)
        if len(sentences) < 2:
            return sentences

        # How different is each sentence from the next? (1 - cosine similarity)
        vectors = np.array(self.embeddings.embed_documents(sentences))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        distances = 1 - (vectors[:-1] * vectors[1:]).sum(axis=1)
        threshold = np.percentile(distances, self.percentile)

        chunks, current = [], [sentences[0]]
        for sentence, distance in zip(sentences[1:], distances):
            too_long = len(" ".join(current)) + len(sentence) > self.max_chars
            if distance > threshold or too_long:
                chunks.append(" ".join(current))
                current = []
            current.append(sentence)
        chunks.append(" ".join(current))
        return chunks


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

    def __init__(self, base: Chunker, llm: OllamaChat, cache_file: Path):
        self.base = base
        self.llm = llm
        self.cache_file = cache_file
        # The LLM calls are slow, so notes are saved and reused on the next build.
        self.cache: dict[str, str] = (
            json.loads(cache_file.read_text()) if cache_file.exists() else {}
        )

    @property
    def name(self) -> str:
        return f"contextual_{self.base.name}"

    def split_page(self, page: Document) -> list[str]:
        chunks = []
        for chunk in self.base.split_page(page):
            chunks.append(f"{self._context_note(page, chunk)}\n\n{chunk}")
        self.cache_file.write_text(json.dumps(self.cache))
        return chunks

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
    IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")

    def __init__(self, chunker: Chunker, data_dir: Path):
        self.chunker = chunker
        self.data_dir = data_dir

    @property
    def name(self) -> str:
        return self.chunker.name

    @classmethod
    def from_config(cls, config: dict) -> "ChunkingPipeline":
        """Build the pipeline that config.toml describes."""
        strategy = config["chunking_strategy"]
        if strategy == "recursive":
            chunker: Chunker = RecursiveChunker(
                config["chunk_size"], config["chunk_overlap"]
            )
        elif strategy == "semantic":
            chunker = SemanticChunker(
                make_embeddings(config),
                config["semantic_percentile"],
                config["semantic_max_chars"],
            )
        else:
            raise ValueError(
                f"Unknown chunking_strategy '{strategy}'. Choose: recursive, semantic"
            )

        if config["contextual"]:
            llm = make_llm(config)
            cache_file = config["vs_dir"] / "context_cache.json"
            chunker = ContextualChunker(chunker, llm, cache_file)
        return cls(chunker, config["data_dir"])

    # 1. LOAD ---- one Document per PDF page, or per image file
    def load_pages(self) -> list[Document]:
        pages = []
        for path in sorted(self.data_dir.iterdir()):
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                texts = self._pdf_texts(path)
            elif suffix in self.IMAGE_SUFFIXES:
                texts = [self._ocr(pymupdf.open(path)[0])]
            else:
                continue
            for page_num, text in enumerate(texts, start=1):
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
            if source.suffix.lower() != ".pdf":
                stem += "_img"  # so a.pdf and a.jpg never share chunk IDs
            for i, text in enumerate(self.chunker.split_page(page)):
                chunk_id = f"{stem}_p{page.metadata['page']}_{i}"
                chunks.append(
                    Document(
                        id=chunk_id,
                        page_content=text,
                        metadata={
                            **page.metadata,
                            "chunk_index": i,
                            "chunk_id": chunk_id,
                        },
                    )
                )
        return chunks


# LOOK AT THEM ---- python src/chunking.py
if __name__ == "__main__":
    from vectorstore import load_config

    pipeline = ChunkingPipeline.from_config(load_config())
    chunks = pipeline.chunks()
    sizes = [len(c.page_content) for c in chunks]
    print(f"strategy: {pipeline.name}")
    print(
        f"{len(chunks)} chunks, characters min/avg/max = "
        f"{min(sizes)}/{sum(sizes) // len(sizes)}/{max(sizes)}"
    )
    print(f"\nExample ({chunks[len(chunks) // 2].id}):\n")
    print(chunks[len(chunks) // 2].page_content)
