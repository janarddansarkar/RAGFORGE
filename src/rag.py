"""RAG: ask questions about one or more PDFs. DocumentRAG retrieves the best chunks and answers from
them, citing the pages; Report.cited finds the pages an answer cites.

The web UI (src/app.py) is the way to use it. It gives every browser session a private folder
(src/workspace.py): the uploaded PDFs are deleted as soon as they are indexed, and the index is
deleted when the documents are replaced or the session ends. Without a private folder (the
command line), each set of documents gets its own store in ./vs_docs, named by a hash of the
files, so editing a PDF re-indexes it.

At most `max_pages` pages (config.toml) are accepted in total across all the PDFs.
"""

import hashlib
import re
import shutil
import sys
import uuid
from pathlib import Path

import pymupdf
from langchain_core.documents import Document

from generator import Generator
from memory import ConversationMemory
from vectorstore import PROJECT_ROOT, VectorStore, load_config

DOCS_DIR = PROJECT_ROOT / "vs_docs"
REFUSAL = (
    "can't find that in the documents"  # what the generator says when it has no answer
)


class PageLimitExceeded(ValueError):
    """The PDFs together have more pages than max_pages allows."""


class DocumentRAG:
    SUFFIXES = (".pdf",)

    @staticmethod
    def page_count(data: bytes) -> int:
        """The number of pages in a PDF (read from its bytes)."""
        with pymupdf.open(stream=data, filetype="pdf") as document:
            return document.page_count

    @classmethod
    def check_page_limit(cls, page_counts: list[int], limit: int | None) -> None:
        """Raise PageLimitExceeded when the PDFs together are longer than the limit (0/None = no limit)."""
        total = sum(page_counts)
        if limit and total > limit:
            raise PageLimitExceeded(
                f"The PDFs have {total} pages in total; the limit is {limit}. Remove some pages or files."
            )

    def __init__(
        self,
        pdfs: Path | list[Path],
        k: int | None = None,
        method: str | None = None,
        settings: dict | None = None,
        workspace: Path | None = None,
    ):
        self.pdfs = [pdfs] if isinstance(pdfs, Path) else list(pdfs)
        if not self.pdfs:
            raise SystemExit("No PDF given")
        for pdf in self.pdfs:
            if pdf.suffix.lower() not in self.SUFFIXES or not pdf.is_file():
                raise SystemExit(f"Not a PDF file: {pdf}")
        base = {
            **load_config(),
            **(settings or {}),
        }  # settings: e.g. the user's own models
        self.check_page_limit(
            [self.page_count(pdf.read_bytes()) for pdf in self.pdfs],
            base.get("max_pages"),
        )
        self.k = k or base["top_k"]
        self.method = method or base["default_method"]
        # workspace: a private folder for this session; it is deleted with the documents
        self.folder = workspace / uuid.uuid4().hex[:12] if workspace else None
        self.config = self.config_for(self.pdfs, base, self.folder)

        store = VectorStore(self.config)
        names = ", ".join(pdf.name for pdf in self.pdfs)
        if store.count() == 0:
            print(f"Indexing {names} (first time only)...", file=sys.stderr)
            store.build()
        # The PDFs are only needed to build the index: do not keep copies of what users upload.
        shutil.rmtree(self.config["data_dir"], ignore_errors=True)
        if store.count() == 0:
            raise SystemExit(f"No text found in {names} (are they blank?)")
        self.generator = Generator(self.config)
        self.memory = ConversationMemory(self.config["memory_turns"])
        self.last_query = ""  # what the last question was rewritten to, for display

    def close(self) -> None:
        """The documents are being replaced: delete this conversation's cache, and the index
        too when it lives in a private session folder."""
        if self.generator.cache:
            self.generator.cache.destroy()
        if self.folder:
            shutil.rmtree(self.folder, ignore_errors=True)

    def is_available(self) -> bool:
        """False when the private folder was deleted (the session ended or was cleaned up)."""
        return self.folder is None or self.folder.exists()

    @staticmethod
    def config_for(pdfs: list[Path], base: dict, folder: Path | None = None) -> dict:
        """A config whose data and store folders belong to exactly this set of documents (the
        given private folder, or a shared one named by a hash of the files)."""
        contents = sorted((pdf.name, pdf.read_bytes()) for pdf in pdfs)
        private = folder is not None
        if folder is None:
            digest = hashlib.sha1()
            for name, data in contents:  # same files in any order -> same store
                digest.update(name.encode() + hashlib.sha1(data).digest())
            folder = DOCS_DIR / digest.hexdigest()[:12]
        (folder / "pdf").mkdir(parents=True, exist_ok=True)
        for name, data in contents:
            staged = folder / "pdf" / name  # the pipeline reads every PDF in data_dir
            if not staged.exists():
                staged.write_bytes(data)
        config = {**base, "data_dir": folder / "pdf", "vs_dir": folder / "vs"}
        if private:  # the answer cache lives in the same private folder, so it is deleted with it
            config["cache_file"] = folder / "cache" / "cache.json"
        return config

    def ask_stream(self, question: str):
        """Like ask(), but yields the answer piece by piece. Afterwards .last_chunks and
        .last_query hold the chunks and the rewritten query."""
        yield from self.generator.chat_stream(
            question, self.memory, self.method, self.k
        )
        self.last_query = self.generator.last_query

    def retry_stream(self, previous: dict, reasons: list[str]):
        """Like ask_stream(), for a second try at an answer that got a thumbs down
        (see Generator.retry_stream)."""
        yield from self.generator.retry_stream(previous, reasons, self.memory, self.k)
        self.last_query = self.generator.last_query

    @property
    def last_chunks(self) -> list[Document]:
        return self.generator.last_chunks


class Report:
    """Which pages an answer cites."""

    @staticmethod
    def refused(answer: str) -> bool:
        """True when the answer says the documents do not contain it: the standard refusal, or
        "The documents do not say ..." with nothing cited (a partial answer cites its passages)."""
        text = answer.lower().replace("\u2019", "'")
        if REFUSAL in text:
            return True
        return "the documents do not say" in text and not re.search(r"\bp\.\d+", text)

    @staticmethod
    def label(chunk: Document) -> str:
        return f"{chunk.metadata['source']} p.{chunk.metadata['page']}"

    @classmethod
    def cited(cls, answer: str, chunks: list[Document]) -> list[str]:
        """The pages the answer actually cites ([LoRA p.10]), in rank order. When it cites
        none that we can match, every retrieved page (what the answer was written from)."""
        found = []
        for chunk in chunks:
            tag = f"{chunk.metadata['source'].removesuffix('.pdf')} p.{chunk.metadata['page']}"
            if re.search(re.escape(tag) + r"(?!\d)", answer):
                found.append(cls.label(chunk))
        return list(dict.fromkeys(found or (cls.label(c) for c in chunks)))
