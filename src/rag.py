"""RAG: ask questions about one or more PDFs or images. Retrieves the best chunks, answers from them,
and shows the citations and the top-k chunks.

    uv run python src/rag.py paper.pdf "What rank r does LoRA use?"
    uv run python src/rag.py paper.pdf                  # no question: ask several, empty line quits
    uv run python src/rag.py paper.pdf "..." --k 3 --full

The PDFs are indexed the first time (a minute or so) and reused afterwards: each set of
documents gets its own store in ./vs_docs, named by a hash of the files, so editing a PDF
re-indexes it. (The CLI takes one file; DocumentRAG and the web UI take several. Scans and photos are
read by the vision model, which makes the first indexing slower.)
"""

import argparse
import hashlib
import sys
import textwrap
from pathlib import Path

from langchain_core.documents import Document

from chunking import ChunkingPipeline
from generator import Generator
from guardrails import InputBlocked
from memory import ConversationMemory
from vectorstore import PROJECT_ROOT, VectorStore, load_config

DOCS_DIR = PROJECT_ROOT / "vs_docs"
REFUSAL = (
    "can't find that in the documents"  # what the generator says when it has no answer
)


class DocumentRAG:
    # Scanned pages and photos are read by the vision model (vision.py).
    SUFFIXES = (".pdf", *ChunkingPipeline.IMAGE_SUFFIXES)

    def __init__(
        self,
        pdfs: Path | list[Path],
        k: int | None = None,
        method: str | None = None,
    ):
        self.pdfs = [pdfs] if isinstance(pdfs, Path) else list(pdfs)
        if not self.pdfs:
            raise SystemExit("No PDF given")
        for pdf in self.pdfs:
            if pdf.suffix.lower() not in self.SUFFIXES or not pdf.is_file():
                raise SystemExit(f"Not a PDF or image file: {pdf}")
        base = load_config()
        self.k = k or base["top_k"]
        self.method = method or base["default_method"]
        self.config = self.config_for(self.pdfs, base)

        store = VectorStore(self.config)
        names = ", ".join(pdf.name for pdf in self.pdfs)
        if store.count() == 0:
            print(f"Indexing {names} (first time only)...", file=sys.stderr)
            store.build()
        if store.count() == 0:
            raise SystemExit(f"No text found in {names} (are they blank?)")
        self.generator = Generator(self.config)
        self.memory = ConversationMemory(self.config["memory_turns"])
        self.last_query = ""  # what the last question was rewritten to, for display

    @staticmethod
    def config_for(pdfs: list[Path], base: dict) -> dict:
        """A config whose data and store folders belong to exactly this set of documents."""
        contents = sorted((pdf.name, pdf.read_bytes()) for pdf in pdfs)
        digest = hashlib.sha1()
        for name, data in contents:  # same files in any order -> same store
            digest.update(name.encode() + hashlib.sha1(data).digest())
        folder = DOCS_DIR / digest.hexdigest()[:12]
        (folder / "pdf").mkdir(parents=True, exist_ok=True)
        for name, data in contents:
            staged = folder / "pdf" / name  # the pipeline reads every PDF in data_dir
            if not staged.exists():
                staged.write_bytes(data)
        return {**base, "data_dir": folder / "pdf", "vs_dir": folder / "vs"}

    def ask(self, question: str) -> tuple[str, list[Document]]:
        """The answer and the top-k chunks it was written from. Remembers the last few turns."""
        answer, chunks, self.last_query = self.generator.chat(
            question, self.memory, self.method, self.k
        )
        return answer, chunks

    def ask_stream(self, question: str):
        """Like ask(), but yields the answer piece by piece. Afterwards .last_chunks and
        .last_query hold the chunks and the rewritten query."""
        yield from self.generator.chat_stream(
            question, self.memory, self.method, self.k
        )
        self.last_query = self.generator.last_query

    @property
    def last_chunks(self) -> list[Document]:
        return self.generator.last_chunks


class Report:
    """Prints an answer, its citations and the chunks behind it."""

    def __init__(self, full: bool = False):
        self.full = full

    @staticmethod
    def label(chunk: Document) -> str:
        return f"{chunk.metadata['source']} p.{chunk.metadata['page']}"

    def show(self, question: str, answer: str, chunks: list[Document]) -> None:
        print(f"\nQ: {question}\n\nAnswer:\n{textwrap.indent(answer, '  ')}")

        print("\nCitations (the pages the answer was written from):")
        if REFUSAL in answer.lower().replace("\u2019", "'"):
            print("  none - the document does not answer this question")
        else:
            for label in dict.fromkeys(
                self.label(c) for c in chunks
            ):  # unique, in rank order
                print(f"  - {label}")

        print(f"\nTop {len(chunks)} chunks:")
        for rank, chunk in enumerate(chunks, start=1):
            score = chunk.metadata.get("rerank_score")
            score_text = f"   rerank score {score:.2f}" if score is not None else ""
            print(f"\n  [{rank}] {self.label(chunk)}{score_text}")
            text = " ".join(chunk.page_content.split())
            text = text if self.full or len(text) <= 400 else text[:400] + " ..."
            print(textwrap.indent(textwrap.fill(text, width=88), "      "))
        print()


# RUN IT ---- python src/rag.py paper.pdf "your question"
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("pdf", type=Path, help="path to the PDF")
    parser.add_argument(
        "question", nargs="*", help="leave empty to ask several questions"
    )
    parser.add_argument("--k", type=int, help="chunks to retrieve (default: top_k)")
    parser.add_argument("--method", help="retrieval method (default: default_method)")
    parser.add_argument("--full", action="store_true", help="show whole chunks")
    args = parser.parse_args()

    rag = DocumentRAG(args.pdf, k=args.k, method=args.method)
    report = Report(full=args.full)

    def ask_and_show(question: str) -> None:
        try:
            report.show(question, *rag.ask(question))
        except InputBlocked as blocked:
            print(f"\n{blocked}\n")

    if args.question:
        ask_and_show(" ".join(args.question))
    else:
        while question := input("Question (empty to quit): ").strip():
            ask_and_show(question)
