"""Judge retrieval for ONE document and ONE question - no golden dataset needed.

It indexes the PDF in its own collection (the main ./vs collections are untouched),
retrieves the top-k chunks, and asks the local LLM whether each chunk helps answer
the question. Only the top-k are judged, so Recall is not possible (we don't know
how many relevant chunks exist); you get Precision@K, HitRate@K and MRR.

Run:
    uv run python eval/judge_retrieval.py "path/to/doc.pdf" "your question"
    uv run python eval/judge_retrieval.py doc.pdf "question" --method bm25 --k 3
    uv run python eval/judge_retrieval.py doc.pdf "question" --method all
"""

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR))

from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from metrics.retriever_metrics import RetrieverMetrics
from retriever import Retriever
from vectorstore import VectorStore, load_config

PROMPT = """You judge search results for a question-answering system.

Question: {question}

Passage:
{passage}

Does the passage contain information that helps answer the question?
Reply with exactly one word on the first line: RELEVANT or NOT_RELEVANT.
On the second line, give a short reason."""


class RelevanceJudge:
    """Asks the local LLM (llm_model in config.toml) if a chunk answers the question."""

    def __init__(self, config: dict):
        self.llm = ChatOllama(
            model=config["llm_model"],
            base_url=config["ollama_base_url"],
            temperature=0,
            reasoning=False,
        )
        self.cache: dict[str, tuple[bool, str]] = {}  # chunk id -> (relevant, reason)

    def judge(self, question: str, chunk: Document) -> tuple[bool, str]:
        key = str(chunk.id)
        if key not in self.cache:
            reply = str(
                self.llm.invoke(
                    PROMPT.format(question=question, passage=chunk.page_content)
                ).content
            ).strip()
            verdict, _, reason = reply.partition("\n")
            # "NOT_RELEVANT" also contains "RELEVANT", so check for the NOT first.
            relevant = "NOT" not in verdict.upper() and "RELEVANT" in verdict.upper()
            self.cache[key] = (relevant, reason.strip())
        return self.cache[key]


class DocumentRetrievalCheck:
    def __init__(self, pdf: Path, config: dict | None = None):
        config = config or load_config()
        # A private folder holding only this PDF, and its own Chroma collection.
        staging = config["vs_dir"] / "adhoc_pdfs" / pdf.stem.strip().replace(" ", "_")
        staging.mkdir(parents=True, exist_ok=True)
        shutil.copy(pdf, staging / pdf.name)
        self.config = {
            **config,
            "data_dir": staging,
            "collection_name": f"adhoc_{staging.name}",
        }
        self.judge = RelevanceJudge(self.config)

    def prepare(self) -> Retriever:
        VectorStore(self.config).build()  # does nothing if this PDF is already indexed
        return Retriever(self.config)

    def check(self, retriever: Retriever, question: str, method: str, k: int) -> None:
        chunks = retriever.retrieve(question, method=method, k=k)
        print(f"\n===== {method}  (top {k})")
        matches: list[str | None] = []
        for rank, chunk in enumerate(chunks, start=1):
            relevant, reason = self.judge.judge(question, chunk)
            matches.append(str(chunk.id) if relevant else None)
            preview = " ".join(chunk.page_content.split())[:150]
            print(f"{rank}. [{'RELEVANT' if relevant else 'not relevant':12}] p.{chunk.metadata['page']}")
            print(f"   {preview}...")
            print(f"   why: {reason}")

        if not chunks:
            print("No chunks retrieved.")
            return
        scores = RetrieverMetrics(matches, sum(m is not None for m in matches))
        print(
            f"-> Precision@{k} {scores.precision(k):.2f}   "
            f"HitRate@{k} {scores.hit(k):.0f}   MRR {scores.reciprocal_rank():.2f}"
        )


# RUN IT ---- python eval/judge_retrieval.py doc.pdf "question" [--method X] [--k N]
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("pdf", type=Path, help="path to the PDF")
    parser.add_argument("question")
    parser.add_argument("--method", help="a method from config.toml, or 'all' (default: default_method)")
    parser.add_argument("--k", type=int, help="chunks to judge (default: top_k)")
    args = parser.parse_args()

    if not args.pdf.is_file():
        sys.exit(f"PDF not found: {args.pdf}")

    check = DocumentRetrievalCheck(args.pdf)
    retriever = check.prepare()
    k = args.k or check.config["top_k"]
    if args.method == "all":
        methods = check.config["methods"]
    else:
        methods = [args.method or check.config["default_method"]]

    print(f"Q: {args.question}")
    for method in methods:
        check.check(retriever, args.question, method, k)
