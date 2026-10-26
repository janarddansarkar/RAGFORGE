"""RAG pipeline: the whole system in one place, using the best settings found by the evaluation.

    uv run python src/rag_pipeline.py "What rank r does LoRA use?"
    uv run python src/rag_pipeline.py                 # no question: ask several, empty line quits
    uv run python src/rag_pipeline.py --settings      # only show the settings in use

The settings come from best_config.json (written by eval/find_best.py) laid over config.toml:
chunking -> embedding model -> retrieval method + reranker -> top-k -> generator.
It answers from the PDFs in ./documents. (For a single PDF of your own, use src/rag.py.)
"""

import argparse
import sys

from langchain_core.documents import Document

from generator import Generator
from memory import ConversationMemory
from vectorstore import BEST_CONFIG_FILE, VectorStore, load_config


class RAGPipeline:
    # The settings that make up the pipeline, in the order a question flows through it.
    STAGES = (
        ("chunking_strategy", "chunking"),
        ("embedding_model", "embedding model"),
        ("default_method", "retrieval method"),
        ("reranker_model", "reranker"),
        ("top_k", "top-k"),
        ("llm_model", "generator"),
    )

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.method = self.config["default_method"]
        self.k = self.config["top_k"]

        # Index the documents the first time (the embedding model is part of the store).
        store = VectorStore(self.config)
        if store.count() == 0:
            print("Indexing ./documents (first time only)...", file=sys.stderr)
            store.build()
        if store.count() == 0:
            raise SystemExit(f"No PDFs found in {self.config['data_dir']}")

        self.generator = Generator(self.config)
        self.memory = ConversationMemory(self.config["memory_turns"])

    @classmethod
    def settings(cls, config: dict) -> str:
        source = BEST_CONFIG_FILE.name if BEST_CONFIG_FILE.exists() else "config.toml"
        lines = [f"Settings (from {source}):"]
        for key, name in cls.STAGES:
            value = config[key]
            if key == "chunking_strategy" and config["contextual"]:
                value = f"{value} + contextual notes"
            lines.append(f"  {name:17} {value}")
        return "\n".join(lines)

    def ask(self, question: str) -> tuple[str, list[Document]]:
        """Retrieve the top-k chunks, then answer from them (follow-ups use the last few turns). Returns the answer and the chunks."""
        answer, chunks, _ = self.generator.chat(
            question, self.memory, self.method, self.k
        )
        return answer, chunks

    def show(self, question: str) -> None:
        answer, chunks = self.ask(question)
        print(f"\nQ: {question}\n\nAnswer:\n{answer}\n\nSources:")
        for label in dict.fromkeys(self.generator.label(c) for c in chunks):
            print(f"  - {label}")
        print()


# RUN IT ---- python src/rag_pipeline.py "your question"
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "question", nargs="*", help="leave empty to ask several questions"
    )
    parser.add_argument(
        "--settings", action="store_true", help="show the settings and exit"
    )
    args = parser.parse_args()

    config = load_config()
    print(RAGPipeline.settings(config))
    if not args.settings:
        pipeline = RAGPipeline(config)
        if args.question:
            pipeline.show(" ".join(args.question))
        else:
            while question := input("Question (empty to quit): ").strip():
                pipeline.show(question)
