"""Generator: answer a question from the retrieved chunks, citing the pages used.

Build the store first:  uv run python src/vectorstore.py
Try it:                 uv run python src/generator.py "What rank r does LoRA use?"
"""

import sys

from langchain_core.documents import Document

from retriever import Retriever
from vectorstore import load_config

ANSWER_PROMPT = """Answer the question using ONLY the passages below. \
If they do not contain the answer, say "I can't find that in the documents." \
Be brief and cite the passages you used like [LoRA p.4].

{passages}

Question: {question}"""


class Generator:
    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.retriever = Retriever(self.config)

    @staticmethod
    def label(chunk: Document) -> str:
        return f"{chunk.metadata['source'].removesuffix('.pdf')} p.{chunk.metadata['page']}"

    def answer(self, question: str) -> tuple[str, list[Document]]:
        """Retrieve chunks, then answer from them. Returns the answer and the chunks."""
        return self.answer_from(question, self.retriever.retrieve(question))

    def answer_from(
        self, question: str, chunks: list[Document]
    ) -> tuple[str, list[Document]]:
        """Answer from the given chunks (lets the eval test the generator on its own)."""
        passages = "\n\n".join(f"[{self.label(c)}]\n{c.page_content}" for c in chunks)
        prompt = ANSWER_PROMPT.format(passages=passages, question=question)
        reply = str(self.retriever.llm.invoke(prompt).content).strip()
        return reply, chunks


# TRY IT ---- python src/generator.py "your question"
if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What rank r does LoRA use?"
    generator = Generator()
    text, chunks = generator.answer(question)
    sources = ", ".join(generator.label(c) for c in chunks)
    print(f"Q: {question}\n\n{text}\n\nSources: {sources}")
