"""Generator: answer a question from the retrieved chunks, citing the pages used.

Build the store first:  uv run python src/vectorstore.py
Try it:                 uv run python src/generator.py "What rank r does LoRA use?"
"""

import sys

from langchain_core.documents import Document

from guardrails import InputGuard
from memory import ConversationMemory
from retriever import Retriever
from vectorstore import load_config

ANSWER_PROMPT = """Answer the question using ONLY the passages below. \
If they do not contain the answer, say "I can't find that in the documents." \
Be brief and cite the passages you used like [LoRA p.4].

{passages}{history}

Question: {question}"""


class Generator:
    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.retriever = Retriever(self.config)
        self.guard = InputGuard(self.config) if self.config["input_guard"] else None

    @staticmethod
    def label(chunk: Document) -> str:
        return f"{chunk.metadata['source'].removesuffix('.pdf')} p.{chunk.metadata['page']}"

    def answer(self, question: str) -> tuple[str, list[Document]]:
        """Retrieve chunks, then answer from them. Returns the answer and the chunks."""
        return self.answer_from(question, self.retriever.retrieve(question))

    def chat(
        self,
        question: str,
        memory: ConversationMemory,
        method: str | None = None,
        k: int | None = None,
    ) -> tuple[str, list[Document], str]:
        """One turn of a conversation. Returns the answer, the chunks and the query searched.
        Raises InputBlocked (before anything is searched or remembered) for a bad question."""
        if self.guard:
            self.guard.check(question)
        search_query = memory.standalone(question, self.retriever.llm)
        chunks = self.retriever.retrieve(
            search_query, method or self.config["default_method"], k
        )
        answer, chunks = self.answer_from(question, chunks, memory.history())
        memory.add(question, answer)
        return answer, chunks, search_query

    def answer_from(
        self, question: str, chunks: list[Document], history: str = ""
    ) -> tuple[str, list[Document]]:
        """Answer from the given chunks (lets the eval test the generator on its own).
        history is the recent conversation, for context only (empty: same prompt as before)."""
        passages = "\n\n".join(f"[{self.label(c)}]\n{c.page_content}" for c in chunks)
        conversation = (
            f"\n\nConversation so far (for context only, not a source):\n{history}"
            if history
            else ""
        )
        prompt = ANSWER_PROMPT.format(
            passages=passages, history=conversation, question=question
        )
        reply = str(self.retriever.llm.invoke(prompt).content).strip()
        return reply, chunks


# TRY IT ---- python src/generator.py "your question"
if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What rank r does LoRA use?"
    generator = Generator()
    text, chunks = generator.answer(question)
    sources = ", ".join(generator.label(c) for c in chunks)
    print(f"Q: {question}\n\n{text}\n\nSources: {sources}")
