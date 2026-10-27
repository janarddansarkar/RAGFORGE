"""Generator: answer a question from the retrieved chunks, citing the pages used.

Build the store first:  uv run python src/vectorstore.py
Try it:                 uv run python src/generator.py "What rank r does LoRA use?"
"""

import sys

from langchain_core.documents import Document

from cache import SessionCache
from guardrails import InputGuard
from memory import ConversationMemory
from retriever import Retriever
from vectorstore import load_config

ANSWER_PROMPT = """Answer the question using ONLY the passages below. \
If they do not contain the answer, say "I can't find that in the documents." \
Be brief. Cite each passage you used by copying the label in square brackets above it, \
exactly as written, and never cite a label that is not above a passage.

{passages}{history}{extra}

Question: {question}"""


# A thumbs down names what was wrong (cache.SessionCache.REASONS). The first two mean the
# chunks were wrong, so the retry searches again, wider. The others change how the answer is
# written from the chunks. Every retry also shows the model the answer that was rejected.
RETRIEVAL_REASONS = ("Wrong or missing source", "Wrong citation")
RETRY_NOTES = {
    "Wrong citation": "Cite each statement with the label of the passage that states it.",
    "Made something up": "Use only facts stated in the passages; leave out any detail that is not there.",
    "Incomplete": "Cover every part of the question, with the details the passages give.",
    "Too long": "Answer in at most two sentences.",
}
RETRY_PROMPT = """

The user rejected an earlier answer to this question ({reasons}). Do not repeat it:
{previous}

{notes}"""


class Generator:
    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.retriever = Retriever(self.config)
        self.guard = InputGuard(self.config) if self.config["input_guard"] else None
        self.cache = SessionCache(self.config) if self.config["use_cache"] else None
        self.cache_entry: dict | None = None  # the cache entry behind the last answer
        self.last_chunks: list[Document] = []  # set by chat_stream() when it finishes
        self.last_query = ""

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
        Same as chat_stream(), just waiting for the whole answer."""
        answer = "".join(self.chat_stream(question, memory, method, k)).strip()
        return answer, self.last_chunks, self.last_query

    def chat_stream(
        self,
        question: str,
        memory: ConversationMemory,
        method: str | None = None,
        k: int | None = None,
    ):
        """One turn of a conversation, yielding the answer piece by piece as it is written.
        Once it is used up, .last_chunks and .last_query hold the chunks and the query searched.
        Raises InputBlocked (before anything is searched or remembered) for a bad question.
        A question like an earlier one in this session is answered from the cache."""
        if self.guard:
            self.guard.check(question)
        search_query = memory.standalone(question, self.retriever.llm)
        hit = self.cache.find(search_query) if self.cache else None
        if hit:  # a similar question was answered earlier in this session
            self.cache_entry = hit
            chunks = self.cache.chunks_of(hit)
            answer = hit["answer"]
            yield answer
        else:
            chunks = self.retriever.retrieve(
                search_query, method or self.config["default_method"], k
            )
            prompt = self.build_prompt(question, chunks, memory.history())
            pieces = []
            for piece in self.retriever.llm.stream(prompt):
                pieces.append(piece)
                yield piece
            answer = "".join(pieces).strip()
            self.cache_entry = (
                self.cache.add(search_query, answer, chunks) if self.cache else None
            )
        memory.add(question, answer)
        self.last_chunks, self.last_query = chunks, search_query

    def retry_stream(
        self,
        previous: dict,
        reasons: list[str],
        memory: ConversationMemory,
        k: int | None = None,
    ):
        """A second try at a question whose answer got a thumbs down. `previous` is that turn
        (its question, text, query and chunks). Wrong-source reasons (or no reason) search again,
        wider; the other reasons write a new answer from the same chunks with extra instructions.
        Yields the answer piece by piece; afterwards .last_chunks and .last_query are set, the
        new answer is in the session cache and it replaces the old one in the memory."""
        k = k or self.config["top_k"]
        search_query = previous["query"]
        search_again = not reasons or any(r in RETRIEVAL_REASONS for r in reasons)
        chunks = (
            self.retriever.retrieve_wider(search_query, k + 2)
            if search_again
            else previous["chunks"]
        )
        notes = "\n".join(RETRY_NOTES[r] for r in reasons if r in RETRY_NOTES)
        extra = RETRY_PROMPT.format(
            reasons=", ".join(reasons) or "no reason given",
            previous=previous["text"],
            notes=notes,
        ).rstrip()
        prompt = self.build_prompt(
            previous["question"], chunks, memory.history(skip_last=True), extra
        )
        # A small model can ignore "two sentences", so "Too long" also lowers the limit.
        limit = self.config["max_new_tokens"]
        if "Incomplete" in reasons:
            limit *= 2
        elif "Too long" in reasons:
            limit //= 3
        pieces = []
        for piece in self.retriever.llm.stream(prompt, max_new_tokens=limit):
            pieces.append(piece)
            yield piece
        answer = "".join(pieces).strip()
        self.cache_entry = (
            self.cache.add(search_query, answer, chunks) if self.cache else None
        )
        memory.replace_last_answer(answer)
        self.last_chunks, self.last_query = chunks, search_query

    def build_prompt(
        self, question: str, chunks: list[Document], history: str = "", extra: str = ""
    ) -> str:
        """The answer prompt. history is the recent conversation, for context only."""
        passages = "\n\n".join(f"[{self.label(c)}]\n{c.page_content}" for c in chunks)
        conversation = (
            f"\n\nConversation so far (for context only, not a source):\n{history}"
            if history
            else ""
        )
        return ANSWER_PROMPT.format(
            passages=passages, history=conversation, extra=extra, question=question
        )

    def answer_from(
        self, question: str, chunks: list[Document], history: str = ""
    ) -> tuple[str, list[Document]]:
        """Answer from the given chunks (lets the eval test the generator on its own)."""
        prompt = self.build_prompt(question, chunks, history)
        reply = str(self.retriever.llm.invoke(prompt).content).strip()
        return reply, chunks


# TRY IT ---- python src/generator.py "your question"
if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What rank r does LoRA use?"
    generator = Generator()
    text, chunks = generator.answer(question)
    sources = ", ".join(generator.label(c) for c in chunks)
    print(f"Q: {question}\n\n{text}\n\nSources: {sources}")
