"""Short-term memory: the last few turns of the conversation, so follow-ups work.

Two jobs:
  - history()  the recent turns, shown to the answer model for context
  - ConversationMemory.standalone() rewrites a follow-up ("and for QLoRA?") into a question
    that makes sense on its own, because the search only sees the question it is given.

Memory is never a source: answers still come only from the retrieved passages.
"""

CONDENSE_PROMPT = """Given the conversation below and a follow-up question, rewrite the \
follow-up as one standalone question that makes sense without the conversation. \
Keep every name and detail. If it is already standalone, repeat it unchanged. \
Reply with the question only.

Conversation:
{history}

Follow-up question: {question}"""

MAX_ANSWER_CHARS = (
    500  # long answers are cut, to keep the prompt small for a small model
)


class ConversationMemory:
    def __init__(self, max_turns: int = 3):
        self.max_turns = max_turns
        self.turns: list[tuple[str, str]] = []  # (question, answer), oldest first

    def add(self, question: str, answer: str) -> None:
        self.turns.append((question, answer))
        self.turns = self.turns[-self.max_turns :]

    def clear(self) -> None:
        self.turns = []

    def history(self) -> str:
        lines = []
        for question, answer in self.turns:
            if len(answer) > MAX_ANSWER_CHARS:
                answer = answer[:MAX_ANSWER_CHARS] + " ..."
            lines += [f"User: {question}", f"Assistant: {answer}"]
        return "\n".join(lines)

    def standalone(self, question: str, llm) -> str:
        """The question rewritten to stand on its own (unchanged on the first turn)."""
        if not self.turns:
            return question  # nothing to resolve, so no model call
        reply = str(
            llm.invoke(
                CONDENSE_PROMPT.format(history=self.history(), question=question)
            ).content
        ).strip()
        reply = reply.splitlines()[0].strip() if reply else ""
        # An empty or rambling rewrite is worse than the original question.
        if not reply or len(reply) > 3 * len(question) + 100:
            return question
        return reply
