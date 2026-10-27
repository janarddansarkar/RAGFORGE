"""Short-term memory: the last few turns of the conversation, so follow-ups work.

Two jobs:
  - history()  the recent turns, shown to the answer model for context
  - ConversationMemory.standalone() rewrites a follow-up ("and for QLoRA?") into a question
    that makes sense on its own, because the search only sees the question it is given.

Memory is never a source: answers still come only from the retrieved passages.
"""

import re

CONDENSE_PROMPT = """Rewrite the follow-up question so it makes sense without the conversation. \
Only fill in what the follow-up leaves out (such as "it" or "that one"). \
If the follow-up already names its own subject, copy it EXACTLY as written, \
even when the conversation was about something else. Reply with the question only.

Example 1
Conversation:
User: What is NF4?
Assistant: A 4-bit data type.
Follow-up: Which paper introduced it?
Question: Which paper introduced NF4?

Example 2
Conversation:
User: What is NF4?
Assistant: A 4-bit data type.
Follow-up: What rank r does LoRA use?
Question: What rank r does LoRA use?

Conversation:
{history}
Follow-up: {question}
Question:"""

# A citation such as "[LoRA p.10]" never belongs in a question: the model has copied an answer.
CITATION = re.compile(r"\[[^\]]*\bp\.\d+[^\]]*\]")

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

    def replace_last_answer(self, answer: str) -> None:
        """A retry replaces the answer of the latest turn instead of adding a turn."""
        if self.turns:
            self.turns[-1] = (self.turns[-1][0], answer)

    def clear(self) -> None:
        self.turns = []

    def history(self, skip_last: bool = False) -> str:
        """The recent turns as text (skip_last: without the latest one, which a retry replaces)."""
        lines = []
        for question, answer in self.turns[: -1 if skip_last else None]:
            if len(answer) > MAX_ANSWER_CHARS:
                answer = answer[:MAX_ANSWER_CHARS] + " ..."
            lines += [f"User: {question}", f"Assistant: {answer}"]
        return "\n".join(lines)

    def standalone(self, question: str, llm) -> str:
        """The question rewritten to stand on its own (unchanged on the first turn)."""
        if not self.turns:
            return question  # nothing to resolve, so no model call
        if any(question.lower() == asked.lower() for asked, _ in self.turns):
            return question  # asked before word for word: it already stands on its own
        reply = str(
            llm.invoke(
                CONDENSE_PROMPT.format(history=self.history(), question=question)
            ).content
        ).strip()
        reply = reply.splitlines()[0].strip() if reply else ""
        # An empty, rambling or answer-like rewrite is worse than the original question.
        if not reply or len(reply) > 3 * len(question) + 100 or CITATION.search(reply):
            return question
        return reply
