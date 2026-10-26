"""Short-term memory with a fake model: the window, the history text and the follow-up rewrite."""

from types import SimpleNamespace

from generator import ANSWER_PROMPT, Generator
from memory import MAX_ANSWER_CHARS, ConversationMemory


class FakeLLM:
    def __init__(self, reply: str):
        self.reply = reply
        self.prompts: list[str] = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.reply)


def test_keeps_only_the_last_turns():
    memory = ConversationMemory(max_turns=2)
    for i in range(4):
        memory.add(f"q{i}", f"a{i}")
    assert memory.turns == [("q2", "a2"), ("q3", "a3")]


def test_history_text_and_long_answer_is_cut():
    memory = ConversationMemory()
    memory.add("What is NF4?", "x" * 2000)
    lines = memory.history().splitlines()
    assert lines[0] == "User: What is NF4?"
    assert len(lines[1]) < MAX_ANSWER_CHARS + 50 and lines[1].endswith("...")


def test_first_question_makes_no_model_call():
    llm = FakeLLM("should not be used")
    assert ConversationMemory().standalone("What is NF4?", llm) == "What is NF4?"
    assert llm.prompts == []


def test_follow_up_is_rewritten_using_the_history():
    memory = ConversationMemory()
    memory.add("What is NF4?", "A 4-bit data type.")
    llm = FakeLLM("Which paper introduced the NF4 data type?")
    assert (
        memory.standalone("Which paper introduced it?", llm)
        == "Which paper introduced the NF4 data type?"
    )
    assert "What is NF4?" in llm.prompts[0]


def test_empty_or_rambling_rewrite_falls_back_to_the_question():
    memory = ConversationMemory()
    memory.add("q", "a")
    assert memory.standalone("and QLoRA?", FakeLLM("")) == "and QLoRA?"
    assert memory.standalone("and QLoRA?", FakeLLM("word " * 200)) == "and QLoRA?"


def test_answer_prompt_is_unchanged_without_history():
    generator = Generator.__new__(Generator)
    generator.retriever = SimpleNamespace(llm=FakeLLM("ok"))
    generator.answer_from("Q?", [])
    assert generator.retriever.llm.prompts[0] == ANSWER_PROMPT.format(
        passages="", history="", question="Q?"
    )
    assert "Conversation so far" not in generator.retriever.llm.prompts[0]
    generator.answer_from("Q?", [], history="User: a\nAssistant: b")
    assert "Conversation so far" in generator.retriever.llm.prompts[1]


def test_chat_stream_gives_the_same_answer_as_chat_and_updates_memory():
    class StreamingLLM(FakeLLM):
        def stream(self, prompt: str):
            self.prompts.append(prompt)
            yield from ("It is ", "r = 8 ", "[LoRA p.4]")

    generator = Generator.__new__(Generator)
    generator.guard = generator.cache = None
    generator.retriever = SimpleNamespace(
        llm=StreamingLLM("unused"), retrieve=lambda query, method, k: []
    )
    generator.config = {"default_method": "hybrid"}
    memory = ConversationMemory()
    pieces = list(generator.chat_stream("What rank?", memory))
    assert pieces == ["It is ", "r = 8 ", "[LoRA p.4]"]
    assert memory.turns == [("What rank?", "It is r = 8 [LoRA p.4]")]
    assert generator.last_query == "What rank?"
    answer, chunks, query = generator.chat("What rank?", ConversationMemory())
    assert answer == "It is r = 8 [LoRA p.4]" and chunks == [] and query == "What rank?"
