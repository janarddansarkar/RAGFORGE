"""The retry after a thumbs down: which reasons search again, what the prompt says, what is kept."""

from types import SimpleNamespace

from langchain_core.documents import Document

from generator import Generator
from memory import ConversationMemory
from retriever import Retriever

OLD = [Document(page_content="old chunk", metadata={"source": "a.pdf", "page": 1})]
NEW = [Document(page_content="new chunk", metadata={"source": "a.pdf", "page": 2})]
PREVIOUS = {
    "question": "What rank r?",
    "query": "LoRA rank r",
    "text": "r = 8 [a p.1]",
    "chunks": OLD,
}


def make_generator():
    """A generator with fake search and model that record what they were asked."""
    generator = Generator.__new__(Generator)
    generator.config = {"top_k": 5, "max_new_tokens": 256}
    generator.seen = SimpleNamespace(searches=[], prompts=[], limits=[], added=[])

    def wider(query, k):
        generator.seen.searches.append((query, k))
        return NEW

    def stream(prompt, max_new_tokens=None):
        generator.seen.prompts.append(prompt)
        generator.seen.limits.append(max_new_tokens)
        yield from ("new ", "answer")

    generator.retriever = SimpleNamespace(
        retrieve_wider=wider, llm=SimpleNamespace(stream=stream)
    )
    generator.cache = SimpleNamespace(
        add=lambda q, a, c: generator.seen.added.append((q, a, c)) or {"answer": a}
    )
    return generator


def memory_with_turn():
    memory = ConversationMemory()
    memory.add("What rank r?", "r = 8 [a p.1]")
    return memory


def retry(reasons):
    generator, memory = make_generator(), memory_with_turn()
    pieces = list(generator.retry_stream(PREVIOUS, reasons, memory))
    return generator, memory, "".join(pieces)


def test_wrong_source_searches_again_wider_and_uses_the_new_chunks():
    generator, _, answer = retry(["Wrong or missing source"])
    assert answer == "new answer"
    assert generator.seen.searches == [("LoRA rank r", 7)]  # k + 2
    assert "new chunk" in generator.seen.prompts[0]
    assert generator.last_chunks == NEW


def test_no_reason_also_searches_again():
    generator, _, _ = retry([])
    assert len(generator.seen.searches) == 1
    assert "no reason given" in generator.seen.prompts[0]


def test_answer_reasons_keep_the_same_chunks_and_add_instructions():
    generator, _, _ = retry(["Too long", "Made something up"])
    assert generator.seen.searches == []  # no new search
    prompt = generator.seen.prompts[0]
    assert "old chunk" in prompt and "at most two sentences" in prompt
    assert "leave out any detail" in prompt
    assert (
        "r = 8 [a p.1]" in prompt
    )  # the rejected answer is shown, to avoid repeating it


def test_incomplete_allows_a_longer_answer_and_too_long_a_shorter_one():
    assert retry(["Incomplete"])[0].seen.limits == [512]
    assert retry(["Too long"])[0].seen.limits == [85]
    assert retry(["Made something up"])[0].seen.limits == [256]


def test_the_extra_instructions_come_before_the_question():
    prompt = retry(["Too long"])[0].seen.prompts[0]
    assert prompt.index("at most two sentences") < prompt.index(
        "Question: What rank r?"
    )


def test_the_new_answer_is_cached_and_replaces_the_old_one_in_the_memory():
    generator, memory, _ = retry(["Wrong citation"])
    assert generator.seen.added == [("LoRA rank r", "new answer", NEW)]
    assert generator.cache_entry == {"answer": "new answer"}
    assert memory.turns == [("What rank r?", "new answer")]  # one turn, new answer
    assert "r = 8" not in memory.history()


def test_the_retry_prompt_leaves_the_rejected_turn_out_of_the_history():
    generator, memory = make_generator(), memory_with_turn()
    memory.add("And QLoRA?", "4-bit [b p.2]")
    previous = {**PREVIOUS, "question": "And QLoRA?", "text": "4-bit [b p.2]"}
    list(generator.retry_stream(previous, ["Too long"], memory))
    prompt = generator.seen.prompts[0]
    assert "User: What rank r?" in prompt  # the earlier turn stays for context
    assert prompt.count("4-bit [b p.2]") == 1  # the rejected answer, shown once


def test_a_wider_search_doubles_fetch_k_for_one_search_only():
    retriever = Retriever.__new__(Retriever)
    retriever.config = {"fetch_k": 20}
    seen = []
    retriever.multi_query_hybrid_rerank = lambda q, k: (
        seen.append((retriever.config["fetch_k"], k)) or NEW
    )
    assert retriever.retrieve_wider("q", 7) == NEW
    assert seen == [(40, 7)]
    assert retriever.config["fetch_k"] == 20  # back to normal afterwards
