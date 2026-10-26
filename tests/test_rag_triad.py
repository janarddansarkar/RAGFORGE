"""RAG triad judge with a fake Ollama chat: the JSON check, schema forwarding and one repair."""

from types import SimpleNamespace

from eval.rag_triad_eval import Judge


class FakeChat:
    def __init__(self, replies: list[str]):
        self.replies = replies
        self.prompts: list[str] = []
        self.schemas: list = []

    def invoke(self, prompt: str, max_new_tokens=None, schema=None):
        self.prompts.append(prompt)
        self.schemas.append(schema)
        return SimpleNamespace(content=self.replies[len(self.prompts) - 1])


def judge_with(replies: list[str]) -> Judge:
    judge = Judge.__new__(Judge)
    judge.chat = FakeChat(replies)
    return judge


def test_is_json_ignores_chatter_around_the_object():
    assert Judge.is_json('Sure! {"verdicts": []} Hope this helps')
    assert not Judge.is_json('{"verdicts": [')
    assert not Judge.is_json("no json at all")


def test_valid_json_is_returned_after_one_call_and_the_schema_is_forwarded():
    judge = judge_with(['{"score": 1}'])
    assert judge.generate("rate it", schema=dict) == '{"score": 1}'
    assert len(judge.chat.prompts) == 1
    assert judge.chat.schemas == [
        dict
    ]  # Ollama is told the schema, so JSON stays valid


def test_broken_json_is_repaired_once():
    judge = judge_with(['{"score": ', '{"score": 1}'])
    assert judge.generate("rate it", schema=dict) == '{"score": 1}'
    assert (
        '{"score": ' in judge.chat.prompts[1]
    )  # the repair prompt quotes the bad reply


def test_without_a_schema_the_prompt_is_untouched():
    judge = judge_with(["plain text"])
    assert judge.generate("hello") == "plain text"
    assert judge.chat.prompts == ["hello"]
    assert judge.chat.schemas == [None]
