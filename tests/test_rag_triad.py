"""RAG triad judge wrapper with a fake model: the JSON check and the one repair attempt."""

from types import SimpleNamespace

from eval.rag_triad_eval import LocalJudge


class FakeChat:
    def __init__(self, replies: list[str]):
        self.replies = replies
        self.prompts: list[str] = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.replies[len(self.prompts) - 1])


def judge_with(replies: list[str]) -> LocalJudge:
    judge = LocalJudge.__new__(LocalJudge)
    judge.chat = FakeChat(replies)
    return judge


def test_is_json_ignores_chatter_around_the_object():
    assert LocalJudge.is_json('Sure! {"verdicts": []} Hope this helps')
    assert not LocalJudge.is_json('{"verdicts": [')
    assert not LocalJudge.is_json("no json at all")


def test_valid_json_is_returned_after_one_call():
    judge = judge_with(['{"score": 1}'])
    assert judge.generate("rate it", schema=dict) == '{"score": 1}'
    assert len(judge.chat.prompts) == 1


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
