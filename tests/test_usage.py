"""Usage tracking: tokens, cost per call, totals per turn, and what a missing price does."""

from types import SimpleNamespace

import litellm
import pytest

from models import LiteLLMChat, LiteLLMEmbeddings, make_llm
from usage import Call, UsageTracker, summarize


def reply(text="hi", prompt=10, completion=5):
    message = SimpleNamespace(content=text)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, logprobs=None)],
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion),
    )


@pytest.fixture
def priced(monkeypatch):
    """Every call costs $0.01, as if LiteLLM knew the model."""
    monkeypatch.setattr(litellm, "completion_cost", lambda **kwargs: 0.01)


def test_a_call_is_recorded_with_its_tokens_and_cost(monkeypatch, priced):
    monkeypatch.setattr(litellm, "completion", lambda **kwargs: reply())
    tracker = UsageTracker()
    LiteLLMChat("openai/gpt-4o", tracker=tracker, role="guard").invoke("x")
    assert tracker.calls == [Call("guard", "openai/gpt-4o", 10, 5, 0.01)]


def test_the_role_comes_from_make_llm_and_the_trackers_from_the_config(
    monkeypatch, priced
):
    monkeypatch.setattr(litellm, "completion", lambda **kwargs: reply())
    tracker = UsageTracker()
    config = {"llm_model": "openai/m", "max_new_tokens": 5, "usage_tracker": tracker}
    make_llm(config, role="judge").invoke("x")
    make_llm(config).invoke("x")
    assert [c.role for c in tracker.calls] == ["judge", "answer"]


def test_a_streamed_answer_is_counted_when_it_ends(monkeypatch, priced):
    def piece(text):
        delta = SimpleNamespace(content=text)
        return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])

    monkeypatch.setattr(
        litellm, "completion", lambda **kwargs: [piece("a"), piece("b")]
    )
    monkeypatch.setattr(
        litellm,
        "stream_chunk_builder",
        lambda parts, messages: reply(prompt=100, completion=2),
    )
    tracker = UsageTracker()
    chat = LiteLLMChat("openai/m", tracker=tracker)
    assert "".join(chat.stream("x")) == "ab"
    assert tracker.calls[0].input_tokens == 100 and tracker.calls[0].output_tokens == 2


def test_embedding_calls_are_counted(monkeypatch, priced):
    monkeypatch.setattr(
        litellm,
        "embedding",
        lambda model, input, api_key=None: SimpleNamespace(
            data=[{"embedding": [1.0]}] * len(input),
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=0),
        ),
    )
    tracker = UsageTracker()
    LiteLLMEmbeddings("openai/e", tracker=tracker).embed_documents(["a"] * 40)
    assert [c.input_tokens for c in tracker.calls] == [7, 7]  # two batches: 32 + 8


def test_a_model_without_a_price_is_counted_but_unpriced(monkeypatch):
    def no_price(**kwargs):
        raise ValueError("model not in the price list")

    monkeypatch.setattr(litellm, "completion_cost", no_price)
    tracker = UsageTracker()
    tracker.record("answer", "openai/brand-new", reply())
    total = summarize(tracker.calls)
    assert total.calls == 1 and total.unpriced == 1 and total.cost == 0.0
    assert "1 unpriced call" in total.money()


def test_a_failure_to_count_never_breaks_the_answer(monkeypatch):
    tracker = UsageTracker()
    tracker.record("answer", "openai/m", None)  # no response at all
    assert tracker.calls[0].input_tokens == 0 and tracker.calls[0].cost is None


def test_summary_totals_and_the_breakdown_by_component():
    calls = [
        Call("answer", "m", 100, 20, 0.5),
        Call("guard", "m", 10, 2, 0.1),
        Call("guard", "m", 10, 2, None),
    ]
    total = summarize(calls)
    assert (total.calls, total.input_tokens, total.output_tokens) == (3, 120, 24)
    assert total.cost == pytest.approx(0.6) and total.unpriced == 1
    assert total.by_role["guard"].calls == 2
    assert (
        total.by_role["answer"].describe().startswith("1 call, 100 tokens in + 20 out")
    )
    assert summarize([]).describe() == "no model calls"


def test_mark_gives_the_calls_made_since(monkeypatch, priced):
    tracker = UsageTracker()
    tracker.record("answer", "m", reply())
    start = tracker.mark()
    tracker.record("guard", "m", reply())
    assert [c.role for c in tracker.calls[start:]] == ["guard"]
