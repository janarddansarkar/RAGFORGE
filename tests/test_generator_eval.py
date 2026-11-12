"""Generator eval helpers that need no model: sentence splitting and verdict parsing."""

from types import SimpleNamespace

from eval.generator_eval import GeneratorMetrics, Judge


def test_sentences_split_on_end_punctuation():
    text = "LoRA freezes the weights. It adds rank r! Why? Because it is cheap."
    assert GeneratorMetrics.sentences(text) == [
        "LoRA freezes the weights.",
        "It adds rank r!",
        "Why?",
        "Because it is cheap.",
    ]


def test_sentences_of_empty_text():
    assert GeneratorMetrics.sentences("   ") == []


def judge_that_says(reply: str) -> Judge:
    judge = Judge.__new__(Judge)  # skip loading a model
    judge.llm = SimpleNamespace(
        invoke=lambda prompt, **kw: SimpleNamespace(content=reply)
    )
    return judge


def test_judge_takes_the_first_option_it_names():
    assert (
        judge_that_says("Correct.").ask("p", ("CORRECT", "PARTLY", "WRONG"))
        == "CORRECT"
    )
    assert judge_that_says("yes, because...").ask("p", ("YES", "NO")) == "YES"


def test_judge_does_not_mistake_incorrect_for_correct():
    assert (
        judge_that_says("INCORRECT").ask("p", ("CORRECT", "PARTLY", "WRONG")) == "WRONG"
    )


def test_unclear_reply_counts_as_the_worst_option():
    assert judge_that_says("hmm").ask("p", ("YES", "PARTLY", "NO")) == "NO"
