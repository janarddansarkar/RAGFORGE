"""The date the answer prompt gives the model."""

from datetime import date

from clock import Clock
from generator import Generator


def test_today_is_written_with_the_month_name():
    today = date.today()
    assert Clock.today() == f"{today.day} {today:%B %Y}"


def test_the_answer_prompt_tells_the_model_the_date():
    prompt = Generator.__new__(Generator).build_prompt("Q?", [])
    assert f"Today's date is {Clock.today()}:" in prompt


def test_a_retry_prompt_has_the_date_too():
    prompt = Generator.__new__(Generator).build_prompt("Q?", [], extra="Do not repeat it.")
    assert f"Today's date is {Clock.today()}:" in prompt
