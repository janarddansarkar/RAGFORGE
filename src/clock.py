"""Today's date, for the prompts that need it.

A model cannot know what day it is. Some questions depend on it: a resume that says
"Feb 2025 - Present", or "how long has she worked there?". Without the date the model
guesses (or refuses), so the answer prompt is told the date (see generator.ANSWER_PROMPT).
"""

from datetime import date


class Clock:
    @staticmethod
    def today() -> str:
        """Today's date in words, e.g. "10 November 2026"."""
        today = date.today()
        return f"{today.day} {today:%B %Y}"
