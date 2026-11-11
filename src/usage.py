"""Usage: every model call made with the user's API key, with its tokens and cost.

A UsageTracker lives for one browser session. The model wrappers in models.py record into it, and
the app shows the totals for a turn and for the whole session. Costs are estimates: LiteLLM
multiplies the tokens by its price list, and a model it has no price for counts as "unpriced".
"""

from dataclasses import dataclass, field
from threading import Lock

ROLE_NAMES = {
    "answer": "Answers",
    "retry": "Retries",
    "notes": "Chunk notes",
    "guard": "Input guard",
    "judge": "Judge (RAG triad)",
    "rerank": "Reranker",
    "embeddings": "Embeddings",
    "voice": "Voice input",
}


@dataclass(frozen=True)
class Call:
    role: str  # what the call was for: answer, guard, judge, ...
    model: str
    input_tokens: int
    output_tokens: int
    cost: float | None  # dollars; None when LiteLLM has no price for the model


@dataclass(frozen=True)
class Summary:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0  # the priced calls only
    unpriced: int = 0  # calls with no known price
    by_role: dict = field(default_factory=dict)  # role -> Summary

    def describe(self) -> str:
        """One line: "4 calls, 5,210 tokens in + 312 out, about $0.0123"."""
        if not self.calls:
            return "no model calls"
        return (
            f"{self.calls} call{'s' if self.calls != 1 else ''}, "
            f"{self.input_tokens:,} tokens in + {self.output_tokens:,} out, {self.money()}"
        )

    def money(self) -> str:
        text = f"about ${self.cost:.4f}"
        if self.unpriced:
            text += (
                f" + {self.unpriced} unpriced call{'s' if self.unpriced != 1 else ''}"
            )
        return text


def summarize(calls: list[Call], split: bool = True) -> Summary:
    priced = [c.cost for c in calls if c.cost is not None]
    by_role = {}
    if split:
        for role in dict.fromkeys(c.role for c in calls):
            by_role[role] = summarize([c for c in calls if c.role == role], split=False)
    return Summary(
        calls=len(calls),
        input_tokens=sum(c.input_tokens for c in calls),
        output_tokens=sum(c.output_tokens for c in calls),
        cost=sum(priced),
        unpriced=len(calls) - len(priced),
        by_role=by_role,
    )


class UsageTracker:
    def __init__(self):
        self.calls: list[Call] = []
        self._lock = Lock()  # the reranker makes several calls at once

    def record(
        self, role: str, model: str, response, call_type: str | None = None
    ) -> None:
        """Note one call, from LiteLLM's response. Never raises: a failure to count must not
        break an answer."""
        try:
            import litellm

            usage = getattr(response, "usage", None)
            input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            extra = {"call_type": call_type} if call_type else {}
            try:
                cost = float(
                    litellm.completion_cost(
                        completion_response=response, model=model, **extra
                    )
                )
            except Exception:  # noqa: BLE001 - e.g. a model that is not in LiteLLM's price list
                cost = None
            call = Call(role, model, input_tokens, output_tokens, cost)
        except Exception:  # noqa: BLE001
            call = Call(role, model, 0, 0, None)
        with self._lock:
            self.calls.append(call)

    def mark(self) -> int:
        """A place in the list: summarize(tracker.calls[mark:]) is everything after it."""
        return len(self.calls)
