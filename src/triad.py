"""RAG triad, scored live with deepeval: the three checks that link question, context and answer.

  Context relevance - are the retrieved chunks relevant to the question?   (question -> context)
  Groundedness      - is every claim in the answer supported by the chunks? (context -> answer)
  Answer relevance  - does the answer address the question?                 (question -> answer)

It needs no golden data: give it the question asked, the top-k chunks retrieved and the answer
generated (the web UI does this after every answer). The judge is a hosted model (judge_model
in config.toml). Judges are noisy; read the scores as a guide.

Run:  uv run python src/triad.py paper.pdf "What rank r does LoRA use?" [--judge ID]
      (asks the question over the PDF, then scores the answer; set OPENAI_API_KEY or the key of
      the provider you use)
"""

import argparse
import json
import os
from pathlib import Path
from typing import ClassVar

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")  # stay local

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from langchain_core.documents import Document

from chunking import chunk_body
from models import make_llm
from vectorstore import load_config


class Judge(DeepEvalBaseLLM):
    """Lets deepeval use a hosted model as its judge (deepeval defaults to OpenAI).
    The reply is forced to match the metric's JSON schema, so JSON rarely breaks."""

    def __init__(self, model: str, config: dict):
        self.model_id = model
        self.chat = make_llm(
            {
                **config,
                "llm_model": model,
                "max_new_tokens": 4096,  # a verdict per statement in 5 chunks is long; cut-off JSON is invalid
            },
            role="judge",
        )

    def load_model(self):
        return self

    def get_model_name(self) -> str:
        return self.model_id

    def generate(self, prompt: str, schema=None) -> str:
        """The metrics ask for JSON in the prompt; deepeval reads it from this text."""
        reply = str(self.chat.invoke(prompt, schema=schema).content)
        if schema is not None and not self.is_json(reply):
            # Rare, e.g. a reply cut off at the token limit: ask the model to repair it.
            repair = (
                f"{reply}\n\nThe text above should be one valid JSON object but it is "
                "not valid JSON. Rewrite it as one valid JSON object only."
            )
            reply = str(self.chat.invoke(repair, schema=schema).content)
        return reply

    @staticmethod
    def is_json(text: str) -> bool:
        try:
            json.loads(text[text.find("{") : text.rfind("}") + 1])
            return True
        except ValueError:
            return False

    async def a_generate(self, prompt: str, schema=None) -> str:
        return self.generate(prompt, schema)


class TriadEvaluator:
    # name shown to the user -> deepeval metric
    METRICS: ClassVar[dict] = {
        "Context relevance": ContextualRelevancyMetric,
        "Groundedness": FaithfulnessMetric,
        "Answer relevance": AnswerRelevancyMetric,
    }

    def __init__(
        self,
        judge_model: str | None = None,
        threshold: float = 0.5,
        config: dict | None = None,
    ):
        config = config or load_config()
        self.judge = Judge(judge_model or config["judge_model"], config)
        self.threshold = threshold

    def evaluate(self, question: str, answer: str, chunks: list[Document]) -> dict:
        """Score one turn. Returns {name: {"score": 0-1 or None, "reason": str}}."""
        case = LLMTestCase(
            input=question,
            actual_output=answer,
            # the document's own text, not the context note: the answer must be grounded in it
            retrieval_context=[chunk_body(chunk) for chunk in chunks],
        )
        results = {}
        for name, metric_class in self.METRICS.items():
            metric = metric_class(
                threshold=self.threshold,
                model=self.judge,
                include_reason=True,
                async_mode=False,
                verbose_mode=False,
            )
            try:
                metric.measure(case, _show_indicator=False)
                results[name] = {"score": metric.score, "reason": metric.reason or ""}
            except Exception as error:  # noqa: BLE001 - a weak judge can break one metric
                results[name] = {"score": None, "reason": f"judge failed: {error}"}
        return results


# TRY IT ---- python src/triad.py paper.pdf "your question"
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("pdf", type=Path, help="path to the PDF")
    parser.add_argument("question", nargs="+")
    parser.add_argument(
        "--judge", help="Hosted model for the judge (default: judge_model)"
    )
    args = parser.parse_args()

    from rag import DocumentRAG

    question = " ".join(args.question)
    rag = DocumentRAG(
        args.pdf, settings={"max_pages": 0}
    )  # the command line has no page limit
    answer = "".join(rag.ask_stream(question)).strip()
    print(f"\nQ: {question}\nA: {answer}\n")
    judge = TriadEvaluator(args.judge, config=rag.config)
    for name, result in judge.evaluate(question, answer, rag.last_chunks).items():
        score = "n/a" if result["score"] is None else f"{result['score']:.2f}"
        print(f"{name:18} {score}   {result['reason']}")
