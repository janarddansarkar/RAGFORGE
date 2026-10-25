"""Generator eval: how good are the answers? Five popular RAG answer metrics, all 0-1.

  Faithfulness        - are the answer's statements supported by the retrieved passages?
                        (low = the model made things up)
  Answer relevancy    - does the answer address the question that was asked?
  Answer correctness  - does the answer agree with the golden answer?
  Semantic similarity - embedding cosine between the answer and the golden answer
  Context recall      - are the golden answer's statements found in the retrieved passages?
                        (low = retrieval missed it, so the generator never had a chance)

A judge LLM (judge_model in config.toml, not the generator) answers short YES/NO-style
questions; nothing leaves your machine. Small judges are noisy, so read the scores as a
comparison between setups, not as absolute truth.

Run:  uv run python eval/generator_eval.py [--limit 10]
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR))

from eval.golden import GoldenQuery
from generator import Generator
from models import make_embeddings, make_llm
from vectorstore import PROJECT_ROOT, load_config

RESULTS_FILE = Path(__file__).parent / "generator_results.json"
REFUSAL = (
    "can't find that in the documents"  # what the generator says when it has no answer
)

VERDICT_SCORE = {"YES": 1.0, "CORRECT": 1.0, "PARTLY": 0.5, "NO": 0.0, "WRONG": 0.0}

SUPPORTED_PROMPT = """Passages:
{passages}

Statement: {statement}

Is the statement supported by the passages? Reply with one word: YES or NO."""

RELEVANCY_PROMPT = """Question: {question}
Answer: {answer}

Does the answer address the question that was asked? Reply with one word: YES, PARTLY or NO."""

CORRECTNESS_PROMPT = """Question: {question}
Reference answer: {reference}
Answer: {answer}

Does the answer agree with the reference answer? Reply with one word: CORRECT, PARTLY or WRONG."""


class Judge:
    """A small LLM that answers one-word questions."""

    def __init__(self, config: dict):
        judge_config = {
            **config,
            "llm_model": config["judge_model"],
            "max_new_tokens": 8,  # a verdict is one word
        }
        self.llm = make_llm(judge_config)

    def ask(self, prompt: str, options: tuple[str, ...]) -> str:
        """The first option the judge names. Unclear replies count as the last (worst) option."""
        reply = str(self.llm.invoke(prompt).content).upper()
        for word in re.findall(r"[A-Z]+", reply):
            if word in options:
                return word
        return options[-1]


class GeneratorMetrics:
    def __init__(self, config: dict):
        self.judge = Judge(config)
        self.embeddings = make_embeddings(config)

    @staticmethod
    def sentences(text: str) -> list[str]:
        return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]

    def share_supported(self, text: str, passages: str) -> float:
        """Fraction of the text's sentences that the passages support."""
        sentences = self.sentences(text)
        if not sentences:
            return 0.0
        verdicts = [
            self.judge.ask(
                SUPPORTED_PROMPT.format(passages=passages, statement=s), ("YES", "NO")
            )
            for s in sentences
        ]
        return verdicts.count("YES") / len(verdicts)

    def faithfulness(self, answer: str, passages: str) -> float:
        if REFUSAL in answer.lower().replace("’", "'"):
            return 1.0  # it claimed nothing, so nothing is unsupported
        return self.share_supported(answer, passages)

    def relevancy(self, question: str, answer: str) -> float:
        prompt = RELEVANCY_PROMPT.format(question=question, answer=answer)
        return VERDICT_SCORE[self.judge.ask(prompt, ("YES", "PARTLY", "NO"))]

    def correctness(self, question: str, answer: str, reference: str) -> float:
        prompt = CORRECTNESS_PROMPT.format(
            question=question, answer=answer, reference=reference
        )
        return VERDICT_SCORE[self.judge.ask(prompt, ("CORRECT", "PARTLY", "WRONG"))]

    def similarity(self, answer: str, reference: str) -> float:
        a, b = np.array(self.embeddings.embed_documents([answer, reference]))
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    def context_recall(self, reference: str, passages: str) -> float:
        return self.share_supported(reference, passages)

    def score(
        self, golden: GoldenQuery, reference: str, answer: str, passages: str
    ) -> dict[str, float]:
        return {
            "faithfulness": self.faithfulness(answer, passages),
            "answer_relevancy": self.relevancy(golden.query, answer),
            "answer_correctness": self.correctness(golden.query, answer, reference),
            "semantic_similarity": self.similarity(answer, reference),
            "context_recall": self.context_recall(reference, passages),
        }


class GeneratorEval:
    def __init__(self, config: dict | None = None, limit: int | None = None):
        self.config = config or load_config()
        with open(PROJECT_ROOT / self.config["golden_file"], encoding="utf-8") as f:
            items = json.load(f)[:limit]
        self.queries = [(GoldenQuery(item), item["golden_ans"]) for item in items]
        self.generator = Generator(self.config)
        self.metrics = GeneratorMetrics(self.config)

    def run(self) -> dict:
        rows = []
        for i, (golden, reference) in enumerate(self.queries, start=1):
            start = time.perf_counter()
            answer, chunks = self.generator.answer(golden.query)
            seconds = time.perf_counter() - start
            passages = "\n\n".join(c.page_content for c in chunks)
            scores = self.metrics.score(golden, reference, answer, passages)
            rows.append(
                {
                    "id": golden.id,
                    "question": golden.query,
                    "answer": answer,
                    "reference": reference,
                    "seconds": seconds,
                    **scores,
                }
            )
            print(
                f"{i}/{len(self.queries)} {golden.id}: "
                + " ".join(f"{k.split('_')[-1]}={v:.2f}" for k, v in scores.items()),
                file=sys.stderr,
            )
        names = [
            "faithfulness",
            "answer_relevancy",
            "answer_correctness",
            "semantic_similarity",
            "context_recall",
        ]
        average = {n: sum(r[n] for r in rows) / len(rows) for n in names}
        average["seconds_per_answer"] = sum(r["seconds"] for r in rows) / len(rows)
        return {
            "generator": self.config["llm_model"],
            "judge": self.config["judge_model"],
            "questions": len(rows),
            "average": average,
            "rows": rows,
        }

    @staticmethod
    def print_summary(results: dict) -> None:
        print(
            f"\ngenerator: {results['generator']}   judge: {results['judge']}   "
            f"questions: {results['questions']}\n"
        )
        for name, value in results["average"].items():
            print(f"  {name:22} {value:6.2f}")


# RUN IT ---- python eval/generator_eval.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--limit", type=int, help="only the first N golden questions")
    args = parser.parse_args()

    evaluation = GeneratorEval(limit=args.limit)
    results = evaluation.run()
    evaluation.print_summary(results)
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {RESULTS_FILE.relative_to(PROJECT_DIR)}")
