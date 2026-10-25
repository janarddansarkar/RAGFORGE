"""Generator eval: how good are the answers? Five popular RAG answer metrics, all 0-1.

  Faithfulness        - are the answer's statements supported by the retrieved passages?
                        (low = the model made things up)
  Answer relevancy    - the judge writes the question the answer responds to; how close
                        is it (embedding cosine) to the real question?
  Answer correctness  - does the answer agree with the golden answer?
  Semantic similarity - embedding cosine between the answer and the golden answer
  Context recall      - are the golden answer's statements found in the retrieved passages?
                        (low = retrieval missed it, so the generator never had a chance)

A judge LLM (judge_model in config.toml, not the generator) answers short YES/NO-style
questions; nothing leaves your machine. Small judges are noisy, so read the scores as a
comparison between setups, not as absolute truth.

Run:  uv run python eval/generator_eval.py [--limit 10] [--oracle]
--oracle gives the generator the golden pages instead of retrieved chunks, so you see
the generator on its own (compare with a normal run to see what retrieval costs).
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

RUNS_DIR = Path(__file__).parent / "generator_runs"  # one JSON file per run
REFUSAL = (
    "can't find that in the documents"  # what the generator says when it has no answer
)

VERDICT_SCORE = {"YES": 1.0, "CORRECT": 1.0, "PARTLY": 0.5, "NO": 0.0, "WRONG": 0.0}

SUPPORTED_PROMPT = """Passages:
{passages}

Statement: {statement}

Is the statement supported by the passages? Reply with one word: YES or NO."""

QUESTION_PROMPT = """Here is an answer from a question-answering system:

{answer}

Write the one question this answer responds to. Reply with the question only."""

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

    def write_question(self, answer: str) -> str:
        """The question the judge thinks this answer responds to."""
        reply = self.llm.invoke(
            QUESTION_PROMPT.format(answer=answer), max_new_tokens=40
        )
        return str(reply.content).strip()


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

    def cosine(self, text_a: str, text_b: str) -> float:
        a, b = np.array(self.embeddings.embed_documents([text_a, text_b]))
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    def relevancy(self, question: str, answer: str) -> float:
        """RAGAS-style: how close is the question the answer *seems* to answer to the real one?"""
        return max(0.0, self.cosine(question, self.judge.write_question(answer)))

    def correctness(self, question: str, answer: str, reference: str) -> float:
        prompt = CORRECTNESS_PROMPT.format(
            question=question, answer=answer, reference=reference
        )
        return VERDICT_SCORE[self.judge.ask(prompt, ("CORRECT", "PARTLY", "WRONG"))]

    def context_recall(self, reference: str, passages: str) -> float:
        return self.share_supported(reference, passages)

    def score(
        self, golden: GoldenQuery, reference: str, answer: str, passages: str
    ) -> dict[str, float]:
        return {
            "faithfulness": self.faithfulness(answer, passages),
            "answer_relevancy": self.relevancy(golden.query, answer),
            "answer_correctness": self.correctness(golden.query, answer, reference),
            "semantic_similarity": self.cosine(answer, reference),
            "context_recall": self.context_recall(reference, passages),
        }


class GeneratorEval:
    def __init__(
        self, config: dict | None = None, limit: int | None = None, oracle: bool = False
    ):
        self.config = config or load_config()
        self.oracle = oracle
        with open(PROJECT_ROOT / self.config["golden_file"], encoding="utf-8") as f:
            items = json.load(f)[:limit]
        self.queries = [(GoldenQuery(item), item["golden_ans"]) for item in items]
        self.generator = Generator(self.config)
        self.metrics = GeneratorMetrics(self.config)

    def run(self) -> dict:
        rows = []
        for i, (golden, reference) in enumerate(self.queries, start=1):
            start = time.perf_counter()
            if self.oracle:
                # The golden pages themselves, so the retriever can't be the cause of a bad answer.
                gold = [c for c in self.generator.retriever.chunks if golden.match(c)]
                answer, chunks = self.generator.answer_from(
                    golden.query, gold[: self.config["top_k"]]
                )
            else:
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
            "context": "oracle (golden pages)" if self.oracle else "retrieved",
            "settings": {  # what produced these numbers, so runs can be compared later
                key: self.config[key]
                for key in (
                    "embedding_model",
                    "reranker_model",
                    "default_method",
                    "top_k",
                    "chunking_strategy",
                    "contextual",
                    "max_new_tokens",
                )
            },
            "questions": len(rows),
            "average": average,
            "rows": rows,
        }

    @staticmethod
    def save(results: dict) -> Path:
        """Write this run to eval/generator_runs/<date>_<time>.json (never overwrites)."""
        RUNS_DIR.mkdir(exist_ok=True)
        suffix = "_oracle" if results["context"].startswith("oracle") else ""
        file = RUNS_DIR / f"{time.strftime('%Y-%m-%d_%H%M%S')}{suffix}.json"
        file.write_text(json.dumps(results, indent=2))
        return file

    @staticmethod
    def print_summary(results: dict) -> None:
        print(
            f"\ngenerator: {results['generator']}   judge: {results['judge']}   "
            f"context: {results['context']}   "
            f"questions: {results['questions']}\n"
        )
        for name, value in results["average"].items():
            print(f"  {name:22} {value:6.2f}")


# RUN IT ---- python eval/generator_eval.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--limit", type=int, help="only the first N golden questions")
    parser.add_argument(
        "--oracle", action="store_true", help="give the generator the golden pages"
    )
    args = parser.parse_args()

    evaluation = GeneratorEval(limit=args.limit, oracle=args.oracle)
    results = evaluation.run()
    evaluation.print_summary(results)
    print(f"\nSaved to {evaluation.save(results).relative_to(PROJECT_DIR)}")
