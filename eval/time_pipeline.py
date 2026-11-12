"""Where does the time go? Seconds per stage for a few questions (no cache).

    uv run python eval/time_pipeline.py              # 3 questions
    uv run python eval/time_pipeline.py --n 5 --no-triad

Stages: guard, search (hybrid, no reranker), rerank (the extra time of the reranker),
answer (the model writing the answer from the chunks), triad (the judge).
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR))

from eval.answering import answer_from
from generator import Generator
from triad import TriadEvaluator
from vectorstore import load_config

QUESTIONS = [
    "What rank r does LoRA use?",
    "What is NF4 and why is it used?",
    "How does the Transformer use attention instead of recurrence?",
    "Which optimizer does QLoRA use for fine-tuning?",
    "What is double quantization?",
]
STAGES = ("guard", "search", "rerank", "answer", "triad")


def seconds(function, *args):
    start = time.perf_counter()
    result = function(*args)
    return time.perf_counter() - start, result


def measure(config: dict, n: int = 3, with_triad: bool = True) -> dict:
    """Seconds per stage for the first n questions: {"rows": [...], "mean": {stage: seconds}}."""
    config = {**config, "use_cache": False, "input_guard": True}
    generator = Generator(config)
    triad = TriadEvaluator(config=config) if with_triad else None

    # One throwaway run so connection set-up is not counted against the first question.
    generator.retriever.retrieve(QUESTIONS[0])
    generator.guard.check(QUESTIONS[0])

    rows = []
    for question in QUESTIONS[:n]:
        guard, _ = seconds(generator.guard.check, question)
        plain, _ = seconds(generator.retriever.retrieve, question, "hybrid")
        full, chunks = seconds(generator.retriever.retrieve, question, "hybrid_rerank")
        answer_time, answer = seconds(answer_from, generator, question, chunks)
        scoring = 0.0
        if triad:
            scoring, _ = seconds(triad.evaluate, question, answer, chunks)
        rows.append(
            dict(zip(STAGES, (guard, plain, full - plain, answer_time, scoring)))
        )
    mean = {stage: sum(r[stage] for r in rows) / len(rows) for stage in STAGES}
    return {"questions": QUESTIONS[:n], "rows": rows, "mean": mean}


def print_table(result: dict) -> None:
    for question, row in zip(result["questions"], result["rows"]):
        print(f"{question[:48]:50} " + "  ".join(f"{s} {row[s]:5.1f}" for s in STAGES))
    print(f"{'MEAN':50} " + "  ".join(f"{s} {result['mean'][s]:5.1f}" for s in STAGES))


# RUN IT ---- python eval/time_pipeline.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=3, help="how many questions")
    parser.add_argument("--no-triad", action="store_true", help="skip the judge (slow)")
    args = parser.parse_args()
    print_table(measure(load_config(), args.n, not args.no_triad))
