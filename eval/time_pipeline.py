"""Where does the time go? Seconds per stage for a few questions (no cache).

    uv run python eval/time_pipeline.py              # 3 questions
    uv run python eval/time_pipeline.py --n 5 --no-triad

Stages: guard, retrieve (search only), rerank (the extra time of the reranker),
answer (the whole answer, and time to the first text when streaming), triad (the judge).
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR))

from generator import Generator
from vectorstore import load_config

QUESTIONS = [
    "What rank r does LoRA use?",
    "What is NF4 and why is it used?",
    "How does the Transformer use attention instead of recurrence?",
    "Which optimizer does QLoRA use for fine-tuning?",
    "What is double quantization?",
]


def seconds(function, *args):
    start = time.perf_counter()
    result = function(*args)
    return time.perf_counter() - start, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=3, help="how many questions")
    parser.add_argument("--no-triad", action="store_true", help="skip the judge (slow)")
    args = parser.parse_args()

    config = {**load_config(), "use_cache": False}
    generator = Generator(config)
    triad = None
    if not args.no_triad:
        from eval.rag_triad_eval import TriadEvaluator

        triad = TriadEvaluator()

    # One throwaway run so model loading is not counted against the first question.
    generator.retriever.retrieve(QUESTIONS[0])
    generator.guard.check(QUESTIONS[0])

    rows = []
    for question in QUESTIONS[: args.n]:
        guard, _ = (
            seconds(generator.guard.check, question) if generator.guard else (0, 0)
        )
        plain, _ = seconds(generator.retriever.retrieve, question, "hybrid")
        full, chunks = seconds(generator.retriever.retrieve, question, "hybrid_rerank")
        answer_time, (answer, _) = seconds(generator.answer_from, question, chunks)
        scoring = 0.0
        if triad:
            scoring, _ = seconds(triad.evaluate, question, answer, chunks)
        rows.append((guard, plain, full - plain, answer_time, scoring))
        print(
            f"{question[:48]:50} guard {guard:5.1f}  search {plain:5.1f}  "
            f"rerank {full - plain:5.1f}  answer {answer_time:5.1f}  triad {scoring:5.1f}"
        )

    means = [sum(column) / len(rows) for column in zip(*rows)]
    print(
        f"{'MEAN':50} guard {means[0]:5.1f}  search {means[1]:5.1f}  "
        f"rerank {means[2]:5.1f}  answer {means[3]:5.1f}  triad {means[4]:5.1f}"
    )
