"""RAG triad over the golden questions: deepeval scores the whole pipeline, and the run is saved.

  Context relevance - are the retrieved chunks relevant to the question?   (question -> context)
  Groundedness      - is every claim in the answer supported by the chunks? (context -> answer)
  Answer relevance  - does the answer address the question?                 (question -> answer)

For every golden question the real pipeline (retriever, reranker, answer model) retrieves and
answers, then the judge model (judge_model in config.toml) scores the triad with deepeval.
Each run is written to eval/triad_runs/<date>_<time>.json with the averages, every score and the
judge's reasons. Judges are noisy; read the scores as a comparison between setups.

Run:  uv run python eval/triad_eval.py [--limit 10] [--generator ID] [--judge ID]
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR))

from eval.golden import GoldenQuery
from eval.rag_triad_eval import TriadEvaluator
from generator import Generator
from vectorstore import PROJECT_ROOT, load_config

RUNS_DIR = Path(__file__).parent / "triad_runs"  # one JSON file per run


class TriadEval:
    def __init__(self, config: dict | None = None, limit: int | None = None):
        self.config = config or load_config()
        with open(PROJECT_ROOT / self.config["golden_file"], encoding="utf-8") as f:
            items = json.load(f)[:limit]
        self.queries = [GoldenQuery(item) for item in items]
        self.generator = Generator(self.config)

    # PHASE 1 ---- the pipeline answers every question (only the answer models are loaded)
    def generate(self) -> list[dict]:
        samples = []
        for i, golden in enumerate(self.queries, start=1):
            start = time.perf_counter()
            answer, chunks = self.generator.answer(golden.query)
            samples.append(
                {
                    "golden": golden,
                    "answer": answer,
                    "chunks": chunks,
                    "seconds": time.perf_counter() - start,
                }
            )
            print(f"generated {i}/{len(self.queries)}", file=sys.stderr)
        return samples

    def release_generator(self) -> None:
        """Free the answer models' memory, so the (large) judge fits next to nothing else."""
        retriever = self.generator.retriever
        retriever.llm.unload()
        if self.config["reranker_type"] == "llm":
            retriever.reranker.unload()

    # PHASE 2 ---- the judge scores the triad for every answer
    def judge(self, samples: list[dict]) -> list[dict]:
        evaluator = TriadEvaluator(self.config["judge_model"])
        rows = []
        for i, sample in enumerate(samples, start=1):
            golden = sample["golden"]
            triad = evaluator.evaluate(golden.query, sample["answer"], sample["chunks"])
            rows.append(
                {
                    "id": golden.id,
                    "question": golden.query,
                    "answer": sample["answer"],
                    "pages": sorted(
                        {
                            f"{c.metadata['source']} p.{c.metadata['page']}"
                            for c in sample["chunks"]
                        }
                    ),
                    "seconds": sample["seconds"],
                    "triad": triad,
                }
            )
            print(
                f"judged {i}/{len(samples)} {golden.id}: "
                + " ".join(
                    f"{name.split()[0]}={'n/a' if r['score'] is None else format(r['score'], '.2f')}"
                    for name, r in triad.items()
                ),
                file=sys.stderr,
            )
        return rows

    def run(self) -> dict:
        samples = self.generate()
        self.release_generator()
        rows = self.judge(samples)
        average, failed = {}, {}
        for name in TriadEvaluator.METRICS:
            scores = [r["triad"][name]["score"] for r in rows]
            good = [s for s in scores if s is not None]
            average[name] = sum(good) / len(good) if good else None
            failed[name] = len(scores) - len(good)  # the judge could not score these
        return {
            "generator": self.config["llm_model"],
            "judge": self.config["judge_model"],
            "settings": {  # what produced these numbers, so runs can be compared later
                key: self.config[key]
                for key in (
                    "embedding_model",
                    "reranker_type",
                    "default_method",
                    "top_k",
                    "chunking_strategy",
                    "contextual",
                    "max_new_tokens",
                )
            },
            "questions": len(rows),
            "average": average,
            "judge_failures": failed,
            "rows": rows,
        }

    @staticmethod
    def save(results: dict) -> Path:
        """Write this run to eval/triad_runs/<date>_<time>.json (never overwrites)."""
        RUNS_DIR.mkdir(exist_ok=True)
        file = RUNS_DIR / f"{time.strftime('%Y-%m-%d_%H%M%S')}.json"
        file.write_text(json.dumps(results, indent=2))
        return file

    @staticmethod
    def print_summary(results: dict) -> None:
        print(
            f"\ngenerator: {results['generator']}   judge: {results['judge']}   "
            f"questions: {results['questions']}\n"
        )
        for name, value in results["average"].items():
            shown = "n/a" if value is None else f"{value:6.2f}"
            failures = results["judge_failures"][name]
            note = f"   ({failures} not scored)" if failures else ""
            print(f"  {name:18} {shown}{note}")


# RUN IT ---- python eval/triad_eval.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--limit", type=int, help="only the first N golden questions")
    parser.add_argument("--generator", help="Ollama model to use instead of llm_model")
    parser.add_argument("--judge", help="Ollama model to use instead of judge_model")
    args = parser.parse_args()

    config = load_config()
    if args.generator:
        config["llm_model"] = args.generator
    if args.judge:
        config["judge_model"] = args.judge

    evaluation = TriadEval(config, limit=args.limit)
    results = evaluation.run()
    evaluation.print_summary(results)
    print(f"\nSaved to {evaluation.save(results).relative_to(PROJECT_DIR)}")
