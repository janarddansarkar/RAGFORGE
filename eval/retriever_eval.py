"""Retriever eval: Recall@K, Precision@K, HitRate@K and MRR (see specs_md/retriever_evaluation_metrics.md).

Chunk IDs are not stable, so relevance is page-based: each golden question names a
document and page(s) in its "source". A retrieved chunk is relevant if it comes
from that document and one of those pages.

Build the store first, then:  uv run python eval/retriever_eval.py
"""

import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR))

from eval.golden import GoldenQuery
from metrics.retriever_metrics import RetrieverMetrics
from retriever import Retriever
from vectorstore import PROJECT_ROOT, load_config

RESULTS_FILE = Path(__file__).parent / "retriever_results.json"


class RetrieverEval:
    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.ks = self.config["eval_ks"]
        self.precision_k = self.config["precision_k"]
        with open(PROJECT_ROOT / self.config["golden_file"], encoding="utf-8") as f:
            self.queries = [GoldenQuery(item) for item in json.load(f)]
        self.retriever = Retriever(self.config)

    def metric_names(self) -> list[str]:
        return [
            *(f"Recall@{k}" for k in self.ks),
            f"Precision@{self.precision_k}",
            "MRR",
            *(f"HitRate@{k}" for k in self.ks),
        ]

    def score(self, s: RetrieverMetrics) -> dict[str, float]:
        scores = {f"Recall@{k}": s.recall(k) for k in self.ks}
        scores[f"Precision@{self.precision_k}"] = s.precision(self.precision_k)
        scores["MRR"] = s.reciprocal_rank()
        scores.update({f"HitRate@{k}": s.hit(k) for k in self.ks})
        return scores

    def run_method(self, method: str) -> dict[str, float]:
        """Average every metric over all golden queries for one retrieval method."""
        fetch = max(*self.ks, self.precision_k)
        totals = dict.fromkeys(self.metric_names(), 0.0)
        for golden in self.queries:
            chunks = self.retriever.retrieve(golden.query, method=method, k=fetch)
            for name, value in self.score(golden.metrics(chunks)).items():
                totals[name] += value
        n = len(self.queries)
        return {name: total / n for name, total in totals.items()}

    def run_all(self) -> dict[str, dict[str, float]]:
        results = {}
        for method in self.config["methods"]:
            results[method] = self.run_method(method)
            print(f"done: {method}", file=sys.stderr)
        return results

    def print_table(self, results: dict[str, dict[str, float]]) -> None:
        columns = self.metric_names()
        print(f"\n{len(self.queries)} golden queries\n")
        print(f"{'method':32}" + "".join(f"{c:>{max(len(c), 7) + 1}}" for c in columns))
        for method, scores in results.items():
            row = "".join(f"{scores[c]:>{max(len(c), 7) + 1}.2f}" for c in columns)
            print(f"{method:32}{row}")


# RUN IT ---- python eval/retriever_eval.py
if __name__ == "__main__":
    evaluation = RetrieverEval()
    all_results = evaluation.run_all()
    evaluation.print_table(all_results)
    RESULTS_FILE.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved to {RESULTS_FILE.relative_to(PROJECT_ROOT)}")
