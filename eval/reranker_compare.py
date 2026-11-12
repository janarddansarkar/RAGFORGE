"""Run the retriever eval for several rerankers (chat models that score each chunk 0-10).

The rerankers come from `rerankers` in config.toml. Each is plugged into the same retriever (the
project files are untouched) and scored with hybrid search + reranking on the golden pages, so the
numbers can be compared with plain hybrid search from eval/retriever_eval.py.

Run:  uv run python eval/reranker_compare.py   (or through eval/run_provider_eval.py)
"""

import json
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR))

from eval.retriever_eval import RetrieverEval
from models import make_llm
from vectorstore import load_config

RESULTS_FILE = Path(__file__).parent / "reranker_results.json"
METHOD = "hybrid_rerank"


class RerankerComparison:
    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.evaluation = RetrieverEval(self.config)

    def run(self) -> dict[str, dict[str, float]]:
        """{reranker model: {metric: value}}"""
        results = {}
        for name in self.config["rerankers"]:
            start = time.time()
            # replaces the cached_property
            self.evaluation.retriever.__dict__["reranker"] = make_llm(
                {**self.config, "llm_model": name, "max_new_tokens": 4}, role="rerank"
            )
            results[name] = self.evaluation.run_method(METHOD)
            print(f"done {name} ({time.time() - start:.0f}s)", file=sys.stderr)
        return results

    def print_table(self, results: dict[str, dict[str, float]]) -> None:
        columns = self.evaluation.metric_names()
        print(
            f"\n{'reranker (' + METHOD + ')':42}" + "".join(f"{c:>12}" for c in columns)
        )
        for name, scores in results.items():
            print(f"{name:42}" + "".join(f"{scores[c]:>12.2f}" for c in columns))


# RUN IT ---- python eval/reranker_compare.py
if __name__ == "__main__":
    comparison = RerankerComparison()
    all_results = comparison.run()
    comparison.print_table(all_results)
    RESULTS_FILE.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved to {RESULTS_FILE.relative_to(PROJECT_DIR)}")
