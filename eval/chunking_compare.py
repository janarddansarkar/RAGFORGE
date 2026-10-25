"""Compare chunking strategies: build a store for each, then run the retriever eval on it.

The variants come from [[chunking_variants]] in config.toml. Each variant gets its own
collection in ./vs, so a store that already exists is reused (not rebuilt).
Relevance is page-based, so every strategy is scored against the same golden pages.

Run:  uv run python eval/chunking_compare.py
The contextual variant makes one LLM call per chunk the first time it is built (slow).
"""

import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR))

from eval.retriever_eval import RetrieverEval
from vectorstore import VectorStore, load_config

RESULTS_FILE = Path(__file__).parent / "chunking_results.json"


class ChunkingComparison:
    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.evaluation: RetrieverEval | None = None  # the last one, used to print tables

    def variant_config(self, variant: dict) -> dict:
        return {**self.config, **variant}

    def run(self) -> dict[str, dict[str, dict[str, float]]]:
        """{method: {chunking name: {metric: value}}}"""
        results: dict[str, dict[str, dict[str, float]]] = {
            m: {} for m in self.config["compare_methods"]
        }
        for variant in self.config["chunking_variants"]:
            config = self.variant_config(variant)
            vector_store = VectorStore(config)
            name = vector_store.pipeline.name
            print(f"== {name}", file=sys.stderr)
            vector_store.build()  # does nothing if this strategy is already built

            evaluation = self.evaluation = RetrieverEval(config)
            for method in self.config["compare_methods"]:
                results[method][name] = evaluation.run_method(method)
                print(f"   done: {method}", file=sys.stderr)
        return results

    def print_tables(self, results: dict[str, dict[str, dict[str, float]]]) -> None:
        assert self.evaluation is not None, "call run() first"
        for method, by_chunking in results.items():
            print(f"\n##### retrieval method: {method}")
            self.evaluation.print_table(by_chunking)


# RUN IT ---- python eval/chunking_compare.py
if __name__ == "__main__":
    comparison = ChunkingComparison()
    all_results = comparison.run()
    comparison.print_tables(all_results)
    RESULTS_FILE.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved to {RESULTS_FILE.relative_to(PROJECT_DIR)}")
