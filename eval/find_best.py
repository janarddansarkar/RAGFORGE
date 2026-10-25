"""Find the best RAG settings in one run, stage by stage (each winner carries forward).

  1. chunking   - build / reuse a store per [[chunking_variants]], score the default method
  2. method     - compare the pipelines in search_methods on the winning chunking
  3. reranker   - compare the Hugging Face rerankers (only if the winning method reranks)

"Best" = highest primary_metric, but anything within tie_margin of the top counts as a
tie and the faster one wins (with 45 queries, small gaps are noise). Stages tune on 2 of
every 3 golden queries; the final pick is checked against the held-out third.
config.toml is never edited: the result goes to eval/best_config.json for you to apply.

Run:  uv run python eval/find_best.py
"""

import json
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR))

from eval.retriever_eval import RetrieverEval
from vectorstore import VectorStore, load_config

RESULTS_FILE = Path(__file__).parent / "best_config.json"
HOLDOUT_EVERY = 3  # every 3rd golden query is held out


class BestConfigSearch:
    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.primary = self.config["primary_metric"]
        self.tie_break = self.config["tie_break_metric"]
        self.margin = self.config["tie_margin"]

    # SCORING ---- one config + one method -> metrics and seconds per query
    def evaluate(self, config: dict, method: str, holdout: bool = False) -> dict[str, float]:
        evaluation = RetrieverEval(config)
        queries = [
            q
            for i, q in enumerate(evaluation.queries)
            if (i % HOLDOUT_EVERY == 0) == holdout
        ]
        evaluation.queries = queries
        # Load the models first, so loading time isn't counted as search time.
        evaluation.retriever.retrieve(queries[0].query, method=method)
        start = time.perf_counter()
        scores = evaluation.run_method(method)
        scores["sec/query"] = (time.perf_counter() - start) / len(queries)
        return scores

    # CHOOSING ---- best score, but a near-tie goes to the faster one
    def pick(self, results: dict[str, dict[str, float]]) -> str:
        top = max(r[self.primary] for r in results.values())
        close = [n for n, r in results.items() if r[self.primary] >= top - self.margin]
        return min(close, key=lambda n: (results[n]["sec/query"], -results[n][self.tie_break]))

    def show(self, title: str, results: dict[str, dict[str, float]], winner: str) -> None:
        columns = [self.primary, self.tie_break, "Recall@1", "Precision@5", "sec/query"]
        columns = list(dict.fromkeys(columns))
        print(f"\n##### {title}")
        print(f"{'':50}" + "".join(f"{c:>13}" for c in columns))
        for name, scores in results.items():
            mark = "*" if name == winner else " "
            print(f"{mark} {name:48}" + "".join(f"{scores[c]:>13.2f}" for c in columns))

    # STAGES ---- each returns the winner's name and config
    def stage_chunking(self) -> tuple[dict, dict]:
        results, configs = {}, {}
        for variant in self.config["chunking_variants"]:
            config = {**self.config, **variant}
            store = VectorStore(config)
            store.build()  # does nothing if this strategy is already built
            name = store.pipeline.name
            print(f"chunking: {name}", file=sys.stderr)
            results[name] = self.evaluate(config, self.config["default_method"])
            configs[name] = {"variant": variant, "config": config}
        winner = self.pick(results)
        self.show(f"1. chunking (method: {self.config['default_method']})", results, winner)
        return configs[winner], results[winner]

    def stage_method(self, config: dict) -> str:
        results = {}
        for method in self.config["search_methods"]:
            print(f"method: {method}", file=sys.stderr)
            results[method] = self.evaluate(config, method)
        winner = self.pick(results)
        self.show("2. retrieval method", results, winner)
        return winner

    def stage_reranker(self, config: dict, method: str) -> str:
        results = {}
        for model in self.config["rerankers"]:
            print(f"reranker: {model}", file=sys.stderr)
            results[model] = self.evaluate({**config, "reranker_model": model}, method)
        winner = self.pick(results)
        self.show(f"3. reranker (method: {method})", results, winner)
        return winner

    # RUN ---- all stages, then check the winner on the held-out queries
    def run(self) -> dict:
        chosen, _ = self.stage_chunking()
        config = chosen["config"]
        method = self.stage_method(config)
        if method.endswith("rerank"):
            config = {**config, "reranker_model": self.stage_reranker(config, method)}

        baseline = self.evaluate(self.config, self.config["default_method"], holdout=True)
        best = self.evaluate(config, method, holdout=True)
        self.show(
            "held-out queries: current config.toml vs. the pick",
            {"current config.toml": baseline, "best found": best},
            "best found",
        )
        keep_current = best[self.primary] < baseline[self.primary]
        if keep_current:
            print(f"\nWARNING: the pick scores lower on {self.primary} than the current "
                  "config.toml on held-out queries. Keep the current settings.")
        return {
            "recommendation": "keep current config.toml" if keep_current else "apply this",
            "chunking": chosen["variant"],
            "default_method": method,
            "reranker_model": config["reranker_model"],
            "held_out_best": best,
            "held_out_current": baseline,
        }


# RUN IT ---- python eval/find_best.py
if __name__ == "__main__":
    best = BestConfigSearch().run()
    RESULTS_FILE.write_text(json.dumps(best, indent=2))
    print(f"\nSaved to {RESULTS_FILE.relative_to(PROJECT_DIR)}")
