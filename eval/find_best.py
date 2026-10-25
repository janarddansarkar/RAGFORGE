"""Find the best RAG settings in one run, stage by stage (each winner carries forward).

  1. chunking   - build / reuse a store per [[chunking_variants]], score the default method
  2. embedder   - compare the Hugging Face embedding models in `embedders` (one store each)
  3. method     - compare the pipelines in `search_methods`
  4. reranker   - compare the Hugging Face rerankers in `rerankers` (only if the method reranks)
  5. generator  - compare the answer models in `generators` on the golden pages (oracle mode),
                  scored by the judge model; this is the slowest stage

Stages 1-4 rank by primary_metric on 2 of every 3 golden queries; the retrieval pick is then
checked on the held-out third. "Best" = highest score, but anything within the tie margin of
the top counts as a tie and the faster one wins (with 45 queries, small gaps are noise).

Output:
  eval/find_best_report.json  every stage's numbers (always written)
  best_config.json            the winning settings, which load_config() lays over config.toml.
                              If the retrieval pick is worse than the current retrieval settings
                              on the held-out queries, the current ones are kept.

Run:  uv run python eval/find_best.py
"""

import json
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR))

from eval.generator_eval import GeneratorEval
from eval.retriever_eval import RetrieverEval
from vectorstore import BEST_CONFIG_FILE, VectorStore, load_config

REPORT_FILE = Path(__file__).parent / "find_best_report.json"
HOLDOUT_EVERY = 3  # every 3rd golden query is held out

RETRIEVAL_SETTINGS = (
    "chunking_strategy",
    "contextual",
    "embedding_model",
    "default_method",
    "reranker_model",
)

# Settings that make up "the best config" (written to best_config.json).
SETTINGS = (
    "chunking_strategy",
    "contextual",
    "embedding_model",
    "default_method",
    "reranker_model",
    "llm_model",
    "top_k",
    "max_new_tokens",
)


class BestConfigSearch:
    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.primary = self.config["primary_metric"]
        self.tie_break = self.config["tie_break_metric"]
        self.margin = self.config["tie_margin"]
        self.report: dict = {}

    # SCORING ---- one config + one method -> metrics and seconds per query
    def evaluate(
        self, config: dict, method: str, holdout: bool = False
    ) -> dict[str, float]:
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
    def pick(
        self,
        results: dict[str, dict[str, float]],
        metric: str,
        margin: float,
        tie_break: str | None = None,
    ) -> str:
        top = max(r[metric] for r in results.values())
        close = [n for n, r in results.items() if r[metric] >= top - margin]
        speed = (
            "sec/query" if "sec/query" in results[close[0]] else "seconds_per_answer"
        )
        return min(
            close,
            key=lambda n: (results[n][speed], -results[n].get(tie_break or metric, 0)),
        )

    def show(
        self,
        title: str,
        results: dict[str, dict[str, float]],
        winner: str,
        columns: list[str],
    ) -> None:
        print(f"\n##### {title}")
        print(f"{'':50}" + "".join(f"{c:>20}" for c in columns))
        for name, scores in results.items():
            mark = "*" if name == winner else " "
            print(f"{mark} {name:48}" + "".join(f"{scores[c]:>20.2f}" for c in columns))
        self.report[title] = {"winner": winner, "results": results}

    def retrieval_stage(
        self, title: str, candidates: dict[str, tuple[dict, str]]
    ) -> str:
        """Score {name: (config, method)}, show the table, return the winning name."""
        results = {}
        for name, (config, method) in candidates.items():
            print(f"{title}: {name}", file=sys.stderr)
            results[name] = self.evaluate(config, method)
        winner = self.pick(results, self.primary, self.margin, self.tie_break)
        columns = [self.primary, self.tie_break, "Recall@1", "Precision@5", "sec/query"]
        self.show(title, results, winner, list(dict.fromkeys(columns)))
        return winner

    # STAGES ---- each returns the config that carries forward
    def stage_chunking(self) -> dict:
        method = self.config["default_method"]
        candidates = {}
        for variant in self.config["chunking_variants"]:
            config = {**self.config, **variant}
            store = VectorStore(config)
            store.build()  # does nothing if this strategy is already built
            candidates[store.pipeline.name] = (config, method)
        winner = self.retrieval_stage(f"1. chunking ({method})", candidates)
        return candidates[winner][0]

    def stage_embedder(self, config: dict) -> dict:
        candidates = {}
        for model in self.config["embedders"]:
            candidate = {**config, "embedding_model": model}
            VectorStore(candidate).build()
            candidates[model] = (candidate, config["default_method"])
        winner = self.retrieval_stage("2. embedding model", candidates)
        return candidates[winner][0]

    def stage_method(self, config: dict) -> dict:
        candidates = {m: (config, m) for m in self.config["search_methods"]}
        winner = self.retrieval_stage("3. retrieval method", candidates)
        return {**config, "default_method": winner}

    def stage_reranker(self, config: dict) -> dict:
        if not config["default_method"].endswith("rerank"):
            print("\n(4. reranker skipped: the winning method does not rerank)")
            return config
        method = config["default_method"]
        candidates = {
            m: ({**config, "reranker_model": m}, method)
            for m in self.config["rerankers"]
        }
        winner = self.retrieval_stage(f"4. reranker ({method})", candidates)
        return candidates[winner][0]

    def stage_generator(self, config: dict) -> dict:
        """Each generator answers from the golden pages; the judge scores the answers."""
        results = {}
        for model in self.config["generators"]:
            print(f"generator: {model}", file=sys.stderr)
            candidate = {**config, "llm_model": model}
            average = GeneratorEval(candidate, oracle=True).run()["average"]
            results[model] = average
        metric = self.config["generator_metric"]
        winner = self.pick(results, metric, self.config["generator_margin"])
        columns = [*GeneratorEval.SCORES, "seconds_per_answer"]
        self.show(
            f"5. generator (judge: {self.config['judge_model']})",
            results,
            winner,
            columns,
        )
        return {**config, "llm_model": winner}

    # RUN ---- all stages, then check the retrieval pick on the held-out queries
    def run(self) -> dict:
        config = self.stage_chunking()
        config = self.stage_embedder(config)
        config = self.stage_method(config)
        config = self.stage_reranker(config)

        method = config["default_method"]
        baseline = self.evaluate(
            self.config, self.config["default_method"], holdout=True
        )
        best = self.evaluate(config, method, holdout=True)
        self.show(
            "held-out queries: current settings vs. the pick",
            {"current settings": baseline, "best found": best},
            "best found",
            list(
                dict.fromkeys([self.primary, self.tie_break, "Recall@1", "sec/query"])
            ),
        )
        keep_current = best[self.primary] < baseline[self.primary]
        if keep_current:
            print(
                f"\nWARNING: the retrieval pick scores lower on {self.primary} than the "
                "current settings on held-out queries, so the current retrieval settings stay."
            )
            config = {**config, **{k: self.config[k] for k in RETRIEVAL_SETTINGS}}

        config = self.stage_generator(config)  # the slowest stage, so it runs last
        return {
            "retrieval": "kept current settings" if keep_current else "updated",
            "settings": {key: config[key] for key in SETTINGS},
            "held_out": {"best": best, "current": baseline},
        }


# RUN IT ---- python eval/find_best.py
if __name__ == "__main__":
    search = BestConfigSearch()
    result = search.run()
    REPORT_FILE.write_text(json.dumps({**result, "stages": search.report}, indent=2))
    BEST_CONFIG_FILE.write_text(json.dumps(result, indent=2))
    print(
        f"\nSaved the winning settings to {BEST_CONFIG_FILE.relative_to(PROJECT_DIR)}"
    )
    print(f"Saved the full report to {REPORT_FILE.relative_to(PROJECT_DIR)}")
