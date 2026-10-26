"""Offline chunk-size sweep: which chunk size (and top-k) gives the most relevant context?

For every chunk size it builds a store (recursive chunking, overlap 15%), then measures
  - retrieval on all golden questions (Recall, MRR, HitRate, Precision; page-based, see
    eval/retriever_eval.py), which shows whether smaller chunks lose information, and
  - context relevance (the deepeval RAG-triad check, judge_model in config.toml) on a sample
    of the golden questions, for the top-3 and top-5 chunks.

The winner is the size with the highest context relevance whose Recall@5 and MRR stay within
`--tolerance` of the baseline (the current chunk_size); top-k is then chosen the same way.
`--apply` writes the winner into best_config.json (laid over config.toml).

This is stage 5 of eval/find_best.py (the script that runs every eval); run it alone with:
      uv run python eval/chunk_size_sweep.py [--sizes 1000 600 400 250 150] [--sample 8] [--apply]
Slow (about an hour): about 4 minutes of retrieval per size, plus the judge. Results are saved
after every size, so a stopped run continues where it left off (--force starts over).
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR))

from eval.rag_triad_eval import TriadEvaluator
from eval.retriever_eval import RetrieverEval
from vectorstore import BEST_CONFIG_FILE, VectorStore, load_config

RESULTS_FILE = Path(__file__).parent / "chunk_size_sweep_results.json"
OVERLAP_SHARE = 0.15  # overlap = 15% of the chunk size
TOP_KS = (3, 5)  # the top-k values compared


class ChunkSizeSweep:
    def __init__(
        self,
        config: dict | None = None,
        sample: int = 8,
        limit: int | None = None,
        holdout_every: int | None = None,
    ):
        self.config = config or load_config()
        self.sample = sample
        self.limit = limit  # only the first N golden questions (for a quick try)
        self.holdout_every = (
            holdout_every  # leave out every Nth question (find_best checks on them)
        )
        self.method = self.config["default_method"]
        self.judge: TriadEvaluator | None = None  # loaded when first needed

    @staticmethod
    def overlap(size: int) -> int:
        return round(size * OVERLAP_SHARE)

    def config_for(self, size: int) -> dict:
        return {
            **self.config,
            "chunk_size": size,
            "chunk_overlap": self.overlap(size),
        }

    @staticmethod
    def sample_indices(total: int, sample: int) -> set[int]:
        """Evenly spread question numbers, so every document is represented."""
        sample = min(sample, total)
        return {round(i * total / sample) for i in range(sample)}

    # ONE SIZE ---- build the store, retrieve for every golden question, judge a sample
    def run_size(self, size: int) -> dict:
        start = time.time()
        config = self.config_for(size)
        store = VectorStore(config)
        store.build()  # does nothing if this size is already built
        evaluation = RetrieverEval(config)
        queries = evaluation.queries
        if self.holdout_every:
            queries = [q for i, q in enumerate(queries) if i % self.holdout_every != 0]
        evaluation.queries = queries[: self.limit]

        totals: dict[str, float] = {}
        sampled = self.sample_indices(len(evaluation.queries), self.sample)
        to_judge = []
        for i, golden in enumerate(evaluation.queries):
            chunks = evaluation.retriever.retrieve(golden.query, self.method, k=10)
            metrics = golden.metrics(chunks)
            scores = evaluation.score(metrics)
            scores["Precision@3"] = metrics.precision(3)
            for name, value in scores.items():
                totals[name] = totals.get(name, 0.0) + value
            if i in sampled:
                to_judge.append((golden.query, chunks))
            print(
                f"   size {size}: retrieved {i + 1}/{len(evaluation.queries)}",
                end="\r",
                file=sys.stderr,
            )
        retrieval = {
            name: total / len(evaluation.queries) for name, total in totals.items()
        }

        relevance, failed = self.judge_context(to_judge)
        return {
            "chunk_size": size,
            "chunk_overlap": self.overlap(size),
            "chunks_in_store": store.count(),
            "retrieval": retrieval,
            "context_relevance": relevance,
            "judge_failures": failed,
            "judged_questions": len(to_judge),
            "seconds": round(time.time() - start),
        }

    def judge_context(self, to_judge: list) -> tuple[dict[str, float | None], int]:
        """Mean context relevance of the top-k chunks, per k. A failed judge call is skipped."""
        if self.judge is None:
            self.judge = TriadEvaluator()
        scores: dict[int, list[float]] = {k: [] for k in TOP_KS}
        failed = 0
        for n, (query, chunks) in enumerate(to_judge, start=1):
            for k in TOP_KS:
                result = self.judge.context_relevance(query, chunks[:k])
                if result["score"] is None:
                    failed += 1
                else:
                    scores[k].append(result["score"])
            print(f"   judged {n}/{len(to_judge)}", end="\r", file=sys.stderr)
        return {
            f"top{k}": (sum(v) / len(v) if v else None) for k, v in scores.items()
        }, failed

    # ALL SIZES ---- resume from the saved file
    def run(self, sizes: list[int], force: bool = False) -> dict[str, dict]:
        results = {}
        if RESULTS_FILE.exists() and not force:
            results = json.loads(RESULTS_FILE.read_text())
        for size in sizes:
            if str(size) in results:
                print(f"== size {size}: already done", file=sys.stderr)
                continue
            print(f"== size {size}", file=sys.stderr)
            results[str(size)] = self.run_size(size)
            RESULTS_FILE.write_text(json.dumps(results, indent=2))
        return {str(size): results[str(size)] for size in sizes}

    # THE CHOICE ---- best context relevance without losing too much recall
    @staticmethod
    def choose(
        results: dict[str, dict], baseline: int, tolerance: float = 0.05
    ) -> dict:
        """{"chunk_size", "top_k", "reason"}: the size and top-k to use."""
        base = results[str(baseline)]["retrieval"]

        def ok(entry: dict) -> bool:
            retrieval = entry["retrieval"]
            return (
                retrieval["Recall@5"] >= base["Recall@5"] - tolerance
                and retrieval["MRR"] >= base["MRR"] - tolerance
            )

        def relevance(entry: dict, k: int = 5) -> float:
            value = entry["context_relevance"].get(f"top{k}")
            return -1.0 if value is None else value

        eligible = [
            e for e in results.values() if ok(e)
        ]  # the baseline always qualifies
        best = max(eligible, key=lambda e: (relevance(e), e["chunk_size"]))
        retrieval = best["retrieval"]
        top_k = 5
        if retrieval["Recall@3"] >= retrieval["Recall@5"] - tolerance and relevance(
            best, 3
        ) > relevance(best, 5):
            top_k = 3
        return {
            "chunk_size": best["chunk_size"],
            "chunk_overlap": best["chunk_overlap"],
            "top_k": top_k,
            "reason": (
                f"highest context relevance ({relevance(best, top_k):.2f} at top-{top_k}) among sizes "
                f"whose Recall@5 and MRR are within {tolerance} of chunk_size {baseline}"
            ),
        }

    @staticmethod
    def print_table(results: dict[str, dict]) -> None:
        columns = ["Recall@3", "Recall@5", "MRR", "Precision@3", "Precision@5"]
        print(
            f"\n{'chunk size':>10} {'chunks':>7}"
            + "".join(f"{c:>13}" for c in columns)
            + f"{'ctx rel @3':>12}{'ctx rel @5':>12}{'judge fails':>12}"
        )
        for entry in results.values():
            row = "".join(f"{entry['retrieval'][c]:>13.2f}" for c in columns)
            rel = [entry["context_relevance"][f"top{k}"] for k in TOP_KS]
            rel_text = "".join(
                f"{'n/a' if v is None else format(v, '.2f'):>12}" for v in rel
            )
            print(
                f"{entry['chunk_size']:>10} {entry['chunks_in_store']:>7}{row}{rel_text}{entry['judge_failures']:>12}"
            )


def apply_choice(choice: dict) -> None:
    """Write the winner into best_config.json, which load_config() lays over config.toml."""
    best = json.loads(BEST_CONFIG_FILE.read_text())
    best["settings"].update(
        {
            "chunk_size": choice["chunk_size"],
            "chunk_overlap": choice["chunk_overlap"],
            "top_k": choice["top_k"],
        }
    )
    best["chunk_size_sweep"] = choice
    BEST_CONFIG_FILE.write_text(json.dumps(best, indent=2))


# RUN IT ---- python eval/chunk_size_sweep.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[1000, 600, 400, 250, 150]
    )
    parser.add_argument(
        "--sample", type=int, default=8, help="golden questions judged per size"
    )
    parser.add_argument(
        "--tolerance", type=float, default=0.05, help="recall/MRR drop allowed"
    )
    parser.add_argument(
        "--apply", action="store_true", help="write the winner to best_config.json"
    )
    parser.add_argument("--force", action="store_true", help="ignore saved results")
    parser.add_argument(
        "--limit",
        type=int,
        help="only the first N golden questions (quick try, not saved)",
    )
    args = parser.parse_args()

    config = load_config()
    if config["chunking_strategy"] != "recursive" or config["contextual"]:
        sys.exit(
            "Chunk size only applies to plain recursive chunking (see config.toml)."
        )
    baseline = config["chunk_size"]
    sizes = list(
        dict.fromkeys([baseline, *args.sizes])
    )  # the baseline is always measured
    sweep = ChunkSizeSweep(config, args.sample, args.limit)
    if args.limit:  # a quick try: show the table, save nothing
        results = {str(size): sweep.run_size(size) for size in sizes}
    else:
        results = sweep.run(sizes, args.force)
    sweep.print_table(results)
    choice = ChunkSizeSweep.choose(results, baseline, args.tolerance)
    print(
        f"\nBest: chunk_size={choice['chunk_size']} overlap={choice['chunk_overlap']} top_k={choice['top_k']}\n  {choice['reason']}"
    )
    if args.apply and not args.limit:
        apply_choice(choice)
        print(f"Written to {BEST_CONFIG_FILE.name}")
