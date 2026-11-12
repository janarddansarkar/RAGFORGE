"""Run every evaluation (golden dataset + RAG triad + latency) with one provider's models.

    uv run python eval/run_provider_eval.py openai
    uv run python eval/run_provider_eval.py openai --limit 10 --stages retriever,generator

Every component uses that provider's default models (src/providers.py): answers, guard, judge,
reranker and embeddings. The key is read from the provider's environment variable or from .env.
Stages (all by default; pick some with --stages):

  retriever  Recall@K, Precision@K, HitRate@K, MRR for dense, bm25, hybrid and hybrid + rerank
  reranker   the same metrics for each reranker model in config.toml (hybrid + rerank)
  chunking   the same metrics for plain vs contextual chunks
  generator  faithfulness, answer relevancy, correctness, similarity, context recall (retrieved chunks)
  oracle     the same, with the golden pages instead of retrieved chunks (the generator on its own)
  triad      the RAG triad (DeepEval) over every golden question
  latency    seconds per stage for a few questions

The numbers and the calls, tokens and estimated cost of each stage are saved to
eval/results/<provider>_<date>_<time>.json (updated after every stage).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR))

import litellm

from eval.chunking_compare import ChunkingComparison
from eval.generator_eval import GeneratorEval
from eval.reranker_compare import RerankerComparison
from eval.retriever_eval import RetrieverEval
from eval.time_pipeline import measure
from eval.time_pipeline import print_table as print_latency
from eval.triad_eval import TriadEval
from providers import COMPONENTS, OFF, PROVIDERS, settings_for
from usage import ROLE_NAMES, UsageTracker, summarize
from vectorstore import VectorStore, load_config

RESULTS_DIR = Path(__file__).parent / "results"
KEY_NAMES = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}
PROVIDER_NAMES = {"openai": "OpenAI", "gemini": "Google Gemini"}


def read_dotenv() -> None:
    """Put KEY=value lines of .env into the environment (values are never printed)."""
    file = PROJECT_DIR / ".env"
    if not file.exists():
        return
    for line in file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            os.environ.setdefault(name.strip(), value.strip().strip("\"'"))


def provider_config(key: str, tracker: UsageTracker) -> dict:
    """config.toml with every component on this provider's first (best) models."""
    provider = PROVIDERS[PROVIDER_NAMES[key]]
    choices = {}
    for component in COMPONENTS:
        if component.kind == "speech":
            choices[component.key] = (OFF, "")  # voice input is not evaluated
        elif component.key == "answer" or component.kind == "embeddings":
            choices[component.key] = (provider.name, provider.models(component.kind)[0])
        else:  # guard, judge, reranker: the cheaper second model
            choices[component.key] = (provider.name, provider.chat[1])
    keys = {provider.name: os.environ[KEY_NAMES[key]]}
    return {
        **load_config(),
        **settings_for(choices, keys),
        "usage_tracker": tracker,
        "vs_dir": PROJECT_DIR
        / "vs_eval"
        / key,  # one store per provider: the runs can go in parallel
        "input_guard": False,  # the evals ask golden questions only
        "use_cache": False,
    }


def usage_report(calls: list) -> dict:
    total = summarize(calls)
    return {
        "calls": total.calls,
        "input_tokens": total.input_tokens,
        "output_tokens": total.output_tokens,
        "estimated_cost_usd": round(total.cost, 4),
        "unpriced_calls": total.unpriced,
        "by_component": {
            ROLE_NAMES.get(role, role): {
                "calls": part.calls,
                "input_tokens": part.input_tokens,
                "output_tokens": part.output_tokens,
                "estimated_cost_usd": round(part.cost, 4),
            }
            for role, part in total.by_role.items()
        },
    }


STAGES = (
    "retriever",
    "reranker",
    "chunking",
    "generator",
    "oracle",
    "triad",
    "latency",
)


def run_stage(name: str, config: dict, limit: int | None, shared: dict):
    """One stage's numbers (and a printout)."""
    if name == "retriever":
        evaluation = shared["retriever_eval"] = RetrieverEval(config)
        if limit:
            evaluation.queries = evaluation.queries[:limit]
        results = evaluation.run_all()
        evaluation.print_table(results)
        return results
    if name == "reranker":
        comparison = RerankerComparison(config)
        if limit:
            comparison.evaluation.queries = comparison.evaluation.queries[:limit]
        results = comparison.run()
        comparison.print_table(results)
        return results
    if name == "chunking":
        comparison = ChunkingComparison(config)
        results = comparison.run_limited(limit) if limit else comparison.run()
        comparison.print_tables(results)
        return results
    if name in ("generator", "oracle"):
        evaluation = GeneratorEval(config, limit=limit, oracle=name == "oracle")
        results = evaluation.run()
        evaluation.print_summary(results)
        return results
    if name == "triad":
        evaluation = TriadEval(config, limit=limit)
        results = evaluation.run()
        evaluation.print_summary(results)
        return results
    if name == "latency":
        result = measure(config, n=3)
        print_latency(result)
        return result
    raise ValueError(name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("provider", choices=sorted(KEY_NAMES))
    parser.add_argument("--limit", type=int, help="only the first N golden questions")
    parser.add_argument(
        "--stages",
        default=",".join(STAGES),
        help=f"comma-separated, from: {', '.join(STAGES)}",
    )
    parser.add_argument(
        "--resume",
        help="a saved results JSON: keep its finished stages and run only the missing ones",
    )
    args = parser.parse_args()
    stages = [x.strip() for x in args.stages.split(",") if x.strip()]
    unknown = [x for x in stages if x not in STAGES]
    if unknown:
        sys.exit(f"unknown stage(s): {unknown}; choose from {STAGES}")

    read_dotenv()
    if not os.environ.get(KEY_NAMES[args.provider]):
        sys.exit(f"{KEY_NAMES[args.provider]} is not set (environment or .env)")
    litellm.num_retries = 4  # ride out rate limits
    litellm.request_timeout = (
        120  # a hung call fails (and is retried) instead of stalling the run
    )

    tracker = UsageTracker()
    config = provider_config(args.provider, tracker)
    shown = {
        k: config[k]
        for k in (
            "llm_model",
            "guard_model",
            "judge_model",
            "reranker_model",
            "embedding_model",
        )
    }
    print(f"provider: {args.provider}\n{json.dumps(shown, indent=2)}", file=sys.stderr)

    RESULTS_DIR.mkdir(exist_ok=True)
    if (
        args.resume
    ):  # continue a saved run: its finished stages and their cost stay as they are
        out = Path(args.resume).resolve()
        report = json.loads(out.read_text())
        if report["provider"] != args.provider or report.get("limit") != args.limit:
            sys.exit("--resume needs the same provider and --limit as the saved run")
        stages = [x for x in stages if x not in report["stages"]]
        report["errors"] = {k: v for k, v in report["errors"].items() if k in stages}
    else:
        out = RESULTS_DIR / f"{args.provider}_{time.strftime('%Y-%m-%d_%H%M%S')}.json"
        report = {
            "provider": args.provider,
            "models": shown,
            "limit": args.limit,
            "stages": {},
            "usage": {},
            "errors": {},
        }
    started = time.time() - report.get("seconds", 0)
    print(f"stages to run: {stages}", file=sys.stderr)

    def save() -> None:
        report["seconds"] = round(time.time() - started)
        parts = [u for name, u in report["usage"].items() if name != "total"]
        report["usage"]["total"] = {
            "calls": sum(u["calls"] for u in parts),
            "input_tokens": sum(u["input_tokens"] for u in parts),
            "output_tokens": sum(u["output_tokens"] for u in parts),
            "estimated_cost_usd": round(sum(u["estimated_cost_usd"] for u in parts), 4),
            "unpriced_calls": sum(u["unpriced_calls"] for u in parts),
        }
        out.write_text(json.dumps(report, indent=2))

    mark = tracker.mark()
    VectorStore(
        config
    ).build()  # chunk, add context notes, embed (skipped if already built)
    if not args.resume:
        report["usage"]["indexing"] = usage_report(tracker.calls[mark:])
    print(f"index ready: {summarize(tracker.calls).describe()}", file=sys.stderr)
    save()

    shared: dict = {}
    for name in stages:
        print(f"\n########## stage: {name}", file=sys.stderr)
        mark, begun = tracker.mark(), time.time()
        try:
            report["stages"][name] = run_stage(name, config, args.limit, shared)
        except Exception as error:  # noqa: BLE001 - keep the other stages' numbers
            report["errors"][name] = f"{type(error).__name__}: {error}"[:500]
            print(f"STAGE {name} FAILED: {report['errors'][name]}", file=sys.stderr)
        report["usage"][name] = {
            **usage_report(tracker.calls[mark:]),
            "seconds": round(time.time() - begun),
        }
        save()

    print(
        f"\ntotal: ${report['usage']['total']['estimated_cost_usd']:.2f} estimated for the whole run\n"
        f"Saved to {out.relative_to(PROJECT_DIR)}"
    )
