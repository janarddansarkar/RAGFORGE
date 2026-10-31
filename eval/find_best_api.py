"""Find the best RAG settings when the models are a cloud API (OpenAI, Gemini or Anthropic).

Same stages as eval/find_best.py (chunking, embedder, method, reranker, generator), but every
model call goes to the provider's API with the key from ./.env, and the models come from
providers.toml. It prints what the run cost at the end.

The reranker stage compares the local cross-encoder (free) with an LLM reranker (one API call
per chunk), because that choice matters more for cost than which LLM does the scoring.

Output (the Ollama results in best_config.json are NOT touched):
  eval/find_best_report_api.json   every stage's numbers
  best_config_api.json             the winning settings; copy it over best_config.json to use it

Run:
  uv run python eval/find_best_api.py                      # OpenAI for everything
  uv run python eval/find_best_api.py --provider gemini
  uv run python eval/find_best_api.py --provider anthropic --embedding-provider openai
  uv run python eval/find_best_api.py --embedders text-embedding-3-small text-embedding-3-large
  uv run python eval/find_best_api.py --judge gpt-4o       # a judge stronger than the generator
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR))

from eval.find_best import BestConfigSearch
from models import use_providers
from vectorstore import VectorStore, load_config

REPORT_FILE = Path(__file__).parent / "find_best_report_api.json"
BEST_CONFIG_API_FILE = PROJECT_DIR / "best_config_api.json"


def api_key(provider: str) -> str:
    """The key for a provider from the environment (loaded from ./.env by load_config)."""
    name = f"{provider.upper()}_API_KEY"
    key = os.environ.get(name, "").strip()
    if not key:
        raise SystemExit(f"{name} is empty. Put your key in .env and run again.")
    return key


class ApiBestConfigSearch(BestConfigSearch):
    """BestConfigSearch with the candidates taken from the provider instead of Ollama."""

    def __init__(self, config: dict, embedders: list[str], generators: list[str]):
        super().__init__(config)
        self.embedders = embedders
        self.generators = generators

    def stage_embedder(self, config: dict) -> dict:
        candidates = {}
        for model in self.embedders:
            candidate = {**config, "embedding_model": model}
            VectorStore(candidate).build()
            candidates[model] = (candidate, config["default_method"])
        winner = self.retrieval_stage("2. embedding model", candidates)
        return candidates[winner][0]

    def stage_reranker(self, config: dict) -> dict:
        if not config["default_method"].endswith("rerank"):
            print("\n(4. reranker skipped: the winning method does not rerank)")
            return config
        method = config["default_method"]
        llm = config["reranker_model"]
        candidates = {
            "cross_encoder (local, free)": (
                {**config, "reranker_type": "cross_encoder"},
                method,
            ),
            f"llm: {llm}": ({**config, "reranker_type": "llm"}, method),
        }
        winner = self.retrieval_stage(f"4. reranker ({method})", candidates)
        return candidates[winner][0]

    def stage_generator(self, config: dict) -> dict:
        self.config = {**self.config, "generators": self.generators}
        return super().stage_generator(config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--provider", default="openai", help="answers, reranking, judge"
    )
    parser.add_argument("--embedding-provider", help="default: same as --provider")
    parser.add_argument("--embedders", nargs="+", help="embedding models to compare")
    parser.add_argument("--generators", nargs="+", help="answer models to compare")
    parser.add_argument(
        "--judge", help="judge model (default: the provider's judge_model)"
    )
    args = parser.parse_args()

    base = load_config()
    embedding_provider = args.embedding_provider or args.provider
    config = use_providers(
        base,
        args.provider,
        api_key(args.provider),
        embedding_provider,
        api_key(embedding_provider),
    )
    if args.judge:
        config["judge_model"] = args.judge

    search = ApiBestConfigSearch(
        config,
        embedders=args.embedders or [config["embedding_model"]],
        generators=args.generators or [config["llm_model"]],
    )
    result = search.run()
    REPORT_FILE.write_text(json.dumps({**result, "stages": search.report}, indent=2))
    BEST_CONFIG_API_FILE.write_text(json.dumps(result, indent=2))

    cost = config["usage"].cost()
    print(
        f"\nThis run used {cost['tokens_in']:,} input and {cost['tokens_out']:,} output tokens,"
        f" about ${cost['dollars']:.2f} at the prices in providers.toml."
    )
    print(f"Saved the winning settings to {BEST_CONFIG_API_FILE.name}")
    print(f"Saved the full report to {REPORT_FILE.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
