"""Run the retriever eval for several Ollama rerankers (LLMs that score chunks; project files untouched)."""

import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT))

from eval.retriever_eval import RetrieverEval
from models import make_llm
from vectorstore import load_config

CONFIG = load_config()
RERANKERS = CONFIG["rerankers"]  # Ollama chat models, from config.toml
METHODS = ["dense_rerank", "hybrid_rerank"]

evaluation = RetrieverEval()
results = {}
for name in RERANKERS:
    try:
        start = time.time()
        # replaces the cached_property
        evaluation.retriever.__dict__["reranker"] = make_llm(
            {**CONFIG, "llm_model": name, "max_new_tokens": 4}
        )
        for method in METHODS:
            results[f"{method} | {name}"] = evaluation.run_method(method)
        print(f"done {name} ({time.time() - start:.0f}s)", file=sys.stderr)
    except Exception as e:  # noqa: BLE001 - keep going if one model is not pulled
        print(f"FAILED {name}: {type(e).__name__}: {str(e)[:150]}", file=sys.stderr)

out = Path(__file__).parent / "reranker_results.json"
out.write_text(json.dumps(results, indent=2))
cols = evaluation.metric_names()
print(f"\n{'pipeline | reranker':62}" + "".join(f"{c:>12}" for c in cols))
for key, s in results.items():
    print(f"{key:62}" + "".join(f"{s[c]:>12.2f}" for c in cols))
