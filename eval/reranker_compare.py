"""Run the retriever eval for several Hugging Face rerankers (project files untouched)."""

import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT))

from sentence_transformers import CrossEncoder

from eval.retriever_eval import RetrieverEval

RERANKERS = [
    "BAAI/bge-reranker-v2-m3",  # current
    "BAAI/bge-reranker-base",
    "BAAI/bge-reranker-large",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "mixedbread-ai/mxbai-rerank-base-v1",
]
METHODS = ["dense_rerank", "hybrid_rerank"]

evaluation = RetrieverEval()
results = {}
for name in RERANKERS:
    try:
        start = time.time()
        evaluation.retriever.__dict__["reranker"] = CrossEncoder(name)  # replaces cached_property
        for method in METHODS:
            results[f"{method} | {name}"] = evaluation.run_method(method)
        print(f"done {name} ({time.time() - start:.0f}s)", file=sys.stderr)
    except Exception as e:  # noqa: BLE001 - keep going if one model fails to download
        print(f"FAILED {name}: {type(e).__name__}: {str(e)[:150]}", file=sys.stderr)

out = Path(__file__).parent / "reranker_results.json"
out.write_text(json.dumps(results, indent=2))
cols = evaluation.metric_names()
print(f"\n{'pipeline | reranker':62}" + "".join(f"{c:>12}" for c in cols))
for key, s in results.items():
    print(f"{key:62}" + "".join(f"{s[c]:>12.2f}" for c in cols))
