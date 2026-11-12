"""Turn a saved evaluation run (eval/results/*.json) into a readable page: docs/EVALUATION_RESULTS.md.

    uv run python eval/make_report.py                       # the latest full run (all golden questions)
    uv run python eval/make_report.py eval/results/openai_2026-11-12_120403.json
    uv run python eval/make_report.py --allow-partial       # the latest run even if it used --limit

It only reads the saved numbers: no model calls, so it costs nothing and can be run any time.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT = PROJECT_DIR / "docs" / "EVALUATION_RESULTS.md"
RETRIEVAL_COLUMNS = [
    "Recall@1",
    "Recall@3",
    "Recall@5",
    "Recall@10",
    "Precision@5",
    "MRR",
    "HitRate@5",
]
GENERATION_COLUMNS = [
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
    "semantic_similarity",
    "context_recall",
]
STAGE_TITLES = {
    "indexing": "Indexing (chunk notes + embeddings, one time)",
    "retriever": "Retriever",
    "reranker": "Reranker",
    "chunking": "Chunking",
    "generator": "Generator (retrieved chunks)",
    "oracle": "Generator (golden pages)",
    "triad": "RAG triad",
    "latency": "Latency",
}


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def f(value, digits=2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def retrieval_table(by_name: dict, label: str) -> str:
    rows = [
        [name, *(f(scores.get(c)) for c in RETRIEVAL_COLUMNS)]
        for name, scores in by_name.items()
    ]
    return table([label, *RETRIEVAL_COLUMNS], rows)


def build(data: dict) -> str:
    st, usage = data["stages"], data["usage"]
    golden = json.loads(
        (
            PROJECT_DIR / "golden_data" / "rag_retriever_golden_dataset_45.json"
        ).read_text()
    )
    questions = data.get("limit") or len(golden)
    out = [
        "# RAGForge: evaluation results",
        "",
        f"Provider: **{data['provider']}**. Golden questions: **{questions}** (45 = the full set"
        + (f"; this run used `--limit {data['limit']}`" if data.get("limit") else "")
        + f"). Run time: {data.get('seconds', 0) // 60} min. Saved by `eval/run_provider_eval.py`; "
        "this page is made from the saved JSON by `eval/make_report.py` (no model calls).",
        "",
        "## Models",
        "",
        table(
            ["component", "model"],
            [[k.replace("_model", ""), f"`{v}`"] for k, v in data["models"].items()],
        ),
        "",
        "Scores are 0 to 1, higher is better. A judge model scores the generator and the triad, so read them as a "
        "comparison between setups, not as absolute truth.",
        "",
    ]
    if "retriever" in st:
        out += [
            "## Retriever",
            "",
            "Page-based relevance against the golden pages. Hybrid + rerank is the app's default.",
            "",
            retrieval_table(st["retriever"], "method"),
            "",
        ]
    if "reranker" in st:
        out += [
            "## Reranker",
            "",
            "Hybrid search + rerank, with each model as the reranker.",
            "",
            retrieval_table(st["reranker"], "reranker"),
            "",
        ]
    if "chunking" in st:
        out += [
            "## Chunking",
            "",
            "Plain recursive chunks vs chunks with an LLM context note, per retrieval method.",
            "",
        ]
        for method, by_chunking in st["chunking"].items():
            out += [f"**{method}**", "", retrieval_table(by_chunking, "chunking"), ""]
    for key in ("generator", "oracle"):
        if key in st:
            avg = st[key]["average"]
            out += [
                f"## {STAGE_TITLES[key]}",
                "",
                f"Answer model `{st[key]['generator']}`, judge `{st[key]['judge']}`, context: {st[key]['context']}.",
                "",
                table(
                    [*GENERATION_COLUMNS, "sec/answer"],
                    [
                        [
                            *(f(avg[c]) for c in GENERATION_COLUMNS),
                            f(avg["seconds_per_answer"], 1),
                        ]
                    ],
                ),
                "",
            ]
    if "triad" in st:
        t = st["triad"]
        out += [
            "## RAG triad",
            "",
            "Scored with DeepEval over every golden question (the real pipeline answers each).",
            "",
            table(
                ["metric", "score", "not scored"],
                [
                    [n, f(v), str(t["judge_failures"][n])]
                    for n, v in t["average"].items()
                ],
            ),
            "",
        ]
    if "latency" in st:
        mean = st["latency"]["mean"]
        out += [
            "## Latency",
            "",
            f"Mean seconds per stage over {len(st['latency']['rows'])} questions (no cache).",
            "",
            table(list(mean), [[f(v, 1) for v in mean.values()]]),
            "",
        ]

    out += [
        "## Cost of this run",
        "",
        "Estimated from LiteLLM's price list (your provider's bill is the truth).",
        "",
    ]
    rows = []
    for name in ("indexing", *st.keys(), "total"):
        u = usage.get(name)
        if not u:
            continue
        rows.append(
            [
                STAGE_TITLES.get(name, name.capitalize()),
                str(u["calls"]),
                f"{u['input_tokens']:,}",
                f"{u['output_tokens']:,}",
                f"${u['estimated_cost_usd']:.2f}",
            ]
        )
    out += [table(["stage", "calls", "tokens in", "tokens out", "cost"], rows), ""]
    if data.get("errors"):
        out += [
            "## Failed stages",
            "",
            *(f"- **{k}**: {v}" for k, v in data["errors"].items()),
            "",
        ]
    out += [
        "## Reproduce",
        "",
        "```bash",
        f"uv run python eval/run_provider_eval.py {data['provider']}",
        "```",
        "",
    ]
    return "\n".join(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "file", nargs="?", help="a results JSON (default: the latest full run)"
    )
    parser.add_argument(
        "--allow-partial", action="store_true", help="accept a run that used --limit"
    )
    args = parser.parse_args()

    files = (
        [args.file]
        if args.file
        else sorted(glob.glob(str(PROJECT_DIR / "eval" / "results" / "*.json")))
    )
    candidates = []
    for name in files:
        data = json.loads(Path(name).read_text())
        if args.file or args.allow_partial or not data.get("limit"):
            candidates.append((name, data))
    if not candidates:
        sys.exit(
            "no saved full run found in eval/results/ (use --allow-partial for a --limit run)"
        )
    name, data = candidates[-1]
    OUT.write_text(build(data))
    print(
        f"Wrote {OUT.relative_to(PROJECT_DIR)} from {Path(name).relative_to(PROJECT_DIR)}"
    )
