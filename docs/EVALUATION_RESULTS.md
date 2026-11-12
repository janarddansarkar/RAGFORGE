# RAGForge: evaluation results

Provider: **openai**. Golden questions: **45** (45 = the full set). Run time: 101 min. Saved by `eval/run_provider_eval.py`; this page is made from the saved JSON by `eval/make_report.py` (no model calls).

## Models

| component | model |
|---|---|
| llm | `openai/gpt-4o` |
| guard | `openai/gpt-4o-mini` |
| judge | `openai/gpt-4o-mini` |
| reranker | `openai/gpt-4o-mini` |
| embedding | `openai/text-embedding-3-small` |

Scores are 0 to 1, higher is better. A judge model scores the generator and the triad, so read them as a comparison between setups, not as absolute truth.

## Retriever

Page-based relevance against the golden pages. Hybrid + rerank is the app's default.

| method | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Precision@5 | MRR | HitRate@5 |
|---|---|---|---|---|---|---|---|
| dense | 0.49 | 0.72 | 0.82 | 0.90 | 0.38 | 0.71 | 0.89 |
| bm25 | 0.57 | 0.73 | 0.82 | 0.93 | 0.41 | 0.74 | 0.91 |
| hybrid | 0.62 | 0.77 | 0.89 | 0.92 | 0.42 | 0.79 | 0.93 |
| hybrid_rerank | 0.72 | 0.88 | 0.92 | 0.96 | 0.46 | 0.88 | 0.96 |

## Reranker

Hybrid search + rerank, with each model as the reranker.

| reranker | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Precision@5 | MRR | HitRate@5 |
|---|---|---|---|---|---|---|---|
| openai/gpt-4o-mini | 0.70 | 0.88 | 0.92 | 0.96 | 0.46 | 0.86 | 0.96 |
| openai/gpt-4o | 0.74 | 0.88 | 0.91 | 0.96 | 0.45 | 0.89 | 0.93 |

## Chunking

Plain recursive chunks vs chunks with an LLM context note, per retrieval method.

**dense**

| chunking | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Precision@5 | MRR | HitRate@5 |
|---|---|---|---|---|---|---|---|
| recursive_600_90 | 0.47 | 0.71 | 0.80 | 0.91 | 0.32 | 0.67 | 0.87 |
| contextual_body_recursive_600_90 | 0.49 | 0.72 | 0.82 | 0.90 | 0.38 | 0.71 | 0.89 |

**bm25**

| chunking | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Precision@5 | MRR | HitRate@5 |
|---|---|---|---|---|---|---|---|
| recursive_600_90 | 0.52 | 0.67 | 0.77 | 0.92 | 0.34 | 0.70 | 0.82 |
| contextual_body_recursive_600_90 | 0.57 | 0.73 | 0.82 | 0.93 | 0.41 | 0.74 | 0.91 |

**hybrid_rerank**

| chunking | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Precision@5 | MRR | HitRate@5 |
|---|---|---|---|---|---|---|---|
| recursive_600_90 | 0.73 | 0.93 | 0.96 | 0.99 | 0.44 | 0.91 | 0.98 |
| contextual_body_recursive_600_90 | 0.71 | 0.87 | 0.92 | 0.96 | 0.47 | 0.87 | 0.96 |

## Generator (retrieved chunks)

Answer model `openai/gpt-4o`, judge `openai/gpt-4o-mini`, context: retrieved.

| faithfulness | answer_relevancy | answer_correctness | semantic_similarity | context_recall | sec/answer |
|---|---|---|---|---|---|
| 0.71 | 0.84 | 0.98 | 0.71 | 0.92 | 18.9 |

## Generator (golden pages)

Answer model `openai/gpt-4o`, judge `openai/gpt-4o-mini`, context: oracle (golden pages).

| faithfulness | answer_relevancy | answer_correctness | semantic_similarity | context_recall | sec/answer |
|---|---|---|---|---|---|
| 0.94 | 0.75 | 0.91 | 0.63 | 0.90 | 1.1 |

## RAG triad

Scored with DeepEval over every golden question (the real pipeline answers each).

| metric | score | not scored |
|---|---|---|
| Context relevance | 0.41 | 0 |
| Groundedness | 0.72 | 0 |
| Answer relevance | 0.94 | 0 |

## Latency

Mean seconds per stage over 3 questions (no cache).

| guard | search | rerank | answer | triad |
|---|---|---|---|---|
| 0.8 | 0.8 | 3.8 | 1.3 | 39.3 |

## Cost of this run

Estimated from LiteLLM's price list (your provider's bill is the truth).

| stage | calls | tokens in | tokens out | cost |
|---|---|---|---|---|
| Indexing (chunk notes + embeddings, one time) | 0 | 0 | 0 | $0.00 |
| Retriever | 1035 | 218,632 | 900 | $0.03 |
| Reranker | 1890 | 434,704 | 1,800 | $0.58 |
| Chunking | 1980 | 393,502 | 1,800 | $0.06 |
| Generator (retrieved chunks) | 1310 | 413,750 | 3,917 | $0.19 |
| Generator (golden pages) | 355 | 181,884 | 2,675 | $0.15 |
| RAG triad | 1581 | 613,357 | 121,153 | $0.29 |
| Latency | 133 | 47,399 | 6,942 | $0.02 |
| Total | 8284 | 2,303,228 | 139,187 | $1.32 |

## Reproduce

```bash
uv run python eval/run_provider_eval.py openai
```
