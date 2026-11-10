# RAGForge: RAG Evaluation

The primary objective of this project was to learn and implement a comprehensive RAG evaluation process.

In the `offline-eval` branch, I prepared a golden evaluation dataset (45 questions over three papers, each with a reference answer and its source pages) and ran offline evaluations of the individual RAG components and of the complete pipeline. All evaluations use the golden dataset to compare configurations, understand component-level performance and identify the best-performing configuration.

## What is evaluated

| # | Evaluation | Metrics | Needs golden data |
| --- | --- | --- | --- |
| 1 | **Retriever** | Recall@K, Precision@K, Hit Rate@K, MRR | Golden pages |
| 2 | **Re-ranker** | The same four metrics, with and without the re-ranker | Golden pages |
| 3 | **Generator** | Faithfulness, Answer Relevancy, Answer Correctness (plus Semantic Similarity and Context Recall) | Golden answers |
| 4 | **End-to-end pipeline** | The RAG triad and latency per stage | None |
| 5 | **RAG triad** | Context Relevance, Groundedness, Answer Relevance (DeepEval) | None |

Not implemented: NDCG@K for the re-ranker and Context Precision for the pipeline.

## RAG pipeline workflow and where each evaluation applies

```
 INDEXING (once per document set)
 PDF -> pages (OCR for scans) -> chunks (600 chars, overlap 90, + LLM context note)
     -> embeddings (all-minilm) -> Chroma vector store

 ANSWERING (every question)
 question
   -> input guard (mistral:7b)                     blocks injection / malicious input
   -> follow-up rewrite + session cache check
   -> RETRIEVER: dense + BM25, fused with RRF      (20 candidates)    <-- 1. retriever eval
   -> RE-RANKER: cross-encoder mxbai-rerank-base   (top 5 kept)       <-- 2. re-ranker eval
   -> GENERATOR: qwen3:14b, cited answer, today's date in the prompt  <-- 3. generator eval
   -> answer + page citations
   -> RAG TRIAD scores the answer (judge: mistral:7b)                 <-- 4 and 5. end-to-end eval

 Whole path, question to scored answer: end-to-end evaluation (RAG triad + latency)
```

## Where the RAG triad is used

The triad scores the **final output of the whole pipeline**, after the generator has produced its answer. It uses the question, the retrieved chunks and the answer, so it needs no golden answer. It is used in two places:

- **In the app (`dev`):** every answer is scored by the judge model in the background, and the three scores appear under the answer with the judge's reasons.
- **Offline:** `eval/triad_eval.py` runs the real pipeline over the golden questions and saves the scores to `eval/triad_runs/`.

| Triad score | Link it checks | What it measures |
| --- | --- | --- |
| Context Relevance | question to retrieved chunks | How much of the retrieved text is relevant to the question (points at the retriever and re-ranker) |
| Groundedness | chunks to answer | Whether the answer's claims contradict the retrieved text (points at the generator) |
| Answer Relevance | question to answer | Whether the answer addresses the question |

The triad is a signal to investigate, not a verdict on correctness: a spot check on the first 3 golden questions with `mistral:7b` as judge scored a correct answer 0.50 on groundedness (saved in `eval/triad_runs/2026-11-10_233525.json`).

## Results

Retrieval (45 questions); hybrid search plus the cross-encoder re-ranker is the best:

| Method | Recall@1 | Recall@5 | Hit Rate@5 | MRR |
| --- | --- | --- | --- | --- |
| dense | 0.47 | 0.71 | 0.78 | 0.65 |
| bm25 | 0.56 | 0.78 | 0.82 | 0.72 |
| hybrid | 0.53 | 0.88 | 0.91 | 0.74 |
| dense + re-rank | 0.73 | 0.86 | 0.89 | 0.87 |
| hybrid + re-rank | 0.76 | 0.91 | 0.96 | 0.92 |

Generator (`qwen3:14b`, 45 questions, `mistral:7b` judge): faithfulness 1.00, answer correctness 0.97, answer relevancy 0.79. The judge changes correctness more than the model does (about 0.2 between judges), so I compare setups under one judge instead of trusting absolute scores.

## From `offline-eval` to `dev`

1. `eval/find_best.py` tests configurations one stage at a time (chunking, embedder, retrieval method, re-ranker, generator), tuning on two of every three golden questions and checking the retrieval pick on the held-out third. Near-ties go to the faster option.
2. The winning settings are saved to `best_config.json`.
3. On `dev`, `load_config()` lays `best_config.json` over `config.toml`, and those settings drive the complete pipeline above: 600 / 90 contextual chunks, `all-minilm`, hybrid search, `mxbai-rerank-base-v1`, `qwen3:14b`, and `mistral:7b` as guard and judge.

## Local models

All models run locally through **Ollama** (the cross-encoder re-ranker downloads from Hugging Face once), so the pipeline and its evaluation need no paid model APIs.

## Reproduce

```bash
git clone -b dev https://github.com/janarddansarkar/RAGFORGE.git && cd RAGFORGE && ./setup.sh
uv run python eval/retriever_eval.py
uv run python eval/reranker_compare.py
uv run python eval/generator_eval.py
uv run python eval/triad_eval.py --limit 5
uv run python eval/find_best.py
```
