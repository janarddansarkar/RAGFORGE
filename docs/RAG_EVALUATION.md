# RAGForge: RAG Evaluation

- The primary objective of this project was to learn and implement a comprehensive RAG evaluation process. 
- A golden dataset (45 questions over three papers, each with a reference answer and its source pages) and built offline evaluations of the individual RAG components on the `offline-eval` branch (retriever, re-ranker, generator and the settings search). 
- The component evaluations use the golden dataset to compare configurations and identify the best-performing one; the RAG triad and latency need no golden data.

## What is evaluated

| # | Evaluation | Metrics | Needs golden data |
| --- | --- | --- | --- |
| 1 | Retriever | Recall@K, Precision@K, Hit Rate@K, MRR | Golden pages |
| 2 | Re-ranker | The same four metrics, with and without the re-ranker | Golden pages |
| 3 | Generator | Faithfulness, Answer Relevancy, Answer Correctness (plus Semantic Similarity and Context Recall) | Golden answers |
| 4 | End-to-end pipeline | The RAG triad and latency per stage | None |
| 5 | RAG triad | Context Relevance, Groundedness, Answer Relevance (DeepEval) | None |


## RAG pipeline workflow and where each evaluation applies

**Indexing (once per document set):** PDF pages (OCR for scans) → chunks of 600 characters, overlap 90, each with an LLM context note → `all-minilm` embeddings → Chroma vector store.

**Answering (every question):**

1. Input guard (`mistral:7b`) blocks injection and malicious input.
2. Follow-up rewrite and session cache check.
3. **Retriever:** dense search plus BM25, fused with RRF, 20 candidates. *Evaluation 1: retriever.*
4. **Re-ranker:** cross-encoder `mxbai-rerank-base-v1`, top 5 kept. *Evaluation 2: re-ranker.*
5. **Generator:** `qwen3:14b` writes a cited answer, with today's date in the prompt. *Evaluation 3: generator.*
6. The answer and its page citations are returned.
7. **RAG triad** scores the answer (judge `mistral:7b`). *Evaluations 4 and 5: end to end.*

The whole path, from question to scored answer, is the end-to-end evaluation (RAG triad plus latency per stage).

## 1. Retriever evaluation

**Question it answers:** 
- does the search find the right pages, and 
- how high does it rank them?

**What counts as a right result:** 
- a retrieved chunk is "relevant" if it comes from the golden document and one of the golden pages. 
- Chunk IDs change whenever chunking changes, so judging by page keeps scores comparable across setups.

**Metrics** (each is worked out per question, then averaged over the 45 questions; K = 1, 3, 5, 10, and Precision at 5):

| Metric | Remember it as | How it is calculated |
| --- | --- | --- |
| Recall@K | "Did we find all the pages we need?" | Golden pages found in the top K ÷ all golden pages |
| Precision@K | "How much of what we fetched is useful?" | Relevant chunks in the top K ÷ K |
| Hit Rate@K | "Did we find at least one right page?" | 1 if any relevant chunk is in the top K, otherwise 0 |
| MRR | "How near the top is the first right page?" | 1 ÷ rank of the first relevant chunk (rank 1 gives 1, rank 2 gives 0.5, rank 3 gives 0.33), 0 if none |

**Example:** the golden pages are 4 and 5, and the results come back as pages 9, 4, 12, 5, 2. Recall@3 = 1 of 2 pages = 0.5. Recall@5 = 2 of 2 = 1.0. Precision@5 = 2 of 5 = 0.4. Hit Rate@5 = 1. The first right page is at rank 2, so MRR = 0.5.

**Results** (45 questions):

| Method | Recall@1 | Recall@5 | Precision@5 | Hit Rate@5 | MRR |
| --- | --- | --- | --- | --- | --- |
| dense (vectors) | 0.47 | 0.71 | 0.34 | 0.78 | 0.65 |
| bm25 (keywords) | 0.56 | 0.78 | 0.36 | 0.82 | 0.72 |
| hybrid (dense + bm25) | 0.53 | 0.88 | 0.40 | 0.91 | 0.74 |

Hybrid search finds more of the right pages (Recall@5 0.71 to 0.88) but does not put them first (Recall@1 0.53, MRR 0.74). Fixing the order is the re-ranker's job.

## 2. Re-ranker evaluation

**Question it answers:** does re-sorting the candidates move the right page up?

**How it works:** 
- The retriever hands over 20 candidates, 
- the cross-encoder re-scores each one against the question, and the best 5 are kept. So it can change which chunks make the top 5, not only their order.

**Metrics:** the same four as the retriever (Recall@K, Precision@K, Hit Rate@K, MRR), run twice: without the re-ranker and with it. The two to watch are **Recall@1** and **MRR**, because they show whether the right page moved up.

**Results** (45 questions):

| Method | Recall@1 | Recall@5 | Precision@5 | Hit Rate@5 | MRR |
| --- | --- | --- | --- | --- | --- |
| dense | 0.47 | 0.71 | 0.34 | 0.78 | 0.65 |
| dense + re-rank | 0.73 | 0.86 | 0.45 | 0.89 | 0.87 |
| hybrid | 0.53 | 0.88 | 0.40 | 0.91 | 0.74 |
| hybrid + re-rank | 0.76 | 0.91 | 0.49 | 0.96 | 0.92 |

The re-ranker is the biggest gain in retrieval. For hybrid search, Recall@1 rises from 0.53 to 0.76 and MRR from 0.74 to 0.92, because the right page was usually already retrieved, just not first. With it, the right page is in the top 5 for 96% of questions.

**Which re-ranker** (hybrid + re-rank, an earlier run in October, so compare rows within this table only):

| Re-ranker | MRR | Recall@5 | Hit Rate@5 |
| --- | --- | --- | --- |
| bge-reranker-v2-m3 | 0.85 | 0.93 | 0.96 |
| mxbai-rerank-base-v1 | 0.84 | 0.93 | 0.98 |
| bge-reranker-base | 0.81 | 0.89 | 0.93 |
| ms-marco-MiniLM-L-12-v2 | 0.78 | 0.90 | 0.93 |
| bge-reranker-large | 0.78 | 0.91 | 0.93 |
| ms-marco-MiniLM-L-6-v2 | 0.74 | 0.89 | 0.93 |

Bigger was not better. I use `mxbai-rerank-base-v1`: within 0.01 MRR of the best and the highest Hit Rate@5.

## 3. Generator evaluation

**Question it answers:** is the answer faithful to its sources, on topic and correct?

**How it is scored:** the pipeline answers each golden question. A judge model (`mistral:7b`, kept different from the answering model so it never grades its own answers) answers short YES / NO style questions, and an embedding model measures closeness of meaning. Every score is between 0 and 1.

**Metrics:**

| Metric | Remember it as | How it is calculated |
| --- | --- | --- |
| Faithfulness | "Does it stick to its sources?" | The answer is split into sentences. The judge says YES or NO to "is this sentence supported by the retrieved passages?". Score = share of YES. A refusal ("can't find that") scores 1.0 because it claims nothing. |
| Answer relevancy | "Does it answer what was asked?" | The judge writes the question the answer seems to respond to. Score = embedding similarity (cosine) between that question and the real one. |
| Answer correctness | "Is it right?" | The judge compares the answer with the golden answer: CORRECT = 1, PARTLY = 0.5, WRONG = 0. |
| Semantic similarity | "Same meaning as the golden answer?" | Embedding similarity between the answer and the golden answer. No judge involved, so it is a steady anchor. |
| Context recall | "Did retrieval give the generator what it needed?" | The golden answer is split into sentences and the judge checks each against the retrieved passages. Score = share found. Low values point at retrieval, not the generator. |

**Results** (`qwen3:14b` answering, 45 questions, retrieved context). The same answers were scored by two judges:

| Metric | Judge `mistral:7b` | Judge `qwen3:14b` |
| --- | --- | --- |
| Faithfulness | 1.00 | 0.99 |
| Answer relevancy | 0.79 | 0.84 |
| Answer correctness | 0.97 | 0.80 |
| Semantic similarity | 0.69 | 0.69 |
| Context recall | 1.00 | 0.98 |
| Seconds per answer | 4.2 | 4.3 |

The answers are faithful and the right information is retrieved. Correctness is where the judge matters: the same answers score 0.97 under `mistral:7b`, which is lenient (it marked none of the 45 wrong), and 0.80 under `qwen3:14b`, which is stricter. Semantic similarity, which has no judge, is the same 0.69 under both. So I compare setups under one judge instead of trusting absolute scores.

### Oracle mode

Oracle mode evaluates the generator on its own, by taking retrieval out of the picture. It is the `--oracle` option of `eval/generator_eval.py`.

- **Normal run:** the pipeline retrieves and re-ranks chunks, and the generator answers from them.
- **Oracle run:** the retriever and re-ranker are skipped. For each golden question, the chunks that come from the golden document and one of the golden pages are taken from the index, the first 5 in document order (the same `top_k` as a normal run). The generator answers from those chunks through the same answer prompt, and the judge scores the answer with the same metrics.
- **How to read it:** the generator now always sees the right pages, so a low score is the generator's fault, not retrieval's. The gap between a normal run and an oracle run is what imperfect retrieval costs.
- **Where it is used:** stage 5 of `eval/find_best.py` compares the candidate answer models in oracle mode, so a model is not penalised for retrieval misses.
- **Saved runs:** an oracle run is saved like any other run in `eval/generator_runs/`, with `_oracle` in the file name and `context: "oracle (golden pages)"` inside. The saved oracle runs are from October, with other models, so they are not comparable with the results above.

## 4. Pipeline-level evaluation: the RAG triad and latency

**Question it answers:** do the question, the retrieved context and the answer fit together, with no golden answer needed?

The triad scores the final output of the whole pipeline, after the generator has produced its answer. It is used in two places:

- **In the app (`dev`):** after each answer is written, the judge model scores it and the three scores appear under the answer with the judge's reasons.
- **Offline:** `eval/triad_eval.py` runs the real pipeline over the golden questions and saves the scores to `eval/triad_runs/`.

**Metrics** (DeepEval, judge `mistral:7b`, pass threshold 0.5). Picture a triangle: one score for each side.

| Metric | Remember it as | How it is calculated | Points at |
| --- | --- | --- | --- |
| Context Relevance | "Is the retrieved text on topic?" (question to context) | The retrieved chunks are split into statements and the judge marks each relevant or not. Score = relevant statements ÷ all statements. | Retriever, re-ranker |
| Groundedness | "Is the answer backed by the context?" (context to answer) | Claims are pulled out of the answer and the judge checks whether each one contradicts the retrieved text. Score = claims not contradicted ÷ all claims. | Generator |
| Answer Relevance | "Does the answer address the question?" (question to answer) | The answer is split into statements and the judge marks each relevant, irrelevant or borderline. Score = (relevant + borderline) ÷ all statements. | Generator, prompt |

Read them together: 
- Low Context Relevance with high Groundedness means retrieval brought noisy chunks and the answer is faithful to them. 
- High Context Relevance with low Groundedness means the generator added claims its sources do not support.

**Latency per stage:** `eval/time_pipeline.py` times the guard, the search, the extra time the re-ranker adds, the full answer and the triad scoring, for a few questions. I did not save a latency run, so this document gives no timing results beyond the seconds per answer above.

**Results:** a spot check on the first 3 golden questions (`qwen3:14b` answering, `mistral:7b` judging):

| Question | Context Relevance | Groundedness | Answer Relevance |
| --- | --- | --- | --- |
| L01 what problem LoRA solves | 0.22 | 0.50 | 1.00 |
| L02 what LoRA does to the weights | 0.75 | 0.50 | 0.50 |
| L03 how the update is parameterized | 0.22 | 0.67 | 0.50 |
| Average | 0.40 | 0.56 | 0.67 |

- The triad is a signal to investigate, not a verdict on correctness. 

## From `offline-eval` to `dev`

1. `eval/find_best.py` tunes the options for every component (chunking, embedder, retrieval method, re-ranker, generator), one stage at a time, and prepares the best configuration.
   - **Tuning and held-out check:** the chunking, embedder, retrieval method and re-ranker stages are tuned on 30 of the 45 golden questions. The winning retrieval setup is then compared with the current settings on the other 15 (every 3rd question), and the current settings are kept if the winner scores lower on MRR.
   - **Generator stage:** it runs in oracle mode on all 45 questions, so a model is not penalised for retrieval misses.
   - **Near-ties:** a candidate within 0.01 MRR of the best (0.03 answer correctness for the generator) counts as a tie, and the faster one wins.
2. The winning settings are saved to `best_config.json`.
3. On `dev`, `load_config()` lays `best_config.json` over `config.toml`, and those settings drive the complete pipeline above: 600 / 90 contextual chunks, `all-minilm`, hybrid search, `mxbai-rerank-base-v1`, `qwen3:14b`, and `mistral:7b` as guard and judge.
4. The final configuration used in the `dev` pipeline is in the table below. `best_config.json` (written by `find_best.py`) overrides `config.toml` for the settings it contains; everything else comes from `config.toml`.

| Component | Setting | Value | Set by |
| --- | --- | --- | --- |
| Chunking | Strategy | Recursive, with an LLM context note added to each chunk | `best_config.json` |
| Chunking | Chunk size / overlap | 600 / 90 characters | `best_config.json` |
| Embeddings | Model | `all-minilm` | `best_config.json` |
| Retriever | Method | `hybrid_rerank`: dense + BM25 fused with RRF | `best_config.json` |
| Retriever | Candidates / RRF constant | 20 / 60 | `config.toml` |
| Re-ranker | Model | Cross-encoder `mxbai-rerank-base-v1`, top 5 kept | `best_config.json` |
| Generator | Model / max new tokens | `qwen3:14b` / 256 | `best_config.json` |
| Input guard | Model | `mistral:7b` | `config.toml` |
| Judge (generator eval and RAG triad) | Model | `mistral:7b`, kept different from the generator | `config.toml` |

## Local models and reproduce

All models run locally through Ollama (the cross-encoder re-ranker downloads from Hugging Face once), so the pipeline and its evaluation need no paid model APIs.

```bash
git clone -b dev https://github.com/janarddansarkar/RAGFORGE.git && cd RAGFORGE && ./setup.sh
uv run python eval/retriever_eval.py
uv run python eval/reranker_compare.py
uv run python eval/generator_eval.py            # add --oracle for oracle mode
uv run python eval/triad_eval.py --limit 5
uv run python eval/find_best.py
```
