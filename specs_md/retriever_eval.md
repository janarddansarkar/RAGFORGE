# Retriever Evaluation (`eval/retriever_eval.py`)

This document explains what the retriever evaluation measures, how each number is
calculated, and how to read it. For the general theory behind the metrics see
`retriever_evaluation_metrics.md`; for the retrieval methods themselves see
`retriever_methods.md`.

Run it:

```bash
uv run python eval/retriever_eval.py      # needs the vector store built first
```

Results are printed as a table and saved to `eval/retriever_results.json`.

---

## 1. What is being evaluated

The **retriever** takes a question and returns a ranked list of chunks from the vector
store. The evaluation asks one thing:

> Did the retriever put the chunks that contain the answer near the top of the list?

It does **not** look at the generated answer. That is the job of the generator
evaluation (`generator_eval.md`). Keeping the two apart tells you which half of the
pipeline is the problem when an answer is bad.

Every retrieval method listed under `methods` in `config.toml` is scored on the same
questions: `dense`, `bm25`, `hybrid`, `dense_rerank` and `hybrid_rerank` (the LLM-based
methods `rewrite_*`, `multi_query_*` and `hyde_*` can be added to the list).

## 2. The test set (golden dataset)

`golden_data/rag_retriever_golden_dataset_45.json` holds **45 questions**. Each has:

| field | meaning |
|---|---|
| `id` | question id, e.g. `L01` |
| `query` | the question |
| `golden_ans` | a reference answer (used by the generator eval, not here) |
| `section` | where the answer is in the paper |
| `source` | document and pages that contain the answer, e.g. `LoRA — pp.4–5` |

## 3. What counts as "relevant"

Chunk IDs change every time the store is rebuilt with a different chunking strategy, so
relevance is decided by **page**, not by chunk:

> A retrieved chunk is **relevant** if it comes from the golden document **and** from
> one of the golden pages.

Example: for `source = "LoRA — pp.4–5"`, any chunk from `LoRA.pdf` page 4 or 5 is
relevant, whichever chunking strategy produced it. This is what lets different chunking
strategies be compared fairly.

The question has `total_relevant` = the number of golden pages (here 2).

## 4. The metrics

All metrics are computed **per question** from the ranked list of results, then
**averaged over all 45 questions**. Every value is between 0 and 1; higher is better.

The code is in `metrics/retriever_metrics.py`. Let `matches[i]` say whether the result
at rank `i+1` is relevant (and which golden page it hit).

### Recall@K  (K = 1, 3, 5, 10)

**What it means:** how much of the information needed was found in the top K results.
This is the most important retriever metric, because a generator can't use a page it
never receives.

**How it is calculated:**

```text
Recall@K = (number of distinct golden pages found in the top K) / (number of golden pages)
```

Several chunks from the same page count once. A question with golden pages {4, 5} where
the top K contains chunks from page 4 only scores 0.5.

### Precision@K  (K = 5)

**What it means:** how clean the top K is, the share of returned chunks that are
relevant. Low precision means the generator is given a lot of irrelevant text.

**How it is calculated:**

```text
Precision@K = (number of relevant chunks in the top K) / K
```

Each relevant chunk counts, including several from the same page. The denominator is
always K, even if the method returned fewer than K chunks (BM25 can).

**Reading it:** the maximum is often below 1. A question with one relevant page and
K = 5 can still have several relevant chunks on that page, but there are only so many
chunks per page. Use Precision@K to **compare methods**, not as an absolute score.

### HitRate@K  (K = 1, 3, 5, 10)

**What it means:** did the retriever get *anything* right in the top K. It is the
forgiving version of Recall@K: one relevant chunk is enough.

**How it is calculated:**

```text
HitRate@K = 1 if at least one relevant chunk is in the top K, otherwise 0
```

Averaged over questions, it is the fraction of questions with at least one hit.
HitRate@K is always at least Recall@K.

### MRR (Mean Reciprocal Rank)

**What it means:** how high the *first* relevant chunk appears. It rewards putting the
right chunk at the very top, which matters when only the first few chunks reach the
generator.

**How it is calculated:**

```text
Reciprocal rank = 1 / (rank of the first relevant chunk)     (0 if none is found)
MRR             = average of the reciprocal ranks over all questions
```

First relevant chunk at rank 1 gives 1.0, at rank 2 gives 0.5, at rank 4 gives 0.25.

## 5. Settings

In `config.toml`:

| setting | value | meaning |
|---|---|---|
| `eval_ks` | `[1, 3, 5, 10]` | K values for Recall@K and HitRate@K |
| `precision_k` | `5` | K for Precision@K |
| `methods` | list | retrieval methods scored |
| `golden_file` | path | the question set |

Each method retrieves `max(eval_ks, precision_k)` = 10 chunks per question, and the
metrics read prefixes of that list.

## 6. Output

The table has one row per method and one column per metric. The JSON file has the same
numbers. Example (45 questions, `recursive_1000_150` chunking, `bge-m3` embeddings):

| method | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Precision@5 | MRR | HitRate@5 |
|---|---|---|---|---|---|---|---|
| dense | 0.59 | 0.78 | 0.89 | 0.96 | 0.34 | 0.78 | 0.93 |
| bm25 | 0.44 | 0.69 | 0.82 | 0.90 | 0.32 | 0.67 | 0.87 |
| hybrid | 0.61 | 0.86 | 0.90 | 0.96 | 0.34 | 0.82 | 0.96 |
| dense_rerank | 0.62 | 0.90 | 0.93 | 0.97 | 0.39 | 0.84 | 0.96 |
| hybrid_rerank | 0.66 | 0.90 | 0.93 | 0.96 | 0.40 | 0.85 | 0.96 |

These are a snapshot, so rerun the evaluation for current numbers.

## 7. How to read the results

- **Pick on MRR and Recall@3/5 first.** Those decide whether the generator gets the
  right context. Recall@10 and HitRate@10 show the ceiling.
- **Ignore gaps of about 0.03 or less.** There are only 45 questions, so one question is
  worth about 0.02. Small differences are noise.
- **A big Recall@10 but small Recall@1** means the right chunk is found but ranked
  badly, so add or improve a reranker.
- **Low Recall@10** means retrieval misses the page entirely. Look at chunking or the
  embedding model.
- **Precision@5 is for comparison only** (see section 4).

## 8. Limitations

- **Page-level relevance.** A chunk on the right page counts as relevant even if it is
  not the paragraph with the answer, so scores are slightly generous.
- **Small test set.** 45 questions from three papers (LoRA, QLoRA, Attention Is All You
  Need). Results may not carry over to other documents.
- **No answer quality.** A perfect retriever score does not mean good answers.
- **The tuning risk.** Choosing settings on the same 45 questions overfits. `find_best.py`
  holds out every third question to check the final pick.

## 9. Related files

| file | role |
|---|---|
| `eval/retriever_eval.py` | runs the evaluation |
| `metrics/retriever_metrics.py` | the metric math |
| `eval/golden.py` | reads a golden question and decides which chunks are relevant |
| `eval/chunking_compare.py` | runs this evaluation for each chunking strategy |
| `eval/find_best.py` | runs it across chunking, methods and rerankers |
| `eval/retriever_results.json` | latest saved numbers |
