# Retriever Methods for RAG

This document explains the five retrieval methods implemented in `src/retriever.py`, and how they differ from chunking strategies.

The five methods are:

1. Dense (vector) retrieval
2. BM25 (keyword) retrieval
3. Hybrid retrieval (dense + BM25 with RRF fusion)
4. Dense + rerank
5. Hybrid + rerank

---

## 1. Dense (Vector) Retrieval

### Definition

The query is embedded with the same model used for the chunks (`bge-m3`). The vector store returns the `k` chunks whose vectors are closest to the query vector.

### Strengths

- Finds chunks that mean the same thing even with different wording ("car" vs "automobile").
- Fast: one embedding call plus one vector lookup.

### Weaknesses

- Can miss exact terms, names, numbers or acronyms.
- Ranks by general similarity, so the very best chunk is not always first.

---

## 2. BM25 (Keyword) Retrieval

### Definition

BM25 is a classic keyword scoring function. Chunks and the query are lowercased and split into words. Chunks that share rare words with the query score highest. The index is built in memory from the chunks in the store. Chunks with a score of 0 are dropped.

### Strengths

- Excellent for exact matches: names, IDs, error codes, rare terms.
- No embedding model needed at query time.

### Weaknesses

- No understanding of meaning: synonyms and paraphrases are missed.
- Can return fewer than `k` results when few chunks share words with the query.

---

## 3. Hybrid Retrieval (Dense + BM25 with RRF)

### Definition

Run dense and BM25 each for `fetch_k` candidates (default 20), then merge the two ranked lists with **Reciprocal Rank Fusion (RRF)**:

```text
score(chunk) = sum over lists of  1 / (rrf_k + rank)      rrf_k = 60
```

A chunk ranked high in either list rises to the top. A chunk found by both lists gets two contributions, so it rises even more. The top `k` after fusion are returned.

### What is RRF?

**Reciprocal Rank Fusion** merges several ranked lists into one. Dense and BM25 score chunks on different scales (for example 0.82 vs 14.3), so the raw scores cannot be added or compared. RRF ignores the scores and uses only each chunk's **rank position** (rank 1 is the best) in each list.

Example with `rrf_k = 60`:

| Chunk | Dense rank | BM25 rank | Calculation | RRF score |
|---|---|---|---|---|
| A | 1 | 5 | 1/61 + 1/65 | 0.0318 |
| B | 3 | 1 | 1/63 + 1/61 | 0.0323 |
| C | 2 | not found | 1/62 | 0.0161 |
| D | not found | 2 | 1/62 | 0.0161 |

Final order: B, A, then C and D tied.

- B wins because it ranks well in both lists. Agreement between two different search methods is a strong sign of relevance.
- C and D each appear in only one list, so they score about half as much.
- The constant `rrf_k` softens the gap between neighbouring ranks, so one list cannot overrule the other just because it put a chunk at rank 1.

In the code this is `Retriever.rrf()` in `src/retriever.py`. It adds up `1 / (rrf_k + rank)` per chunk ID across the lists and sorts by the total.

### Strengths

- Covers both meaning (dense) and exact words (BM25).
- RRF uses ranks, not raw scores, so there is no need to normalize two different score scales.

### Weaknesses

- Roughly the cost of two searches.
- The order is still decided by rank position, not by reading query and chunk together.

---

## 4. Dense + Rerank

### Definition

Get `fetch_k` candidates from dense search, then a **cross-encoder** (`BAAI/bge-reranker-v2-m3`) scores each (query, chunk) pair and re-orders them. The top `k` are returned. The score is stored in the chunk metadata as `rerank_score`.

### Why it helps

Dense search compares two separately made vectors. A cross-encoder reads the query and the chunk together, so it judges relevance more accurately. It is too slow to run on every chunk, so it only re-orders the small candidate list.

### Weaknesses

- Slower: one model call per candidate.
- It can only re-order what dense search already found. A relevant chunk missing from the 20 candidates cannot be recovered.

---

## 5. Hybrid + Rerank

### Definition

Take the `fetch_k` candidates from **hybrid** search (dense + BM25 + RRF), then rerank them with the cross-encoder. The top `k` are returned. This is the default (`default_method = "hybrid_rerank"`).

### Why it helps

- Hybrid gives the widest candidate pool (meaning + keywords), so fewer relevant chunks are missed.
- The reranker then puts the best ones at the top.

### Weaknesses

- The slowest and most complex pipeline: two searches, a fusion, and a reranker pass.

---

## Comparison

| Method | Finds by | Candidate pool | Final order by | Relative cost |
|---|---|---|---|---|
| `dense` | meaning | top k | vector similarity | low |
| `bm25` | keywords | top k | BM25 score | low |
| `hybrid` | meaning + keywords | 20 + 20 | RRF rank | medium |
| `dense_rerank` | meaning | top 20 | cross-encoder | medium-high |
| `hybrid_rerank` | meaning + keywords | 20 + 20 merged | cross-encoder | high |


---

## Configuration (`config.toml`)

```toml
default_method = "hybrid_rerank"  # used by retrieve() when no method is given
methods = ["dense", "bm25", "hybrid", "dense_rerank", "hybrid_rerank"]
top_k = 5        # results returned to the caller
fetch_k = 20     # candidates gathered before fusion / reranking
rrf_k = 60       # RRF constant
reranker_model = "BAAI/bge-reranker-v2-m3"
```

Usage:

```python
retriever = Retriever()
retriever.retrieve("What rank r does LoRA use?")                 # default method
retriever.retrieve("What rank r does LoRA use?", method="bm25")  # pick one
```
