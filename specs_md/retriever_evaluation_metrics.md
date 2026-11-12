# Retriever Evaluation Metrics

This document defines four core metrics for evaluating a RAG retriever:

- Recall@K
- Precision@K
- MRR (Mean Reciprocal Rank)
- HitRate@K

These metrics evaluate whether the retriever finds relevant chunks and how well it ranks them.

---

## 1. Recall@K

### Definition

**Recall@K** measures whether the retriever successfully retrieves the relevant chunks within the top `K` results.

It answers:

> **"How much of the relevant information did the retriever retrieve in its top K results?"**

### Formula

For a query:

```text
Recall@K = (# relevant chunks retrieved in top K)
          / (# total relevant chunks)
```

If each query has exactly one relevant chunk, Recall@K becomes:

```text
Recall@K = 1  if the relevant chunk is in the top K
           0  otherwise
```

### Example

Suppose the relevant chunk is:

```text
chunk_42
```

The retriever returns:

```text
Top 5:
1. chunk_10
2. chunk_17
3. chunk_42   <-- relevant
4. chunk_31
5. chunk_08
```

Then:

```text
Recall@5 = 1
```

If `chunk_42` is not present in the top 5:

```text
Recall@5 = 0
```

### Dataset-level Recall@K

For `N` queries:

```text
Recall@K = sum(query-level Recall@K) / N
```

### Implementation note

For a RAG system, commonly evaluate:

```text
Recall@1
Recall@3
Recall@5
Recall@10
```

`Recall@5` is particularly useful when the generator receives the top 5 retrieved chunks as context.

---

## 2. Precision@K

### Definition

**Precision@K** measures the proportion of the top `K` retrieved chunks that are relevant.

It answers:

> **"Of the chunks I retrieved in the top K, how many are actually relevant?"**

### Formula

```text
Precision@K = (# relevant chunks in top K) / K
```

### Example

Suppose the retriever returns:

```text
Top 5:
1. Relevant
2. Relevant
3. Irrelevant
4. Irrelevant
5. Relevant
```

There are 3 relevant chunks out of 5:

```text
Precision@5 = 3 / 5
            = 0.60
```

### Dataset-level Precision@K

For `N` queries:

```text
Precision@K = average of query-level Precision@K
```

### Implementation note

High precision means the retrieved context contains less irrelevant information.

This matters in RAG because irrelevant chunks can:

- increase context size,
- increase token usage,
- distract the generator,
- potentially increase hallucination or incorrect answers.

---

## 3. MRR — Mean Reciprocal Rank

### Definition

**MRR (Mean Reciprocal Rank)** measures how highly the **first relevant retrieved result** appears in the ranking.

It answers:

> **"How high is the first relevant chunk ranked?"**

### Reciprocal Rank

For one query:

```text
Reciprocal Rank = 1 / rank_of_first_relevant_result
```

Examples:

```text
Relevant result at rank 1 → RR = 1/1 = 1.00
Relevant result at rank 2 → RR = 1/2 = 0.50
Relevant result at rank 3 → RR = 1/3 = 0.33
Relevant result at rank 5 → RR = 1/5 = 0.20
```

If no relevant result is retrieved:

```text
RR = 0
```

### Formula

For `N` queries:

```text
MRR = (1/N) × Σ ReciprocalRank_i
```

### Example

Suppose three queries have their first relevant result at:

```text
Query 1 → rank 1 → RR = 1.00
Query 2 → rank 2 → RR = 0.50
Query 3 → rank 5 → RR = 0.20
```

Then:

```text
MRR = (1.00 + 0.50 + 0.20) / 3
    = 0.567
```

### Implementation note

MRR is especially useful when you care about getting the **first relevant chunk as high as possible**.

---

## 4. HitRate@K

### Definition

**HitRate@K** measures the percentage of queries for which **at least one relevant chunk appears in the top K retrieved results**.

It answers:

> **"For how many queries did the retriever find at least one relevant chunk?"**

### Formula

For each query:

```text
Hit@K = 1  if at least one relevant chunk is in top K
        0  otherwise
```

Across the dataset:

```text
HitRate@K = (# queries with at least one relevant result in top K)
            / (# total queries)
```

### Example

Suppose there are 5 queries:

```text
Query 1 → relevant chunk in top 5 → 1
Query 2 → relevant chunk in top 5 → 1
Query 3 → no relevant chunk in top 5 → 0
Query 4 → relevant chunk in top 5 → 1
Query 5 → relevant chunk in top 5 → 1
```

Then:

```text
HitRate@5 = 4 / 5
          = 0.80
          = 80%
```

### Implementation note

HitRate@K is particularly intuitive for RAG because it directly measures:

> **Did the retriever give the LLM at least one relevant piece of context?**

---

# 5. Difference Between the Metrics

| Metric | Main Question | Ranking Sensitive? |
|---|---|---|
| Recall@K | Did we retrieve the relevant information within top K? | Partially |
| Precision@K | How much of top K is relevant? | No |
| MRR | How high is the first relevant result? | Yes |
| HitRate@K | Did we retrieve at least one relevant result? | No |

### Example

Suppose the relevant chunk is `chunk_42`.

Retriever A:

```text
1. chunk_42       Relevant
2. chunk_10
3. chunk_17
4. chunk_31
5. chunk_08
```

Retriever B:

```text
1. chunk_10
2. chunk_17
3. chunk_31
4. chunk_08
5. chunk_42       Relevant
```

For both:

```text
Recall@5 = 1
HitRate@5 = 1
```

But:

```text
MRR(A) = 1/1 = 1.00
MRR(B) = 1/5 = 0.20
```

This demonstrates why multiple metrics should be used together.

---

# 6. Recommended Metrics for This RAG Implementation

For a retriever that returns the top `K` chunks to an LLM, track:

```text
Recall@1
Recall@3
Recall@5
Recall@10

Precision@5

MRR

HitRate@1
HitRate@3
HitRate@5
HitRate@10
```

A practical summary could look like:

```text
Retriever Evaluation
---------------------
Recall@1       : 0.71
Recall@3       : 0.87
Recall@5       : 0.93
Recall@10      : 0.98

Precision@5    : 0.62

MRR            : 0.81

HitRate@1      : 0.71
HitRate@3      : 0.87
HitRate@5      : 0.93
HitRate@10     : 0.98
```

These values are only an example; they are not evaluation results for the current dataset.

---

# 7. Golden Dataset Requirements

To calculate these metrics reliably, the golden dataset should identify which retrieved chunks are considered relevant.

For example:

```json
{
  "id": "A04",
  "query": "What is the formula for scaled dot-product attention?",
  "golden_ans": "Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V",
  "relevant_chunks": [
    "chunk_123"
  ],
  "source": "Attention Is All You Need — §3.2.1, p.4"
}
```

The retriever should return results containing stable chunk IDs:

```json
[
  {
    "chunk_id": "chunk_091",
    "score": 0.91
  },
  {
    "chunk_id": "chunk_123",
    "score": 0.87
  },
  {
    "chunk_id": "chunk_045",
    "score": 0.82
  }
]
```

The evaluator can then compare:

```text
golden_dataset.relevant_chunks
              ↓
        vs.
              ↓
retriever.top_k.chunk_id
```

and calculate the metrics automatically.

---

# 8. Important Implementation Detail

Do **not** calculate retriever metrics by comparing the generated answer with `golden_ans`.

Retriever metrics evaluate the **retrieval stage**.

The evaluation flow should be:

```text
Golden Query
     |
     v
Retriever
     |
     v
Top-K Retrieved Chunk IDs
     |
     v
Compare with Golden Relevant Chunk IDs
     |
     +----> Recall@K
     +----> Precision@K
     +----> MRR
     +----> HitRate@K
```

Answer-level metrics such as answer correctness, faithfulness, and answer relevance belong to the **generation/RAG evaluation stage**, not the pure retriever evaluation stage.
