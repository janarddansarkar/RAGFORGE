# Top 3 Popular Chunking Strategies for RAG

This document is a practical reference for selecting and implementing chunking strategies in a Retrieval-Augmented Generation (RAG) system.

The four strategies covered here are:

1. Recursive / hierarchical text chunking
2. Fixed-size token chunking with overlap
3. Semantic chunking
4. Contextual Chunking

---

## 1. Recursive / Hierarchical Text Chunking

### Definition

Recursive chunking splits a document using a hierarchy of separators. Instead of cutting text at an arbitrary point immediately, it attempts to preserve larger logical units first and recursively splits them when they exceed the target chunk size.

Typical hierarchy:

```text
Document
    ↓
Paragraph
    ↓
Sentence
    ↓
Word
    ↓
Character
```

### Example

If the target chunk size is 500 tokens:

```text
Document
  ↓
Paragraphs
  ↓
Paragraph <= 500 tokens
  → keep as chunk

Paragraph > 500 tokens
  → split into smaller units
  → continue recursively
```

Typical configuration:

```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)
```

### Advantages

- Simple to implement
- Fast
- Good general-purpose baseline
- Attempts to preserve natural text boundaries
- Easy to reproduce and benchmark

### Limitations

- Does not truly understand document meaning
- A logical section can still be split across chunks
- Relationships between headings, tables, and equations may be lost
- Quality depends on separators and chunk-size configuration

### Best suited for

- General PDF/document RAG
- Articles
- Documentation
- Plain text
- A baseline for evaluating more advanced strategies

---

## 2. Fixed-Size Token Chunking with Overlap

### Definition

Fixed-size chunking divides a document into chunks containing approximately the same number of tokens. Adjacent chunks overlap by a specified number of tokens.

Example:

```text
Chunk size  = 500 tokens
Overlap     = 100 tokens
```

Conceptually:

```text
0 ───────────────── 500
        │
        └───────────────┐
                        ↓
                    400 ─────────────── 900
                              │
                              └───────────────┐
                                              ↓
                                          800 ──────── 1300
```

### Example

```python
chunk_size = 500
chunk_overlap = 100
```

Result:

```text
chunk_001 → tokens 0–499
chunk_002 → tokens 400–899
chunk_003 → tokens 800–1299
```

### Advantages

- Very simple
- Predictable chunk sizes
- Fast
- Easy to implement
- Easy to reproduce
- Overlap reduces information loss at boundaries

### Limitations

- Does not understand semantic boundaries
- Can split sentences, paragraphs, equations, or sections
- Overlap increases storage and embedding cost
- May retrieve incomplete context

### Best suited for

- Baseline RAG systems
- Large volumes of relatively uniform text
- Benchmarking
- Situations where predictable chunk sizes matter

---

## 3. Semantic Chunking

### Definition

Semantic chunking attempts to split a document according to changes in meaning rather than only character or token counts.

A common approach is:

```text
Document
    ↓
Sentences / paragraphs
    ↓
Generate embeddings
    ↓
Calculate similarity between adjacent units
    ↓
Detect significant semantic changes
    ↓
Create chunks at those boundaries
```

Example:

```text
Paragraph 1 ─┐
Paragraph 2 ─┤  High semantic similarity
Paragraph 3 ─┘
       ↓
    Chunk A

Paragraph 4 ─┐
Paragraph 5 ─┘  Different topic
       ↓
    Chunk B
```

### Conceptual implementation

```python
sentences = split_into_sentences(document)
embeddings = embed(sentences)
similarities = compare_adjacent_embeddings(embeddings)
boundaries = detect_semantic_breaks(similarities)
chunks = create_chunks(sentences, boundaries)
```

### Advantages

- Better semantic coherence
- Can preserve topic boundaries
- Less dependent on arbitrary token boundaries
- Useful when topic transitions are important

### Limitations

- More computationally expensive
- Requires an embedding model during chunking
- More parameters and thresholds to tune
- Chunk sizes can vary substantially
- More difficult to reproduce consistently

### Best suited for

- Research papers
- Long-form documents
- Knowledge bases
- Documents with meaningful topic transitions

---

## 4. Contextual Chunking

### Definition

Contextual chunking first splits the document with any base strategy (recursive,
fixed-size or semantic). Then an LLM reads each chunk together with the document
(or page) it came from and writes a short note that says where the chunk fits.
The note is **prepended to the chunk** before it is embedded and indexed.

```text
Document / page
    ↓
Base chunker (recursive, fixed-size or semantic)
    ↓
Chunk
    ↓
LLM: "Situate this chunk within its document"
    ↓
Context note + Chunk   →  embed + index
```

### Example

Original chunk:

```text
The rank r was set to 8 for all experiments.
```

Contextualized chunk:

```text
This chunk is from the LoRA paper, section 7 (Understanding the Low-Rank Updates),
and describes the hyperparameters of the GPT-3 experiments.

The rank r was set to 8 for all experiments.
```

A query such as "What rank does LoRA use for GPT-3?" now matches the chunk even
though the chunk itself never names the paper or the model.

### Advantages

- Fixes chunks that lose meaning once cut from their surroundings
- Helps both vector search and BM25, because the note adds key terms
- Works with any base chunker
- Published results show one of the largest retrieval gains of these techniques

### Limitations

- One LLM call per chunk, so building the index is slow and costs compute
- Needs the whole document (or page) in the LLM prompt; long documents may not fit
- The note can be wrong or add noise (an LLM can hallucinate)
- Notes should be cached, so re-indexing does not repeat the LLM calls

### Best suited for

- Documents where chunks rely on surrounding context (tables, pronouns, "the model")
- Collections with many similar documents
- After a base strategy is already evaluated, as an optimization

---

# Comparison

| Strategy | Simplicity | Semantic Awareness | Speed | Chunk Size | Good Baseline |
|---|---|---|---|---|---|
| Recursive / Hierarchical | High | Medium | High | Controlled | Yes |
| Fixed-size + Overlap | Very High | Low | Very High | Very predictable | Excellent |
| Semantic | Medium/Low | High | Lower | Variable | Good advanced option |
| Contextual (on top of any above) | Low | High | Lowest (LLM call per chunk) | Base size + note | No, an add-on |

---

# Choosing a Strategy

A practical progression for a RAG project is:

```text
Start
  ↓
Fixed-size / Recursive chunking
  ↓
Evaluate Retriever
  ↓
Recall@K / Precision@K / MRR / HitRate@K
  ↓
Try Semantic Chunking
  ↓
Try Contextual Chunking on the best base strategy
  ↓
Compare results
```

For a general-purpose PDF RAG system, recursive or fixed-size chunking is a sensible baseline. Semantic chunking can then be evaluated as an optimization rather than assumed to be better.

---