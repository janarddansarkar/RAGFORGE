# Evaluation: how quality was measured

A RAG system can fail in two places: it can **retrieve the wrong passages**, or it can **write a poor answer from good
passages**. Evaluation exists to tell those apart. This document explains the approach and what it found. For the full tables see
[EVALUATION_RESULTS.md](EVALUATION_RESULTS.md); for the detailed method see [RAG_EVALUATION.md](RAG_EVALUATION.md).

## The approach in one page

Two kinds of measurement, because they answer different questions:

| Kind | Needs reference answers? | Answers the question | When it runs |
|---|---|---|---|
| **Component evaluation** against a golden set | Yes | Which version of each part is best? | Offline, on demand |
| **RAG triad** | No | Is *this* answer on topic, grounded and relevant? | Live, on every answer |

### The golden set

45 questions over three papers (LoRA, QLoRA, "Attention Is All You Need"). Each has a reference answer and the pages that
contain it. It is small, so differences of a few hundredths between two setups are noise (one question moves a score by about
0.02). Treat the numbers as a way to compare setups, not as absolute truth.

### What is measured, and where it points

| Stage | Question | Measures |
|---|---|---|
| Retriever | Does search find the right pages, and near the top? | Recall, Precision, Hit Rate, MRR |
| Reranker | Does re-sorting move the right page up? | The same four, with and without |
| Chunking | Does adding context notes to chunks help? | The same four, per method |
| Generator | Is the answer faithful, relevant and correct? | Faithfulness, relevancy, correctness, similarity, context recall |
| Generator, "oracle" | Same, but handed the right pages | Isolates the answer model from search |
| RAG triad | Does the whole pipeline hang together? | Context relevance, groundedness, answer relevance |
| Latency | Where does time go? | Seconds per stage |

**Reading the generator against the oracle.** If answers are much better when handed the right pages, the fix is in
retrieval; if not, the fix is in the answer model or its prompt. This is the most useful diagnostic in the project.

**Reading the triad.** Low context relevance with high groundedness: search brought noisy passages and the answer stayed
faithful to them. High context relevance with low groundedness: the model added claims its sources do not support.

### Safeguards in the method
- A **different model judges** than answers, so a model never grades itself. This matters: the same answers scored 0.97 for
  correctness under a lenient judge and 0.80 under a stricter one. Only compare runs that used the same judge.
- Relevance is judged **by page**, so different chunking strategies can be compared on equal terms.
- Settings were **tuned on two thirds of the questions and checked on the remaining third**; near-ties go to the faster option.

## What it found

### Local models (`offline-eval` and `dev`)
- **Hybrid search beats either search alone**, and **adding a reranker was the single biggest gain**: the right page was usually
  already retrieved, just not first (MRR 0.74 to 0.92; right page in the top 5 for 96% of questions).
- **A purpose-built cross-encoder reranker beat asking a small chat model to rank**; the chat-model rerankers actually made
  results worse than no reranker.
- **Bigger rerankers were not better.** The chosen one was within 0.01 of the best and highest on Hit Rate.
- **Answers were faithful** (about 0.99) and the right information was retrieved. The correctness number depended heavily on the
  judge, which is why runs are only compared under one judge.
- Final local configuration: contextual chunks of 600 characters with 90 overlap, `all-minilm` embeddings, hybrid search with
  20 candidates, the `mxbai-rerank-base-v1` cross-encoder keeping 5, and `qwen3:14b` writing answers.

### Hosted models (`ragforge_api`, OpenAI)
Full run on all 45 questions, `gpt-4o` answering and `gpt-4o-mini` as guard, judge and reranker:

| Result | Finding |
|---|---|
| Best retriever | Hybrid plus reranker: MRR 0.88, Recall@5 0.92, Hit Rate@5 0.96 (hybrid alone 0.79, dense alone 0.71) |
| Reranker size | `gpt-4o` 0.89 against `gpt-4o-mini` 0.86: not worth the cost |
| Context notes | Helped single searches but not the reranked pipeline (plain 0.91 against 0.87 with notes). They cost a one-time indexing charge |
| Answers | Correctness 0.98, context recall 0.92, faithfulness 0.71 with retrieved passages and 0.94 with the right pages, so most of the gap is retrieval |
| Triad | Context relevance 0.41, groundedness 0.72, answer relevance 0.94 (the top 5 passages usually include some unrelated ones) |
| Latency per question | guard 0.8 s, search 0.8 s, rerank 3.8 s, answer 1.3 s, **triad scoring 39 s** |

The triad is by far the slowest part of a turn and a large share of its cost, because each of its three metrics makes several
judge calls. It runs in the background so the answer appears first.

## Checks that led to changes after the full run

The full run is expensive, so later fixes were checked smaller and separately:

1. **A 15-question test through the real app.** 9 of 11 answerable questions were answered correctly, both out-of-scope
   questions were refused, a prompt injection was blocked and a repeated question came from the cache. Two answerable
   questions were wrongly refused (a two-part question and a follow-up). The answer prompt was changed to answer the parts
   the documents support and say which part they don't cover.
2. **A wrong answer ("what rank did the GPT-3 experiments use?").** The right passage never reached the reranker: it was not
   in the top 20 candidates. Widening to 30 fixed it and, on the whole set, improved Recall@3 from 0.878 to 0.922 and MRR from
   0.866 to 0.888 (about 2 seconds slower per query).
3. **A stricter answer prompt** (answer supported parts, never use outside knowledge, prefer the values the authors actually
   used) raised faithfulness from 0.71 to 0.93 with correctness unchanged, and out-of-scope questions were still refused.
4. **A multi-query retry search** gained within noise (MRR 0.871 against 0.866) for about 50% more time, so the default did not
   change.

One question stays unreliable: a vague phrasing like "what rank r does LoRA use?", because the paper uses different ranks in
different places. This is a known limitation of the approach, not a bug that was fixed.

## What has not been measured
- Quality for Gemini, Cohere and Anthropic on the full golden set. They were tested live end to end (Anthropic: 7 questions with
  no errors, about $0.011 per question against about $0.007 with OpenAI) but not scored.
- Latency of the local version (no timing run was saved).
- Anything beyond three research papers; a different kind of document may behave differently.
