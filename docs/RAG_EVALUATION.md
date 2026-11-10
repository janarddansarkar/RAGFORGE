# RAGForge: Evaluating a RAG System, Component by Component

Nov 10, 2026 · Janarddan Sarkar

## Summary

I built a local RAG app (RAGForge) and evaluated it one component at a time on 45 golden questions: retrieval is strong, answers are mostly right, and the weakest part of the whole system turned out to be the LLM judge I used to score it.

- **Retrieval:** hybrid search plus a cross-encoder reranker puts the right page in the top 5 for 96% of questions (Recall@5 0.91, MRR 0.92). Dense search alone reaches 0.71 and 0.65.
- **Generation:** answers are faithful to their sources (0.99) and about 80% correct under a strict judge.
- **The judge:** the same 45 answers scored 0.80 correctness under one judge and 0.97 under another. Neither number is the truth, so I checked the disputed answers by hand against the PDFs.
- **A real failure found by evaluation:** the app answered "9 years" to a resume question whose correct answer is about 6 years. The cause was a missing date in the prompt, not retrieval. Fixing it was a one-line change, and the new answer is correct.

The point of this project is the evaluation method: separate metrics for retrieval, reranking, generation and the whole pipeline, saved runs that can be compared, and a habit of reading the failures instead of trusting the scores.

## What I built

RAGForge answers questions about uploaded PDFs with page citations, runs entirely on a laptop through Ollama, and was built as a vehicle for measuring each stage of a RAG pipeline separately.

| Stage | What it does | Technique and model (local version) |
| --- | --- | --- |
| Indexing | Reads pages (OCR for scans), splits them, adds an optional LLM context note per chunk, embeds into Chroma | Recursive chunks of 600 characters, overlap 90, contextual notes; embeddings `all-minilm` |
| Input guard | Blocks prompt injection and malicious input before anything is searched | `mistral:7b` plus pattern checks |
| Retrieval | Finds candidate chunks | Dense vectors plus BM25 keywords, fused with reciprocal rank fusion (RRF) |
| Reranking | Re-scores the candidates against the question | Cross-encoder `mxbai-rerank-base-v1`, top 5 kept |
| Generation | Writes a cited answer from the passages only, streamed | `qwen3:14b`, prompt told today's date |
| Scoring | Scores each answer live (RAG triad) | DeepEval with a judge model, `mistral:7b` |

Three branches serve three purposes. `dev` is the free local version, `main` is a hosted version where each user brings their own API key, and `offline-eval` holds the evaluation lab that searches for the best settings. The code uses plain classes throughout, including the evaluation code, so each metric and experiment can be read and rerun on its own.

## How I evaluate it

I measure RAG at four levels with different metrics, so a bad answer can be traced to the stage that caused it instead of just being observed.

| Level | Question it answers | Metrics | Needs golden data |
| --- | --- | --- | --- |
| Retriever | Are the right pages found, and how high? | Recall@K, Precision@K, HitRate@K, MRR (K = 1, 3, 5, 10) | Golden pages |
| Reranker | Does re-ordering move the right page up? | The same metrics, with and without the reranker | Golden pages |
| Generator | Is the answer faithful, relevant and correct? | Faithfulness, answer relevancy, answer correctness, semantic similarity, context recall, seconds per answer | Golden answers |
| Whole pipeline | Do context, answer and question fit together? | RAG triad via DeepEval: context relevance, groundedness, answer relevance; latency per stage | None |

*(In the Claude Doc version there is an evaluation-map diagram here: which metric measures which stage. The dashed box is the gap I have not closed.)*

Golden-data metrics score retrieval, reranking and generation one at a time, while the triad scores the whole pipeline without any golden data.

**Golden set.** 45 questions over three papers (LoRA, QLoRA, Attention Is All You Need), 15 each. Every question has a reference answer, the section it comes from, and the pages that contain it.

**Page-based relevance.** Chunk IDs change whenever chunking changes, so a retrieved chunk counts as relevant if it comes from the right document and one of the golden pages. This keeps scores comparable across chunking strategies.

**Oracle mode.** The generator can be given the golden pages instead of retrieved chunks. Comparing a normal run with an oracle run separates what retrieval costs from what the generator can do on its own.

**Saved, comparable runs.** Every run writes a JSON file with the settings that produced it, the average scores, and every per-question row with the answer. I can re-read any past run and diff it against a new one.

**Staged search with a held-out check.** `find_best.py` tunes chunking, embedder, retrieval method, reranker and generator one stage at a time on two of every three questions, then confirms the winner on the rest. Near-ties go to the faster option.

**Gaps I know about.** The input guard and the judge have no accuracy metric of their own. The next sections show why the judge matters most.

## The metrics, stage by stage

Each stage gets metrics that can fail independently of the others: the retriever and reranker are scored against golden pages, the generator against golden answers, and the whole pipeline needs neither.

### Retriever and reranker

A golden question names the pages that contain its answer. Retrieved chunks are ranked, and a chunk counts as relevant if it is on one of those pages.

| Metric | How it is calculated (per question, then averaged) | What it tells us |
| --- | --- | --- |
| Recall@K | Golden pages found in the top K divided by all golden pages | Whether retrieval finds everything the answer needs, which matters when two pages are required |
| Precision@K | Relevant chunks in the top K divided by K | How much irrelevant text reaches the generator |
| HitRate@K | 1 if any relevant chunk is in the top K, otherwise 0 | Whether the generator sees at least one right page |
| MRR | 1 divided by the rank of the first relevant chunk, 0 if none | How close to the top the first right page sits |

Worked example: a question whose golden pages are 4 and 5, with results ranked page 9, 4, 12, 5, 2. Recall@3 is 1 of 2 pages, 0.5. Recall@5 is 2 of 2, 1.0. Precision@5 is 2 of 5, 0.4. HitRate@5 is 1. The first relevant result is at rank 2, so the reciprocal rank is 0.5.

The reranker is scored with the same four metrics, run with and without it. It receives 20 candidates and keeps 5, so it can change which chunks make the top 5 as well as their order. The numbers to watch are MRR and Recall@1, which show whether the right page moved up.

### Generator

A judge model and an embedding model compare the answer to the retrieved passages and to the golden answer.

| Metric | How it is calculated | What it tells us |
| --- | --- | --- |
| Faithfulness | The answer is split into sentences. The judge says YES or NO to "is this sentence supported by the passages?". Score is the share of YES. A refusal scores 1.0. | Whether the answer stays within its sources, meaning no hallucination |
| Answer relevancy | The judge writes the question the answer seems to respond to. Score is the embedding cosine between that question and the real one. | Whether the answer addresses what was asked |
| Answer correctness | The judge compares the answer to the golden answer: CORRECT is 1, PARTLY is 0.5, WRONG is 0 | Whether the answer is right |
| Semantic similarity | Embedding cosine between the answer and the golden answer, with no judge | How close the meaning is, and a judge-free anchor |
| Context recall | The golden answer is split into sentences and the judge checks each against the retrieved passages. Score is the share supported. | Whether retrieval gave the generator what it needed |
| Seconds per answer | Wall-clock time to answer | The cost of a better model |

Faithfulness and context recall use the same support check aimed at different texts: the answer in one case, the golden answer in the other. Context recall is the bridge between the two halves of the evaluation, since low context recall points at retrieval and not at the generator.

### Whole pipeline

The pipeline is scored end to end by the RAG triad (next section) and by latency per stage: guard, retrieval, reranking, the answer and its time to first text, and the triad scoring itself. Oracle mode (golden pages given to the generator) isolates the generator from retrieval.

## The RAG triad

The RAG triad is three checks, one on each link between the question, the retrieved context and the answer, and none of them needs a golden answer, so it can score any live answer.

*(In the Claude Doc version there is a triangle diagram here: the RAG triad, three checks, one per link.)*

Each side of the triangle is one score, and a RAG system is only as good as its weakest side.

### How each score is calculated

I use DeepEval's three metrics (version 4.2.7) with a judge model and a pass threshold of 0.5. Each one makes several judge calls and then divides. I give it the document's own text, not the context note added at indexing, so groundedness is tested against the real source.

| Score (DeepEval name) | Link | Steps | Score is |
| --- | --- | --- | --- |
| Context relevance (ContextualRelevancy) | Question to context | The retrieved chunks are split into statements. The judge marks each statement relevant or not to the question. | Relevant statements divided by all statements |
| Groundedness (Faithfulness) | Context to answer | Claims are extracted from the answer. The judge asks whether each claim contradicts the retrieved context and answers yes, no or borderline. | Claims judged yes or borderline divided by all claims |
| Answer relevance (AnswerRelevancy) | Question to answer | The answer is split into statements. The judge marks each relevant, irrelevant or borderline to the question. | Relevant plus borderline statements divided by all statements |

Every metric also writes a short reason in words. Those reasons are how I found the problems described in the failure analysis.

### What the scores tell us

I read the three together, because the pattern points at the stage to fix.

| Pattern | Likely cause | Where to look |
| --- | --- | --- |
| Low context relevance, high groundedness | Retrieval brought noisy or wrong chunks, and the answer is faithful to them | Retriever, chunk size, reranker |
| High context relevance, low groundedness | The generator added claims its sources do not support | Answer prompt, answer model |
| High on both, low answer relevance | The answer is true to the text but responds to something else, such as a copied nearby sentence | Answer prompt, query rewriting |
| All three high | Looks healthy, but is not proof of correctness | Spot-check against the source |

### What they cannot tell us

- **Correctness.** Groundedness only lowers the score for claims that contradict the context, and borderline claims pass. An invented figure that nothing contradicts scores 1.00, which is what happened with the wrong "9 years" answer.
- **Evidence spread across chunks.** Context relevance counts every sentence in every chunk as relevant or not. Dates that must be added up, or an answer surrounded by other text, lower the score even when retrieval worked. The same resume example scored 0.00.
- **Judge quality.** All three depend on one judge model, so the judge problems described below apply. They are also slow: about two minutes per question with a 14B judge, so I use them for live scoring and spot checks, not on every run.

## Retrieval experiments

Adding a cross-encoder reranker to hybrid search was the biggest retrieval gain: MRR rose from 0.74 to 0.92, and the right page is in the top 5 for 96% of questions.

*(In the Claude Doc version there is an MRR chart here for the five retrieval methods: RAGForge retriever eval, 45 golden questions, dev branch, 10 Nov 2026.)*

| Method | Recall@1 | Recall@5 | HitRate@5 |
| --- | --- | --- | --- |
| dense | 0.47 | 0.71 | 0.78 |
| bm25 | 0.56 | 0.78 | 0.82 |
| hybrid | 0.53 | 0.88 | 0.91 |
| dense + rerank | 0.73 | 0.86 | 0.89 |
| hybrid + rerank | 0.76 | 0.91 | 0.96 |

Hybrid search helps recall (0.71 to 0.88 at K = 5) but not ranking. The reranker fixes ranking: Recall@1 jumps from 0.53 to 0.76 because the right page is usually already retrieved, just not first.

### Which reranker

I compared six cross-encoders under hybrid retrieval. Bigger was not better: `bge-reranker-large` scored below the smaller `bge-reranker-v2-m3` and `bge-reranker-base`. I use `mxbai-rerank-base-v1`, within 0.01 MRR of the best and with the highest HitRate@5.

| Reranker | MRR | Recall@5 | HitRate@5 |
| --- | --- | --- | --- |
| bge-reranker-v2-m3 | 0.85 | 0.93 | 0.96 |
| mxbai-rerank-base-v1 | 0.84 | 0.93 | 0.98 |
| bge-reranker-base | 0.81 | 0.89 | 0.93 |
| ms-marco-MiniLM-L-12-v2 | 0.78 | 0.90 | 0.93 |
| bge-reranker-large | 0.78 | 0.91 | 0.93 |
| ms-marco-MiniLM-L-6-v2 | 0.74 | 0.89 | 0.93 |

### Which chunking

Contextual chunks (an LLM-written note per chunk) help weak retrievers: BM25 MRR rises from 0.67 to 0.76. Once a reranker is added, the strategies land within 0.07 MRR of each other, so the reranker hides most chunking differences.

| Chunking | Dense MRR | BM25 MRR | Hybrid + rerank MRR |
| --- | --- | --- | --- |
| recursive, 1000 / 150 | 0.78 | 0.67 | 0.85 |
| tokens, 500 / 100 | 0.73 | 0.71 | 0.87 |
| semantic, percentile 90 | 0.71 | 0.73 | 0.80 |
| contextual recursive, 1000 / 150 | 0.81 | 0.76 | 0.85 |

### Which chunk size

Smaller chunks improve ranking (MRR 0.51 at 1000 characters, 0.66 at 150) but create many more chunks to store and judge. Context relevance was scored by an LLM judge on only 8 sampled questions, and it does not move steadily with size (0.37, 0.52, 0.37, 0.45, 0.52), so I treat it as noise and not as a ranking. The sweep selected 600 / 90.

| Chunk size / overlap | Chunks stored | Recall@5 | MRR | Context relevance (top 5) |
| --- | --- | --- | --- | --- |
| 1000 / 150 | 264 | 0.81 | 0.51 | 0.37 |
| 600 / 90 | 422 | 0.81 | 0.55 | 0.52 |
| 400 / 60 | 629 | 0.87 | 0.58 | 0.37 |
| 250 / 38 | 1077 | 0.93 | 0.65 | 0.45 |
| 150 / 22 | 2030 | 0.83 | 0.66 | 0.52 |

## Generator and judge

The judge model changed my results more than the generator model did, so I treated the judge as something to evaluate, not a fixed ruler.

*(In the Claude Doc version there is a judge-comparison chart here: RAGForge generator eval, 45 golden questions, retrieved context, dev branch, 10 Nov 2026.)*

Under the strict judge, correctness runs from 0.61 to 0.77 and the models are clearly ordered. Under the lenient judge every model lands between 0.92 and 0.99 and the order barely means anything. The spread between judges (about 0.2) is larger than the spread between models.

### Three judges on the same generator

I scored the same generator (`qwen3:14b`) with three judges. Three things stand out:

| Judge | Faithfulness | Correctness | Context recall |
| --- | --- | --- | --- |
| `qwen2.5:3b` | 0.69 | 0.76 | 0.56 |
| `qwen3:14b` | 1.00 | 0.77 | 0.96 |
| `mistral:7b` | 1.00 | 0.92 | 0.96 |

- **The 3B judge was too weak to use.** It tied a 3B and a 14B generator at 0.76, and it reported context recall of 0.56, meaning retrieval missed the answer for 44% of questions. My retriever eval said the right page was in the top 5 for 96%. Two independent measurements disagreed, which was the clue that the judge was wrong, not the retriever. A stronger judge moved context recall to 0.96.
- **`mistral:7b` was too lenient.** In the final run it marked none of the 45 answers wrong (0.97), and in an earlier run it scored `gemma3:4b` 0.99, yet reading that model's answers I found at least five that were wrong.
- **`qwen3:14b` was the most discriminating** but it was grading its own answers when it also generated them, and it scored two correct answers 0.0 (next section).

One metric did not move with the judge: semantic similarity, an embedding comparison with no LLM involved, was 0.69 under both. I keep at least one judge-free metric as an anchor.

### Prompt and model changes

From the first review I rewrote the answer prompt (answer directly in your own words, give the exact facts, do not copy passages) and moved the answerer from `gemma3:4b` to `qwen3:14b`. Under the same judge, answer relevancy rose from 0.62 to 0.84 and correctness from 0.73 to 0.80. Those two changes happened together, so the run alone cannot say which helped. A spot check of the new prompt on `gemma3:4b` for the nine wrongly scored questions fixed three of them, which is evidence the prompt contributed. The model upgrade cost time: 4.3 seconds per answer against 2.0.

## Reading the failures

Scores told me where to look; reading the wrong answers against the source PDFs told me what was actually wrong, and in several cases it was the score, not the answer.

### Nine answers the strict judge marked wrong

I checked each against the document. Five were real failures, two were judge errors, and two were borderline.

| Question | Verdict | What I found |
| --- | --- | --- |
| L01 what problem LoRA solves | Real failure | Quoted an unrelated sentence about full-rank updates and never stated the problem |
| L10 checkpoint-size reduction | Real failure | Gave the VRAM figure instead of the checkpoint size (about 10,000x, 350 GB to 35 MB) |
| L14 LoRA vs full fine-tuning | Real failure | A garbled quote and no statement that higher rank recovers full fine-tuning |
| A07 three uses of multi-head attention | Real failure | Named one of the three and filled the rest with general properties |
| Q15 QLoRA loss and training | Real failure, retrieval | `context_recall` 0.00: the chunk naming cross-entropy and supervised learning was never retrieved |
| L05 how LoRA matrices are initialized | Judge error | The answer matched the reference; it only omitted that the initial update is zero |
| A04 attention formula | Judge error | Correct formula, scored 0 because PDF line breaks made it look different |
| L12 trainable parameters | Borderline | Contained the right fact (10,000x fewer) buried between unrelated numbers |
| A11 complexity of self-attention | Borderline | First line exactly right, followed by contradictory extra lines |

The real failures share one pattern: the answer model copies a nearby sentence or lists loosely related citations instead of answering the question. That became the target for the prompt rewrite described above.

Two more from the final run. For Q13 (data quality versus size) the answer matched the paper's appendix and only missed the headline example, so the judge's 0.0 was too harsh. For A14 (BLEU scores) the answer said 41.0 for English-French while the reference said 41.8. The paper contradicts itself: the abstract and Table 2 say 41.8, but a sentence in Section 6.1 says 41.0. The answer quoted the sentence faithfully. I updated the golden answer to accept both figures.

### A wrong answer that every score missed

On a resume outside the golden set I asked for total experience. The app answered "9 years" with a per-company breakdown. The dates in the document give about 6 years 1 month. The RAG triad scored it: context relevance 0.00, groundedness 1.00, answer relevance 1.00. Two of the three scores were wrong. Groundedness passed an invented figure because, as I read the judge's own reason ("no contradictions"), it checks for contradiction with the context, not for support. Context relevance failed good chunks because no sentence states the total, even though the dates needed to compute it were all there.

To find the cause I held the five retrieved chunks fixed and varied one thing at a time:

| Answer model | Thinking | Date told to the model | Answer |
| --- | --- | --- | --- |
| `gemma3:4b` | off | no | 9 years (invented) |
| `qwen3:14b` | off | no | Refuses |
| `qwen3:14b` | on | no | Refuses; its reasoning says the current date is not in the passages |
| `qwen3:14b` | on | yes | 6 years 1 month (correct) |
| `qwen3:14b` | off | yes | 6 years 9 months (wrong) |
| `gemma3:4b` | off | yes | 9 years 8 months (wrong) |

Retrieval was fine: all five chunks came from the resume and held the dates. Two separate causes remained. The resume says "Feb 2025 to Present" and the model had no way to know today's date. And a 4B model cannot reliably add durations across chunks. I added a small `Clock` class that puts today's date in the answer prompt, with an instruction to work out durations from the dates. Through the real pipeline, `qwen3:14b` without thinking then answered 6 years 1 month for total experience and 1 year 9 months for time at the current employer, both correct. `gemma3:4b` still got the arithmetic wrong (8 years 3 months).

## What I learned about evaluating RAG

The most useful lesson was to evaluate the evaluator: an LLM judge is a model with its own failure modes, and most of my wrong conclusions came from trusting one.

1. **Test the judge before trusting its scores.** The same 45 answers scored 0.80 and 0.97 under two judges, and a 3B judge tied a 3B and a 14B model. A judge can be too weak (cannot separate models), too lenient (marks everything right), or biased (grading its own output).
2. **Triangulate with independent measures.** A judge-based context recall of 0.56 disagreed with a retriever HitRate@5 of 0.96. Disagreement between two methods that should agree is the fastest way to find a broken metric. I also keep one metric with no judge in it (embedding similarity) as an anchor.
3. **Read the failures.** Of nine answers marked wrong, five were real, two were judge errors, and two were borderline. The score alone would have misreported the system in both directions.
4. **Test outside the golden set.** 45 questions over academic papers never asked about "present" or "how long". A resume question exposed an invented answer and a missing-date bug that no golden question could.
5. **Do not treat the RAG triad as a correctness check.** Groundedness passed an invented figure, and context relevance failed correct chunks because the answer needed calculation across them. It is a signal to investigate, not a verdict.
6. **Evaluate stages separately.** Splitting retrieval, reranking and generation showed the reranker was the largest retrieval gain, that chunking differences mostly disappear behind it, and that a larger reranker was not a better one.
7. **Change one thing at a time.** I changed the prompt and the model together, so the run cannot say which improved relevancy. Save settings with every run, and isolate changes when the question is attribution.
8. **Golden data can be wrong, and small samples are noisy.** One reference contradicted its own paper. Context relevance on 8 questions did not move steadily with chunk size. A few hundredths on 45 questions is noise.

## Limits and next steps

The retrieval results are solid for the three papers tested, but the generation scores depend on judges I could not fully validate, so they are comparisons between setups and not absolute accuracy.

**Limits**

- **Small, narrow test set.** 45 questions over three academic papers, plus two informal questions on a resume. Scanned PDFs, long documents and tables are untested.
- **No trusted judge.** The local judges were too weak, too lenient or self-grading. I have no human-labelled answers to measure any judge's agreement with a person.
- **Guard unmeasured.** I have not measured how often the input guard blocks a harmless question or misses an injection.
- **Triad run incomplete.** The DeepEval triad over all 45 questions did not finish (about two minutes per question), so my triad conclusions rest on live and spot-checked examples.
- **Results from different setups.** The chunking and reranker comparisons ran in October with other embedding models; the retrieval-method numbers are from 10 November. Compare rows within a table, not across tables.
- **Not tested under load.** Latency was measured for one user at a time.

**Next steps**

1. Hand-label about 100 answers and report each judge's agreement with me, so judge quality becomes a number.
2. Add a stronger independent judge (a cloud model) and keep the local one as a cheap check.
3. Show the support-based faithfulness check from the generator eval in the live app as well, since it flags invented claims that the triad lets through, and finish the triad run.
4. Grow the golden set with other document types and with time-relative and aggregation questions.
5. Build a small adversarial set to measure the guard.
6. Rerun the prompt change and the model change as separate experiments.

## Reproduce it

The scores above come from these commands on the `dev` branch (the chunk-size sweep lives on `offline-eval`). Each run writes a JSON file with its settings and per-question rows. The resume experiment used a short one-off script that is not in the repository.

```bash
git clone -b dev https://github.com/janarddansarkar/RAGFORGE.git
cd RAGFORGE
./setup.sh

uv run python src/vectorstore.py --rebuild

uv run python eval/retriever_eval.py
uv run python eval/chunking_compare.py
uv run python eval/reranker_compare.py

uv run python eval/generator_eval.py
uv run python eval/generator_eval.py --oracle
uv run python eval/generator_eval.py --generator gemma3:4b --judge mistral:7b

uv run python eval/triad_eval.py --limit 5
uv run python eval/time_pipeline.py
uv run pytest
```

On `offline-eval`: `uv run python eval/chunk_size_sweep.py` and `uv run python eval/find_best.py` run the staged search.
