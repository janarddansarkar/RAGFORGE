# RAGForge: RAG Evaluation

How RAGForge's pipeline is evaluated component by component and end to end, what the scores say, and how the winning configuration moves from the `offline-eval` branch into the `dev` pipeline.

**Summary.** On 45 golden questions, hybrid search plus a cross-encoder reranker puts the right page in the top 5 for 96% of questions (Recall@5 0.91, MRR 0.92). Answers are faithful to their sources (0.99) and about 80% correct under a strict judge. The least reliable part of the evaluation is the LLM judge: the same answers scored 0.80 or 0.97 correctness depending on the judge.

## 1. Evaluation workflow

Three branches: `offline-eval` is where experiments run, `dev` is the working Ollama pipeline, and `main` is identical to `dev` for now.

```
offline-eval                                          dev
golden set (45 Q) -> eval each component -> staged search (find_best.py)
                                                |
                                      best_config.json  ------>  load_config() lays it over config.toml
                                                                  -> indexing, retrieval, reranking, generation
```

- **Golden set:** 45 questions over three papers (LoRA, QLoRA, Attention Is All You Need), 15 each. Each has a reference answer, its source section and the pages that contain it.

How each eval uses it:

| Eval | Uses from each golden question | How |
| --- | --- | --- |
| Retriever and reranker | The question and the golden pages | The question is run through the retriever; Recall, Precision, HitRate and MRR compare the ranked chunks' pages with the golden pages |
| Generator | The question and the golden answer | The pipeline answers the question; the judge compares the answer with the golden answer (correctness, semantic similarity) and checks the golden answer against the retrieved passages (context recall) |
| Oracle mode | The golden pages | The first 5 chunks on those pages, in document order, are given to the generator instead of retrieved chunks |
| `find_best.py` | All questions, split 2 : 1 | Two of every three tune each stage; the remaining third checks the retrieval pick |
| RAG triad | The question only | No golden answer is needed |

Further points on the evaluation:

- **Golden updates:** when a check against the PDFs shows a reference answer is wrong or incomplete, the golden file is updated and later runs use the new reference.
- **Page-based relevance:** chunk IDs change whenever chunking changes, so a retrieved chunk counts as relevant if it comes from the right document and one of the golden pages. This keeps scores comparable across chunking strategies.
- **Per-component evals:** the retriever, reranker and generator are scored separately (section 2), so a bad answer can be traced to the stage that caused it.
- **Oracle mode:** comparing a normal run with an oracle run separates what retrieval costs from what the generator can do on its own.
- **Whole pipeline:** scored end to end with the RAG triad, which needs no golden answers.
- **Saved runs:** every generator run writes a timestamped JSON file to `eval/generator_runs/` with its settings, average scores and per-question rows. The retriever, reranker and chunking evals each keep one results file that the next run overwrites.

## 2. Metrics

### Retriever and reranker (against golden pages)

| Metric | Calculation (per question, then averaged) | What it tells us |
| --- | --- | --- |
| Recall@K | Golden pages found in the top K ÷ all golden pages | Whether retrieval finds everything the answer needs |
| Precision@K | Relevant chunks in the top K ÷ K | How much irrelevant text reaches the generator |
| HitRate@K | 1 if any relevant chunk is in the top K, else 0 | Whether the generator sees at least one right page |
| MRR | 1 ÷ rank of the first relevant chunk, 0 if none | How close to the top the first right page sits |

K = 1, 3, 5, 10 (Precision at 5). Example: golden pages 4 and 5, results ranked 9, 4, 12, 5, 2. Recall@3 = 0.5, Recall@5 = 1.0, Precision@5 = 0.4, HitRate@5 = 1, MRR = 0.5 (first hit at rank 2).

The reranker is scored with the same metrics, run with and without it. It receives 20 candidates and keeps 5. Watch MRR and Recall@1: they show whether the right page moved up.

### Generator (against golden answers and retrieved passages)

| Metric | Calculation | What it tells us |
| --- | --- | --- |
| Faithfulness | Answer split into sentences; the judge says YES/NO to "supported by the passages?". Score = share of YES. A refusal scores 1.0. | Whether the answer stays within its sources (no hallucination) |
| Answer relevancy | The judge writes the question the answer seems to respond to; score = embedding cosine with the real question | Whether the answer addresses what was asked |
| Answer correctness | The judge compares with the golden answer: CORRECT 1, PARTLY 0.5, WRONG 0 | Whether the answer is right |
| Semantic similarity | Embedding cosine between the answer and the golden answer, no judge | Closeness of meaning; a judge-free anchor |
| Context recall | Golden answer split into sentences, each checked against the retrieved passages; score = share supported | Whether retrieval gave the generator what it needed. Low values point at retrieval, not the generator |
| Seconds per answer | Wall-clock time | Cost of a better model |

### Whole pipeline: the RAG triad (DeepEval 4.2.7, threshold 0.5)

Three checks, one per link between question, context and answer. None needs a golden answer.

| Score | Link | Calculation |
| --- | --- | --- |
| Context relevance | question → context | Retrieved chunks split into statements; judge marks each relevant or not. Relevant ÷ all. |
| Groundedness | context → answer | Claims extracted from the answer; judge says whether each contradicts the context (yes, no, borderline). Yes + borderline ÷ all. |
| Answer relevance | question → answer | Answer split into statements; judge marks each relevant, irrelevant or borderline. Relevant + borderline ÷ all. |

How to read them together:

| Pattern | Likely cause |
| --- | --- |
| Low context relevance, high groundedness | Retrieval brought noisy chunks; the answer is faithful to them |
| High context relevance, low groundedness | The generator added unsupported claims |
| High on both, low answer relevance | True to the text but answers something else |
| All three high | Healthy, but not proof of correctness |

Limits: groundedness only penalises claims that contradict the context, so an invented figure nothing contradicts still scores 1.00. Context relevance penalises answers that need calculation across chunks even when retrieval worked. All three depend on one judge and are slow (3 questions took about 3 minutes with the 7B judge, generation included; I did not time a 14B judge), so I use them for spot checks, not every run. The triad has been run only on 3 questions (section 3); the run over all 45 did not finish.

## 3. Results

### Retrieval methods (45 questions)

| Method | Recall@1 | Recall@5 | HitRate@5 |
| --- | --- | --- | --- |
| dense | 0.47 | 0.71 | 0.78 |
| bm25 | 0.56 | 0.78 | 0.82 |
| hybrid | 0.53 | 0.88 | 0.91 |
| dense + rerank | 0.73 | 0.86 | 0.89 |
| hybrid + rerank | 0.76 | 0.91 | 0.96 |

Hybrid search improves recall (0.71 to 0.88 at K = 5) but not ranking. The reranker fixes ranking: Recall@1 rises from 0.53 to 0.76, and MRR from 0.74 (hybrid) to 0.92, because the right page was usually already retrieved, just not first.

### Generator and judge

The judge changed results more than the generator did. Under the strict judge, correctness runs from 0.61 to 0.77 across models and the order is clear. Under the lenient judge every model lands between 0.92 and 0.99 and the order means little.

Same generator (`qwen3:14b`), three judges:

| Judge | Faithfulness | Correctness | Context recall |
| --- | --- | --- | --- |
| `qwen2.5:3b` | 0.69 | 0.76 | 0.56 |
| `qwen3:14b` | 1.00 | 0.77 | 0.96 |
| `mistral:7b` | 1.00 | 0.92 | 0.96 |

- **3B judge: too weak.** Context recall 0.56 would mean retrieval missed the answer for 44% of questions, while the retriever eval said the right page was in the top 5 for 96%. Two independent measurements disagreed, which showed the judge, not the retriever, was wrong.
- **`mistral:7b`: too lenient.** It marked none of the 45 answers wrong in the final run (0.97).
- **`qwen3:14b`: most discriminating**, but it grades its own answers when it is also the generator, so the judge is kept a different model from the generator.
- **Semantic similarity** (no LLM) was 0.69 under both judges, so I keep one judge-free metric as an anchor.

Hand-checking the nine answers the strict judge marked wrong against the PDFs: five were real failures (the answer model copied a nearby sentence or listed loosely related citations instead of answering; one was a retrieval miss), two were judge errors, two were borderline. For one question the golden answer itself contradicted the paper (abstract and Table 2 say 41.8 BLEU, Section 6.1 says 41.0), so it now accepts both.

Prompt rewrite and model change (`gemma3:4b` to `qwen3:14b`), same judge: answer relevancy 0.62 to 0.84, correctness 0.73 to 0.80. Both changed together, so the run cannot say which helped; the cost was 4.3 seconds per answer against 2.0.

### RAG triad spot check (first 3 golden questions)

Saved run `eval/triad_runs/2026-11-10_233525.json`: `qwen3:14b` answering, `mistral:7b` judging, threshold 0.5.

| Question | Context relevance | Groundedness | Answer relevance |
| --- | --- | --- | --- |
| L01 what problem LoRA solves | 0.22 | 0.50 | 1.00 |
| L02 what LoRA does to the pretrained weights | 0.75 | 0.50 | 0.50 |
| L03 how the LoRA update is parameterized | 0.22 | 0.67 | 0.50 |
| Average | 0.40 | 0.56 | 0.67 |

These scores reflect the judge more than the pipeline. L02's answer was correct ("LoRA freezes the pre-trained model weights"), yet groundedness was 0.50: the judge's reason says the output "appears to modify" the weights, which is wrong. Its answer relevance was 0.50 for a direct answer. Context relevance was 0.22 on L01 and L03 although the retrieved pages were the right ones, because every unrelated sentence in the chunks counts against the score. With 3 questions this is an anecdote, but it shows the triad can mislead in both directions: it passed an invented figure in the resume check, and here `mistral:7b` penalised correct answers.

## 4. Best configuration selection

`eval/find_best.py` searches one stage at a time, and each stage's winner carries forward:

| Stage | Candidates | Ranked by |
| --- | --- | --- |
| 1. Chunking | recursive, semantic, contextual recursive | MRR |
| 2. Embedder | `bge-m3`, `nomic-embed-text`, `mxbai-embed-large`, `all-minilm` | MRR |
| 3. Retrieval method | dense, bm25, hybrid, dense/hybrid + rerank, plus rewrite, multi-query and HyDE variants | MRR |
| 4. Reranker | Ollama LLM rerankers (when the method reranks); the cross-encoders were compared separately, below | MRR |
| 5. Generator | `qwen3:14b`, `qwen2.5:3b`, `llama3.2:3b`, `gemma3:4b`, in oracle mode | answer correctness (judge) |

Rules that keep the choice honest:

- Stages 1 to 4 tune on two of every three golden questions; the retrieval pick is then checked on the held-out third. Stage 5 (generator) is scored on all 45 questions, with no held-out check.
- Anything within a tie margin of the best (0.01 on retrieval, 0.03 on generator) counts as a tie and the faster option wins.
- If the pick is worse than the current settings on the held-out questions, the current settings are kept.
- Every stage's numbers are written to `eval/find_best_report.json`.

Separate comparisons I ran on the retriever (hybrid + rerank unless stated):

| Reranker (cross-encoder) | MRR | Recall@5 | HitRate@5 |
| --- | --- | --- | --- |
| bge-reranker-v2-m3 | 0.85 | 0.93 | 0.96 |
| mxbai-rerank-base-v1 | 0.84 | 0.93 | 0.98 |
| bge-reranker-base | 0.81 | 0.89 | 0.93 |
| ms-marco-MiniLM-L-12-v2 | 0.78 | 0.90 | 0.93 |
| bge-reranker-large | 0.78 | 0.91 | 0.93 |
| ms-marco-MiniLM-L-6-v2 | 0.74 | 0.89 | 0.93 |

Bigger was not better. I chose `mxbai-rerank-base-v1`: within 0.01 MRR of the best and the highest HitRate@5.

Saved October run (`eval/chunking_results.json`). The current `config.toml` no longer lists the token and 1000 / 150 variants, so this table cannot be regenerated as shown.

| Chunking | Dense MRR | BM25 MRR | Hybrid + rerank MRR |
| --- | --- | --- | --- |
| recursive, 1000 / 150 | 0.78 | 0.67 | 0.85 |
| tokens, 500 / 100 | 0.73 | 0.71 | 0.87 |
| semantic, percentile 90 | 0.71 | 0.73 | 0.80 |
| contextual recursive, 1000 / 150 | 0.81 | 0.76 | 0.85 |

Contextual chunks help weak retrievers (BM25 MRR 0.67 to 0.76); once a reranker is added the strategies land within 0.07 MRR, so the reranker hides most chunking differences.

| Chunk size / overlap | Chunks stored | Recall@5 | MRR |
| --- | --- | --- | --- |
| 1000 / 150 | 264 | 0.81 | 0.51 |
| 600 / 90 | 422 | 0.81 | 0.55 |
| 400 / 60 | 629 | 0.87 | 0.58 |
| 250 / 38 | 1077 | 0.93 | 0.65 |
| 150 / 22 | 2030 | 0.83 | 0.66 |

Smaller chunks rank better but multiply the chunk count. The sweep ran in October with an earlier reranker setup (hence Recall@1 of only 0.27 to 0.41, below the 0.76 above). By Recall@5 the best size was 250 and by MRR 150, so 600 / 90 is a middle choice that keeps chunk count and indexing time moderate, not the sweep's top scorer.

**Chosen configuration:** recursive chunks 600 / 90 with contextual notes, `all-minilm` embeddings, hybrid search (dense + BM25, RRF) with `mxbai-rerank-base-v1` (20 candidates, top 5), `qwen3:14b` answering, `mistral:7b` judging (kept separate from the answer model).

## 5. From `offline-eval` to `dev`

1. `find_best.py` writes the winning settings to `best_config.json`.
2. That file is copied to `dev`, where `load_config()` lays it over `config.toml`. The effective values are the ones above; `config.toml` holds everything else (folders, Ollama host, retrieval methods, eval lists).
3. On `dev` the settings drive the full pipeline: PDF pages are chunked and embedded into Chroma, a question is checked by the input guard, searched with hybrid retrieval, reranked, answered with page citations, and scored live with the RAG triad.
4. `dev` adds what the lab does not need: today's date in the answer prompt, the input guard, the session cache and the thumbs-down retry. `offline-eval` lacks these, so code and settings are moved between the branches by cherry-pick, not merge.
5. After any change on `dev`, rerun the golden evals and compare against the saved runs.

## 6. Limits

- 45 questions over three papers; scanned PDFs, long documents and tables are untested.
- No trusted judge: the local judges were too weak, too lenient or self-grading, and I have no human labels to measure any judge's agreement with a person.
- The input guard and the judge have no accuracy metric. The triad has been run only on 3 questions (section 3); the run over all 45 did not finish.
- Chunking and reranker comparisons ran in October with other embedding models; retrieval-method numbers are from 10 November. Compare rows within a table, not across tables.
- With 45 questions, gaps of a few hundredths are noise.

**Next steps:** hand-label about 100 answers and report each judge's agreement; add a stronger independent judge; finish the triad run; grow the golden set; build a small adversarial set for the guard; rerun the prompt and model changes as separate experiments.

## Reproduce

```bash
git clone -b dev https://github.com/janarddansarkar/RAGFORGE.git && cd RAGFORGE && ./setup.sh
uv run python src/vectorstore.py --rebuild

uv run python eval/retriever_eval.py
uv run python eval/chunking_compare.py
uv run python eval/reranker_compare.py
uv run python eval/generator_eval.py            # add --oracle, --generator, --judge
uv run python eval/triad_eval.py --limit 5
uv run python eval/find_best.py                 # staged search (on dev and offline-eval)
```

`eval/chunk_size_sweep.py` exists only on `offline-eval`.
