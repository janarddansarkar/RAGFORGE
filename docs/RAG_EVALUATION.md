# RAGForge: RAG Evaluation

RAGForge is evaluated in two ways. The **RAG triad** scores every answer as it is written and needs no reference data.
An **offline component evaluation** uses a golden dataset of 45 questions to measure each part of the pipeline on its
own and the pipeline as a whole. Every model in both is hosted and called through LiteLLM with the user's own key.

## What is evaluated

| # | Evaluation (stage) | Component | Metrics | Needs golden data |
| --- | --- | --- | --- | --- |
| 1 | **Retriever** (`retriever`) | dense, BM25, hybrid, hybrid + rerank | Recall@K, Precision@K, Hit Rate@K, MRR | golden pages |
| 2 | **Reranker** (`reranker`) | each reranker model | the same four metrics, hybrid search + rerank | golden pages |
| 3 | **Chunking** (`chunking`) | plain vs context-note chunks | the same four metrics, per retrieval method | golden pages |
| 4 | **Generator** (`generator`) | the answer model, given retrieved chunks | faithfulness, answer relevancy, answer correctness, semantic similarity, context recall | golden answers |
| 5 | **Generator, oracle** (`oracle`) | the answer model alone, given the golden pages | the same five | golden answers |
| 6 | **RAG triad** (`triad`) | the whole pipeline | context relevance, groundedness, answer relevance (DeepEval) | none |
| 7 | **Latency** (`latency`) | each pipeline stage | seconds per stage | none |

Comparing 4 with 5 shows whether to fix retrieval or the generator: if the oracle scores are much better, retrieval is
losing the answers.

## Pipeline and where each evaluation applies

```
 INDEXING (once per document set)
 PDF -> pages (OCR for scans) -> chunks (600 chars, overlap 90, + LLM context note)
     -> embeddings (hosted) -> Chroma vector store                                  <-- 3. chunking

 ANSWERING (every question)
 question
   -> input guard (small LLM)                       blocks injection / malicious input
   -> follow-up rewrite + session cache check
   -> RETRIEVER: dense + BM25, fused with RRF       (20 candidates)                 <-- 1. retriever
   -> RERANKER: an LLM scores each chunk, or Cohere's cross-encoder (top 5 kept)    <-- 2. reranker
   -> GENERATOR: answer model, cited answer, today's date in the prompt             <-- 4 and 5. generator
   -> answer + page citations
   -> RAG TRIAD scores the answer (judge model)                                     <-- 6. RAG triad

 Whole path, with seconds per stage                                                 <-- 7. latency
```

## The RAG triad

The triad scores the **final output of the whole pipeline** from the question, the retrieved chunks and the answer,
so it needs no reference answer and works on any uploaded document. In the web app every answer is scored in the
background: the three scores appear under the answer with the judge's reasons, and the sidebar shows the latest score
and the session average.

| Triad score | Link it checks | What it measures |
| --- | --- | --- |
| Context Relevance | question to retrieved chunks | How much of the retrieved text is relevant to the question (points at the retriever and reranker) |
| Groundedness | chunks to answer | Whether the answer's claims are supported by the retrieved text (points at the generator) |
| Answer Relevance | question to answer | Whether the answer addresses the question |

Each score is 0 to 1, with the judge's reasons. They are computed with **DeepEval**'s `ContextualRelevancyMetric`,
`FaithfulnessMetric` and `AnswerRelevancyMetric`, using the hosted judge model you pick (`src/triad.py`). Each metric
makes several judge calls per answer, so the triad is a large share of a turn's cost and by far the slowest part
(about 40 seconds per answer in the OpenAI run). From the command line:
`uv run python src/triad.py your.pdf "your question"`.

## The offline evaluation

**Test set.** `golden_data/rag_retriever_golden_dataset_45.json`: 45 questions over three papers (LoRA, QLoRA, Attention
Is All You Need, in `documents/`). Each has a reference answer and the document pages that contain the answer.
Relevance is page-based, so different chunking strategies are compared fairly (see `specs_md/retriever_eval.md`).

**Running it.** `eval/run_provider_eval.py` builds the index with one provider's models, runs the stages above, and
saves everything to `eval/results/<provider>_<date>_<time>.json` after each stage, including the calls, tokens and
estimated cost of each stage.

```bash
uv run python eval/run_provider_eval.py openai                          # all stages
uv run python eval/run_provider_eval.py openai --limit 10               # first 10 questions only
uv run python eval/run_provider_eval.py openai --stages retriever,generator
uv run python eval/run_provider_eval.py openai --resume eval/results/<file>.json   # run only the missing stages
uv run python eval/make_report.py                                        # rebuild docs/EVALUATION_RESULTS.md (free)
```

The key comes from `OPENAI_API_KEY` (or `GEMINI_API_KEY`) in the environment or a `.env` file. The same scripts can also
be run one by one: `eval/retriever_eval.py`, `eval/reranker_compare.py`, `eval/chunking_compare.py`,
`eval/generator_eval.py` (`--oracle`), `eval/triad_eval.py` and `eval/time_pipeline.py`.

**Cost.** The full OpenAI run cost about $1.3 in model calls for the stages, plus about $0.7 once to build the index
(one context note per chunk). These are LiteLLM estimates, not a bill. Because it costs real money, the numbers are
saved: `docs/EVALUATION_RESULTS.md` is built from the saved JSON and costs nothing to rebuild.

## Results

The full tables, the models used and the cost of every stage are in **`docs/EVALUATION_RESULTS.md`** (OpenAI:
`gpt-4o` for answers, `gpt-4o-mini` for guard, judge and reranker, `text-embedding-3-small` for embeddings). The headline
findings of that run on the 45 questions:

- **Hybrid search plus the reranker is the best retriever:** MRR 0.88, Recall@5 0.92, Hit Rate@5 0.96, against MRR 0.79
  for hybrid search alone and 0.71 for dense search alone.
- **A bigger reranker model barely helps:** `gpt-4o` reaches MRR 0.89 against 0.86 for `gpt-4o-mini`, at far higher cost.
- **Context notes help the single searches but not the reranked pipeline:** MRR 0.67 to 0.71 for dense and 0.70 to
  0.74 for BM25, but with the reranker plain chunks score 0.91 against 0.87 with notes. The notes cost a one-time
  indexing charge.
- **Answers are correct but not always faithful:** correctness 0.98 and context recall 0.92. Faithfulness is 0.71 with
  retrieved chunks and 0.94 with the golden pages, so most of the gap comes from retrieval.
- **Triad:** context relevance 0.41, groundedness 0.72, answer relevance 0.94. Context relevance is low because the
  top 5 chunks usually include some unrelated ones.
- **Latency per question:** guard 0.8 s, search 0.8 s, rerank 3.8 s, answer 1.3 s, triad scoring 39 s.

The Cohere cross-encoder reranker was added to the app after this run, so it is **not** part of these results.

## Reading the scores

- A judge model scores the generator and the triad, so those numbers are a comparison between setups, not absolute
  truth. Compare runs that used the same judge, and keep the judge a different model from the answer model.
- With 45 questions, one question is worth about 0.02, so gaps of a few hundredths are noise.
- Relevance is judged by page, so scores are slightly generous (a chunk on the right page counts even if it is not the
  paragraph with the answer).
- The run used OpenAI only. Other providers can be run the same way if their keys allow enough requests per minute.

## Files

| file | role |
| --- | --- |
| `src/triad.py` | the RAG triad (DeepEval) |
| `eval/run_provider_eval.py` | runs every stage with one provider and saves the results |
| `eval/make_report.py` | builds `docs/EVALUATION_RESULTS.md` from the saved JSON |
| `eval/retriever_eval.py`, `eval/reranker_compare.py`, `eval/chunking_compare.py` | retrieval stages |
| `eval/generator_eval.py`, `eval/triad_eval.py`, `eval/time_pipeline.py` | answer, triad and latency stages |
| `eval/golden.py`, `metrics/retriever_metrics.py` | which chunks are relevant, and the metric math |
| `golden_data/` | the 45 questions |
| `eval/results/` | the saved runs |
| `specs_md/retriever_eval.md`, `specs_md/generator_eval.md`, `specs_md/retriever_evaluation_metrics.md` | how each number is calculated |
