# RAGForge

Ask questions about your PDFs and get answers with **citations**, the **chunks they came from**, and **live quality scores**.
RAGForge is a small, readable retrieval-augmented generation (RAG) system built with plain classes, plus an evaluation
framework that finds the best settings for it.

- Upload one or more PDFs (scanned pages are read with OCR), click Ingest, then ask by typing or by voice.
- Hybrid search (vectors + BM25 keywords, fused with RRF), then a **cross-encoder reranker**, then an answer model.
- Every answer shows the pages it cites, the retrieved and reranked chunks with their scores, and the
  **RAG triad** (context relevance, groundedness, answer relevance) scored by a judge model.
- A question like an earlier one in the same conversation is answered from a private session cache.
- An input guard checks every question for prompt injection and other malicious input.
- An evaluation suite (45 golden questions over three papers) measures retrieval, answers and chunking, and writes the best settings to `best_config.json`.

## Branches: what each one is for

| branch | purpose | models run on | who it is for |
|---|---|---|---|
| **`main`** | The **hosted version**. Each user brings their own API key (OpenAI, Gemini or Anthropic) and the key pays for embeddings, answers, the guard and the judge. Shows the **cost** of every turn and of the whole conversation. | the user's cloud provider, plus a local cross-encoder and Whisper | people using a deployed app |
| **`dev`** | The **local, free version** and the day-to-day development branch. Everything runs on your machine with [Ollama](https://ollama.com); no keys, no per-call cost, works offline once models are downloaded. New features land here first and are then ported to `main`. | Ollama, plus a local cross-encoder and Whisper | developers, local use, demos |
| **`offline-eval`** | The **evaluation lab**. Holds the full `eval/find_best.py` search (chunking, embedder, retrieval method, reranker, chunk size and top-k, generator) and the chunk-size sweep. Its output, `best_config.json`, is copied to `dev` and `main`. It has the core pipeline but not the session cache, input guard or timing script. | Ollama (the judge and candidates), plus Hugging Face cross-encoders | tuning settings, comparing models |

The flow between them: run experiments on `offline-eval`, copy the winning settings and code into `dev`, then port to `main`.
`dev` and `offline-eval` split from a common commit and `main` was fast-forwarded from `dev`, so changes between them are cherry-picked by hand rather than merged.

### Running each branch

**`dev` and `offline-eval` (local, Ollama)**

```bash
ollama serve                                   # if Ollama is not already running
ollama pull all-minilm gemma3:4b mistral:7b    # embeddings, answers, guard + judge (see best_config.json / config.toml)
uv sync
uv run streamlit run src/app.py                # the web app
uv run python src/rag.py documents/LoRA.pdf "What rank r does LoRA use?"   # or the command line
```

The first question downloads the reranker (`mixedbread-ai/mxbai-rerank-base-v1`, about 1 GB) from Hugging Face.
Scanned pages need Tesseract (`brew install tesseract`).

**`main` (bring your own key)**

```bash
uv sync
uv run streamlit run src/app.py
```

In the sidebar pick a provider and paste your key. The key stays in that browser session's memory and is never written
to disk. Anthropic has no embeddings API, so choose OpenAI or Gemini for embeddings. The models and their prices per
provider are in `providers.toml`; the sidebar shows an approximate cost per turn and for the conversation.
The command line scripts and the eval on `main` still use Ollama (`chat_provider = "ollama"` in `config.toml`).

## How it works

```
PDF -> pages (OCR for scans) -> chunks (+ optional LLM context note) -> embeddings -> Chroma
question -> input guard -> (rewrite follow-ups) -> cache check
         -> hybrid search (dense + BM25, fused) -> cross-encoder reranker -> top-k chunks
         -> answer model (streamed) -> answer + citations -> RAG-triad scores
```

| file | role |
|---|---|
| `src/chunking.py` | PDFs to chunks (recursive or semantic; optional LLM context notes; OCR for scanned pages) |
| `src/vectorstore.py` | builds and opens the Chroma store; `load_config()` |
| `src/retriever.py` | dense, BM25, hybrid, reranked (LLM or cross-encoder) and LLM-assisted retrieval |
| `src/generator.py` | answers a question from chunks, citing the pages used |
| `src/rag.py` | `DocumentRAG` for PDFs, the citation report and the command line |
| `src/rag_pipeline.py` | the full pipeline over `documents/` with the best settings |
| `src/app.py` | the Streamlit UI |
| `src/models.py` | the model wrappers (`dev`: Ollama; `main`: also OpenAI, Gemini, Anthropic and cost tracking) |
| `src/guardrails.py` | the input guard (rules plus a guard model) |
| `src/cache.py` | the per-conversation session cache |
| `src/memory.py` | short-term memory: the last few turns, follow-ups rewritten into standalone questions |
| `src/voice.py` | speech to text (faster-whisper on CPU), macOS `say` for reading answers aloud |
| `config.toml` | all settings |
| `best_config.json` | settings found best by the evaluation; laid over `config.toml` |
| `providers.toml` (`main`) | models and prices per cloud provider |

## Settings

`config.toml` holds every setting. After `eval/find_best.py` has run, the settings it found best are in
`best_config.json` and **override** the matching `config.toml` values, so the pipeline uses them without code changes.
Delete that file to go back to `config.toml` alone. `reranker_type` chooses between `cross_encoder` (a Hugging Face
model scoring all chunks in one batch, the current best) and `llm` (a chat model scoring each chunk 0-10).

## Evaluation

| script | what it does |
|---|---|
| `eval/find_best.py` | the main one: searches chunking, embedder, retrieval method, reranker, chunk size and answer model, then writes `best_config.json` and `eval/find_best_report.json` (`--stages` runs only some stages) |
| `eval/retriever_eval.py` | scores the retrieval methods (Recall, Precision, HitRate, MRR) |
| `eval/generator_eval.py` | scores answers (faithfulness, relevancy, correctness, similarity, context recall); `--oracle` gives the generator the golden pages |
| `eval/rag_triad_eval.py` | the RAG triad, scored with deepeval and a judge model (the UI scores every answer) |
| `eval/chunking_compare.py` | scores each chunking strategy |
| `eval/reranker_compare.py` | compares rerankers |
| `eval/judge_retrieval.py` | ask a question of any PDF and have the judge rate each retrieved chunk |

The 45 golden questions are in `golden_data/`. Metric definitions: `specs_md/retriever_eval.md` and `specs_md/generator_eval.md`.

```bash
uv run python eval/find_best.py     # slow (hours): needs the Ollama models in config.toml pulled
uv run pytest                       # unit tests, no models needed
uv run ruff check .                 # lint
```

### What the latest run found (reranker stage, 30 tuning queries)

| reranker | MRR | Recall@5 | sec/query |
|---|---|---|---|
| no reranker (plain hybrid) | 0.87 | 0.95 | 0.01 |
| LLM `qwen2.5:1.5b` / `3b` / `llama3.2:3b` | 0.58 / 0.61 / 0.66 | 0.82 / 0.93 / 0.82 | 1.1 to 2.0 |
| cross-encoder `bge-reranker-v2-m3` | 0.83 | 0.88 | 0.78 |
| **cross-encoder `mxbai-rerank-base-v1`** | **0.90** | 0.93 | 0.52 |

LLM rerankers made retrieval worse than no reranker; the `mxbai` cross-encoder is a small gain over plain hybrid and is
what `best_config.json` uses. On the 15 held-out queries the pick scored MRR 0.79 against 0.61 for the old settings.

## Notes

- With 45 questions, gaps of a few hundredths are noise.
- The judge model must differ from the answer model, so a model never grades its own answers.
- Each answer has a thumbs up/down. A thumbs down asks what was wrong and offers a reason-based second try; the ratings live in the session cache and disappear with it (`user_feedback` branch, details in `specs_md/user_feedback.md`).
- Only PDFs are accepted (scanned PDF pages are read with OCR); standalone images are not supported.
- Uploaded documents are indexed in `vs_docs/`, named by a hash of the files; the session cache lives in its own temporary
  folder and is deleted when the documents are replaced or the program exits.
- `main` has not yet been run against live provider APIs (its tests use fake clients), and deploying it still needs
  HTTPS, a login or rate limit, cleanup of stored uploads, and a container setup. Prices in `providers.toml` are list prices
  to be checked against each provider's page.
- `RAGForge_RAG_Pipeline_Design.md` and `RAGForge_Tasks.md` describe a larger production design (Postgres, Docker).
  Much of it is deferred; this repo is the free, local version plus a bring-your-own-key variant.
