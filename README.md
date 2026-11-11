# RAGForge

Ask questions about your PDFs and get answers with **citations**, the **chunks they came from**, and **live quality scores**.
RAGForge is a small, readable retrieval-augmented generation (RAG) system built with plain classes. This is the
**bring-your-own-key** version: every model is hosted, and you choose the provider and model for each component.

- Upload one or more PDFs (scanned pages are read with OCR), click Ingest, then ask by typing (or by voice, with OpenAI).
- Hybrid search (vectors + BM25 keywords, fused with RRF), then an **LLM reranker**, then an answer model.
- Every answer shows the pages it cites, the retrieved and reranked chunks with their scores, and the
  **RAG triad** (context relevance, groundedness, answer relevance) scored by a judge model. No reference data is needed.
- **Your usage:** every model call made with your key is counted. Each turn shows its calls, tokens and estimated cost with a
  breakdown by component, the sidebar shows the session total, and the ingest cost is shown after you click Ingest.
- Thumbs up/down on every answer; a thumbs down asks what was wrong and offers a second try.
- A question like an earlier one in the same conversation is answered from a private session cache.
- An input guard checks every question for prompt injection and other malicious input.

## Branches

| branch | purpose | models |
|---|---|---|
| **`main`** (this one) | The **bring-your-own-key (BYOK) version, with no Ollama and no local models.** Users pick OpenAI, Anthropic or Gemini and paste their own key in the web app. Only the RAG triad is evaluated. | Hosted, through [LiteLLM](https://docs.litellm.ai/) |
| **`dev`** | The free, local version: everything runs on your machine with [Ollama](https://ollama.com), plus a local cross-encoder and Whisper. | Local |
| **`offline-eval`** | The evaluation lab: golden-dataset evaluations and the `find_best.py` settings search. | Ollama |

The earlier BYOK work is kept as the tag `byok-snapshot`.

## Running it

```bash
git clone -b main https://github.com/janarddansarkar/RAGFORGE.git && cd RAGFORGE
# install uv first (https://docs.astral.sh/uv/)
./setup.sh      # installs the Python packages; no model is downloaded
./run.sh        # the web app at http://localhost:8501
```

In the web app's sidebar, pick a **provider and a model for each component**, then paste a key for each provider you
chose. Keys stay in the session's memory only: they are never written to disk or logged.

| Component | What it does | OpenAI | Anthropic | Google Gemini |
|---|---|---|---|---|
| Answers | answers, retries, chunk notes | `gpt-4o`, `gpt-4o-mini`, `gpt-4.1`, `gpt-4.1-mini` | `claude-sonnet-5-5`, `claude-opus-5-5`, `claude-haiku-5-5` | `gemini-2.0-flash`, `-flash-lite`, `2.5-flash`, `2.5-pro` |
| Input guard | blocks prompt injection | same chat models | same | same |
| Judge | scores the RAG triad | same | same | same |
| Reranker | scores each chunk 0-10 | same | same | same |
| Embeddings | vectors for the search | `text-embedding-3-small`, `-3-large` | not offered | `text-embedding-004`, `gemini-embedding-001` |
| Voice input (optional) | speech to text | `whisper-1`, `gpt-4o-transcribe` | not offered | not offered |

"Custom..." lets you type any other model name. The defaults are all OpenAI, so one key is enough to start. For the
command line, set the provider's key as an environment variable (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`GEMINI_API_KEY`); the defaults in `config.toml` use OpenAI.

```bash
uv run python src/rag.py documents/LoRA.pdf "What rank r does LoRA use?"   # the command line
```

Scanned pages need Tesseract (`brew install tesseract`). Everything you send (questions and document text) goes to the
providers you chose, and you pay for the calls. Building the index adds one model call per chunk for its context note.

## How it works

```
PDF -> pages (OCR for scans) -> chunks (+ optional LLM context note) -> embeddings -> Chroma
question -> input guard -> (rewrite follow-ups) -> cache check
         -> hybrid search (dense + BM25, fused) -> LLM reranker -> top-k chunks
         -> answer model (streamed) -> answer + citations -> RAG-triad scores
```

| file | role |
|---|---|
| `src/chunking.py` | PDFs to chunks (recursive or semantic; optional LLM context notes; OCR for scanned pages) |
| `src/vectorstore.py` | builds and opens the Chroma store; `load_config()` |
| `src/retriever.py` | dense, BM25, hybrid, reranked and LLM-assisted retrieval |
| `src/generator.py` | answers a question from chunks, citing the pages used; the retry after a thumbs down |
| `src/rag.py` | `DocumentRAG` for PDFs, the citation report and the command line |
| `src/rag_pipeline.py` | the full pipeline over `documents/` with the tuned settings |
| `src/app.py` | the Streamlit UI, including the provider and model choices |
| `src/providers.py` | the providers (OpenAI, Anthropic, Gemini), their models, and the per-component choices |
| `src/models.py` | the model wrappers: chat, embeddings and rerank through LiteLLM |
| `src/usage.py` | counts every model call, its tokens and cost (`UsageTracker`) |
| `src/guardrails.py` | the input guard (rules plus a guard model) |
| `src/cache.py` | the per-conversation session cache |
| `src/memory.py` | short-term memory: the last few turns, follow-ups rewritten into standalone questions |
| `src/clock.py` | today's date for the prompt |
| `src/voice.py` | speech to text (a hosted API), macOS `say` for reading answers aloud |
| `eval/rag_triad_eval.py` | the RAG triad, scored with deepeval and a judge model |
| `config.toml` | all settings (the defaults for the command line) |
| `best_config.json` | the tuned settings, laid over `config.toml` |

## Settings

`config.toml` holds every setting. `best_config.json` holds the tuned settings and **overrides** the matching `config.toml`
values, so the pipeline uses them without code changes. Delete that file to go back to `config.toml` alone. In the web
app, the sidebar's model choices override the model settings for that session.
`reranker_type` chooses between `llm` (a chat model scoring each chunk 0-10, the default) and `cross_encoder`
(a rerank API such as `cohere/rerank-v3.5` scoring all chunks in one call).

## Evaluation

The only evaluation on `main` is the **RAG triad**, scored by a judge model with deepeval. The web app scores every
answer; `uv run python eval/rag_triad_eval.py "your question"` scores one from the command line. See
`docs/RAG_EVALUATION.md`.

```bash
uv run pytest                       # unit tests, no API calls
uv run ruff check .                 # lint
```

## Notes

- The judge model should differ from the answer model, so a model never grades its own answers.
- Each answer has a thumbs up/down; the ratings live in the session cache and disappear with it (details in `specs_md/user_feedback.md`).
- Only PDFs are accepted (scanned PDF pages are read with OCR); standalone images are not supported.
- Uploaded documents are indexed in `vs_docs/`, named by a hash of the files and the embedding model; the session cache
  lives in its own temporary folder and is deleted when the documents are replaced or the program exits.
- Costs are **estimates**: LiteLLM multiplies the tokens by its price list. A model missing from that list is counted as
  an "unpriced call" and left out of the cost, so your provider's bill is the truth. An LLM reranker makes about 20 small
  calls per question; a `cohere/rerank` model would make one.
- Changing a provider or model in the sidebar resets the ingested documents: click Ingest again.
- `main` has not yet been run against live provider APIs (its tests use fake clients), and deploying it still needs
  HTTPS, a login or rate limit, cleanup of stored uploads, and a container setup.
- `RAGForge_RAG_Pipeline_Design.md` describes a larger production design (Postgres, Docker, local models) that this repo
  does not follow; see the note at its top.
