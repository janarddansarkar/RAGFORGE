# RAGForge

Ask questions about your PDFs and get answers with **citations**, the **chunks they came from**, **live quality scores** and
**what each answer cost**. RAGForge is a small, readable retrieval-augmented generation (RAG) system built with plain
classes. It is **bring-your-own-key**: every model is hosted and called through [LiteLLM](https://docs.litellm.ai/), and
you choose the provider and the model for each component. Nothing runs on your machine except the app itself.

- Upload one or more PDFs (at most **20 pages in total**; scanned pages are read with OCR), click Ingest, then ask by
  typing (or by voice, with OpenAI).
- Hybrid search (vectors + BM25 keywords, fused with RRF), then a **reranker** (an LLM, or Cohere's cross-encoder),
  then an answer model.
- Every answer shows the pages it cites, the retrieved and reranked chunks with their scores, and the **RAG triad**
  (context relevance, groundedness, answer relevance) scored by a judge model. The triad needs no reference answers.
- **Your usage:** every model call made with your key is counted. Each turn shows its calls, tokens and estimated cost with
  a breakdown by component, the sidebar shows the session total, and the ingest cost is shown after you click Ingest.
- Thumbs up/down on every answer; a thumbs down asks what was wrong and offers a second try.
- A question like an earlier one in the same conversation is answered from a private session cache.
- An input guard checks every question for prompt injection and other malicious input.
- **Uploads are private and short-lived:** each browser session gets its own folder, the PDFs are deleted as soon as they
  are indexed, and the index is deleted when the documents are replaced or the session ends (a reload, a closed tab or
  a new session).
- An offline evaluation on a 45-question golden dataset measures each component (retriever, reranker, chunking,
  generator) and the whole pipeline. The saved results are in `docs/EVALUATION_RESULTS.md`.

## Running it

```bash
git clone -b main https://github.com/janarddansarkar/RAGFORGE.git && cd RAGFORGE
# install uv first (https://docs.astral.sh/uv/)
uv run streamlit run src/app.py   # installs the Python packages on the first run (no model is
                                  # downloaded), then starts the web app at http://localhost:8501
```

Scanned pages need Tesseract (`brew install tesseract`, or `apt install tesseract-ocr` on a server). To run it on a
server (no Docker), see `docs/DEPLOY_VPS.md`.

In the web app's sidebar, pick a **provider and a model for each component**, then paste a key for each provider you
chose. Keys stay in the session's memory only: they are never written to disk or logged.

| Component | What it does | OpenAI | Anthropic | Google Gemini |
|---|---|---|---|---|
| Answers | answers, retries, chunk notes | `gpt-4o`, `gpt-4o-mini`, `gpt-4.1`, `gpt-4.1-mini` | `claude-sonnet-5-5`, `claude-opus-5-5`, `claude-haiku-5-5` | `gemini-3.5-flash`, `gemini-3.5-flash-lite`, `gemini-3.6-flash`, `gemini-3.1-flash-lite` |
| Input guard | blocks prompt injection | same chat models | same | same |
| Judge | scores the RAG triad | same | same | same |
| Reranker | an LLM rates each chunk 0-10 (about 30 calls), or a cross-encoder scores all chunks in one call | same chat models | same | same, **or Cohere** `rerank-v3.5`, `rerank-v4.0-pro`, `rerank-v4.0-fast` (the cross-encoder; needs a Cohere key) |
| Embeddings | vectors for the search | `text-embedding-3-small`, `-3-large` | not offered | `gemini-embedding-001`, `gemini-embedding-2` |
| Voice input (optional) | speech to text | `whisper-1`, `gpt-4o-transcribe` | not offered | not offered |

"Custom..." lets you type any other model name. The defaults are all OpenAI, so one key is enough to start.

Everything you send (questions and document text) goes to the providers you chose, and you pay for the calls. Building
the index adds one model call per chunk for its context note.

**The Cohere reranker.** Choosing Cohere switches the reranker from the LLM scorer to a cross-encoder: one rerank call
scores all 30 candidates, and a Cohere key field appears. If the Cohere call fails, the answer keeps the plain search
order and the failure is logged. A Cohere **trial** key allows 10 calls a minute; set `rerank_calls_per_minute = 9` in
`config.toml` to stay under it.

## How it works

```
PDF -> pages (OCR for scans) -> chunks (+ optional LLM context note) -> embeddings -> Chroma
question -> input guard -> (rewrite follow-ups) -> cache check
         -> hybrid search (dense + BM25, fused) -> reranker (LLM or Cohere) -> top-k chunks
         -> answer model (streamed) -> answer + citations -> RAG-triad scores
```

| file | role |
|---|---|
| `src/app.py` | the Streamlit UI, including the provider and model choices |
| `src/providers.py` | the providers (OpenAI, Anthropic, Gemini, Cohere), their models, and the per-component choices |
| `src/models.py` | the model wrappers: chat, embeddings and rerank through LiteLLM |
| `src/usage.py` | counts every model call, its tokens and cost (`UsageTracker`) |
| `src/rag.py` | `DocumentRAG` for PDFs and the citation helper |
| `src/chunking.py` | PDFs to chunks (recursive splitting; optional LLM context notes; OCR for scanned pages) |
| `src/vectorstore.py` | builds and opens the Chroma store; `load_config()` |
| `src/retriever.py` | dense, BM25, hybrid, reranked and multi-query retrieval |
| `src/generator.py` | answers a question from chunks, citing the pages used; the retry after a thumbs down |
| `src/guardrails.py` | the input guard (rules plus a guard model) |
| `src/cache.py` | the per-conversation session cache |
| `src/workspace.py` | the private, short-lived folders for each session's uploads and their cleanup |
| `src/memory.py` | short-term memory: the last few turns, follow-ups rewritten into standalone questions |
| `src/clock.py` | today's date for the prompt |
| `src/voice.py` | speech to text (a hosted API) |
| `src/triad.py` | the RAG triad, scored with DeepEval and a judge model |
| `config.toml` | all settings (the defaults when no sidebar choice overrides them) |
| `.streamlit/config.toml` | Streamlit's server settings (headless, a 25 MB upload cap per file) |
| `eval/`, `metrics/`, `golden_data/`, `documents/` | the offline evaluation (see below) |
| `docs/`, `specs_md/` | the documentation |

## Settings

`config.toml` holds every setting: chunk size, top-k, the cache, the guard, the reranker type and the default models.
In the web app, the sidebar's model choices override the model settings for that session.

Every model request has a timeout (`request_timeout`, 60 seconds) and is retried `request_retries` times (2), so one
stalled API call cannot freeze the page.

## Evaluation

There are two kinds, described in `docs/RAG_EVALUATION.md`:

- **RAG triad (live, no reference answers).** The web app scores every answer with DeepEval and the judge model you
  chose. `uv run python src/triad.py your.pdf "your question"` scores one question from the command line.
- **Offline component evaluation (golden dataset).** `golden_data/` holds 45 questions over the three papers in
  `documents/`, each with its source pages and a reference answer. `eval/run_provider_eval.py` builds the index and runs
  the retriever, reranker, chunking, generator, oracle-generator, triad and latency stages with one provider's models,
  then saves the numbers and the cost to `eval/results/`. It costs real API money, so the saved results are kept
  and `docs/EVALUATION_RESULTS.md` is built from them without any model call.

```bash
uv run python eval/run_provider_eval.py openai                     # everything (about 100 minutes)
uv run python eval/run_provider_eval.py openai --limit 10          # a quick, cheaper trial
uv run python eval/run_provider_eval.py openai --stages retriever,generator
uv run python eval/make_report.py                                   # rebuild docs/EVALUATION_RESULTS.md (free)
uv run pytest                                                       # unit tests, no API calls
uv run ruff check .                                                 # lint
```

The runner reads `OPENAI_API_KEY` (or `GEMINI_API_KEY`) from the environment or a `.env` file. Metric definitions are
in `specs_md/retriever_eval.md` and `specs_md/generator_eval.md`.

## Notes

- The judge model should differ from the answer model, so a model never grades its own answers.
- If the documents answer only part of a question, the app answers that part and writes "The documents do not say ..."
  for the rest. It replies "I can't find that in the documents." only when none of the question can be answered.
- Each answer has a thumbs up/down; the ratings live in the session cache and disappear with it (details in
  `specs_md/user_feedback.md`).
- Only PDFs are accepted (scanned pages are read with OCR); standalone images are not supported.
- Nothing a user uploads is kept: the PDFs are deleted right after indexing, and the index and the answer cache live in
  the session's private folder in the system temp directory, which is deleted when the documents are replaced, and
  whenever any session starts or reruns for sessions that are no longer connected. A server restart removes whatever an
  earlier run left behind. If a connection drops and the folder is cleaned up meanwhile, the app asks the user to upload
  again. (The command line, `src/triad.py`, keeps its index in `vs_docs/`, and the offline evaluation reads `documents/`
  directly; neither has the 20-page limit.)
- The 20-page limit (`max_pages` in `config.toml`) counts all uploaded PDFs together. It is checked on the page and again
  before indexing, so a file over the limit never reaches a model.
- Changing a provider or model in the sidebar resets the ingested documents: click Ingest again.
- Costs are **estimates**: LiteLLM multiplies the tokens by its price list. A model missing from that list is counted as
  an "unpriced call" and left out of the cost, so your provider's bill is the truth.
- The offline evaluation has been run with OpenAI only. Gemini and Cohere were checked with a few live calls. Anthropic
  was tested end to end in the app (Claude Sonnet for answers, Claude Haiku for guard, judge and reranker, OpenAI
  embeddings): 7 questions, no errors, about $0.011 per question against about $0.007 with OpenAI. None of their
  quality has been measured on the golden set. A free-tier Gemini key (5 requests a minute) cannot complete the
  evaluation.
- Claude models "think" before answering and the thinking counts against the token limit, so the app asks for minimal
  thinking and adds 512 tokens of headroom for Claude (otherwise some answers came back empty).
- Deploying it for other people still needs HTTPS (see `docs/DEPLOY_VPS.md`) and, if it is public, a login or rate
  limit.
