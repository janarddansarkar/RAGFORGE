# How to use RAGForge

## First time

```bash
git clone -b ragforge_api https://github.com/janarddansarkar/RAGFORGE.git
cd RAGFORGE
uv run streamlit run src/app.py   # installs the Python packages on the first run (no model is downloaded)
```

Install [uv](https://docs.astral.sh/uv/) first. Scanned PDFs also need Tesseract (`brew install tesseract`).
Open http://localhost:8501

## In the app

1. In the sidebar, pick a **provider** (OpenAI, Anthropic or Google Gemini) and a **model** for each component:
   answers, input guard, judge, reranker, embeddings, and optional voice input. The defaults are all OpenAI.
2. Paste an API key for each provider you picked. Keys stay in this session's memory only.
3. Upload one or more PDFs (**at most 20 pages in total**; the page counter shows how many you have) and click
   **Ingest**. The first time is slow, because every chunk gets a context note from the answer model. Your PDFs are
   deleted from the server as soon as they are indexed, and the index is deleted when you replace the documents or your
   session ends, so you upload them again next time.
4. Ask questions. Each answer shows its citations, the chunks behind it, and the RAG triad scores.
5. Watch your cost. Under each answer, **This turn** shows the model calls, tokens and estimated cost, with a breakdown
   by component (guard, embeddings, reranker, answer, judge). The sidebar's **Your usage** shows the session total, and
   the line under *Ready* shows what ingesting cost. Costs are estimates; your provider's bill is the truth.
6. Rate an answer with the thumbs. A thumbs down asks what was wrong and offers **Try again**.

Choices to know about:

- **Anthropic** is available for chat only (answers, guard, judge, reranker). Embeddings need OpenAI or Gemini, and
  voice input needs OpenAI.
- **Reranker:** an LLM rates each retrieved chunk (about 30 calls per question), or choose **Cohere** to use its
  cross-encoder (one call per question; needs a Cohere key).
- Changing a provider or model resets the documents: click Ingest again.

## Scoring one question from the command line

Set the provider's key first (the defaults use OpenAI):

```bash
export OPENAI_API_KEY=...
uv run python src/triad.py your.pdf "What rank r does LoRA use?"   # answer + RAG triad
```

## Running the evaluation

The offline evaluation uses the 45 golden questions and the sample papers in `documents/`. It costs real API money
(about $1.3 for the evaluation stages plus a one-time index of about $0.7 on OpenAI), so read the saved results first:
`docs/EVALUATION_RESULTS.md`.

```bash
uv run python eval/run_provider_eval.py openai --limit 10      # a cheap trial
uv run python eval/run_provider_eval.py openai                 # the full run (about 100 minutes)
uv run python eval/make_report.py                              # rebuild the results page (free)
```

Put the key in the environment or in a `.env` file (`OPENAI_API_KEY=...`). See `docs/RAG_EVALUATION.md`.

## Next times

```bash
cd RAGFORGE
uv run streamlit run src/app.py
```

## Update to the latest version

```bash
cd RAGFORGE
git pull
uv run streamlit run src/app.py
```
