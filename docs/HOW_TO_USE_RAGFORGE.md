# How to use RAGForge

## First time

```bash
git clone -b main https://github.com/janarddansarkar/RAGFORGE.git
cd RAGFORGE
./setup.sh      # installs the Python packages; no model is downloaded
./run.sh
```

Open http://localhost:8501

## In the app

1. In the sidebar, pick a **provider** (OpenAI, Anthropic or Google Gemini) and a **model** for each component:
   answers, input guard, judge, reranker, embeddings, and optional voice input. The defaults are all OpenAI.
2. Paste an API key for each provider you picked. Keys stay in this session's memory only.
3. Upload one or more PDFs and click **Ingest**. The first time is slow, because every chunk gets a context note
   from the answer model.
4. Ask questions. Each answer shows its citations, the chunks behind it, and the RAG triad scores.
5. Watch your cost. Under each answer, **This turn** shows the model calls, tokens and estimated cost, with a breakdown
   by component (guard, embeddings, reranker, answer, judge). The sidebar's **Your usage** shows the session total, and
   the line under *Ready* shows what ingesting cost. Costs are estimates; your provider's bill is the truth.

Anthropic is available for chat only (answers, guard, judge, reranker). Embeddings need OpenAI or Gemini, and voice
input needs OpenAI. Changing a provider or model resets the documents: click Ingest again.

## From the command line

Set the provider's key first (the defaults use OpenAI):

```bash
export OPENAI_API_KEY=...
uv run python src/rag.py documents/LoRA.pdf "What rank r does LoRA use?"
uv run python eval/rag_triad_eval.py "What rank r does LoRA use?"     # the RAG triad for one question
```

## Next times

```bash
cd RAGFORGE
./run.sh
```

## Update to the latest version

```bash
cd RAGFORGE
git pull
./setup.sh
./run.sh
```
