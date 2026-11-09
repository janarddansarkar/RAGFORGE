# RAGForge

Ask questions about PDFs. This is the **hosted version** (`main`): every user brings their own API key
(OpenAI, Gemini or Anthropic) and the key pays for embeddings, answers, reranking, the input guard and the judge.
The reranker (a Hugging Face cross-encoder) and spoken questions (faster-whisper) run locally. For a fully local,
free version that runs on [Ollama](https://ollama.com) with no keys, use the `dev` branch.

## Quick start

```bash
uv sync
echo 'OPENAI_API_KEY=...' > .env       # command line + eval only (GEMINI_API_KEY / ANTHROPIC_API_KEY also work)
uv run streamlit run src/app.py        # the web app: paste your key in the sidebar
uv run python src/rag.py documents/LoRA.pdf "What rank r does LoRA use?"   # command line: key from .env
```

`rag.py` indexes the PDF the first time (about a minute, stored in `vs_docs/`), then prints the
answer, its citations and the top-k chunks it was written from. Leave the question out to ask
several. Options: `--k 3` (chunks), `--method hybrid` (retrieval method), `--full` (whole chunks).

The command line scripts and the eval use `chat_provider` and `embedding_provider` from `config.toml`
(default `openai`) and the key of that provider from `.env`. The models of each provider are in
`providers.toml`. Reranking asks a chat model to score each chunk 0-10 (ties keep the search order),
or uses a local cross-encoder when `reranker_type = "cross_encoder"`.

## Bring your own API key (web app)

In the web app's sidebar the user picks a provider (OpenAI, Gemini or Anthropic) and pastes their own key; a second provider can be chosen for embeddings (Anthropic has no embeddings API). The key is held in that browser session's memory only and is never written to disk. It pays for everything the app asks a model: embeddings, contextual chunk notes, query rewriting, reranking, the answer, the input guard and the RAG-triad judge. The models and their prices per provider are in `providers.toml`. The UI shows the approximate cost of every turn and of the whole conversation (list prices, from the token counts the providers report).

## How it works

```
PDF -> pages -> chunks -> embeddings -> Chroma (./vs)
question -> hybrid search (dense + BM25, fused) -> reranker -> top-k chunks -> answer model -> answer + citations
```

| file | role |
|---|---|
| `src/chunking.py` | PDFs to chunks (recursive, tokens, semantic; optional LLM context notes) |
| `src/vectorstore.py` | builds and opens the Chroma store; `load_config()` |
| `src/retriever.py` | dense, BM25, hybrid, reranked and LLM-assisted retrieval |
| `src/generator.py` | answers a question from chunks |
| `src/rag.py` | puts it together for PDFs (scanned pages are read with Tesseract: `brew install tesseract`) |
| `src/rag_pipeline.py` | the full pipeline over `documents/`, using the best settings from `best_config.json` |
| `src/app.py` | Streamlit UI: upload one or more PDFs, Ingest, ask, see answer + citations + reranked chunks (`uv run streamlit run src/app.py`) |
| `src/voice.py` | speech to text (faster-whisper, on CPU) for spoken questions, macOS `say` for reading answers aloud |
| `src/memory.py` | short-term memory: the last few turns, and follow-ups rewritten into standalone questions |
| `src/models.py` | the cloud embedder and chat models (OpenAI, Gemini, Anthropic), and the token and cost tracking |
| `config.toml` | all settings |
| `best_config.json` | settings found best by the evaluation; laid over `config.toml` |

## Settings

`config.toml` holds every setting. After `eval/find_best.py` has run, the settings it found best
are in `best_config.json` and **override** the matching `config.toml` values, so the pipeline
uses them without code changes. Delete that file to go back to `config.toml` alone.

## Evaluation

| script | what it does |
|---|---|
| `eval/find_best.py` | the main one: searches chunking, embedder, retrieval method, reranker and answer model, then writes `best_config.json` and `eval/find_best_report.json` |
| `eval/rag_triad_eval.py` | RAG triad (context relevance, groundedness, answer relevance) scored with deepeval and a judge model (your key); the UI scores every answer |
| `eval/retriever_eval.py` | scores the retrieval methods (Recall, Precision, HitRate, MRR) |
| `eval/generator_eval.py` | scores answers (faithfulness, relevancy, correctness, similarity, context recall); `--oracle` gives the generator the golden pages |
| `eval/chunking_compare.py` | scores each chunking strategy |
| `eval/judge_retrieval.py` | ask a question of any PDF and have the judge rate each retrieved chunk |

The 45 test questions are in `golden_data/`. Metric definitions and how to read them:
`specs_md/retriever_eval.md` and `specs_md/generator_eval.md`.

```bash
uv run python eval/find_best.py        # slow (hours): needs the models in config.toml pulled, rebuilds stores
uv run pytest                          # unit tests (no models needed)
```

## Notes

- `best_config.json` and `eval/*_report.json` were measured with local Ollama models on `dev`/`offline-eval`; the chunking and retrieval settings carry over, the model names do not. Re-run `eval/find_best_api.py` to tune for your provider.
- The judge model must differ from the answer model, so a model never grades its own answers.
- Evaluation scores come from 45 questions over three papers: gaps of a few hundredths are noise.
- `RAGForge_RAG_Pipeline_Design.md` and `RAGForge_Tasks.md` describe a larger production design
  (Ollama, Postgres, Docker). Much of it is deferred.
