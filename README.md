# RAGForge

Ask questions about PDFs and images. Everything runs **locally and free**: embeddings, reranker,
answer model, speech-to-text and the evaluation judge are all [Ollama](https://ollama.com) models
(no API keys, no per-call cost). Each model is pulled once, then works offline.

## Quick start

```bash
ollama serve                           # if Ollama is not already running
ollama pull bge-m3 qwen2.5:3b          # embeddings, answers and reranking (see config.toml)
ollama pull mistral:7b                 # the evaluation judge
ollama pull gemma4:e2b                 # only for spoken questions in the UI
uv sync
uv run python src/rag.py documents/LoRA.pdf "What rank r does LoRA use?"
```

`rag.py` indexes the PDF the first time (about a minute, stored in `vs_docs/`), then prints the
answer, its citations and the top-k chunks it was written from. Leave the question out to ask
several. Options: `--k 3` (chunks), `--method hybrid` (retrieval method), `--full` (whole chunks).

Models and the Ollama address (`ollama_host`) are set in `config.toml`; a missing model gives a
clear `ollama pull ...` message. Ollama has no cross-encoder reranker, so reranking asks a chat
model to score each chunk 0-10 (ties keep the search order).

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
| `src/rag.py` | puts it together for PDFs and images (scans are read with Tesseract: `brew install tesseract`) |
| `src/rag_pipeline.py` | the full pipeline over `documents/`, using the best settings from `best_config.json` |
| `src/app.py` | Streamlit UI: upload one or more PDFs, Ingest, ask, see answer + citations + reranked chunks (`uv run streamlit run src/app.py`) |
| `src/voice.py` | speech to text (an Ollama audio model) for spoken questions, macOS `say` for reading answers aloud |
| `src/memory.py` | short-term memory: the last few turns, and follow-ups rewritten into standalone questions |
| `src/models.py` | the Ollama embedder, chat and audio models |
| `config.toml` | all settings |
| `best_config.json` | settings found best by the evaluation; laid over `config.toml` |

## Settings

`config.toml` holds every setting. After `eval/find_best.py` has run, the settings it found best
| `eval/rag_triad_eval.py` | RAG triad (context relevance, groundedness, answer relevance) scored live with deepeval and a local judge; the UI has a checkbox for it |
are in `best_config.json` and **override** the matching `config.toml` values, so the pipeline
uses them without code changes. Delete that file to go back to `config.toml` alone.

## Evaluation

| script | what it does |
|---|---|
| `eval/find_best.py` | the main one: searches chunking, embedder, retrieval method, reranker and answer model, then writes `best_config.json` and `eval/find_best_report.json` |
| `eval/rag_triad_eval.py` | RAG triad (context relevance, groundedness, answer relevance) scored live with deepeval and a local judge; the UI has a checkbox for it |
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

- `best_config.json` and `eval/*_report.json` from before the Ollama switch were measured with other (Hugging Face) models; re-run `eval/find_best.py` to measure the Ollama setup.
- The judge model must differ from the answer model, so a model never grades its own answers.
- Evaluation scores come from 45 questions over three papers: gaps of a few hundredths are noise.
- `RAGForge_RAG_Pipeline_Design.md` and `RAGForge_Tasks.md` describe a larger production design
  (Ollama, Postgres, Docker). Much of it is deferred; this repo is the free, local version.
