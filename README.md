# RAGForge

Ask questions about PDFs. Everything runs **locally and free**: embeddings, reranker, answer
model and the evaluation judge are Hugging Face models that run inside Python (no server, no
API keys, no per-call cost). Models download once to `~/.cache/huggingface`, then work offline.

## Quick start

```bash
uv sync
uv run python src/rag.py documents/LoRA.pdf "What rank r does LoRA use?"
```

`rag.py` indexes the PDF the first time (about a minute, stored in `vs_docs/`), then prints the
answer, its citations and the top-k chunks it was written from. Leave the question out to ask
several. Options: `--k 3` (chunks), `--method hybrid` (retrieval method), `--full` (whole chunks).

A free Hugging Face token speeds up downloads: put `HF_TOKEN=...` in `.env`
(then `export HF_TOKEN` or load it in your shell).

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
| `src/rag.py` | puts it together for one PDF |
| `src/rag_pipeline.py` | the full pipeline over `documents/`, using the best settings from `best_config.json` |
| `src/app.py` | Streamlit UI: upload a PDF, Ingest, ask, see answer + citations + reranked chunks (`uv run streamlit run src/app.py`) |
| `src/models.py` | loads the Hugging Face embedder and answer model |
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
| `eval/retriever_eval.py` | scores the retrieval methods (Recall, Precision, HitRate, MRR) |
| `eval/generator_eval.py` | scores answers (faithfulness, relevancy, correctness, similarity, context recall); `--oracle` gives the generator the golden pages |
| `eval/chunking_compare.py` | scores each chunking strategy |
| `eval/judge_retrieval.py` | ask a question of any PDF and have the judge rate each retrieved chunk |

The 45 test questions are in `golden_data/`. Metric definitions and how to read them:
`specs_md/retriever_eval.md` and `specs_md/generator_eval.md`.

```bash
uv run python eval/find_best.py        # slow (hours): downloads models, rebuilds stores
uv run pytest                          # unit tests (no models needed)
```

## Notes

- The judge model must differ from the answer model, so a model never grades its own answers.
- Evaluation scores come from 45 questions over three papers: gaps of a few hundredths are noise.
- `RAGForge_RAG_Pipeline_Design.md` and `RAGForge_Tasks.md` describe a larger production design
  (Ollama, Postgres, Docker). Much of it is deferred; this repo is the free, local version.
