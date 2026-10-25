# RAGForge — Task Breakdown

Built from `RAGForge_RAG_Pipeline_Design.md`, in the order given in §11.9. Each task is sized so it can be finished in **one session**: one clear goal, a handful of files, and tests.

## How to use this file

- Work top to bottom. A task starts only when everything in its **Depends on** list is done.
- At the start of a session: *"Implement task Tx.y from RAGForge_Tasks.md"*.
- A task is done only when its **Done when** checks pass. Then tick its box in the tracker below.
- Keep the scope tight. If new work turns up, add it as a new task; don't grow the current one.
- `§x.y` refers to sections of the design doc.

## Progress Tracker

| Phase | Tasks |
|---|---|
| 0 · Setup | [ ] T0.1 [ ] T0.2 [ ] T0.3 [ ] T0.4 [ ] T0.5 |
| 1 · Core RAG | [ ] T1.1 [ ] T1.2 [ ] T1.3 [ ] T1.4 [ ] T1.5 [ ] T1.6 [ ] T1.7 [ ] T1.8 [ ] T1.9 [ ] T1.10 [ ] T1.11 [ ] T1.12 [ ] T1.13 [ ] T1.14 [ ] T1.15 |
| 2 · Eval baseline | [ ] T2.1 [ ] T2.2 [ ] T2.3 [ ] T2.4 [ ] T2.5 [ ] T2.6 |
| 3 · Retrieval quality | [ ] T3.1 [ ] T3.2 [ ] T3.3 [ ] T3.4 [ ] T3.5 [ ] T3.6 |
| 4 · Reliability | [ ] T4.1 [ ] T4.2 [ ] T4.3 [ ] T4.4 [ ] T4.5 [ ] T4.6 [ ] T4.7 [ ] T4.8 [ ] T4.9 [ ] T4.10 [ ] T4.11 |
| 5 · Memory | [ ] T5.1 [ ] T5.2 [ ] T5.3 [ ] T5.4 |
| 6 · Observability | [ ] T6.1 [ ] T6.2 [ ] T6.3 [ ] T6.4 [ ] T6.5 [ ] T6.6 [ ] T6.7 |
| 7 · Hardening | [ ] T7.1 [ ] T7.2 [ ] T7.3 [ ] T7.4 |

---

## Phase 0 — Setup and Local Models

### T0.1 — Project scaffold
- **Goal:** An empty FastAPI app that runs and has tests.
- **Build:** `pyproject.toml` (uv or poetry), the `app/` folder tree from §11.2 (empty packages), `app/main.py` with `GET /health`, ruff + pytest config, `.gitignore`, `git init`.
- **Done when:** `uvicorn app.main:app` serves `/health` → `200`; `pytest` and `ruff check` pass.
- **Depends on:** —

### T0.2 — Configuration
- **Goal:** One config source for the whole app (§11.6).
- **Build:** `config/default.yaml` copied from §11.6; `app/core/config.py` using pydantic-settings, with env var overrides (e.g. `RAGFORGE__LLM__NUM_CTX`); `.env.example`. Secrets only from env.
- **Done when:** A unit test loads the config, checks the typed values, and checks that an env override works.
- **Depends on:** T0.1

### T0.3 — Docker Compose infrastructure
- **Goal:** Postgres + pgvector, Redis and MinIO running locally.
- **Build:** `docker-compose.yml` with `pgvector/pgvector` Postgres, Redis and MinIO, plus volumes and healthchecks. A `make up` / `make down` target. Ollama is **not** in Compose (it runs natively on macOS, §11.4).
- **Done when:** `docker compose up` gives three healthy services; `CREATE EXTENSION vector` works.
- **Depends on:** T0.1

### T0.4 — Ollama setup and model check
- **Goal:** Local models pulled, checked and benchmarked (§11.9 step 0).
- **Build:** `scripts/setup_models.sh` (`ollama pull qwen3:8b qwen3:1.7b bge-m3`); `scripts/bench_llm.py`, which measures tokens/sec and time to first token at `num_ctx=8192`; a startup check in `app/main.py` that fails with a clear error if a required model tag is missing. Write down the host env vars (`OLLAMA_FLASH_ATTENTION`, `OLLAMA_KV_CACHE_TYPE`, `OLLAMA_NUM_PARALLEL`) in the README.
- **Done when:** The benchmark prints tok/s for the generator (target ≥ 15–20); the startup check fails when a model is missing.
- **Depends on:** T0.2

### T0.5 — Logging and request IDs
- **Goal:** Structured logs with a `request_id` on every request.
- **Build:** `app/core/logging.py` (JSON logs via structlog or stdlib); middleware that creates or passes on `X-Request-ID` and `trace_id`.
- **Done when:** Every log line for a request has the same `request_id`; a test checks the response header.
- **Depends on:** T0.1

---

## Phase 1 — Core RAG (synchronous, minimal)

### T1.1 — Database schema and migrations
- **Goal:** The core tables with isolation keys from day one.
- **Build:** SQLAlchemy models plus Alembic for `users, knowledge_bases, documents, chunks (embedding vector(1024)), ingestion_jobs, traces`. The chunk metadata columns follow §4.5. Indexes on `(user_id, knowledge_base_id)` and an HNSW/IVFFlat index on `embedding`.
- **Done when:** `alembic upgrade head` works on a fresh DB; a test inserts and reads back a chunk with a vector.
- **Depends on:** T0.3

### T1.2 — Dev auth and knowledge-base CRUD
- **Goal:** Users own knowledge bases; ownership is checked in the data layer.
- **Build:** A simple dev auth dependency (API key or signed dev token → `user_id`; full JWT comes in T4.1). `POST/GET/DELETE /api/v1/knowledge-bases`. A repository layer where **every** query takes `user_id`.
- **Done when:** User A gets `404` on User B's KB (test).
- **Depends on:** T1.1

### T1.3 — Upload endpoint and validation
- **Goal:** Fail-fast validation from §4.1, steps 1–6.
- **Build:** `app/ingestion/validator.py` (extension/MIME/PDF validity, size, ≤ 5 docs, ≤ 50 total pages, duplicate SHA-256 within the KB); a storage abstraction (local disk now, MinIO-ready); `POST /knowledge-bases/{kb_id}/documents` saves the file and a document record with status `UPLOADED`.
- **Done when:** A unit test covers each rejection rule, using small fixture PDFs in `tests/fixtures/`.
- **Depends on:** T1.2

### T1.4 — PDF parser
- **Goal:** PDF → structured `Document` (§4.3).
- **Build:** `app/ingestion/parser.py` using PyMuPDF; dataclasses `Document`, `Page(page_number, text, tables, layout_metadata)`; keeps font size and bold info per text block for T1.6.
- **Done when:** Tests on fixture PDFs check page count, text and page numbers.
- **Depends on:** T0.1

### T1.5 — Text cleaner
- **Goal:** Clean text without losing structure.
- **Build:** `app/ingestion/cleaner.py`: removal of repeated headers/footers across pages, Unicode/whitespace normalization, hyphenation and broken-line repair, empty-page removal.
- **Done when:** Table-driven unit tests cover each cleaning rule.
- **Depends on:** T1.4

### T1.6 — Structure detection
- **Goal:** Headings and sections become metadata.
- **Build:** `app/ingestion/structure.py`: heading detection from font size and bold (with a numbered-heading regex as backup); gives each block a `section` path; marks table blocks.
- **Done when:** Tests on a fixture with known headings give the right section per page.
- **Depends on:** T1.5

### T1.7 — Chunker
- **Goal:** Structure-aware recursive chunking (§4.4).
- **Build:** `app/ingestion/chunker.py`: split on section → paragraph → sentence; ~650 tokens, 80 overlap, min 100 (from config); keep tables whole; fill `page_start/page_end/section/chunk_index/chunking_version`. Token counting with a tokenizer helper.
- **Done when:** Tests: no chunk over the limit, no chunk under the minimum (except the last one), chunks don't cross sections, page ranges are correct, tables are intact.
- **Depends on:** T1.6

### T1.8 — Provider interfaces and Ollama embeddings
- **Goal:** Swappable model providers (§11.1).
- **Build:** `app/providers/base.py` (`LLMProvider`, `EmbeddingProvider`, `Reranker` protocols); `ollama_embeddings.py` with batching (32), per-model task prefixes, timeout and retry.
- **Done when:** An integration test embeds 3 texts → 1024-d vectors; a unit test with a mocked HTTP client checks the prefixes and batching.
- **Depends on:** T0.4

### T1.9 — Synchronous ingestion pipeline
- **Goal:** Upload → Parse → Clean → Chunk → Embed → Index, in one call.
- **Build:** `app/ingestion/pipeline.py`, which moves the document through the status machine (§4.2) and records `FAILED` with the error message. For now it runs inline after upload (it becomes async in T4.2).
- **Done when:** An integration test uploads a PDF, the document reaches `COMPLETED`, and the chunks have embeddings in the DB.
- **Depends on:** T1.3, T1.7, T1.8

### T1.10 — Dense retrieval with isolation
- **Goal:** Scoped vector search (§5.2).
- **Build:** `app/retrieval/dense.py`: embed the query, run a pgvector cosine top-K **filtered by `user_id` and `knowledge_base_id` in SQL**; optional `document_id` filter.
- **Done when:** Tests show relevant chunks rank at the top, and User B's identical content is **never** returned for User A.
- **Depends on:** T1.9

### T1.11 — Ollama LLM provider
- **Goal:** A reliable local generation call.
- **Build:** `app/providers/ollama_llm.py`: `num_ctx` and `keep_alive` set explicitly on every call; `format` = JSON schema; thinking turned off; timeout (120 s); streaming support; returns the token counts from the Ollama response. (The Redis gate is added in T4.6.)
- **Done when:** An integration test gets valid JSON for a toy schema; token counts are returned.
- **Depends on:** T1.8

### T1.12 — Prompt builder and generator
- **Goal:** A grounded answer in the §5.8 contract.
- **Build:** `app/generation/prompts.py` (a versioned system prompt with the untrusted-document rule from §5.7, chunks shown with their IDs); `generator.py` checks the output against a pydantic `Answer` model and retries once if it doesn't match.
- **Done when:** A unit test with a mock LLM covers the retry on bad JSON; a live test returns an `Answer` with citations.
- **Depends on:** T1.11

### T1.13 — Citation validation and rendering
- **Goal:** Invalid citations never reach the user (§5.9).
- **Build:** `app/generation/citations.py`: check that cited IDs exist, were in the supplied context, and belong to this KB; remove or trigger a regenerate; render `[Doc name, p.14]` from chunk metadata (never from LLM text).
- **Done when:** Unit tests cover a made-up ID, an ID from outside the context, an ID from another KB, and correct rendering.
- **Depends on:** T1.12

### T1.14 — Chat API v1 and RAGState pipeline
- **Goal:** End-to-end question answering.
- **Build:** `RAGState` (§11.5); `app/rag_pipeline.py` with the stage functions (stages not built yet do nothing); `POST /api/v1/knowledge-bases/{kb_id}/chat`; save a basic trace row (query, chunk IDs + scores, prompt, raw output, answer, latency).
- **Done when:** An integration test uploads a PDF, asks a question, and gets an answer with a valid citation; the trace row exists.
- **Depends on:** T1.10, T1.13

### T1.15 — Deletion
- **Goal:** Delete a document or KB completely (§4.8).
- **Build:** `DELETE` for documents and KBs: remove the file, chunk rows and embeddings; add a hook to invalidate caches (used by BM25 in T3.1).
- **Done when:** A test deletes a document and then gets no hits for its unique content.
- **Depends on:** T1.14

---

## Phase 2 — Evaluation Baseline

### T2.1 — Eval dataset format and tooling
- **Goal:** A versioned golden dataset format (§7.1).
- **Build:** `app/evaluation/datasets.py` (pydantic schema, JSONL loader); relevant chunks labeled by **document + page + a text snippet** and resolved to chunk IDs when the eval runs; `datasets/rag_eval_v1/`.
- **Done when:** The loader checks the schema; the resolver maps snippet labels to current chunk IDs (test).
- **Depends on:** T1.9

### T2.2 — Write rag_eval_v1 (30–50 questions)
- **Goal:** The actual dataset.
- **Build:** Pick 3–5 real test PDFs; write 30–50 questions covering the §7.1 categories (at least 5 no-answer, 3 exact-match, 3 table, 3 prompt-injection). An LLM-assisted draft is fine; a human reviews every item.
- **Done when:** The dataset loads and every label resolves to at least one chunk.
- **Depends on:** T2.1

### T2.3 — Retriever metrics and runner
- **Goal:** Offline retrieval scores (§7.2).
- **Build:** `app/evaluation/retriever.py`: Recall@K, Precision@K, Hit@K, MRR, NDCG, **per stage** (dense now; BM25/fused/reranked later); `python -m app.evaluation.runner retriever --dataset v1`; results saved to the `eval_runs` table and as JSON.
- **Done when:** Unit tests check the metrics on hand-computed examples; the runner prints a table for dense retrieval.
- **Depends on:** T2.2, T1.10

### T2.4 — End-to-end runner (deterministic metrics)
- **Goal:** Full-pipeline eval without a judge.
- **Build:** `app/evaluation/end_to_end.py`: runs the chat pipeline for every item and records citation validity rate, abstention on no-answer questions, answer rate on answerable questions, latency, and tokens.
- **Done when:** The runner produces a report and a saved `eval_run`.
- **Depends on:** T2.3, T1.14

### T2.5 — Local LLM judge
- **Goal:** Faithfulness, relevance and correctness via a local judge (§7.3–7.4).
- **Build:** DeepEval or Ragas wired to an Ollama judge model (a different family from the generator); `scripts/judge_handcheck.py` exports 30 judgments to CSV for human labeling and reports agreement.
- **Done when:** The E2E report includes the judge metrics; the hand-check CSV is produced.
- **Depends on:** T2.4

### T2.6 — Whole-KB-in-context baseline
- **Goal:** A reference point for how good retrieval is (§1, §7.4).
- **Build:** An eval mode that puts every chunk of the KB in the prompt (bigger `num_ctx`), skipping retrieval.
- **Done when:** The report compares RAG and whole-KB on the same metrics (or notes that the hardware can't run it).
- **Depends on:** T2.5

---

## Phase 3 — Retrieval Quality

### T3.1 — BM25 sparse retrieval
- **Goal:** Exact-term retrieval (§4.6, §5.3).
- **Build:** `app/retrieval/sparse.py`: an in-memory `rank_bm25` index per `(user_id, kb_id)`, built from the chunks table and cached; the cache is cleared on ingest/delete (hook from T1.15); a simple tokenizer that keeps IDs and codes.
- **Done when:** Tests show an exact code/ID query finds the right chunk, and the cache is rebuilt after delete. BM25 stage metrics appear in the eval.
- **Depends on:** T2.3

### T3.2 — RRF fusion and hybrid retrieval
- **Goal:** Dense + BM25 → fused list (§5.3).
- **Build:** `app/retrieval/fusion.py` (RRF, k=60); `retrieval/pipeline.py` runs both retrievers, falls back to one if the other fails, and records each stage's results in `RAGState`.
- **Done when:** Unit tests check RRF math and the fallback; the eval shows fused vs dense vs BM25.
- **Depends on:** T3.1

### T3.3 — Cross-encoder reranker
- **Goal:** Rerank the fused candidates (§5.4).
- **Build:** `app/providers/cross_encoder_reranker.py` (sentence-transformers, device auto mps/cuda/cpu, loaded once as a singleton); keep the top-N (5); config flag to turn it off.
- **Done when:** The eval shows reranked metrics **and the reranker's latency**; you write down whether it's worth keeping.
- **Depends on:** T3.2

### T3.4 — Retrieval quality gate and abstention
- **Goal:** Don't generate without evidence (§5.5, §5.10).
- **Build:** `app/guardrails/retrieval.py`: a gate on the top reranker score and the number of chunks above the threshold; the standard abstention response; `scripts/calibrate_gate.py`, which sweeps thresholds over the eval set (focus on no-answer questions) and suggests a value.
- **Done when:** No-answer questions abstain in the eval; the calibration script outputs a precision/recall table.
- **Depends on:** T3.3

### T3.5 — Context selection
- **Goal:** A clean, compact context (§5.6).
- **Build:** `app/generation/context.py`: remove near-duplicates, merge adjacent chunks from the same section, order the chunks, enforce the token budget (fits `num_ctx`), keep citation metadata.
- **Done when:** Unit tests cover dedup, merge and budget trimming.
- **Depends on:** T3.3

### T3.6 — Experiment configs and comparison report
- **Goal:** Repeatable experiments (§9).
- **Build:** `experiments/*.yaml` (format from §9); the runner accepts `--experiment`, stamps all component versions on the run, and `compare` prints the differences between two runs.
- **Done when:** Two experiments (e.g. reranker on vs off) run and the comparison table prints.
- **Depends on:** T3.4, T3.5

---

## Phase 4 — Production Reliability

### T4.1 — Real authentication
- **Goal:** JWT-based auth (§10.2).
- **Build:** Register/login endpoints, password hashing (argon2/bcrypt), JWT access tokens; replace the dev auth dependency.
- **Done when:** Protected routes reject missing or invalid tokens; the isolation tests still pass.
- **Depends on:** T1.14

### T4.2 — Async ingestion worker
- **Goal:** Upload returns right away; a worker does the processing (§4.2, §4.7).
- **Build:** Redis + RQ (or Celery); `app/workers/ingestion_worker.py`; `ingestion_jobs` rows; retries with exponential backoff; a dead-letter queue; idempotent stages keyed on `document_id + stage + version`; `GET /documents/{id}` shows the status.
- **Done when:** A test forces an embedding failure and the retry resumes without re-parsing; the upload response time doesn't depend on PDF size.
- **Depends on:** T1.9, T0.3

### T4.3 — OCR fallback and sandboxed parsing
- **Goal:** Scanned PDFs and hostile PDFs are handled (§4.3, §10.2).
- **Build:** Detect pages with no text layer → OCRmyPDF/Tesseract; parse in a subprocess with a timeout and memory limit; basic malware/structure checks (e.g. reject JavaScript/embedded files, optional ClamAV).
- **Done when:** A scanned fixture gets text; a malformed PDF fails cleanly with `PARSING_FAILURE`.
- **Depends on:** T4.2

### T4.4 — Input guardrails
- **Goal:** Cheap checks before retrieval (§5.1).
- **Build:** `app/guardrails/input.py`: length limits, an injection-pattern list, an abuse keyword check; the result is stored in `guardrail_results`; an optional hook for `llama-guard3` behind a flag.
- **Done when:** Unit tests with a list of good and bad inputs pass; the prompt-injection eval items are recorded.
- **Depends on:** T1.14

### T4.5 — Output guardrails
- **Goal:** Check answers before they are returned (§5.9).
- **Build:** `app/guardrails/output.py`: a grounding check (lexical/embedding overlap between each cited sentence and its chunk), system-prompt leakage detection, a sensitive-data regex; on failure, regenerate once and otherwise return a controlled response.
- **Done when:** Unit tests: a leaked system prompt is blocked, and an unsupported claim is flagged.
- **Depends on:** T1.13

### T4.6 — Redis LLMGate
- **Goal:** At most 10 concurrent LLM calls across all processes (§10.1).
- **Build:** `app/core/llm_gate.py`: a Redis semaphore with leases (150 s), a priority queue (answer > rewrite > judge), judge calls capped at 2 slots and paused while users are waiting, 30 s max wait; wired into `OllamaLLM` so no call can skip it.
- **Done when:** The capacity tests from §11.7 pass: the 11th call waits and times out after 30 s; a killed holder's slot comes back when its lease expires; judge calls never exceed 2.
- **Depends on:** T1.11

### T4.7 — Active-user slots
- **Goal:** At most 15 active users, and 1 chat in flight per user (§10.1).
- **Build:** Session slots in Redis with a 600 s TTL that each request refreshes; release on logout; `503` + `Retry-After` for the 16th user; a per-user in-flight lock; admin and health routes don't count.
- **Done when:** Tests: the 16th user gets `503`; the TTL frees a slot; a second message sent at the same time is rejected.
- **Depends on:** T4.1

### T4.8 — Per-user rate limits
- **Goal:** Upload 10/h, chat 20/min, eval 10/h (§10.2).
- **Build:** A Redis sliding-window limiter as a FastAPI dependency; `429` with `Retry-After`.
- **Done when:** Tests pass for each limit.
- **Depends on:** T4.7

### T4.9 — Timeouts, retries, fallback model
- **Goal:** Controlled failures (§10.2).
- **Build:** A shared retry/timeout helper for Ollama, the reranker and DB calls (retry only idempotent operations); LLM failure path: retry → fallback model if configured → controlled error; the failure category is recorded.
- **Done when:** Tests with injected faults hit each path.
- **Depends on:** T4.6

### T4.10 — Streaming responses
- **Goal:** Stream generated tokens to the client (§10.2).
- **Build:** An SSE chat endpoint. Because citations need validation, stream the answer text, then send a final event with the validated citations (or a correction if validation fails).
- **Done when:** A curl/httpx test gets token events, then a final citations event.
- **Depends on:** T4.9, T4.5

### T4.11 — Load test
- **Goal:** Choose `OLLAMA_NUM_PARALLEL` from real numbers (§10.1, §11.7).
- **Build:** A Locust (or k6) script: 15 users asking questions. Record P95 latency, time to first token, per-stream tok/s and gate wait time for `OLLAMA_NUM_PARALLEL` ∈ {2, 4, 6, 10}.
- **Done when:** The results table is written to `docs/load_test.md` with the chosen setting.
- **Depends on:** T4.10, T4.7

---

## Phase 5 — Memory

### T5.1 — Conversations and short-term memory
- **Goal:** Conversation history (§6.1).
- **Build:** `conversations` and `messages` tables (cited chunk IDs per turn); `app/memory/short_term.py` loads the last 10 turns; `conversation_id` in the chat API.
- **Done when:** A test shows turns being saved and a bounded load.
- **Depends on:** T1.14

### T5.2 — Query rewriting
- **Goal:** Follow-up questions become standalone queries (§5.1).
- **Build:** `app/retrieval/rewrite.py` using the small rewrite model through the gate (rewrite priority); skipped on the first turn; the rewritten query is used for retrieval and the original for generation. Add follow-up items to the eval set.
- **Done when:** The eval shows better retrieval on follow-up items with the rewrite turned on than with it off.
- **Depends on:** T5.1, T4.6

### T5.3 — Long-term memory extraction
- **Goal:** Durable user preferences only (§6.2).
- **Build:** `memories` table; `app/memory/long_term.py`: candidate extraction (LLM) → durability check → safety check (no sensitive data, no document facts) → store, with category and source.
- **Done when:** Tests: "I prefer concise answers" is stored; a document fact and an email address are **not** stored.
- **Depends on:** T5.1

### T5.4 — Memory API and prompt injection of memory
- **Goal:** Users can see and control memory; memory is used safely.
- **Build:** `GET/DELETE /api/v1/memory` plus an audit log; memory is added to the prompt in a delimited, marked-untrusted block (§5.7).
- **Done when:** Tests: a deleted memory no longer affects prompts; a memory containing an instruction isn't followed (eval item).
- **Depends on:** T5.3

---

## Phase 6 — Observability

### T6.1 — OpenTelemetry tracing
- **Goal:** One span per stage (§8.1).
- **Build:** OTel SDK; spans for guardrails, rewrite, embed, dense, BM25, fusion, rerank, context, LLM, output guard; optional self-hosted Langfuse in Compose.
- **Done when:** One request shows the full span tree in the trace viewer.
- **Depends on:** T1.14

### T6.2 — Full trace record and inspection endpoint
- **Goal:** An engineer can inspect the whole chain for any request (§8.1).
- **Build:** Extend the trace row to the §8.1 JSON (stage results with scores, final context, prompt, raw output, validation results, versions); `GET /api/v1/monitoring/traces/{trace_id}` (admin only).
- **Done when:** The endpoint returns the full chain for a real request.
- **Depends on:** T6.1

### T6.3 — Compute and token tracking
- **Goal:** Know what each request costs in local compute (§8.2).
- **Build:** Record prompt/output tokens, prefill/decode time, tok/s and time to first token from the Ollama responses; **warn when the prompt tokens get close to `num_ctx`** (truncation detection, §11.4); embedding token counts.
- **Done when:** The trace contains the compute fields; a test fires the truncation warning.
- **Depends on:** T6.2

### T6.4 — Prometheus metrics and Grafana
- **Goal:** Dashboards (§8.2).
- **Build:** `/metrics` (prometheus-client): request/error counts, per-stage latency histograms, reranker score distribution, abstention rate, active users, LLM slots in use, gate wait time/timeouts. Prometheus + Grafana in Compose with a provisioned dashboard.
- **Done when:** The dashboard shows live data during a small load run.
- **Depends on:** T6.3, T4.7

### T6.5 — Failure taxonomy
- **Goal:** Every failure gets a category (§8.3).
- **Build:** An enum from §8.3; each stage's error handling sets `failure_category` on the trace; a metric per category.
- **Done when:** Tests with injected faults produce the right category for each stage.
- **Depends on:** T6.2

### T6.6 — User feedback
- **Goal:** 👍/👎 linked to traces (§7.6).
- **Build:** `feedback` table; `POST /api/v1/feedback` (trace_id, rating, reason, comment); `scripts/export_negative_feedback.py` → draft golden-dataset items.
- **Done when:** A test covers feedback storage and export.
- **Depends on:** T6.2

### T6.7 — Online evaluation worker
- **Goal:** Reference-free quality checks on a sample of traffic (§7.5).
- **Build:** Sample 5% of traces → queue → `app/workers/evaluation_worker.py` runs the judge at judge priority (≤ 2 slots, paused when users are waiting) → quality store → dashboard panels.
- **Done when:** Sampled traces get scores; a test shows the worker backs off while user requests are waiting.
- **Depends on:** T6.4, T4.6, T2.5

---

## Phase 7 — Production Hardening

### T7.1 — CI with regression gates
- **Goal:** Stop quality regressions (§7.7).
- **Build:** A GitHub Actions (or local `make ci`) pipeline: lint, unit and integration tests, retriever eval, deterministic E2E metrics; fail if Recall@5, faithfulness or citation correctness drop > 5% or P95 latency rises > 20% against the stored baseline run.
- **Done when:** A deliberately worse config makes CI fail.
- **Depends on:** T3.6

### T7.2 — Versioning and side-by-side re-indexing
- **Goal:** Change the chunker or embedding model safely (§9).
- **Build:** Version stamps on chunks and traces (pipeline, chunker, embedding, reranker, LLM + quantization + num_ctx, prompt, guardrails); a `reindex` command that builds a new index version next to the current one; switch over through config.
- **Done when:** Re-indexing with a new chunk size keeps the old index working until the switch.
- **Depends on:** T3.6

### T7.3 — Containerize the app and harden security
- **Goal:** A deployable stack (§11.8, §10.2).
- **Build:** Dockerfiles for the API and workers (HF cache as a volume); the app reaches Ollama via `host.docker.internal`; reverse proxy (Caddy/nginx); the Ollama port not exposed; secrets from env; dependency audit (pip-audit).
- **Done when:** `docker compose up` runs the full stack and a smoke test passes end to end.
- **Depends on:** T4.11

### T7.4 — Alerts and backups
- **Goal:** Operate it safely.
- **Build:** Grafana alerts (error rate, P95 latency, gate timeouts, ingestion DLQ size); a nightly `pg_dump` and MinIO backup script with a restore check.
- **Done when:** A test alert fires; a restore into a fresh DB passes the smoke test.
- **Depends on:** T7.3, T6.4
