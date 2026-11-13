# The branches

Four branches, each with one job. Work flows in one direction: experiments on `offline-eval`, the winners into `dev`, the
hosted rebuild on `ragforge_api`. They are **not merged** into each other. They split from a shared history early on and
changes are carried across by hand, because each branch changes the models underneath the same pipeline and a merge
would keep fighting over those lines.

```
 offline-eval ──▶ dev ──▶ ragforge_api            main = these docs
   lab          local        hosted
```

---

## `offline-eval`: the evaluation lab

**Role.** The place where RAGForge's components were measured against each other. It is where the question "what settings
should the product use?" was answered with evidence rather than taste.

**What it provides**
- A **golden dataset**: 45 questions over three papers (LoRA, QLoRA, "Attention Is All You Need"), each with a reference
  answer and the pages that contain it.
- **Evaluation scripts** for each component: the retriever, the reranker, chunking strategies and the answer generator.
- **A settings search** (`find_best`) that tries candidates stage by stage (chunking, embedding model, retrieval method,
  reranker, chunk size, answer model), keeps each stage's winner for the next, and writes the result to a config file.
- The core pipeline, so every experiment ran on the real thing.

**Features not here (on purpose).** No session cache, input guard, feedback buttons or timing tools. The lab needed the
pipeline, not the product around it.

**Decisions made here**
- *Judge relevance by page, not by chunk.* Chunk IDs change whenever chunking changes, so scoring on pages is the only way to
  compare chunking strategies fairly. The cost: scores are slightly generous.
- *Tune on two thirds, check on one third.* The search ranks candidates on 30 of the 45 questions and then compares its
  pick with the current settings on the other 15. If the pick loses, the current settings stay. This guards against
  picking a winner that only fits the questions it was tuned on.
- *Treat near-ties as ties.* With 45 questions, one question moves a score by about 0.02. A candidate within 0.01 of the
  best counts as tied, and the faster one wins.
- *Test the generator on its own ("oracle" mode).* The answer model is handed the correct pages directly, so a model is
  not blamed for retrieval misses.
- *Keep the judge model different from the answering model*, so a model never grades its own work.
- *Everything local.* Embeddings, reranker, answer model and judge all ran on Ollama or Hugging Face models, so the whole
  lab cost nothing to run.

**How it helps.** It turned "I think hybrid search with a reranker is better" into numbers, and produced the
configuration that `dev` ships. It is also the reference for how to evaluate a RAG system: see
[EVALUATION.md](EVALUATION.md).

**State.** Frozen. Its last commit is from late October; later evaluation work moved to `dev` and `ragforge_api`.

---

## `dev`: the local, free product

**Role.** The day-to-day development branch and the version anyone can run for free. Everything runs on your own machine
through Ollama: no API keys, no per-call cost, works offline once the models are downloaded, and your documents never
leave the computer.

**What it provides**
- The full question-answering app (web UI plus a command-line mode): upload PDFs, ingest, ask by typing or by voice.
- The pipeline: scan-aware PDF reading, chunking with an AI-written context note per chunk, hybrid search, a cross-encoder
  reranker, a cited answer, then live RAG-triad scoring.
- The product features around it: input guard, follow-up rewriting, session cache, thumbs up/down with a reason-based retry,
  today's date supplied to the model (so "present" and durations work out), and speech input.
- One-command setup scripts that install and start Ollama if needed and download exactly the models the settings name.
- The results of the lab, applied: the winning configuration is stored in a config file that overrides the defaults.

**Decisions made here**
- *Local first.* Free, private and offline were worth more than the strongest possible answers. The price is hardware: the
  default configuration wants roughly 24 GB of RAM and about 14 GB of downloads, with a smaller option for modest machines.
- *Separate models for separate jobs.* A larger model writes answers; a smaller one acts as input guard and judge. Using a
  different model to judge keeps the scores honest.
- *A cross-encoder reranker over an LLM reranker.* In testing, asking a small chat model to score each chunk made retrieval
  *worse* than no reranker at all; a purpose-built cross-encoder improved it and was faster. Bigger was not better either.
- *Context notes on each chunk.* An AI-written sentence of context per chunk helps search find it. This was kept on `dev`
  (and later re-tested on the hosted models; see `ragforge_api`).
- *Answers use the original passage, not the AI's note,* and a thumbs-down retry reuses the same evidence with a
  different instruction depending on the reason (wrong citation, made something up, incomplete, too long).
- *Feedback lives and dies with the conversation.* Ratings sit in the session cache and are deleted with it, so nothing
  about a user persists.
- *Simple storage.* A local vector store in a folder; no database server, no containers. An early design for a production
  stack (Postgres, Docker) was deliberately deferred.

**How it helps.** It is the proof that the whole system works with no spend, the safe place to try changes, and the
baseline the hosted version was built from. If you want to keep documents private, this is the branch.

**State.** Stable and feature-complete. Last worked on in early November.

---

## `ragforge_api`: the hosted, bring-your-own-key product

**Role.** The version meant for other people. Nothing runs on the host except the app itself: every model call goes to a
hosted provider using the *user's own key*, so the operator carries no model cost.

(Until this documentation took over, this code lived on `main`; `ragforge_api` is its continuation, two commits further
on. The last code on `main` is the commit just before the cleanup.)

**What it provides** (everything in `dev`, rebuilt on hosted models, plus)
- **Provider and model choice per component.** In the sidebar the user picks OpenAI, Anthropic or Gemini for answers, input
  guard, judge, reranker, embeddings and voice, and pastes a key per provider. Keys stay in the session's memory and are
  never written to disk or logged. A Cohere cross-encoder is an optional reranker.
- **Usage and cost tracking.** Every model call made with the user's key is counted. Each answer shows its calls, tokens and
  estimated cost broken down by component, and the sidebar shows the session total and the cost of ingesting.
- **Hardening for a public server.** A 20-page total limit per user, 25 MB per file, request timeouts and retries so one
  stalled provider cannot freeze the page, and private per-session uploads that are deleted right after indexing.
- **Deployment guide** for a plain server with HTTPS in front, no containers.
- **A re-run of the evaluation** on the hosted models, with results saved so they never need to be paid for again.

**Decisions made here**
- *Bring your own key.* The operator pays nothing and holds no secrets. The user sees exactly what they spend. The trade-off
  is that every user needs an account with a provider, and everything they upload goes to that provider.
- *One gateway for every provider* (LiteLLM). Adding a provider is a configuration change rather than new code, and cost
  accounting comes for free.
- *No local models at all.* Removing Ollama entirely kept the install to a single command (no model downloads) and the
  server small (about 2 GB of RAM is comfortable). `dev` keeps the local story.
- *Uploads are private and short-lived.* PDFs are deleted as soon as they are indexed. The index and cache live in a private
  per-session folder that is removed when the documents are replaced, the session ends, or the server restarts. A
  limit of 20 pages is enforced on the page *and* again before indexing, so an oversized file never reaches a model.
- *Refuse rarely, and only when warranted.* Early testing showed two false refusals (a two-part question and a follow-up).
  The answer prompt was changed to answer the parts the documents support and say plainly which part they don't.
- *Widen the candidate pool.* One wrong answer (a vague question about a "rank") traced to the right passage never reaching
  the reranker. Raising candidates from 20 to 30 fixed it and improved retrieval on the whole test set, at the cost of
  more rerank work per question.
- *Handle model quirks explicitly.* Claude models "think" before answering and the thinking counts against the token limit;
  the app asks for minimal thinking and adds headroom, otherwise some answers came back empty.
- *Costs are estimates and the app says so.*

**How it helps.** It is the shareable product: it can be put on a small server, used by anyone with an API key, and it
keeps the user informed about quality and cost on every answer.

**State.** Current and the most advanced branch. Known gaps: no login or rate limit, quality measured on OpenAI only, and
HTTPS is the operator's job (see [SETUP.md](SETUP.md)).

---

## `main`: this documentation

Code was removed from `main` so that it can serve as a stable front door: it says what the project is, where each part
lives and how to run it, without being tied to any one version of the implementation. The code that used to be here is
preserved on `ragforge_api` (and in `main`'s history).

## Comparing the three code branches

| | `offline-eval` | `dev` | `ragforge_api` |
|---|---|---|---|
| Purpose | Find the best settings | Free, private product | Shareable hosted product |
| Models | Local | Local | Hosted, user's own key |
| Costs the user | Nothing | Nothing | API usage (shown per answer) |
| Documents leave the machine | No | No | Yes, to the chosen provider |
| Web app | Basic | Full | Full, with provider choice and cost |
| Input guard, cache, feedback | No | Yes | Yes |
| Evaluation | Full search and component evals | Component evals, triad, timing | Component evals, triad, timing on hosted models |
| Hardware needed | Large | Large (about 24 GB RAM by default) | Small server |
| Typical reader | Someone studying the method | Someone running it privately | Someone deploying it for others |
