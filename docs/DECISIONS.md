# Decision log

The significant choices in RAGForge, in one place: what was decided, why, and what it cost. Where a decision was tested, the
evidence is in [EVALUATION.md](EVALUATION.md). "Branch" says where the decision shows up.

## Shape of the system

| Decision | Why | Trade-off | Branch |
|---|---|---|---|
| Build the evaluation first-class, not as an afterthought | The project's primary goal was learning to evaluate RAG properly; a system you cannot measure cannot be improved | Slower to a first demo | all |
| Keep the lab, the local product and the hosted product on separate branches | Each answers a different question and changes the models under the same pipeline | Changes are carried by hand, not merged | all |
| Plain, readable classes over heavy frameworks | The code must stay understandable months later | Some things are written by hand that a framework would give | all |
| Store vectors in a local folder; no database server, no containers | Small scale (tens of pages), simple to run; an early production design with Postgres and Docker was deferred | Not built for many concurrent users | all |
| One pipeline shape everywhere: guard, rewrite, cache, hybrid search, rerank, answer, score | The same flow lets results from the lab carry to both products | | all |

## Retrieval and answering

| Decision | Why | Trade-off | Branch |
|---|---|---|---|
| Hybrid search (vectors plus keywords) | Each alone misses things the other finds; combined they found more of the right pages | Two indexes to maintain | all |
| Always rerank | The biggest single retrieval gain: the right page was usually retrieved but not first | Extra latency and, when the reranker is an LLM, extra cost | all |
| Reranker is a purpose-built cross-encoder, not a chat model (local) | A small chat model used as reranker made retrieval worse than none; the cross-encoder improved it and was faster | Needs a one-time model download | `dev`, `offline-eval` |
| Reranker choice is the user's (hosted): an LLM scorer or Cohere's cross-encoder | Hosted LLM scoring works but costs about 30 calls per question; a cross-encoder is one call | Cohere trial keys are rate-limited | `ragforge_api` |
| Do not pay for a bigger reranker model | The larger model gained about 0.03 MRR at much higher cost | | `ragforge_api` |
| Score relevance by page | Lets different chunking strategies be compared fairly | Slightly generous scores | `offline-eval` |
| Rank 30 candidates, not 20 | A real wrong answer traced to the right passage never reaching the reranker; 30 fixed it and improved the whole test set | About 2 more seconds per query | `ragforge_api` |
| Answer the supported parts of a question and say what the documents do not cover | Early testing showed false refusals on two-part and follow-up questions | A slightly longer prompt | `ragforge_api` |
| Answers quote the original passage, not the AI-written context note | The note helps search but should never be what the user reads or what is cited | | `dev`, `ragforge_api` |
| Today's date is given to the model | "Present" and durations cannot be worked out otherwise | | `dev`, `ragforge_api` |
| Add an AI-written context note to every chunk | Helps single searches (about +0.04 MRR). But with a reranker, plain chunks scored slightly *better* (0.91 against 0.87), so the benefit is not clear-cut, and it adds a one-time indexing cost | Kept as the default; revisit if indexing cost matters | `dev`, `ragforge_api` |
| Multi-query retry search was not made the default | Gain within noise (MRR 0.871 against 0.866) for about 50% more latency | | `ragforge_api` |

## Evaluation method

| Decision | Why |
|---|---|
| Golden set of 45 questions with reference answers and pages | Gives something objective to measure against |
| Tune on two thirds, validate on one third | Guards against settings that only fit the tuning questions |
| Near-ties go to the faster candidate | With 45 questions, differences of a few hundredths are noise |
| Test the generator separately from retrieval ("oracle" mode) | Tells you whether to fix search or the answer model |
| A different model judges than answers | A model should not grade its own answers; the same answers scored 0.97 or 0.80 for correctness depending on the judge, so compare runs only under one judge |
| The RAG triad runs on every live answer | Needs no reference answers, so it works on any uploaded document |
| Save results and rebuild reports from the saved data | A full hosted run costs real money (about $2 and 100 minutes); the report costs nothing |

## Trust, safety and privacy

| Decision | Why | Branch |
|---|---|---|
| Input guard on every question | Screens for prompt injection and other malicious input before anything is retrieved | `dev`, `ragforge_api` |
| The model refuses rather than invents | A wrong confident answer is worse than "the documents do not say" | all |
| Bring your own key | The operator pays nothing and holds no secrets; the user sees their own spend | `ragforge_api` |
| API keys are held in session memory only | Never written to disk or logged | `ragforge_api` |
| Uploaded PDFs are deleted right after indexing; the index lives in a private per-session folder and is removed on replace, session end or restart | Nothing a user uploads should outlive their use of it | `ragforge_api` |
| 20-page and 25 MB limits, checked on the page and again before indexing | Protects a small server and the user's wallet; an oversized file never reaches a model | `ragforge_api` |
| Timeouts and retries on every model call | One stalled provider call must not freeze the page | `ragforge_api` |
| Ratings are stored with the session and deleted with it | No user data persists | `dev`, `ragforge_api` |

## Hosting

| Decision | Why | Trade-off |
|---|---|---|
| One gateway (LiteLLM) for all providers | A new provider is configuration, not code, and cost tracking comes with it | Depends on the gateway's price list; unpriced models are counted but not costed |
| No local models in the hosted version | One-command install, small server | Everything uploaded goes to the chosen provider |
| Anthropic offered for chat only | It has no embeddings or speech-to-text | Mixed-provider setups are common (for example Claude for answers, OpenAI for embeddings) |
| Plain server, systemd and a Caddy proxy; no Docker | Matches the small scale; the fewest moving parts | No automatic scaling |
| No login or rate limit built in | Out of scope for the project so far | If public, put one in front |
| Costs are shown as estimates | Token counts times a price list are not a bill | Users should check their provider's dashboard |

## What was deliberately not done

- No cloud database, containers or background job queue. Designed on paper, deferred as unnecessary at this scale.
- No long-term user memory or accounts. Only short-term conversation memory.
- No measured quality for providers other than OpenAI on the full test set.
- No standalone image upload; only PDFs (scanned pages are read with OCR).
