# RAGForge

RAGForge answers questions about PDF documents and shows its work: every answer comes with page citations, the passages it
was written from, a live quality score, and (in the hosted version) what the answer cost.

This branch, `main`, holds **no code**. It is the project's front page: what was built, how the work is split across
branches, why each major decision was made, and how to run each version. The code lives on the branches below.

## The project in one paragraph

RAGForge started as a learning project with a clear goal: build a retrieval-augmented generation (RAG) system *and* learn
how to evaluate one properly. That produced three kinds of work that are deliberately kept apart: a **lab** where
components were measured against a hand-built test set, a **local, free product** that runs entirely on the developer's
machine, and a **hosted, bring-your-own-key product** that other people can use with their own API keys. Each lives on its
own branch.

## Branch map

| Branch | What it is | Models | Status | Read this if you want to... |
|---|---|---|---|---|
| **`ragforge_api`** | The hosted, bring-your-own-key (BYOK) product, ready to deploy on a small server | Hosted (OpenAI, Anthropic, Gemini, Cohere) via LiteLLM | **Current, most advanced** | run or deploy RAGForge for other people |
| **`dev`** | The local, free product | Local ([Ollama](https://ollama.com)), no keys, works offline | Stable, feature-complete | run it privately at zero cost, or tinker |
| **`offline-eval`** | The evaluation lab that found the best settings | Local (Ollama) | Frozen (last work late October) | see how components were compared and why the defaults were chosen |
| **`main`** | This documentation | none | Docs only | understand the whole project |

How they relate:

```
 offline-eval ──(winning settings and code copied by hand)──▶ dev ──(rebuilt on hosted models)──▶ ragforge_api
   the lab                                                 local product                            hosted product
                                                                                                      (main held this
                                                                                                       code until it
                                                                                                       became docs-only)
```

Details per branch are in [docs/BRANCHES.md](docs/BRANCHES.md).

## Which branch should I use?

| I want to... | Use |
|---|---|
| Try RAGForge on my own documents without paying per call, or keep documents on my machine | `dev` |
| Host it for others, or use the strongest hosted models | `ragforge_api` |
| Understand how the settings were chosen, or re-run the experiments | `offline-eval` (and [docs/EVALUATION.md](docs/EVALUATION.md)) |
| Understand the reasoning behind the project | [docs/DECISIONS.md](docs/DECISIONS.md) |

## What every version does

- **Ask questions about uploaded PDFs**, including scanned pages (read with OCR). Answers cite pages.
- **Hybrid search**: meaning-based (vector) search and keyword (BM25) search, combined, then a **reranker** that re-sorts the
  best candidates before an answer model writes the response.
- **Show the evidence**: the retrieved passages and their scores sit next to each answer.
- **Score itself live** with the *RAG triad*: is the retrieved text on topic, is the answer backed by it, does it answer the
  question. No reference answers needed, so it works on any document.
- **Protect itself**: an input guard screens questions for prompt injection, and the answer prompt is told to refuse
  rather than invent.
- **Remember the conversation** just enough to handle follow-up questions, and answer repeated questions from a private
  session cache.
- **Learn from feedback**: thumbs up/down on every answer; a thumbs down asks what was wrong and offers a better second try.

The hosted version adds usage and cost tracking, a choice of provider and model per component, upload limits and
privacy handling for a public server.

## Documentation

| Document | What it covers |
|---|---|
| [docs/BRANCHES.md](docs/BRANCHES.md) | Each branch in depth: role, features, decisions, how it helps |
| [docs/DECISIONS.md](docs/DECISIONS.md) | The decision log across the project, with the reasoning and trade-offs |
| [docs/SETUP.md](docs/SETUP.md) | How to set up and run each branch, and how to deploy the hosted one |
| [docs/EVALUATION.md](docs/EVALUATION.md) | How quality was measured and what the numbers say, in plain language |
| [docs/RAG_EVALUATION.md](docs/RAG_EVALUATION.md) | Detailed evaluation write-up from the hosted version (reference) |
| [docs/EVALUATION_RESULTS.md](docs/EVALUATION_RESULTS.md) | Full tables of the saved OpenAI evaluation run (reference) |

The two reference documents were written next to the code and mention file names; read them with the
`ragforge_api` branch checked out if you want to follow along.

## Known limits (stated up front)

- The evaluation set is small: 45 questions over three research papers. Differences of a few hundredths between two setups
  are noise.
- The hosted version's quality was measured with OpenAI only. Gemini, Cohere and Anthropic were exercised end to end but not
  scored on the full test set.
- Costs shown in the hosted version are estimates from a price list; the provider's bill is the truth.
- The hosted version has no login or rate limit; if it is public, put one in front of it.
