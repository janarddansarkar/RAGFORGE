# Setup

`main` contains no code, so you always clone one of the code branches. Repository:
`https://github.com/janarddansarkar/RAGFORGE`.

Common to all branches:
- Install [uv](https://docs.astral.sh/uv/), which installs the right Python and the packages for you.
- **Scanned PDFs** need Tesseract (`brew install tesseract` on macOS, `apt install tesseract-ocr` on Debian/Ubuntu). Without it,
  text PDFs still work.
- The web app opens at `http://localhost:8501`.

---

## Hosted version (`ragforge_api`)

The right choice if you want light hardware, the strongest models, or to host it for others.

**You need:** an API key from at least one provider. The defaults are all OpenAI, so one OpenAI key is enough to start.
Anthropic can answer but cannot embed, so pair it with OpenAI or Gemini for embeddings. A Cohere key is optional (for its
reranker).

```bash
git clone -b ragforge_api https://github.com/janarddansarkar/RAGFORGE.git && cd RAGFORGE
uv run streamlit run src/app.py
```

The first run installs the packages. No model is downloaded.

**Using it**
1. In the sidebar, pick a provider and model for each component (answers, input guard, judge, reranker, embeddings and
   optional voice), then paste a key for each provider you chose. Keys stay in the session's memory only.
2. Upload PDFs (up to 20 pages in total, 25 MB per file) and click **Ingest**. The first ingest is the slow one because every
   chunk gets a short AI-written context note.
3. Ask questions. Each answer shows citations, the passages behind it, quality scores, and the cost of that turn.

**Things to know**
- Everything you ask and upload goes to the providers you chose, and you pay them for it. A typical question is under a cent
  on OpenAI.
- Changing a provider or model resets the documents; click Ingest again.
- Cost figures are estimates.
- The command line can score one question: set the provider's key, then run `uv run python src/triad.py your.pdf "your question"`.

### Deploying it on a server

Written for Ubuntu or Debian, no Docker. It was tested locally with a real server and browser, not on a VPS.

| Item | Guidance |
|---|---|
| Memory | About 2 GB is comfortable (around 560 MB for one user; the scoring library adds about 160 MB). 1 GB works only with swap. |
| Disk | About 1 GB for the Python packages. Uploads are not kept. |
| Needed | `git`, `curl`, `tesseract-ocr`, and a domain name so the page can be served over HTTPS. |

Why HTTPS is not optional: users type their API keys into the page, so never serve it over plain HTTP.

1. **Install and run.** Install the packages above and uv, clone the branch, and start the app listening only on the
   server itself (`127.0.0.1`), so it is not reachable from outside directly.
2. **Keep it running.** Run it as a systemd service with `Restart=always` and `PrivateTmp=true`. The private temp folder is
   deleted when the service stops, which is a second guarantee that no upload survives a restart.
3. **Put HTTPS in front.** A reverse proxy such as Caddy gets and renews a certificate by itself and passes the WebSocket that
   Streamlit needs. Open only SSH, 80 and 443 in the firewall; keep 8501 closed.
4. **Update** with `git pull` and a service restart.

The complete commands and the service file are in `docs/DEPLOY_VPS.md` on the `ragforge_api` branch.

**What the server keeps:** nothing a user uploads. PDFs are deleted right after indexing; the index and cache sit in a private
per-session folder that is removed when documents are replaced, when the session ends, and on every restart. API keys live
only in session memory.

**What it does not protect against:** there is no login or rate limit. Costs fall on each user's own key, but anyone who can
reach the page can use your server's CPU. If it is not meant to be public, put it behind a login (Caddy `basic_auth`), a VPN
or an IP allow-list.

---

## Local, free version (`dev`)

The right choice for privacy (documents never leave your machine), zero cost, or offline use.

**You need:** [Ollama](https://ollama.com) and a reasonably strong computer. The default configuration downloads about
14 GB of models and wants about 24 GB of RAM. On a smaller machine, switch the answer model to the small option (`gemma3:4b`,
roughly 8 GB of models and 16 GB of RAM) in the two config files, then run setup again.

```bash
git clone -b dev https://github.com/janarddansarkar/RAGFORGE.git && cd RAGFORGE
./setup.sh     # offers to install and start Ollama if needed, installs packages, downloads the models (skips ones you have)
./run.sh       # starts the web app
```

The first question also downloads the reranker (about 1 GB) from Hugging Face. After that the whole thing works offline.

A command-line mode is also available: `uv run python src/rag.py documents/LoRA.pdf "What rank r does LoRA use?"`. Run
`uv run python src/rag.py --help` for options such as the number of passages and the retrieval method.

**What the default models do**
| Job | Model |
|---|---|
| Writes answers and the second try | `qwen3:14b` |
| Input guard and quality judge | `mistral:7b` (kept different from the answer model on purpose) |
| Embeddings | `all-minilm` |
| Reranker | `mxbai-rerank-base-v1` cross-encoder |
| Speech to text | faster-whisper on the CPU |

---

## Evaluation lab (`offline-eval`)

For re-running the experiments that chose the settings. Same Ollama requirement as `dev`.

```bash
git clone -b offline-eval https://github.com/janarddansarkar/RAGFORGE.git && cd RAGFORGE
ollama serve                         # if Ollama is not already running
ollama pull bge-m3 qwen2.5:3b        # embeddings and answers (see config.toml)
ollama pull mistral:7b               # the judge
uv sync
uv run python eval/find_best.py --limit 5 --no-save   # quick smoke test on 5 questions, writes nothing
```

The full search takes hours. Useful switches:

| Switch | Effect |
|---|---|
| `--stages chunk_size` | Run only some stages; the others keep today's settings |
| `--no-save` | Print everything, write nothing |
| `--limit 5` | Use only 5 golden questions (a smoke test) |
| `--skip-missing` | Leave out candidate models you have not downloaded |

Before it starts, the script checks that every model it needs is downloaded and prints the exact commands for any that are not.

This branch's own README predates the later changes, and some saved reports in it were measured with other models. Re-run
the search to measure your setup.

---

## Running the evaluation on the hosted version

The hosted branch has its own runner that tests everything against one provider's models and saves the numbers. It costs
real money (about $1.3 for the stages plus about $0.7 once to build the index on OpenAI, and roughly 100 minutes), so read
the saved results first ([EVALUATION_RESULTS.md](EVALUATION_RESULTS.md)). A cheaper trial on 10 questions is available. The
report is rebuilt from the saved data at no cost. A free-tier Gemini key (5 requests a minute) cannot complete the evaluation.

---

## Checking that it works

On any code branch, unit tests need no models or keys: `uv run pytest`. Lint with `uv run ruff check .`.
