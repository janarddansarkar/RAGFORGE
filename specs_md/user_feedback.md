# User feedback (thumbs up / down) and the retry

How the thumbs on an answer work, what they change, and where the code is. Written for future reference.
Branch: `user_feedback` (built from `dev`). Not ported to `main` yet.

## The idea in one paragraph

Under every answer the user can press thumbs up or thumbs down. A thumbs down asks **what was wrong** and offers
**Try again**, which answers the same question a second time in a way that depends on the reason. Everything
is kept **for one conversation only**: the ratings live in the session cache's own JSON file, and a new upload
(or a new Ingest) deletes that file, so the next conversation starts with no ratings and no cache.

## What the user sees

```
answer + citations + chunks + triad scores
[ thumbs up ] [ thumbs down ]
   (after a thumbs down)
   What was wrong?  [Wrong or missing source] [Wrong citation] [Made something up] [Incomplete] [Too long]
   [ Try again with these reasons ]        <- only on the latest answer
```

A retried answer appears as a new assistant message headed "Second try after your thumbs down (reasons)",
with its own citations, chunks, triad scores and thumbs. The rejected answer stays visible above it.

## What is stored, and where

The session cache (`src/cache.py`) keeps one entry per answered question in a JSON file inside its own
temporary folder (`ragforge_cache_*/cache.json`). The thumbs go into the same entry:

```json
{
  "question": "What rank r does LoRA use?",
  "answer": "...",
  "chunks": [ { "text": "...", "metadata": { "source": "LoRA.pdf", "page": 10, "chunk_id": "..." } } ],
  "triad": { "Groundedness": { "score": 0.8, "reason": "..." } },
  "feedback": { "rating": "down", "reasons": ["Wrong citation", "Incomplete"] }
}
```

- `feedback` is `null` until the answer is rated; rating again replaces it.
- Vectors are not written to the file (they stay in memory).
- No database and no separate feedback file. The file is deleted by `SessionCache.destroy()`, which the app
  calls when the uploaded files change, when Ingest is pressed again, and when the program exits.
- If `use_cache = false` in `config.toml` there is no cache, so the thumbs are hidden.

## What each thumb does

| action | effect |
|---|---|
| thumbs up | `feedback = {"rating": "up"}` is saved. The answer stays in the cache, so the same (or a very similar) question is answered from it instantly: no search, no model call. |
| thumbs down | `feedback = {"rating": "down", "reasons": [...]}` is saved. The answer **stays in the file but is never reused** (`SessionCache.find` skips disliked entries), so asking the same question again runs the full pipeline. |
| Try again | Runs the retry below. The new answer is added to the cache as its own entry (unrated), and it replaces the rejected answer in the chat memory. |

Rating the same answer up after a thumbs down makes it reusable again.

## The retry: what changes depending on the reason

Code: `Generator.retry_stream` in `src/generator.py` (called through `DocumentRAG.retry_stream`).

| reason | search again? | what changes for the answer |
|---|---|---|
| Wrong or missing source | **yes, wider** | new chunks (see below) |
| Wrong citation | **yes, wider** | note: cite each statement with the label of the passage that states it |
| Made something up | no, same chunks | note: use only facts stated in the passages; leave out any detail that is not there |
| Incomplete | no, same chunks | note: cover every part of the question; the reply may be twice as long (`max_new_tokens` x 2) |
| Too long | no, same chunks | note: answer in at most two sentences; the reply limit is cut to a third (`max_new_tokens` / 3), because a small model can ignore the note |
| none chosen | yes, wider | only the "do not repeat" instruction |

Every retry also shows the model the rejected answer ("Do not repeat it") and puts the extra notes **before** the
question in the prompt. The chat history in the prompt leaves out the rejected turn, so it appears only once.

**The wider search** (`Retriever.retrieve_wider` in `src/retriever.py`): the question is also searched in other
words (the multi-query method) and twice as many candidates (`fetch_k` x 2) go to the reranker; `top_k + 2` chunks
come back. `fetch_k` is restored right afterwards.

Retrieval and generation, in short: a "source" reason redoes **retrieval and generation**; an "answer" reason redoes
**generation only**. The retry skips the input guard (the question already passed it) and the cache lookup.
Only the latest answer has the Try again button, so the chat memory stays consistent
(`ConversationMemory.replace_last_answer`).

## A fix found while testing: repeated questions were rewritten into answers

The chat memory rewrites follow-ups into standalone questions (`ConversationMemory.standalone`). When the same
question was asked again, the 4B model once turned it into the text of the previous answer
(`r = 4 [LoRA p.26], [LoRA p.10]`), which broke both the cache lookup and the search. Now a question asked before
word for word is not rewritten at all, and a rewrite that contains a citation like `[LoRA p.10]` is rejected.

## Files

| file | role |
|---|---|
| `src/cache.py` | `SessionCache.rate()` stores the thumbs, `disliked()`, `REASONS`, `find()` skips disliked entries |
| `src/generator.py` | `retry_stream`, `RETRY_NOTES`, `RETRIEVAL_REASONS`, `build_prompt(..., extra)` |
| `src/retriever.py` | `retrieve_wider` |
| `src/rag.py` | `DocumentRAG.retry_stream` |
| `src/memory.py` | `replace_last_answer`, `history(skip_last)`, the repeated-question and citation guards |
| `src/app.py` | `show_feedback` (thumbs, reason chips, Try again), `answer_turn`, the `retry` hand-off in `st.session_state` |
| `tests/test_cache.py`, `tests/test_retry.py`, `tests/test_memory.py` | unit tests for all of the above |

## Limits

- **Session only, by design.** Nothing survives a new upload, so the ratings cannot be used to tune the system later.
  Keeping them would need a store that outlives the session (and, on a hosted app, the users' consent).
- A repeated question answered from the cache shares the earlier entry, so it has one rating per cached answer,
  not one per displayed message.
- Only the latest answer can be retried.
- A small local model (`gemma3:4b`) may ignore the extra notes; the token limit is the only hard control.
- Cloud use (`main`) would add API cost for the retry: a wider search is about 20 more reranker calls if the
  reranker is an LLM (none with the cross-encoder) plus the rewrite and the answer.

## How it was tested

- Unit tests with fake models (`pytest`): the cache file contents, a disliked answer not being reused, the retry's
  prompt and settings per reason, the wider search, the memory guards.
- End to end through the real Streamlit page (`streamlit.testing`) with the real models and index: thumbs down plus
  reasons were stored, the question was then answered afresh with a different answer, "Wrong or missing source" ran a
  wider search (7 chunks, new pages), "Too long" reused the chunks, and the cache file was deleted on a new upload.
  The RAG-triad judge was stubbed in the replay runs to keep them fast.

## Ideas for later

- Keep ratings beyond the session (with consent) to build a golden set and tune the retriever.
- Per-chunk thumbs in the "Top chunks" list, for direct relevance labels.
- Retry more than once, or retry an older answer.
- Port to `main`: the retry cost should appear in the per-turn cost, and the same cache and UI changes apply.
