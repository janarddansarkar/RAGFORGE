"""Minimal web UI: upload one or more PDFs, click Ingest, then ask questions about them (typed or spoken), as a chat that remembers the last few turns.

    uv run streamlit run src/app.py

It uses DocumentRAG from rag.py (the settings from config.toml and the sidebar), so the answer comes
with its citations and the retrieved + reranked chunks behind it.
"""

import hashlib
import tempfile
import uuid
from itertools import chain
from pathlib import Path

import streamlit as st

from chunking import chunk_body, chunk_note
from guardrails import InputBlocked
from models import api_key_for
from providers import (
    COMPONENTS,
    OFF,
    Component,
    models_of,
    needed_keys,
    providers_for,
    settings_for,
)
from rag import DocumentRAG, Report
from usage import ROLE_NAMES, UsageTracker, summarize
from voice import SpeechToText

st.set_page_config(
    page_title="RAGForge", page_icon="📄", initial_sidebar_state="expanded"
)
st.title("RAGForge")

# ---- 0. Bring your own key: pick a provider and a model for every component ----
# The keys live in this browser session's memory only: never saved to disk, never logged,
# and never shared with other sessions.
CUSTOM = "Custom..."


def choose(component: Component) -> tuple[str, str]:
    """A provider and a model for one component: (OFF, "") when it is switched off."""
    st.markdown(f"**{component.label}**")
    st.caption(component.help)
    names = providers_for(component.kind, component.also)
    if component.kind == "speech":
        names = [OFF, *names]  # voice input is optional
    default = names.index(component.provider)
    provider = st.selectbox(
        "Provider",
        names,
        default,
        key=f"provider_{component.key}",
        label_visibility="collapsed",
    )
    if provider == OFF:
        return OFF, ""
    models = list(models_of(component, provider))
    index = models.index(component.model) if component.model in models else 0
    model = st.selectbox(
        "Model",
        [*models, CUSTOM],
        index,
        key=f"model_{component.key}_{provider}",
        label_visibility="collapsed",
    )
    if model == CUSTOM:
        model = st.text_input("Model name", key=f"custom_{component.key}_{provider}")
    return provider, model.strip()


with st.sidebar:
    st.header("Models")
    choices = {component.key: choose(component) for component in COMPONENTS}
    st.divider()
    keys = {
        provider: st.text_input(
            f"Your {provider} API key", type="password", key=f"key_{provider}"
        )
        for provider in needed_keys(choices)
    }
    st.caption(
        "Your keys stay in this session's memory only. Your questions and document text "
        f"are sent to {', '.join(keys)}, and you pay for the calls. Nothing runs on this machine."
    )
    if not all(keys.values()) or not all(
        model for _, model in choices.values() if _ != OFF
    ):
        st.info("Choose a model for every component and enter your API key(s).")
        st.stop()
    settings = settings_for(choices, keys)
    # Counts every model call made with the user's keys, for the whole session.
    tracker = st.session_state.setdefault("usage", UsageTracker())
    settings["usage_tracker"] = tracker

# ---- 1. Upload + ingest ----
uploaded = st.file_uploader(
    "Upload PDFs",
    type=[suffix.removeprefix(".") for suffix in DocumentRAG.SUFFIXES],
    accept_multiple_files=True,
)

# A different set of uploads starts fresh: forget the old documents, answer and question.
# Changing the models does the same: the ingested documents were set up with the old ones.
files = (*(file.file_id for file in uploaded), repr(settings))
if files != st.session_state.get("files"):
    if st.session_state.get("rag"):
        st.session_state.rag.close()  # the old conversation's cache goes with it
    for key in (
        "rag",
        "ingested",
        "messages",
        "spoken",
        "heard",
        "triad",
        "stt",
        "ingest_calls",
    ):
        st.session_state.pop(key, None)
    st.session_state.fresh = st.session_state.get("fresh", 0) + 1  # new recorder, empty
    st.session_state.files = files

if uploaded and st.button("Ingest"):
    # DocumentRAG wants real file paths; keep the original names so citations read well.
    # A name uploaded twice is dropped (the first one wins).
    if st.session_state.get("rag"):
        st.session_state.rag.close()  # re-ingesting starts a new conversation
    with tempfile.TemporaryDirectory() as folder:  # DocumentRAG keeps its own copy
        paths = {}
        for file in uploaded:
            if file.name not in paths:
                paths[file.name] = Path(folder) / file.name
                paths[file.name].write_bytes(file.getvalue())
        ingest_start = tracker.mark()
        with st.spinner(
            "Preparing the vector store (a minute or so the first time)..."
        ):
            st.session_state.rag = DocumentRAG(list(paths.values()), settings=settings)
            st.session_state.ingest_calls = tracker.calls[ingest_start:]
            st.session_state.ingested = ", ".join(paths)
            st.session_state.messages = []
            st.session_state.conversation_id = uuid.uuid4().hex  # ties ratings together

rag = st.session_state.get("rag")
if rag is None:
    st.info("Upload one or more PDFs and click Ingest to begin.")
    st.stop()
st.success(f"Ready: {st.session_state.ingested}")
if st.session_state.get("ingest_calls"):
    st.caption(
        "Ingest cost: "
        + summarize(st.session_state.ingest_calls).describe()
        + " (chunk notes and embeddings; a document already indexed costs nothing)"
    )

# ---- 2. Ask (typed or spoken); the chat remembers the last few turns ----
turn_start = tracker.mark()  # everything the models do from here is this turn's cost
recording = (
    st.audio_input("Or ask by voice", key=f"voice_{st.session_state.get('fresh', 0)}")
    if rag.config["stt_model"]  # not every provider has speech-to-text
    else None
)
if recording:
    audio = recording.getvalue()
    digest = hashlib.sha1(audio).hexdigest()
    if digest != st.session_state.get("heard"):  # transcribe each recording once
        with st.spinner("Listening..."):
            if "stt" not in st.session_state:
                st.session_state.stt = SpeechToText(
                    rag.config["stt_model"],
                    api_key_for(rag.config, rag.config["stt_model"]),
                    rag.config.get(
                        "usage_tracker"
                    ),  # so voice input shows up in the cost
                )
            st.session_state.spoken = st.session_state.stt.transcribe(audio)
        st.session_state.heard = digest

with st.sidebar:
    st.header("RAG triad")
    st.caption(
        f"Every answer is scored by {rag.config['judge_model']}. "
        "The scores appear after the answer."
    )
if st.button("Clear chat"):
    rag.memory.clear()
    st.session_state.messages = []
messages = st.session_state.setdefault("messages", [])


def show_usage(calls: list, title: str) -> None:
    """The calls, tokens and cost of one turn, with a breakdown by component."""
    total = summarize(calls)
    st.caption(f"{title}: {total.describe()}")
    if total.by_role:
        with st.expander("Cost breakdown"):
            for role, part in total.by_role.items():
                st.markdown(f"- **{ROLE_NAMES.get(role, role)}:** {part.describe()}")


def score_turn(question: str, answer: str, chunks: list) -> dict:
    """The RAG-triad scores for one answer (the judge loads the first time)."""
    with st.spinner(
        "Scoring the answer (context relevance, groundedness, answer relevance)..."
    ):
        if "triad" not in st.session_state:
            from triad import TriadEvaluator

            st.session_state.triad = TriadEvaluator(config=rag.config)
        return st.session_state.triad.evaluate(question, answer, chunks)


# ---- 3. The conversation: answers with citations and the chunks behind them ----
def show_triad(triad: dict) -> None:
    """The three RAG-triad scores (0-1) with the judge's reasons."""
    columns = st.columns(len(triad))
    for column, (name, result) in zip(columns, triad.items()):
        score = result["score"]
        column.metric(name, "n/a" if score is None else f"{score:.2f}")
    with st.expander("Why these scores"):
        for name, result in triad.items():
            st.markdown(f"**{name}:** {result['reason']}")


# The retriever's own score for a chunk, from whichever search put it in the results.
RETRIEVER_SCORES = (  # (metadata key, what it is)
    ("rrf_score", "fused rank score, higher = better"),
    ("dense_distance", "vector distance, lower = closer"),
    ("bm25_score", "keyword score, higher = better"),
)


def score_line(chunk) -> str:
    """The retriever score, and the reranker score when a reranker ran."""
    parts = []
    for key, note in RETRIEVER_SCORES:
        if key in chunk.metadata:
            parts.append(f"retriever score {chunk.metadata[key]:.4f} ({note})")
            break
    if "rerank_score" in chunk.metadata:
        parts.append(
            f"reranker score {chunk.metadata['rerank_score']:.4f} (0-1, higher = better)"
        )
    return " | ".join(parts) or "no scores"


def retry_note(reasons: list[str]) -> str:
    return "Second try after your thumbs down" + (
        f" ({', '.join(reasons)})" if reasons else ""
    )


def show_feedback(message: dict, index: int) -> None:
    """Thumbs up / down on an answer, kept in the session cache next to the answer. A thumbs
    down asks what was wrong, and the same question then gets a fresh answer."""
    cache, entry = rag.generator.cache, message.get("cache_entry")
    if not cache or not entry:
        return  # without the session cache there is nowhere to keep it
    key = f"{st.session_state.conversation_id}_{index}"  # a new conversation starts unrated
    choice = st.feedback("thumbs", key=f"thumbs_{key}")
    if choice is None:
        return
    rating, reasons = ("up", []) if choice == 1 else ("down", [])
    if rating == "down":
        reasons = st.pills(
            "What was wrong?", cache.REASONS, selection_mode="multi", key=f"why_{key}"
        )
    saved = {"rating": rating, "reasons": list(reasons)}
    if entry.get("feedback") != saved:
        cache.rate(entry, rating, saved["reasons"])
    st.caption("Thanks, your feedback is saved for this session.")
    if rating == "down" and index == len(messages) - 1:  # only the latest answer
        label = "Try again" + (" with these reasons" if reasons else "")
        if st.button(label, key=f"retry_{key}"):
            st.session_state.retry = {"index": index, "reasons": saved["reasons"]}
            st.rerun()


def show_answer(message: dict, index: int, streamed: bool = False) -> None:
    """streamed=True: the answer text is already on screen and the caller scores it."""
    answer, chunks = message["text"], message["chunks"]
    if message.get("retry_reasons") is not None and not streamed:
        st.caption(retry_note(message["retry_reasons"]))
    if not streamed:
        st.write(answer)
    if message["query"] != previous_question:
        st.caption(f"Searched for: {message['query']}")
    if Report.refused(answer):
        st.write("**Citations:** none - the document does not answer this question.")
    else:
        st.write("**Citations:** " + ", ".join(Report.cited(answer, chunks)))
    if "triad" in message:
        show_triad(message["triad"])
        if message.get("usage_calls") and not streamed:
            show_usage(message["usage_calls"], "This turn")
    elif not streamed and st.button(
        "Score this answer (RAG triad)", key=f"score_{index}"
    ):
        scoring_start = tracker.mark()
        message["triad"] = score_turn(message["question"], answer, chunks)
        message["usage_calls"] = [
            *message.get("usage_calls", []),
            *tracker.calls[scoring_start:],
        ]
        st.rerun()
    with st.expander(f"Top {len(chunks)} chunks (retrieved, then reranked)"):
        for rank, chunk in enumerate(chunks, start=1):
            st.markdown(f"**[{rank}] {Report.label(chunk)}**")
            st.caption(score_line(chunk))
            st.write(chunk_body(chunk))
            if chunk_note(chunk):
                st.caption(
                    f"Context note (helps the search, not the answer): {chunk_note(chunk)}"
                )
    if not streamed:
        show_feedback(message, index)


previous_question = ""
for index, message in enumerate(messages):
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["text"])
            previous_question = message["text"]
        else:
            show_answer(message, index)


def answer_turn(question: str, stream, first: str, retry_reasons=None) -> None:
    """Show one answer as it streams, then its citations, scores, cost and thumbs.
    retry_reasons is None for a new question and the thumbs-down reasons for a second try."""
    global previous_question
    if retry_reasons is None:
        messages.append({"role": "user", "text": question})
        with st.chat_message("user"):
            st.write(question)
    with st.chat_message("assistant"):
        if retry_reasons is not None:
            st.caption(retry_note(retry_reasons))
        answer = st.write_stream(chain([first], stream)).strip()
        turn = {
            "role": "assistant",
            "question": question,
            "text": answer,
            "chunks": rag.last_chunks,
            "query": rag.last_query,
        }
        if retry_reasons is not None:
            turn["retry_reasons"] = retry_reasons
        messages.append(turn)
        previous_question = question
        show_answer(turn, len(messages) - 1, streamed=True)  # citations and chunks now
        entry = rag.generator.cache_entry  # the cache keeps the scores with the answer
        if entry and entry["triad"]:
            turn["triad"] = entry["triad"]
        else:
            turn["triad"] = score_turn(question, answer, turn["chunks"])
            if entry:
                entry["triad"] = turn["triad"]
                rag.generator.cache.save()
        show_triad(turn["triad"])
        turn["usage_calls"] = tracker.calls[turn_start:]
        show_usage(turn["usage_calls"], "This turn")
        turn["cache_entry"] = entry
        show_feedback(turn, len(messages) - 1)


retry = st.session_state.pop("retry", None)
question = st.chat_input("Ask a question about the documents") or st.session_state.pop(
    "spoken", None
)
if retry and retry["index"] == len(messages) - 1:
    # A thumbs down: answer the same question again, in the way the reasons ask for.
    previous = messages[retry["index"]]
    question = previous["question"]
    stream = rag.retry_stream(previous, retry["reasons"])
    with st.spinner("Trying again..."):
        first = next(stream, "")
    answer_turn(question, stream, first, retry["reasons"])
elif question and question.strip():
    question = question.strip()
    stream = rag.ask_stream(question)
    try:
        with st.spinner("Checking, retrieving and answering..."):
            first = next(stream, "")  # the guard, the search and the first words
    except InputBlocked as blocked:
        st.warning(str(blocked))
        question = None  # nothing was asked: no scoring
    else:
        answer_turn(question, stream, first)


# ---- 4. Sidebar: the triad scores, always in view ----
def average(scored: list[dict], name: str) -> float | None:
    values = [m["triad"][name]["score"] for m in scored]
    values = [v for v in values if v is not None]  # a failed judge call is left out
    return sum(values) / len(values) if values else None


def show_score(name: str, value: float | None) -> None:
    st.metric(name, "n/a" if value is None else f"{value:.2f}")


with st.sidebar:
    st.header("Your usage")
    session = summarize(tracker.calls)
    columns = st.columns(3)
    columns[0].metric("Calls", f"{session.calls:,}")
    columns[1].metric("Tokens", f"{session.input_tokens + session.output_tokens:,}")
    columns[2].metric("Cost", f"${session.cost:.4f}")
    st.caption(
        f"{session.input_tokens:,} tokens in, {session.output_tokens:,} out, this session"
        + (
            f". {session.unpriced} call(s) have no known price and are not in the cost."
            if session.unpriced
            else "."
        )
        + " Costs are estimates from LiteLLM's price list; your provider's bill is the truth."
    )
    with st.expander("By component"):
        for role, part in session.by_role.items():
            st.markdown(f"- **{ROLE_NAMES.get(role, role)}:** {part.describe()}")
    st.divider()
    scored = [m for m in messages if "triad" in m]
    if not scored:
        st.info("Scores appear here after the first answer.")
    else:
        latest = scored[-1]
        st.subheader("Latest answer")
        st.caption(latest["question"])
        for name, result in latest["triad"].items():
            show_score(name, result["score"])
        with st.expander("Why these scores"):
            for name, result in latest["triad"].items():
                st.markdown(f"**{name}:** {result['reason']}")
        if len(scored) > 1:
            st.subheader(f"Average of {len(scored)} answers")
            for name in latest["triad"]:
                show_score(name, average(scored, name))
        st.caption(
            "0 = bad, 1 = good. A small judge model is noisy: use it as a guide."
        )

    # The session's chat history: every question asked since the last Ingest, newest last.
    st.divider()
    st.subheader("Chat history")
    answered = [m for m in messages if m["role"] == "assistant"]
    if not answered:
        st.caption("No questions asked in this session yet.")
    for number, turn in enumerate(answered, start=1):
        with st.expander(f"{number}. {turn['question']}"):
            st.write(turn["text"])
            if "triad" in turn:
                st.caption(
                    " | ".join(
                        f"{name} {'n/a' if r['score'] is None else format(r['score'], '.2f')}"
                        for name, r in turn["triad"].items()
                    )
                )
