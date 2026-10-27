"""Minimal web UI: upload one or more PDFs, click Ingest, then ask questions about them (typed or spoken), as a chat that remembers the last few turns.

    uv run streamlit run src/app.py

It uses DocumentRAG from rag.py (the best settings from best_config.json), so the answer comes
with its citations and the retrieved + reranked chunks behind it.
"""

import hashlib
import sys
import tempfile
from itertools import chain
from pathlib import Path

import streamlit as st

from guardrails import InputBlocked
from rag import REFUSAL, DocumentRAG, Report
from vectorstore import PROJECT_ROOT, load_config
from voice import SpeechToText, TextToSpeech

sys.path.insert(0, str(PROJECT_ROOT))  # so the eval/ package can be imported
st.set_page_config(
    page_title="RAGForge", page_icon="📄", initial_sidebar_state="expanded"
)
st.title("RAGForge")

# ---- 1. Upload + ingest ----
uploaded = st.file_uploader(
    "Upload PDFs",
    type=[suffix.removeprefix(".") for suffix in DocumentRAG.SUFFIXES],
    accept_multiple_files=True,
)

# A different set of uploads starts fresh: forget the old documents, answer and question.
files = tuple(file.file_id for file in uploaded)
if files != st.session_state.get("files"):
    if st.session_state.get("rag"):
        st.session_state.rag.close()  # the old conversation's cache goes with it
    for key in ("rag", "ingested", "messages", "spoken", "heard"):
        st.session_state.pop(key, None)
    st.session_state.fresh = st.session_state.get("fresh", 0) + 1  # new recorder, empty
    st.session_state.files = files

if uploaded and st.button("Ingest"):
    # DocumentRAG wants real file paths; keep the original names so citations read well.
    # A name uploaded twice is dropped (the first one wins).
    if st.session_state.get("rag"):
        st.session_state.rag.close()  # re-ingesting starts a new conversation
    folder = Path(tempfile.mkdtemp())
    paths = {}
    for file in uploaded:
        if file.name not in paths:
            paths[file.name] = folder / file.name
            paths[file.name].write_bytes(file.getvalue())
    with st.spinner("Preparing the vector store (a minute or so the first time)..."):
        st.session_state.rag = DocumentRAG(list(paths.values()))
        st.session_state.ingested = ", ".join(paths)
        st.session_state.messages = []

rag = st.session_state.get("rag")
if rag is None:
    st.info("Upload one or more PDFs and click Ingest to begin.")
    st.stop()
st.success(f"Ready: {st.session_state.ingested}")

# ---- 2. Ask (typed or spoken); the chat remembers the last few turns ----
recording = st.audio_input(
    "Or ask by voice", key=f"voice_{st.session_state.get('fresh', 0)}"
)
if recording:
    audio = recording.getvalue()
    digest = hashlib.sha1(audio).hexdigest()
    if digest != st.session_state.get("heard"):  # transcribe each recording once
        with st.spinner("Listening..."):
            if "stt" not in st.session_state:
                st.session_state.stt = SpeechToText(load_config()["stt_model"])
            st.session_state.spoken = st.session_state.stt.transcribe(audio)
        st.session_state.heard = digest

read_aloud = st.checkbox("Read answers aloud")
with st.sidebar:
    st.header("RAG triad")
    st.caption(
        f"Every answer is scored by {load_config()['judge_model']} (an Ollama model, "
        "runs on this machine). The scores appear about a minute after the answer."
    )
if st.button("Clear chat"):
    rag.memory.clear()
    st.session_state.messages = []
messages = st.session_state.setdefault("messages", [])


def score_turn(question: str, answer: str, chunks: list) -> dict:
    """The RAG-triad scores for one answer (the judge loads the first time)."""
    with st.spinner(
        "Scoring the answer (context relevance, groundedness, answer relevance)..."
    ):
        if "triad" not in st.session_state:
            from eval.rag_triad_eval import TriadEvaluator

            st.session_state.triad = TriadEvaluator()
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


def show_answer(message: dict, index: int, streamed: bool = False) -> None:
    """streamed=True: the answer text is already on screen and the caller scores it."""
    answer, chunks = message["text"], message["chunks"]
    if not streamed:
        st.write(answer)
    if message["query"] != previous_question:
        st.caption(f"Searched for: {message['query']}")
    if REFUSAL in answer.lower().replace("\u2019", "'"):
        st.write("**Citations:** none - the document does not answer this question.")
    else:
        labels = dict.fromkeys(Report.label(c) for c in chunks)
        st.write("**Citations:** " + ", ".join(labels))
    if "triad" in message:
        show_triad(message["triad"])
    elif not streamed and st.button(
        "Score this answer (RAG triad)", key=f"score_{index}"
    ):
        message["triad"] = score_turn(message["question"], answer, chunks)
        st.rerun()
    with st.expander(f"Top {len(chunks)} chunks (retrieved, then reranked)"):
        for rank, chunk in enumerate(chunks, start=1):
            st.markdown(f"**[{rank}] {Report.label(chunk)}**")
            st.caption(score_line(chunk))
            st.write(chunk.page_content)


previous_question = ""
for index, message in enumerate(messages):
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["text"])
            previous_question = message["text"]
        else:
            show_answer(message, index)

question = st.chat_input("Ask a question about the documents") or st.session_state.pop(
    "spoken", None
)
if question and question.strip():
    question = question.strip()
    stream = rag.ask_stream(question)
    try:
        with st.spinner("Checking, retrieving and answering..."):
            first = next(stream, "")  # the guard, the search and the first words
    except InputBlocked as blocked:
        st.warning(str(blocked))
        question = None  # nothing was asked: no scoring, no reading aloud
    else:
        messages.append({"role": "user", "text": question})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            answer = st.write_stream(chain([first], stream)).strip()
            turn = {
                "role": "assistant",
                "question": question,
                "text": answer,
                "chunks": rag.last_chunks,
                "query": rag.last_query,
            }
            messages.append(turn)
            previous_question = question
            show_answer(
                turn, len(messages) - 1, streamed=True
            )  # citations and chunks now
            entry = (
                rag.generator.cache_entry
            )  # the cache keeps the scores with the answer
            if entry and entry["triad"]:
                turn["triad"] = entry["triad"]
            else:
                turn["triad"] = score_turn(question, answer, turn["chunks"])
                if entry:
                    entry["triad"] = turn["triad"]
                    rag.generator.cache.save()
            show_triad(turn["triad"])

if read_aloud and question and messages and messages[-1]["role"] == "assistant":
    st.audio(TextToSpeech().speak(messages[-1]["text"]), autoplay=True)


# ---- 4. Sidebar: the triad scores, always in view ----
def average(scored: list[dict], name: str) -> float | None:
    values = [m["triad"][name]["score"] for m in scored]
    values = [v for v in values if v is not None]  # a failed judge call is left out
    return sum(values) / len(values) if values else None


def show_score(name: str, value: float | None) -> None:
    st.metric(name, "n/a" if value is None else f"{value:.2f}")


with st.sidebar:
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
            "0 = bad, 1 = good. A small local judge is noisy: use it as a guide."
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
