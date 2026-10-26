"""Minimal web UI: upload one or more PDFs, click Ingest, then ask questions about them (typed or spoken), as a chat that remembers the last few turns.

    uv run streamlit run src/app.py

It uses DocumentRAG from rag.py (the best settings from best_config.json), so the answer comes
with its citations and the retrieved + reranked chunks behind it.
"""

import hashlib
import sys
import tempfile
from pathlib import Path

import streamlit as st

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
    "Upload PDFs or images",
    type=[suffix.removeprefix(".") for suffix in DocumentRAG.SUFFIXES],
    accept_multiple_files=True,
)

# A different set of uploads starts fresh: forget the old documents, answer and question.
files = tuple(file.file_id for file in uploaded)
if files != st.session_state.get("files"):
    for key in ("rag", "ingested", "messages", "spoken", "heard"):
        st.session_state.pop(key, None)
    st.session_state.fresh = st.session_state.get("fresh", 0) + 1  # new recorder, empty
    st.session_state.files = files

if uploaded and st.button("Ingest"):
    # DocumentRAG wants real file paths; keep the original names so citations read well.
    # A name uploaded twice is dropped (the first one wins).
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
    st.info("Upload one or more PDFs or images and click Ingest to begin.")
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
    score_answers = st.checkbox(
        "Score each new answer",
        help="deepeval + a local judge; adds about a minute per answer",
    )
    st.caption(
        f"Judge: {load_config()['judge_model']} (an Ollama model, runs on this machine)."
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


question = st.chat_input("Ask a question about the documents") or st.session_state.pop(
    "spoken", None
)
if question and question.strip():
    messages.append({"role": "user", "text": question.strip()})
    with st.spinner("Retrieving and answering..."):
        answer, chunks = rag.ask(question.strip())
    turn = {
        "role": "assistant",
        "question": question.strip(),
        "text": answer,
        "chunks": chunks,
        "query": rag.last_query,
    }
    if score_answers:
        turn["triad"] = score_turn(question.strip(), answer, chunks)
    messages.append(turn)


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


def show_answer(message: dict, index: int) -> None:
    answer, chunks = message["text"], message["chunks"]
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
    elif st.button("Score this answer (RAG triad)", key=f"score_{index}"):
        message["triad"] = score_turn(message["question"], answer, chunks)
        st.rerun()
    with st.expander(f"Top {len(chunks)} chunks (retrieved, then reranked)"):
        for rank, chunk in enumerate(chunks, start=1):
            score = chunk.metadata.get("rerank_score")
            score_text = f" - rerank score {score:.4f}" if score is not None else ""
            st.markdown(f"**[{rank}] {Report.label(chunk)}{score_text}**")
            st.write(chunk.page_content)


previous_question = ""
for index, message in enumerate(messages):
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["text"])
            previous_question = message["text"]
        else:
            show_answer(message, index)

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
        st.info(
            "No scored answers yet. Tick the box above for new answers, "
            "or click 'Score this answer' under an answer."
        )
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
