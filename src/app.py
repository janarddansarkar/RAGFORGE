"""Minimal web UI: upload one or more PDFs, click Ingest, then ask questions about them (typed or spoken).

    uv run streamlit run src/app.py

It uses DocumentRAG from rag.py (the best settings from best_config.json), so the answer comes
with its citations and the retrieved + reranked chunks behind it.
"""

import hashlib
import tempfile
from pathlib import Path

import streamlit as st

from rag import REFUSAL, DocumentRAG, Report
from vectorstore import load_config
from voice import SpeechToText, TextToSpeech

st.set_page_config(page_title="RAGForge", page_icon="📄")
st.title("RAGForge")

# ---- 1. Upload + ingest ----
uploaded = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

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
        st.session_state.pop("result", None)

rag = st.session_state.get("rag")
if rag is None:
    st.info("Upload one or more PDFs and click Ingest to begin.")
    st.stop()
st.success(f"Ready: {st.session_state.ingested}")

# ---- 2. Ask (typed or spoken) ----
recording = st.audio_input("Or ask by voice")
if recording:
    audio = recording.getvalue()
    digest = hashlib.sha1(audio).hexdigest()
    if digest != st.session_state.get("heard"):  # transcribe each recording once
        with st.spinner("Listening..."):
            if "stt" not in st.session_state:
                st.session_state.stt = SpeechToText(load_config()["stt_model"])
            st.session_state.question = st.session_state.stt.transcribe(audio)
        st.session_state.heard = digest

with st.form("ask"):
    question = st.text_input("Ask a question about the document", key="question")
    asked = st.form_submit_button("Ask")

if asked and question.strip():
    with st.spinner("Retrieving and answering..."):
        st.session_state.result = (question.strip(), *rag.ask(question.strip()))

# ---- 3. Answer, citations and chunks ----
if "result" in st.session_state:
    question, answer, chunks = st.session_state.result
    st.subheader("Answer")
    st.write(answer)
    if st.checkbox("Read the answer aloud"):
        st.audio(TextToSpeech().speak(answer), autoplay=True)

    st.subheader("Citations")
    if REFUSAL in answer.lower().replace("’", "'"):
        st.write("None - the document does not answer this question.")
    else:
        for label in dict.fromkeys(Report.label(c) for c in chunks):
            st.write(f"- {label}")

    st.subheader(f"Top {len(chunks)} chunks (retrieved, then reranked)")
    for rank, chunk in enumerate(chunks, start=1):
        score = chunk.metadata.get("rerank_score")
        score_text = f" - rerank score {score:.4f}" if score is not None else ""
        with st.expander(f"[{rank}] {Report.label(chunk)}{score_text}"):
            st.write(chunk.page_content)
