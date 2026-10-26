"""Voice helpers that need no model: what gets read aloud."""

import shutil

import pytest

from voice import TextToSpeech


def test_spoken_drops_citations_and_markdown():
    text = "LoRA uses **rank r = 4** [LoRA p.4] and `r = 8` [LoRA p.10, QLoRA p.2]."
    assert TextToSpeech.spoken(text) == "LoRA uses rank r = 4 and r = 8 ."


@pytest.mark.skipif(shutil.which("say") is None, reason="needs macOS `say`")
def test_speak_returns_wav_audio():
    assert TextToSpeech().speak("hello there")[:4] == b"RIFF"
