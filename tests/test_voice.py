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


def test_transcribe_joins_the_segments_with_a_fake_recognizer():
    from types import SimpleNamespace

    from voice import SpeechToText

    speech = SpeechToText("small")
    fake = SimpleNamespace(
        transcribe=lambda audio, beam_size: (
            [
                SimpleNamespace(text=" What rank "),
                SimpleNamespace(text="does LoRA use? "),
            ],
            None,
        )
    )
    speech.__dict__["recognizer"] = fake  # replaces the cached model
    assert speech.transcribe(b"audio") == "What rank does LoRA use?"
