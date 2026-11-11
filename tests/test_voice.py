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


def test_transcribe_sends_the_audio_and_the_key_to_the_speech_api(monkeypatch):
    from types import SimpleNamespace

    import litellm

    from voice import SpeechToText

    sent = {}
    monkeypatch.setattr(
        litellm,
        "transcription",
        lambda **kwargs: sent.update(kwargs) or SimpleNamespace(text=" What rank? "),
    )
    assert (
        SpeechToText("openai/whisper-1", "sk-me").transcribe(b"audio") == "What rank?"
    )
    assert sent["model"] == "openai/whisper-1" and sent["api_key"] == "sk-me"
