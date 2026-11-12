"""Voice: the audio and the key go to the speech API."""


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
