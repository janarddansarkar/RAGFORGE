"""Voice: turn speech into a question with a hosted speech-to-text API.

stt_model in config.toml is a LiteLLM name such as openai/whisper-1 (needs that provider's key).
Providers without speech-to-text leave it empty, and the app hides voice input.
"""


class SpeechToText:
    def __init__(self, model: str, api_key: str | None = None, tracker=None):
        self.model = model
        self.api_key = api_key
        self.tracker = tracker  # counts the calls (see usage.py)

    def transcribe(self, audio: bytes) -> str:
        """The words in the audio (WAV, MP3, ...)."""
        import litellm

        reply = litellm.transcription(
            model=self.model, file=("audio.wav", audio), api_key=self.api_key
        )
        if self.tracker:
            self.tracker.record("voice", self.model, reply, "transcription")
        return reply.text.strip()
