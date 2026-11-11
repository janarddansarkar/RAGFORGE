"""Voice: turn speech into a question (a hosted speech-to-text API) and an answer into speech (macOS `say`).

    uv run python src/voice.py "What rank r does LoRA use?"   # speaks it, then transcribes it back

stt_model in config.toml is a LiteLLM name such as openai/whisper-1 (needs that provider's key).
Providers without speech-to-text leave it empty, and the app hides voice input.
"""

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from models import api_key_for
from vectorstore import load_config


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


class TextToSpeech:
    @staticmethod
    def available() -> bool:
        """Reading aloud needs macOS's `say`."""
        return shutil.which("say") is not None

    @staticmethod
    def spoken(text: str) -> str:
        """The answer without what a person wouldn't say aloud: [LoRA p.4] tags and markdown."""
        text = re.sub(r"\[[^\]]*\bp\.\d+[^\]]*\]", "", text)
        text = re.sub(r"[*_`#>]", "", text)
        return " ".join(text.split())

    def speak(self, text: str) -> bytes:
        """WAV audio of the text (macOS only)."""
        with tempfile.TemporaryDirectory() as folder:
            wav = Path(folder) / "speech.wav"
            subprocess.run(
                ["say", "-o", str(wav), "--data-format=LEI16@22050", self.spoken(text)],
                check=True,
            )
            return wav.read_bytes()


# TRY IT ---- python src/voice.py "your sentence"
if __name__ == "__main__":
    sentence = " ".join(sys.argv[1:]) or "What rank r does LoRA use?"
    audio = TextToSpeech().speak(sentence)
    print(f"Spoke {len(audio)} bytes of audio")
    config = load_config()
    print(
        "Heard:",
        SpeechToText(
            config["stt_model"], api_key_for(config, config["stt_model"])
        ).transcribe(audio),
    )
