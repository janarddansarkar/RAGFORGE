"""Voice: turn speech into a question (faster-whisper) and an answer into speech (macOS `say`).

    uv run python src/voice.py "What rank r does LoRA use?"   # speaks it, then transcribes it back

faster-whisper is Whisper running on CPU (no PyTorch): free, and offline once the model has
downloaded the first time (stt_model in config.toml; "small" is about 500 MB).
"""

import io
import re
import shutil
import subprocess
import sys
import tempfile
from functools import cached_property
from pathlib import Path

from vectorstore import load_config


class SpeechToText:
    def __init__(self, model: str):
        self.model = model

    @cached_property
    def recognizer(self):
        from faster_whisper import WhisperModel

        return WhisperModel(self.model, device="cpu", compute_type="int8")

    def transcribe(self, audio: bytes) -> str:
        """The words in the audio (WAV, MP3, ...)."""
        segments, _ = self.recognizer.transcribe(io.BytesIO(audio), beam_size=1)
        return " ".join(segment.text.strip() for segment in segments).strip()


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
    print("Heard:", SpeechToText(load_config()["stt_model"]).transcribe(audio))
