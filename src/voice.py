"""Voice: turn speech into a question (Whisper) and an answer into speech (macOS `say`).

    uv run python src/voice.py "What rank r does LoRA use?"   # speaks it, then transcribes it back

Whisper is a Hugging Face model that runs inside this process: free, and offline once downloaded.
"""

import re
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
        import torch
        from transformers import pipeline

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        return pipeline("automatic-speech-recognition", model=self.model, device=device)

    def transcribe(self, audio: bytes) -> str:
        """The words in the audio (WAV, MP3, ...; ffmpeg decodes it)."""
        return str(self.recognizer(audio)["text"]).strip()


class TextToSpeech:
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
