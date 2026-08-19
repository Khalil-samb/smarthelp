import os
import subprocess
import torch
from transformers import pipeline
from app.config import settings


class AudioService:
    _instance = None
    _pipe = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Charge le modèle Whisper une seule fois (Singleton)."""
        device = 0 if torch.cuda.is_available() and settings.INFERENCE_DEVICE == "cuda" else -1

        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=settings.WHISPER_MODEL,
            device=device
        )

    def _convert_webm_to_wav(self, source_path: str) -> str:
        """Convertit un fichier WebM/ogg en WAV lisible par Whisper."""
        wav_path = os.path.splitext(source_path)[0] + ".wav"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            source_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            wav_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return wav_path

    def transcribe(self, audio_bytes: bytes, filename: str = "audio.webm") -> str:
        """Transcrit un fichier audio en texte, en convertissant WebM/ogg si nécessaire."""
        os.makedirs("uploads", exist_ok=True)

        audio_path = os.path.join("uploads", filename or "audio.webm")
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        working_path = audio_path
        try:
            lower_name = audio_path.lower()
            if lower_name.endswith((".webm", ".ogg", ".opus")):
                working_path = self._convert_webm_to_wav(audio_path)

            result = self._pipe(working_path, return_timestamps=True)
            text = result.get("text", "") if isinstance(result, dict) else str(result)
            return text.strip()
        finally:
            for path in {audio_path, working_path}:
                if path and os.path.exists(path):
                    os.remove(path)
