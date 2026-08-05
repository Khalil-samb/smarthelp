import os
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

    def transcribe(self, audio_bytes: bytes, filename: str = "audio.mp3") -> str:
        """Transcrit un fichier audio en texte."""
        # Créer le dossier uploads s'il n'existe pas
        os.makedirs("uploads", exist_ok=True)
        
        # Sauvegarder le fichier temporairement
        audio_path = f"uploads/{filename}"
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)
        
        try:
            # Transcrire
            result = self._pipe(audio_path,
            return_timestamps = True)
            return result["text"]
        finally:
            # Nettoyer
            if os.path.exists(audio_path):
                os.remove(audio_path)