"""
Orchestrateur du pipeline de traitement des réclamations.
"""

from typing import Optional
from fastapi import UploadFile, HTTPException
from app.services.audio_service import AudioService
from app.services.vision_service import VisionService
from app.services.rag_service import RAGService
from app.services.diagnostic_service import DiagnosticService
from app.utils.validators import (
    validate_audio_file,
    validate_image_file,
    read_file_within_limit
)


class Orchestrator:
    _instance = None
    _audio_service = None
    _vision_service = None
    _rag_service = None
    _diagnostic_service = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialise tous les services."""
        self._audio_service = AudioService()
        self._vision_service = VisionService()
        self._rag_service = RAGService()
        self._diagnostic_service = DiagnosticService()

    async def process(
        self,
        audio: Optional[UploadFile],
        image: Optional[UploadFile],
        description: Optional[str]
    ) -> dict:
        """
        Exécute le pipeline complet.
        """
        transcription = None
        image_diagnostic = None
        rag_rule = None

        # === 1. TRAITEMENT AUDIO ===
        if audio:
            audio_bytes = await read_file_within_limit(audio)
            validate_audio_file(audio, audio_bytes)
            transcription = self._audio_service.transcribe(audio_bytes, audio.filename)

        # === 2. TRAITEMENT IMAGE ===
        if image:
            image_bytes = await read_file_within_limit(image)
            validate_image_file(image, image_bytes)
            image_diagnostic = self._vision_service.analyser(image_bytes)

        # === 3. RAG ===
        if transcription or description:
            query_text = description or transcription
            rag_results = self._rag_service.query(query_text, top_k=3)
            rag_rule = "\n\n---\n".join(rag_results) if rag_results else "Aucune règle trouvée"
        else:
            rag_rule = "Pas de texte à analyser"

        # === 4. DIAGNOSTIC FINAL ===
        diagnostic = self._diagnostic_service.diagnostiquer(
            transcription=transcription,
            description_image=image_diagnostic,  # ← Changé : description_image
            rag_rule=rag_rule
        )

        return {
            "transcription": transcription,
            "image_diagnostic": image_diagnostic,
            "rag_rule": rag_rule,
            "ticket_status": diagnostic["ticket_status"],
            "confidence": diagnostic["confidence"],
            "reasoning": diagnostic["reasoning"]
        }