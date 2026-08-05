"""
Service de vision - Description et détection de défauts.
"""

import io
import torch
from PIL import Image
from transformers import pipeline
from app.config import settings


class VisionService:
    _instance = None
    _pipe = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        device = 0 if torch.cuda.is_available() and settings.INFERENCE_DEVICE == "cuda" else -1
        
        self._pipe = pipeline(
            "image-to-text",
            model="microsoft/git-base-coco",
            device=device
        )

    def analyser(self, image_bytes: bytes) -> str:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            result = self._pipe(image, max_new_tokens=50)
            description = result[0]["generated_text"]
            
            # Détection de défauts par mots-clés
            mots_defauts = ["broken", "crack", "scratch", "damaged", "defect", "chip", "stain", "rust"]
            a_defaut = any(mot in description.lower() for mot in mots_defauts)
            
            status = "avec défaut" if a_defaut else "conforme"
            return f"Description : {description} | Statut : Produit {status}"
        
        except Exception as e:
            raise Exception(f"Erreur lors de l'analyse : {str(e)}")