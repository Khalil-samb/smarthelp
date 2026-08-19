"""
Service de diagnostic - Utilise un LLM (Groq) pour determiner le statut final.
"""

import json
import re
from groq import Groq
from app.config import settings
from typing import Optional


class DiagnosticService:
    _instance = None
    _client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        if not settings.GROQ_API_KEY:
            print("ERREUR: GROQ_API_KEY non definie dans .env")
            self._client = None
            return

        try:
            self._client = Groq(api_key=settings.GROQ_API_KEY)
        except Exception as e:
            print(f"ERREUR: impossible d'initialiser le client Groq : {e}")
            self._client = None

    @staticmethod
    def _normalize_status(value: Optional[str]) -> str:
        if value is None:
            return "needs_review"

        normalized = str(value).strip().lower()
        mapping = {
            "accepted": "accepted",
            "approve": "accepted",
            "approved": "accepted",
            "reimbursable": "accepted",
            "rejected": "rejected",
            "refuse": "rejected",
            "refused": "rejected",
            "a verifier": "needs_review",
            "needs_review": "needs_review",
            "needs review": "needs_review",
            "review": "needs_review",
            "pending": "needs_review",
            "verify": "needs_review",
            "vérifier": "needs_review",
        }
        return mapping.get(normalized, "needs_review")

    def diagnostiquer(self, transcription: str, description_image: str, rag_rule: str) -> dict:
        # CAS 1 : Client non initialisé (clé ajoutée ici)
        if not self._client:
            return {
                "ticket_status": "needs_review",
                "confidence": 0.3,
                "reasoning": "GROQ_API_KEY non definie",
                "description_image": description_image
            }

        prompt = self._construire_prompt(transcription, description_image, rag_rule)

        try:
            response = self._client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Tu es un assistant. Retourne UNIQUEMENT un JSON valide avec les champs : "
                            "status (Remboursable, A verifier, Refuse), "
                            "confidence (nombre entre 0 et 1), "
                            "reasoning (explication)."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )

            content = response.choices[0].message.content.strip()
            if not content:
                raise ValueError("Reponse vide")
            
            data = self._parse_response_content(content)
            
            return {
                "ticket_status": self._normalize_status(data.get("status", "needs_review")),
                "confidence": float(data.get("confidence", 0.3)),
                "reasoning": data.get("reasoning", "Pas de justification fournie."),
                "description_image": description_image
            }

        except Exception as e:
            # CAS 2 : Passe description_image au fallback
            return self._fallback(str(e), description_image)   

    def _parse_response_content(self, content: str) -> dict:
        """Parse la réponse JSON du LLM avec fallback."""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            if json_match := re.search(r"\{.*\}", content, re.DOTALL):
                try:
                    return json.loads(json_match[0])  # ← Changé ici
                except json.JSONDecodeError:
                    pass
            raise ValueError(f"Reponse non JSON: {content[:100]}")

    def _construire_prompt(self, transcription: str, description_image: str, rag_rule: str) -> str:
        prompt = "Analyse la reclamation client suivante :\n\n"

        if transcription:
            prompt += f"Message vocal : {transcription}\n\n"
        if description_image:
            prompt += f"Description de l'image : {description_image}\n\n"
        if rag_rule and rag_rule not in ("Pas de texte a analyser", "Aucune regle trouvee"):
            prompt += f"Regle interne : {rag_rule}\n\n"

        prompt += "Retourne le diagnostic sous forme de JSON."
        return prompt


   # Fallback mis à jour pour toujours inclure description_image
    def _fallback(self, error: str, description_image: Optional[str] = None) -> dict:
        return {
            "ticket_status": "needs_review",
            "confidence": 0.3,
            "reasoning": f"Erreur: {error}",
            "description_image": description_image
        }
        
        