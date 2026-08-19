from fastapi.testclient import TestClient

from app.main import app
import app.routes as routes

client = TestClient(app)


def test_support_ticket_accepts_description_and_returns_schema_fields(monkeypatch):
    async def fake_process(audio, image, description):
        return {
            "transcription": "Je veux un remboursement",
            "image_diagnostic": "une image de facture",
            "rag_rule": "Règle de remboursement",
            "ticket_status": "A verifier",
            "confidence": 0.8,
            "reasoning": "Analyse OK",
        }

    monkeypatch.setattr(routes.orchestrator, "process", fake_process)

    response = client.post(
        "/support-ticket",
        data={"description": "Je veux un remboursement"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["description_image"] == "une image de facture"
    assert payload["ticket_status"] == "A verifier"
    assert "image_diagnostic" not in payload
