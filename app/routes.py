from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
from app.models.schemas import TicketResponse
from app.core.orchestrator import Orchestrator

router = APIRouter(tags=["Support Tickets"])
orchestrator = Orchestrator()


@router.post("/test/rag")
async def test_rag(question: str):
    """
    Teste uniquement le service RAG.
    """
    try:
        from app.services.rag_service import RAGService
        
        service = RAGService()
        results = service.query(question, top_k=3)
        
        return {
            "success": True,
            "question": question,
            "results": results
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/test/diagnostic")
async def test_diagnostic(
    transcription: Optional[str] = None,
    description_image: Optional[str] = None,
    rag_rule: Optional[str] = None
):
    """
    Teste uniquement le service diagnostic (LLM Groq).
    """
    try:
        from app.services.diagnostic_service import DiagnosticService
        
        service = DiagnosticService()
        result = service.diagnostiquer(
            transcription=transcription,
            description_image=description_image,
            rag_rule=rag_rule
        )
        
        return {
            "success": True,
            "diagnostic": result
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
