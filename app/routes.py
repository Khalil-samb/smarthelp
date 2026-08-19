from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
from app.models.schemas import TicketResponse
from app.core.orchestrator import Orchestrator

router = APIRouter(tags=["Support Tickets"])
orchestrator = Orchestrator()


@router.post("/support-ticket", response_model=TicketResponse)
async def create_support_ticket(
    audio: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    description: Optional[str] = Form(None)
):
    if description is not None:
        description = description.strip() or None

    if not audio and not image and not description:
        raise HTTPException(400, detail="Au moins un fichier ou une description est requis")

    try:
        result = await orchestrator.process(audio, image, description)

        return TicketResponse(
            transcription=result.get("transcription"),
            description_image=result.get("description_image"),
            rag_rule=result.get("rag_rule"),
            ticket_status=result.get("ticket_status", "needs_review"),
            confidence=float(result.get("confidence", 0.0)),
            reasoning=result.get("reasoning")
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Erreur interne : {str(e)}")


@router.post("/test/audio")
async def test_audio(audio: UploadFile = File(...)):
    """
    Teste uniquement le service audio (Whisper).
    """
    try:
        from app.utils.validators import read_file_within_limit, validate_audio_file
        from app.services.audio_service import AudioService
        
        audio_bytes = await read_file_within_limit(audio)
        validate_audio_file(audio, audio_bytes)
        
        service = AudioService()
        transcription = service.transcribe(audio_bytes, audio.filename)
        
        return {
            "success": True,
            "filename": audio.filename,
            "transcription": transcription
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/test/image")
async def test_image(image: UploadFile = File(...)):
    """
    Teste uniquement le service vision (description d'image).
    """
    try:
        from app.utils.validators import read_file_within_limit, validate_image_file
        from app.services.vision_service import VisionService
        
        image_bytes = await read_file_within_limit(image)
        validate_image_file(image, image_bytes)
        
        service = VisionService()
        description = service.analyser(image_bytes)
        
        return {
            "success": True,
            "filename": image.filename,
            "description": description
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


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