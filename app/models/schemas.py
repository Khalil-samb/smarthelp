from pydantic import BaseModel
from typing import Optional

class TicketResponse(BaseModel):
    transcription: Optional[str] = None
    description_image: Optional[str] = None 
    rag_rule: Optional[str] = None
    ticket_status: str
    confidence: float
    reasoning: Optional[str] = None    
class ErrorResponse(BaseModel):
    detail: str
    status_code: int