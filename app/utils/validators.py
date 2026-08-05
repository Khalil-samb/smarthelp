import os
import io
from fastapi import UploadFile, HTTPException
from PIL import Image
from app.config import settings

# ============================
# CONSTANTES
# ============================

ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav"}
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

ALLOWED_AUDIO_MIME = {"audio/mpeg", "audio/wav", "audio/wave", "audio/x-wav"}
ALLOWED_IMAGE_MIME = {"image/png", "image/jpeg", "image/jpg", "image/bmp", "image/webp"}


# ============================
# FONCTIONS DE VALIDATION
# ============================

async def read_file_within_limit(file: UploadFile) -> bytes:
    """Lit le fichier en mémoire avec limite de taille."""
    max_bytes = settings.MAX_FILE_SIZE_BYTES
    contents = await file.read(max_bytes + 1)
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Fichier trop lourd (max {settings.MAX_FILE_SIZE_MB} Mo)"
        )
    return contents


def validate_audio_file(file: UploadFile, contents: bytes) -> None:
    """
    Valide un fichier audio sur 2 niveaux :
    1. Extension (.mp3 ou .wav)
    2. Type MIME (audio/mpeg, audio/wav, etc.)
    """
    # 1. Vérifier l'extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Extension audio non supportée. Utilisez: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}"
        )

    # 2. Vérifier le MIME
    if file.content_type is None or file.content_type not in ALLOWED_AUDIO_MIME:
        raise HTTPException(
            status_code=415,
            detail="Type MIME audio non supporté ou manquant."
        )


def validate_image_file(file: UploadFile, contents: bytes) -> None:
    """
    Valide une image sur 3 niveaux :
    1. Extension (.png, .jpg, etc.)
    2. Type MIME (image/png, image/jpeg, etc.)
    3. Contenu réel (via PIL)
    """
    # 1. Vérifier l'extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Extension image non supportée. Utilisez: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}"
        )

    # 2. Vérifier le MIME
    if file.content_type is None or file.content_type not in ALLOWED_IMAGE_MIME:
        raise HTTPException(
            status_code=415,
            detail="Type MIME image non supporté ou manquant."
        )

    # 3. Vérifier le contenu réel avec PIL
    try:
        Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(
            status_code=415,
            detail="Le contenu du fichier n'est pas une image valide."
        )