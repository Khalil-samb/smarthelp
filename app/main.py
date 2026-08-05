from fastapi import FastAPI
from app.config import settings  # ← Modifié
from app.routes import router

app = FastAPI(
    title=settings.APP_NAME,  # ← Modifié
    debug=settings.DEBUG      # ← Modifié
)

app.include_router(router)

@app.get("/")
async def root():
    return {"message": "Support Ticket Assistant API"}

@app.get("/health")
async def health():
    return {"status": "healthy"}