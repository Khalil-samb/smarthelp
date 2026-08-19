from fastapi import FastAPI
from app.config import settings  # ← Modifié
from app.routes import router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title=settings.APP_NAME,  # ← Modifié
    debug=settings.DEBUG      # ← Modifié
)
# le cors pour que le navigateur ne bloque pas les requetes
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:4200'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/")
async def root():
    return {"message": "Support Ticket Assistant API"}

@app.get("/health")
async def health():
    return {"status": "healthy"}