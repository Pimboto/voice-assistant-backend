# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from src.config.settings import settings
from src.core.stt_engine import SimpleSTTEngine
from src.api import stt_routes

# Crear motor STT global
stt_engine = SimpleSTTEngine(
    model_name=settings.model_name,
    device=settings.device
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializar recursos al arrancar"""
    # Startup
    print("🚀 Iniciando servidor...")
    await stt_engine.initialize()
    
    # Asignar motor a las rutas
    stt_routes.stt_engine = stt_engine
    
    yield
    
    # Shutdown
    print("👋 Cerrando servidor...")

# Crear app
app = FastAPI(
    title="Voice Assistant STT Backend",
    version="0.1.0",
    lifespan=lifespan
)

# CORS para desarrollo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rutas
app.include_router(stt_routes.router, prefix="/api/stt", tags=["STT"])

# Health check
@app.get("/")
async def root():
    return {
        "message": "STT Backend funcionando",
        "model": settings.model_name,
        "device": settings.device,
        "ready": stt_engine.is_ready
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True  # Auto-reload en desarrollo
    )
