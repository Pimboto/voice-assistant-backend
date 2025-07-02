# main.py
import os
# Deshabilitar optimizaciones PyTorch que requieren Triton (problemas en Windows)
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Para mejor debugging

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import sys

# Configuración adicional de PyTorch después de importar
import torch
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

# Deshabilitar CUDA graphs que causan problemas con Moshi
if torch.cuda.is_available():
    torch.cuda.set_sync_debug_mode(1)

# Verificar versiones instaladas
print("🔍 Verificando entorno:")
print(f"   - Python: {sys.version}")
print(f"   - PyTorch: {torch.__version__}")
print(f"   - CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   - GPU: {torch.cuda.get_device_name(0)}")

# Verificar Moshi
try:
    import moshi
    print(f"   - Moshi instalado: ✅ (versión: {getattr(moshi, '__version__', 'desconocida')})")
except ImportError:
    print("   - Moshi instalado: ❌")
    print("   💡 Instala con: pip install moshi>=0.2.6")

from src.config.settings import settings
from src.api.stt_routes import router as stt_router

# Global engine instance
engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestión del ciclo de vida de la aplicación"""
    global engine
    
    print("\n🚀 Iniciando servidor...")
    
    # Mostrar configuración de dispositivo
    print(f"📱 Dispositivo configurado: {settings.device}")
    print(f"📱 Dispositivo real a usar: {settings.actual_device}")
    
    # Determinar qué engine usar
    if settings.stt_engine == "kyutai":
        print(f"🎯 Intentando usar Kyutai STT con modelo: {settings.kyutai_model}")
        
        try:
            # Primero intentar importar para verificar disponibilidad
            from src.core.kyutai_real_engine import KyutaiRealSTTEngine, MOSHI_AVAILABLE
            
            if not MOSHI_AVAILABLE:
                print("⚠️ Moshi no está disponible, usando fallback...")
                raise ImportError("Moshi no disponible")
            
            # Crear engine Kyutai con el dispositivo REAL
            engine = KyutaiRealSTTEngine(
                model_name=settings.kyutai_model,
                device=settings.actual_device  # Usar actual_device en lugar de device
            )
            
            # Inicializar con timeout
            try:
                await asyncio.wait_for(engine.initialize(), timeout=60.0)
                print("✅ Motor Kyutai STT inicializado correctamente")
                
                # Mostrar info del modelo
                model_info = engine.get_model_info()
                print(f"📊 Info del modelo:")
                for key, value in model_info.items():
                    print(f"   - {key}: {value}")
                    
            except asyncio.TimeoutError:
                print("⏱️ Timeout inicializando Kyutai STT")
                raise
            except Exception as e:
                print(f"❌ Error inicializando Kyutai STT: {type(e).__name__}: {e}")
                raise
                
        except Exception as e:
            print(f"\n⚠️ No se pudo cargar Kyutai STT: {e}")
            print("📥 Usando Whisper como fallback...")
            
            # Fallback a Whisper
            try:
                from src.core.stt_engine import SimpleSTTEngine
                engine = SimpleSTTEngine(
                    model_name="openai/whisper-base",
                    device=settings.actual_device  # Usar actual_device
                )
                await engine.initialize()
                print("✅ Whisper fallback inicializado")
            except Exception as whisper_error:
                print(f"❌ Error con Whisper fallback: {whisper_error}")
                raise
    else:
        print("🎯 Usando Whisper STT directamente")
        from src.core.stt_engine import SimpleSTTEngine
        engine = SimpleSTTEngine(
            model_name="openai/whisper-base",
            device=settings.actual_device  # Usar actual_device
        )
        await engine.initialize()
        print("✅ Motor Whisper STT inicializado")
    
    # Hacer el engine disponible globalmente
    app.state.stt_engine = engine
    
    yield
    
    print("\n👋 Cerrando servidor...")
    if engine and hasattr(engine, 'cleanup'):
        await engine.cleanup()

# Crear aplicación FastAPI
app = FastAPI(
    title="Voice Assistant Backend - Kyutai STT",
    description="Backend para asistente de voz con Kyutai Labs STT. Prueba la transcripción en /docs",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc"  # ReDoc
)

# Configurar CORS - ARREGLADO
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,  # Usar la propiedad que convierte a lista
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir rutas
app.include_router(stt_router, prefix="/api/stt", tags=["Speech to Text"])

# Servir archivos estáticos
@app.get("/", tags=["UI"])
async def read_root():
    """Servir la página principal con interfaz de transcripción en tiempo real"""
    return FileResponse("client_realtime.html")

@app.get("/health", tags=["System"])
async def health_check():
    """
    Verificar el estado del sistema y los modelos cargados.
    
    Retorna información sobre:
    - Estado del motor STT
    - Modelo cargado
    - Uso de memoria
    - Configuración actual
    """
    global engine
    
    # Información básica
    health_info = {
        "status": "healthy",
        "stt_engine": settings.stt_engine,
        "configured_model": settings.kyutai_model if settings.stt_engine == "kyutai" else "whisper-base",
        "configured_device": settings.device,
        "actual_device": settings.actual_device,  # Mostrar el dispositivo real
        "engine_ready": False,
        "actual_model": None,
        "model_info": {},
        "system_info": {
            "python_version": sys.version.split()[0],
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }
    }
    
    # Información del engine si está disponible
    if engine:
        health_info["engine_ready"] = getattr(engine, 'is_ready', False)
        health_info["actual_model"] = getattr(engine, 'model_name', 'unknown')
        
        if hasattr(engine, 'get_model_info'):
            health_info["model_info"] = engine.get_model_info()
    
    # Verificar Moshi
    try:
        import moshi
        health_info["moshi_available"] = True
        health_info["moshi_version"] = getattr(moshi, '__version__', 'unknown')
    except ImportError:
        health_info["moshi_available"] = False
        health_info["moshi_message"] = "Install with: pip install moshi>=0.2.6"
    
    return health_info

@app.get("/test-transcription", tags=["Testing"])
async def test_transcription():
    """
    Endpoint de prueba para verificar la transcripción con audio sintético.
    
    Genera un audio de prueba y lo transcribe. Útil para verificar que el sistema funciona.
    """
    global engine
    
    if not engine or not engine.is_ready:
        return {"error": "Engine not ready"}
    
    try:
        # Crear audio de prueba (1 segundo de habla simulada)
        import numpy as np
        
        # Crear un audio más complejo que simule habla
        sample_rate = 16000  # Se resampleará automáticamente si es necesario
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Simular formantes de voz (frecuencias típicas del habla)
        f1, f2, f3 = 700, 1700, 2700  # Formantes típicos
        audio = (0.3 * np.sin(2 * np.pi * f1 * t) + 
                0.2 * np.sin(2 * np.pi * f2 * t) + 
                0.1 * np.sin(2 * np.pi * f3 * t))
        
        # Agregar modulación de amplitud (simular ritmo del habla)
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)  # 3 Hz modulation
        audio = audio * envelope
        
        # Agregar algo de ruido
        noise = np.random.normal(0, 0.02, len(audio))
        audio = audio + noise
        audio = audio.astype(np.float32)
        
        # Normalizar
        audio = audio / np.abs(audio).max() * 0.8
        
        # Transcribir
        result = await engine.transcribe(audio, sample_rate)
        
        return {
            "success": True,
            "transcription": result or "(audio procesado, sin texto detectado)",
            "engine_type": type(engine).__name__,
            "model": getattr(engine, 'model_name', 'unknown'),
            "audio_duration": duration,
            "audio_samples": len(audio),
            "note": "Este es audio sintético. Para mejores resultados, usa audio de voz real."
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

# Documentación adicional para Swagger
app.tags_metadata = [
    {
        "name": "Speech to Text",
        "description": "Endpoints para transcripción de audio usando Kyutai STT"
    },
    {
        "name": "System",
        "description": "Endpoints de estado y salud del sistema"
    },
    {
        "name": "Testing",
        "description": "Endpoints para pruebas y verificación"
    },
    {
        "name": "UI",
        "description": "Interfaces de usuario"
    }
]

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🎙️  Voice Assistant Backend con Kyutai STT")
    print("="*50)
    print(f"🌐 Servidor: http://localhost:{settings.port}")
    print(f"📊 Health: http://localhost:{settings.port}/health")
    print(f"🧪 Test: http://localhost:{settings.port}/test-transcription")
    print(f"📚 API Docs: http://localhost:{settings.port}/docs")
    print(f"📖 ReDoc: http://localhost:{settings.port}/redoc")
    print("="*50 + "\n")
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info" if not settings.debug else "debug"
    )
