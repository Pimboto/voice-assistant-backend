# src/config/settings.py
from pydantic_settings import BaseSettings
from typing import Optional, List, Union
import torch

class Settings(BaseSettings):
    """Configuración de la aplicación"""
    
    # Motor STT
    stt_engine: str = "kyutai"  # "whisper" o "kyutai"
    kyutai_model: str = "kyutai/stt-1b-en_fr"  # Kyutai STT oficial con 500ms latency
    
    # Configuración de GPU - AHORA CON CUDA
    device: str = "cuda"  # cuda, cpu, auto
    use_gpu: bool = True
    
    # Audio
    sample_rate: int = 16000
    chunk_duration: float = 1.0  # segundos
    
    # Servidor
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # CORS - Arreglado para aceptar string o lista
    cors_origins: Union[str, List[str]] = "*"
    
    # Variables de entorno
    openai_api_key: Optional[str] = None
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Convertir cors_origins a lista si es string"""
        if isinstance(self.cors_origins, str):
            if self.cors_origins == "*":
                return ["*"]
            # Si es string con comas, dividir
            return [origin.strip() for origin in self.cors_origins.split(",")]
        return self.cors_origins
    
    @property
    def actual_device(self) -> str:
        """Obtener el dispositivo real (cuda o cpu) manejando 'auto'"""
        if self.device.lower() == "auto":
            # Detectar automáticamente
            return "cuda" if torch.cuda.is_available() else "cpu"
        elif self.device.lower() in ["cuda", "gpu"]:
            # Verificar que CUDA esté disponible
            if torch.cuda.is_available():
                return "cuda"
            else:
                print("⚠️ CUDA no disponible, usando CPU")
                return "cpu"
        else:
            # cpu o cualquier otra cosa
            return "cpu"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
