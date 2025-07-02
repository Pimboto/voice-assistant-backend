# src/config/settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Modelo
    model_name: str = "openai/whisper-base"
    device: str = "cuda"
    
    # Server
    host: str = "127.0.0.1"
    port: int = 8000
    
    class Config:
        env_file = ".env"
        # Agregar esta línea para resolver el warning
        protected_namespaces = ('settings_',)

settings = Settings()
