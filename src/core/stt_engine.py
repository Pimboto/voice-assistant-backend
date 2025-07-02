# src/core/stt_engine.py
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np
from typing import Optional
import asyncio

class SimpleSTTEngine:
    def __init__(self, model_name: str = "openai/whisper-base", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.processor = None
        self.model = None
        self.is_ready = False
        
    async def initialize(self):
        """Cargar modelo"""
        print(f"🔄 Cargando modelo {self.model_name}...")
        
        # Determinar device y dtype reales
        if self.device == "cuda" and torch.cuda.is_available():
            actual_device = "cuda"
            model_dtype = torch.float16
            print(f"🔧 Usando CUDA con float16")
        else:
            actual_device = "cpu"
            model_dtype = torch.float32
            print(f"🔧 Usando CPU con float32")
            
        # Guardar configuración real
        self.actual_device = actual_device
        self.model_dtype = model_dtype
        
        # Cargar processor y modelo
        self.processor = WhisperProcessor.from_pretrained(self.model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=model_dtype
        )
        
        # Mover a device correcto
        self.model = self.model.to(actual_device)
        
        self.is_ready = True
        print(f"✅ Modelo cargado en {actual_device} con dtype {model_dtype}")
        
    async def transcribe(self, audio_array: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribir audio a texto"""
        if not self.is_ready:
            await self.initialize()
        
        try:
            # Asegurar que el audio esté en el formato correcto
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            
            # Procesar audio
            inputs = self.processor(
                audio_array, 
                sampling_rate=sample_rate, 
                return_tensors="pt"
            )
            
            # IMPORTANTE: inputs es un diccionario
            input_features = inputs["input_features"]
            
            # Mover a device correcto y con dtype correcto para que coincida con el modelo
            input_features = input_features.to(self.actual_device, dtype=self.model_dtype)
            
            # Generar transcripción
            with torch.no_grad():
                # Generar IDs predichos
                predicted_ids = self.model.generate(input_features)
            
            # Decodificar - predicted_ids ya está en CPU después de generate
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            return transcription.strip()
            
        except Exception as e:
            print(f"❌ Error en transcripción: {e}")
            print(f"   Tipo de audio: {type(audio_array)}, shape: {audio_array.shape}")
            print(f"   Dtype: {audio_array.dtype}, sample_rate: {sample_rate}")
            raise e
