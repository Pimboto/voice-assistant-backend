# src/api/stt_routes.py
from fastapi import APIRouter, UploadFile, File, WebSocket, HTTPException
from pydantic import BaseModel
import numpy as np
import soundfile as sf
import librosa
import io
from typing import Dict

class TranscriptionResponse(BaseModel):
    status: str
    text: str
    filename: str
    duration_seconds: float
    original_sample_rate: int

router = APIRouter()

# Instancia global del motor (se inicializará en main.py)
stt_engine = None

@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)) -> TranscriptionResponse:
    """Endpoint para transcribir archivo de audio"""
    try:
        # Verificar tipo de archivo
        if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg')):
            raise HTTPException(
                status_code=400, 
                detail="Formato no soportado. Use: wav, mp3, m4a, flac, ogg"
            )
        
        # Leer archivo
        print(f"📁 Procesando archivo: {file.filename}")
        audio_bytes = await file.read()
        
        # Convertir a numpy array
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
        print(f"🎵 Audio original: {sample_rate} Hz, shape: {audio_data.shape}")
        
        # Si es stereo, convertir a mono
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
            print("🔊 Convertido a mono")
        
        # IMPORTANTE: Resamplear a 16kHz si es necesario
        if sample_rate != 16000:
            print(f"🔄 Convirtiendo de {sample_rate} Hz a 16000 Hz...")
            audio_data = librosa.resample(
                audio_data, 
                orig_sr=sample_rate, 
                target_sr=16000
            )
            sample_rate = 16000
        
        # Asegurar que el audio sea float32
        audio_data = audio_data.astype(np.float32)
        
        print(f"✅ Audio listo: 16000 Hz, {len(audio_data)/16000:.2f} segundos")
        
        # Transcribir
        text = await stt_engine.transcribe(audio_data, sample_rate)
        
        return TranscriptionResponse(
            status="success",
            text=text,
            filename=file.filename,
            duration_seconds=round(len(audio_data) / sample_rate, 2),
            original_sample_rate=sample_rate
        )
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    """WebSocket para streaming en tiempo real"""
    await websocket.accept()
    
    try:
        audio_buffer = []
        
        while True:
            # Recibir datos de audio
            data = await websocket.receive_bytes()
            
            # Convertir bytes a numpy
            audio_chunk = np.frombuffer(data, dtype=np.float32)
            audio_buffer.extend(audio_chunk)
            
            # Cuando tengamos suficiente audio (ej: 3 segundos)
            if len(audio_buffer) > 48000:  # 16000 * 3
                audio_array = np.array(audio_buffer[:48000])
                audio_buffer = audio_buffer[16000:]  # Overlap de 1 segundo
                
                # Transcribir
                text = await stt_engine.transcribe(audio_array)
                
                # Enviar resultado
                await websocket.send_json({
                    "type": "transcription",
                    "text": text
                })
                
    except Exception as e:
        print(f"Error en WebSocket: {e}")
    finally:
        await websocket.close()
