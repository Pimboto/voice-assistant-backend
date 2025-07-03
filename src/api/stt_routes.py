# src/api/stt_routes.py - VERSIÓN MEJORADA
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi import File, UploadFile
from pydantic import BaseModel
import numpy as np
import json
import asyncio
import time
import io
import soundfile as sf
from typing import Dict, Any, Optional, List
from collections import deque

router = APIRouter()

class TranscriptionResponse(BaseModel):
    text: str
    confidence: float
    processing_time: float
    model: str

class StreamingSession:
    """Clase mejorada para manejar sesiones de streaming"""
    def __init__(self):
        self.start_time = time.time()
        self.transcription_count = 0
        self.words_buffer = deque(maxlen=50)  # Últimas 50 palabras
        self.audio_buffer = []
        self.last_activity = time.time()
        self.silence_threshold = 2.0  # Reducido para mejor respuesta
        self.current_sentence = []
        self.last_final_text = ""
        self.processing_lock = asyncio.Lock()
        
        # Configuración optimizada para Kyutai
        self.buffer_size = 12000  # 0.75 segundos a 16kHz
        self.overlap_size = 4000  # 0.25 segundos de overlap
        self.min_audio_level = 0.003  # Más sensible
        
    def is_duplicate(self, text: str) -> bool:
        """Verificar si el texto es duplicado con algoritmo mejorado"""
        clean_text = text.strip().lower()
        
        # Verificar contra las últimas palabras
        recent_text = ' '.join(self.words_buffer)
        if clean_text in recent_text:
            return True
            
        # Verificar similitud con último texto final
        if self.last_final_text and clean_text in self.last_final_text.lower():
            return True
            
        return False
    
    def update_words_buffer(self, text: str):
        """Actualizar buffer de palabras"""
        words = text.strip().split()
        self.words_buffer.extend(words)

@router.websocket("/stream")
async def websocket_stt_stream(websocket: WebSocket):
    """WebSocket mejorado para transcripción en tiempo real"""
    await websocket.accept()
    
    # Obtener engine
    stt_engine = websocket.app.state.stt_engine
    
    if not stt_engine or not stt_engine.is_ready:
        await websocket.send_text(json.dumps({
            "error": "STT engine no está listo"
        }))
        await websocket.close()
        return
    
    print("🎤 Nueva sesión WebSocket iniciada")
    session = StreamingSession()
    
    try:
        while True:
            # Recibir datos
            message = await websocket.receive()
            
            if message["type"] == "websocket.disconnect":
                break
                
            data = message.get("bytes")
            if not data:
                continue
            
            try:
                # Procesar audio Float32
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                
                if len(audio_chunk) == 0:
                    continue
                
                # Agregar al buffer con lock para evitar condiciones de carrera
                async with session.processing_lock:
                    session.audio_buffer.extend(audio_chunk)
                
                # Procesar cuando tengamos suficiente audio
                if len(session.audio_buffer) >= session.buffer_size:
                    await process_audio_buffer(
                        session, stt_engine, websocket
                    )
                    
            except Exception as e:
                print(f"⚠️ Error procesando chunk: {e}")
                # Continuar con el siguiente chunk
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"❌ Error en WebSocket: {e}")
    finally:
        duration = time.time() - session.start_time
        print(f"🔚 Sesión terminada. Duración: {duration:.1f}s, "
              f"Transcripciones: {session.transcription_count}")

async def process_audio_buffer(
    session: StreamingSession, 
    stt_engine: Any, 
    websocket: WebSocket
):
    """Procesar buffer de audio con algoritmo mejorado"""
    async with session.processing_lock:
        # Tomar chunk del buffer
        audio_array = np.array(
            session.audio_buffer[:session.buffer_size], 
            dtype=np.float32
        )
        
        # Calcular nivel de audio (RMS)
        audio_level = float(np.sqrt(np.mean(audio_array**2)))
        current_time = time.time()
        
        if audio_level > session.min_audio_level:
            # Audio detectado
            session.last_activity = current_time
            
            try:
                # Transcribir con el motor
                transcription = await stt_engine.transcribe(audio_array, 16000)
                
                if transcription and len(transcription.strip()) > 0:
                    # Procesar transcripción
                    result = process_transcription_enhanced(
                        transcription, session, audio_level
                    )
                    
                    if result:
                        # Enviar resultado
                        await websocket.send_text(json.dumps(result))
                        session.transcription_count += 1
                        
                        # Log para debug
                        print(f"📤 Transcripción: '{result['text']}' "
                              f"(final: {result['is_final']}, "
                              f"confianza: {result['confidence']:.2f})")
                        
            except Exception as e:
                print(f"⚠️ Error en transcripción: {e}")
        
        else:
            # Silencio detectado
            silence_duration = current_time - session.last_activity
            
            # Si hay una oración en curso y silencio prolongado, finalizarla
            if (session.current_sentence and 
                silence_duration > session.silence_threshold):
                
                final_text = ' '.join(session.current_sentence)
                session.last_final_text = final_text
                session.current_sentence = []
                
                result = {
                    "text": final_text + ".",
                    "is_final": True,
                    "confidence": 0.9,
                    "timestamp": current_time,
                    "type": "sentence_end"
                }
                
                await websocket.send_text(json.dumps(result))
                print(f"📤 Oración completa: '{final_text}.'")
        
        # Mantener overlap para continuidad
        session.audio_buffer = session.audio_buffer[-session.overlap_size:]

def process_transcription_enhanced(
    text: str, 
    session: StreamingSession, 
    audio_level: float
) -> Optional[Dict[str, Any]]:
    """Procesamiento mejorado de transcripción"""
    
    # Limpiar texto
    clean_text = text.strip()
    if not clean_text or len(clean_text) < 2:
        return None
    
    # Verificar duplicados
    if session.is_duplicate(clean_text):
        return None
    
    # Actualizar buffer de palabras
    session.update_words_buffer(clean_text)
    
    # Detectar si es final de oración
    is_sentence_end = any(clean_text.endswith(p) for p in ['.', '?', '!'])
    
    # Detectar pausas naturales (palabras que suelen terminar frases)
    pause_indicators = [
        'gracias', 'adiós', 'hola', 'bien', 'vale', 'okay', 
        'sí', 'no', 'claro', 'perfecto', 'entendido'
    ]
    is_natural_pause = any(
        word.lower() in pause_indicators 
        for word in clean_text.split()[-2:]  # Últimas 2 palabras
    )
    
    # Determinar si es transcripción final
    word_count = len(clean_text.split())
    is_final = (
        is_sentence_end or 
        is_natural_pause or 
        word_count > 8  # Frases largas se marcan como finales
    )
    
    # Gestionar oraciones
    if is_final:
        session.current_sentence.append(clean_text)
        complete_sentence = ' '.join(session.current_sentence)
        session.current_sentence = []
        session.last_final_text = complete_sentence
        result_text = complete_sentence
    else:
        session.current_sentence.append(clean_text)
        result_text = clean_text
    
    # Calcular confianza mejorada
    confidence = calculate_enhanced_confidence(
        clean_text, audio_level, word_count, is_final
    )
    
    return {
        "text": result_text,
        "is_final": is_final,
        "confidence": confidence,
        "timestamp": time.time(),
        "audio_level": float(audio_level),
        "word_count": word_count,
        "type": "final" if is_final else "partial"
    }

def calculate_enhanced_confidence(
    text: str, 
    audio_level: float, 
    word_count: int, 
    is_final: bool
) -> float:
    """Cálculo mejorado de confianza"""
    
    # Factor de nivel de audio (normalizado)
    audio_factor = min(audio_level * 40, 1.0)
    
    # Factor de longitud
    length_factor = min(word_count / 5.0, 1.0)
    
    # Factor de calidad del texto
    quality_factor = 1.0
    if text:
        # Penalizar repeticiones
        words = text.lower().split()
        if len(words) > 1:
            unique_ratio = len(set(words)) / len(words)
            quality_factor = max(0.6, unique_ratio)
        
        # Bonus por mayúsculas al inicio (indica inicio de oración)
        if text[0].isupper():
            quality_factor *= 1.1
    
    # Factor de finalidad
    final_factor = 1.0 if is_final else 0.85
    
    # Combinar factores
    confidence = (
        audio_factor * 0.3 + 
        length_factor * 0.2 + 
        quality_factor * 0.3 + 
        final_factor * 0.2
    )
    
    return float(max(0.4, min(0.98, confidence)))
