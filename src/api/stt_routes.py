# src/api/stt_routes.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi import File, UploadFile
from pydantic import BaseModel
import numpy as np
import json
import asyncio
import time
import io
import soundfile as sf
from typing import Dict, Any, Optional

router = APIRouter()

class TranscriptionResponse(BaseModel):
    text: str
    confidence: float
    processing_time: float
    model: str

@router.post("/transcribe")
async def transcribe_audio(request: Request, file: UploadFile = File(...)):
    """Endpoint REST para transcripción de audio"""
    try:
        # Obtener engine del estado de la aplicación
        stt_engine = request.app.state.stt_engine
        
        if not stt_engine or not stt_engine.is_ready:
            raise HTTPException(status_code=503, detail="STT engine no está listo")
        
        # Leer archivo de audio
        audio_data = await file.read()
        
        # Decodificar audio
        audio_array, sample_rate = sf.read(io.BytesIO(audio_data))
        
        if len(audio_array) == 0:
            raise HTTPException(status_code=400, detail="Audio vacío")
        
        # Transcribir
        start_time = time.time()
        transcription = await stt_engine.transcribe(audio_array, sample_rate)
        processing_time = time.time() - start_time
        
        # Calcular confianza básica
        confidence = _calculate_basic_confidence(audio_array, processing_time)
        
        # Obtener información del modelo
        model_name = getattr(stt_engine, 'model_name', 'unknown')
        
        return TranscriptionResponse(
            text=transcription,
            confidence=confidence,
            processing_time=processing_time,
            model=model_name
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en transcripción: {str(e)}")

@router.websocket("/stream")
async def websocket_stt_stream(websocket: WebSocket):
    """WebSocket para transcripción en tiempo real"""
    await websocket.accept()
    
    # Obtener engine del estado de la aplicación  
    stt_engine = websocket.app.state.stt_engine
    
    if not stt_engine or not stt_engine.is_ready:
        await websocket.send_text(json.dumps({
            "error": "STT engine no está listo"
        }))
        await websocket.close()
        return
    
    print("🎤 Nueva sesión WebSocket iniciada")
    
    # Estado de la sesión
    session_state = {
        "start_time": time.time(),
        "transcription_count": 0,
        "words_sent": set(),  # Palabras ya enviadas para evitar duplicados
        "last_activity": time.time(),
        "silence_threshold": 3.0,  # Segundos de silencio para reset
        "last_transcription": "",
        "partial_text": "",
        "audio_buffer": []  # Buffer para chunks de audio Float32
    }
    
    try:
        while True:
            # Recibir datos de audio
            message = await websocket.receive()
            
            if message["type"] == "websocket.disconnect":
                break
                
            data = message.get("bytes")
            if not data:
                continue
            
            try:
                # Los datos vienen como Float32Array desde ScriptProcessorNode
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                
                if len(audio_chunk) == 0:
                    continue
                
                # Acumular chunks hasta tener suficiente para transcribir
                session_state["audio_buffer"].extend(audio_chunk)
                
                # Procesar cuando tengamos aproximadamente 0.5 segundos de audio (8000 samples) - optimizado para CUDA
                if len(session_state["audio_buffer"]) >= 8000:  # Reducido para CUDA
                    # Convertir lista a numpy array
                    audio_array = np.array(session_state["audio_buffer"][:8000], dtype=np.float32)
                    
                    # Detectar nivel de audio para gestión de silencio
                    audio_level = float(np.sqrt(np.mean(audio_array**2)))  # Asegurar Python float
                    current_time = time.time()
                    
                    if audio_level > 0.005:  # Umbral más sensible para mejor detección
                        session_state["last_activity"] = current_time
                        
                        # Transcribir el chunk de audio con mejor manejo de errores
                        try:
                            transcription = await stt_engine.transcribe(audio_array, 16000)
                            
                            if transcription and len(transcription.strip()) > 0:
                                # Procesar transcripción para evitar duplicados
                                processed_result = _process_transcription_with_deduplication(
                                    transcription, 
                                    session_state,
                                    audio_level
                                )
                                
                                if processed_result:
                                    # Enviar resultado
                                    await websocket.send_text(json.dumps(processed_result))
                                    session_state["transcription_count"] += 1
                                    
                                    print(f"📤 Enviado: '{processed_result['text']}' "
                                          f"(final: {processed_result['is_final']}, "
                                          f"conf: {processed_result['confidence']:.2f})")
                        except Exception as transcribe_error:
                            print(f"⚠️ Error en transcripción específica: {transcribe_error}")
                            # No usar continue, solo registrar el error y continuar procesando
                    
                    else:
                        # Silencio detectado
                        silence_duration = current_time - session_state["last_activity"]
                        
                        if silence_duration > session_state["silence_threshold"]:
                            # Reset por silencio prolongado
                            if session_state["words_sent"] or session_state["partial_text"]:
                                print("🔄 Reset por silencio prolongado")
                                session_state["words_sent"].clear()
                                session_state["partial_text"] = ""
                                session_state["last_transcription"] = ""
                    
                    # Mantener overlap optimizado para CUDA (0.25 segundos)
                    overlap_samples = 4000  # 0.25 segundos a 16kHz - más eficiente con CUDA
                    session_state["audio_buffer"] = session_state["audio_buffer"][-overlap_samples:]
            
            except Exception as e:
                # Error más específico para debug
                error_type = type(e).__name__
                print(f"⚠️ Error procesando audio ({error_type}): {e}")
                # No hacer continue aquí, continuar con el siguiente mensaje
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"❌ Error en WebSocket: {e}")
    finally:
        duration = time.time() - session_state["start_time"]
        print(f"🔚 Sesión terminada. Duración: {duration:.1f}s, "
              f"Transcripciones: {session_state['transcription_count']}")

def _process_transcription_with_deduplication(
    transcription: str, 
    session_state: Dict[str, Any], 
    audio_level: float
) -> Optional[Dict[str, Any]]:
    """Procesar transcripción con algoritmo avanzado de deduplicación"""
    
    # Limpiar y normalizar texto
    clean_text = transcription.strip().lower()
    
    if not clean_text or len(clean_text) < 2:
        return None
    
    # Filtrar palabras de relleno comunes
    filler_words = {"uh", "um", "er", "ah", "hmm", "like", "you know"}
    words = clean_text.split()
    filtered_words = [w for w in words if w not in filler_words]
    
    if not filtered_words:
        return None
    
    # Convertir de vuelta a string original (con capitalización)
    filtered_text = " ".join(filtered_words)
    
    # Verificar si es substring de transcripción anterior (evitar duplicados)
    if (session_state["last_transcription"] and 
        filtered_text in session_state["last_transcription"].lower()):
        return None
    
    # Verificar si alguna palabra nueva no ha sido enviada
    current_words = set(filtered_words)
    new_words = current_words - session_state["words_sent"]
    
    if not new_words and not session_state["partial_text"]:
        return None
    
    # Determinar si es transcripción final o parcial
    is_final = len(filtered_text) > 10 or "." in transcription or "?" in transcription
    
    # Calcular confianza
    confidence = _calculate_advanced_confidence(
        transcription, audio_level, len(filtered_words), is_final
    )
    
    # Actualizar estado
    if is_final:
        session_state["words_sent"].update(current_words)
        session_state["last_transcription"] = filtered_text
        session_state["partial_text"] = ""
        result_text = transcription.strip()  # Mantener capitalización original
    else:
        session_state["partial_text"] = filtered_text
        result_text = transcription.strip()
    
    return {
        "text": result_text,
        "is_final": is_final,
        "confidence": confidence,
        "timestamp": time.time(),
        "audio_level": float(audio_level),
        "word_count": len(filtered_words)
    }

def _calculate_basic_confidence(audio_array: np.ndarray, processing_time: float) -> float:
    """Calcular confianza básica para endpoint REST"""
    try:
        # Nivel de señal
        rms = np.sqrt(np.mean(audio_array**2))
        signal_strength = min(float(rms) * 5, 1.0)  # Convertir a Python float
        
        # Factor de velocidad de procesamiento
        speed_factor = max(0.1, 1.0 - max(0, processing_time - 0.5) / 2.0)
        
        confidence = (signal_strength * 0.6 + speed_factor * 0.4)
        return float(max(0.1, min(0.95, confidence)))  # Asegurar que sea Python float
    except:
        return 0.5

def _calculate_advanced_confidence(
    text: str, 
    audio_level: float, 
    word_count: int, 
    is_final: bool
) -> float:
    """Calcular confianza avanzada para streaming - optimizado y robusto"""
    try:
        # Asegurar que audio_level es Python float
        audio_level = float(audio_level) if audio_level is not None else 0.0
        word_count = int(word_count) if word_count is not None else 0
        
        # Factor de nivel de audio (más sensible para CUDA)
        audio_factor = min(audio_level * 30, 1.0)  # Más sensible con CUDA
        
        # Factor de longitud de texto
        length_factor = min(word_count / 3.0, 1.0)  # Más rápido con buffers pequeños
        
        # Factor de finalidad
        final_factor = 1.0 if is_final else 0.8  # Más confianza en parciales
        
        # Factor de calidad del texto (evitar repeticiones)
        quality_factor = 1.0
        if text and len(text.strip()) > 0:
            words = text.lower().split()
            if len(words) > 1:
                unique_ratio = len(set(words)) / len(words)
                quality_factor = max(0.5, unique_ratio)  # Más tolerante
        
        # Combinar factores (optimizado para CUDA)
        confidence = (
            audio_factor * 0.35 + 
            length_factor * 0.25 + 
            final_factor * 0.25 + 
            quality_factor * 0.15
        )
        
        return float(max(0.2, min(0.95, confidence)))  # Asegurar Python float
    except Exception as e:
        print(f"⚠️ Error calculando confianza: {e}")
        return 0.6  # Confianza predeterminada más alta para CUDA
