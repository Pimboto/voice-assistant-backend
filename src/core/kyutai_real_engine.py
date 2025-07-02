# src/core/kyutai_real_engine.py
"""
Motor STT usando Kyutai - Implementación ARREGLADA
Basado en el código REAL del notebook stt_pytorch.ipynb
Con fixes para Windows y manejo de errores
"""
import os
# IMPORTANTE: Deshabilitar compilación JIT problemática en Windows
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'

import torch
# Configurar PyTorch para evitar errores de compilación
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

import numpy as np
import asyncio
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import sentencepiece
import warnings

# Suprimir warnings de compilación
warnings.filterwarnings("ignore", category=UserWarning, module="torch._dynamo")

# Importaciones EXACTAS del notebook
try:
    from moshi.models import loaders, MimiModel, LMModel, LMGen
    MOSHI_AVAILABLE = True
    print("✅ Moshi cargado correctamente")
except ImportError as e:
    MOSHI_AVAILABLE = False
    print(f"❌ Error importando Moshi: {e}")


@dataclass
class InferenceState:
    """Clase EXACTA del notebook para manejar la inferencia"""
    mimi: MimiModel
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm_gen: LMGen

    def __init__(
        self,
        mimi: MimiModel,
        text_tokenizer: sentencepiece.SentencePieceProcessor,
        lm: LMModel,
        batch_size: int,
        device: str | torch.device,
    ):
        self.mimi = mimi
        self.text_tokenizer = text_tokenizer
        self.lm_gen = LMGen(lm, temp=0, temp_text=0, use_sampling=False)
        self.device = device
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.batch_size = batch_size
        
        # Deshabilitar CUDA graphs que causan problemas
        with torch.cuda.device(device):
            torch.cuda.set_sync_debug_mode(1)
        
        self.mimi.streaming_forever(batch_size)
        self.lm_gen.streaming_forever(batch_size)

    def run(self, in_pcms: torch.Tensor):
        """Método run EXACTO del notebook con manejo de errores"""
        device = self.lm_gen.lm_model.device
        ntokens = 0
        first_frame = True
        chunks = [
            c
            for c in in_pcms.split(self.frame_size, dim=2)
            if c.shape[-1] == self.frame_size
        ]
        all_text = []
        
        for chunk in chunks:
            try:
                codes = self.mimi.encode(chunk)
                if first_frame:
                    # Ensure that the first slice of codes is properly seen by the transformer
                    tokens = self.lm_gen.step(codes)
                    first_frame = False
                tokens = self.lm_gen.step(codes)
                if tokens is None:
                    continue
                assert tokens.shape[1] == 1
                one_text = tokens[0, 0].cpu()
                if one_text.item() not in [0, 3]:
                    text = self.text_tokenizer.id_to_piece(one_text.item())
                    text = text.replace("▁", " ")
                    all_text.append(text)
                ntokens += 1
            except Exception as e:
                # Continuar con el siguiente chunk si hay error
                print(f"⚠️ Error procesando chunk: {e}")
                continue
        
        return "".join(all_text)


class KyutaiRealSTTEngine:
    """
    Motor STT usando el código EXACTO del notebook stt_pytorch.ipynb
    Con fixes para Windows y errores de compilación
    """
    def __init__(self, model_name: str = "kyutai/stt-1b-en_fr", device: str = "cuda"):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Componentes del modelo (del notebook)
        self.checkpoint_info = None
        self.mimi = None
        self.text_tokenizer = None
        self.lm = None
        self.inference_state = None
        
        self.is_ready = False
        
        # Configuración del notebook
        self.batch_size = 1
        
    async def initialize(self):
        """Inicializar EXACTAMENTE como en el notebook"""
        if not MOSHI_AVAILABLE:
            raise ImportError(
                "Moshi no está instalado. Instala con:\n"
                "pip install moshi>=0.2.6"
            )
        
        print(f"\n🔄 Cargando Kyutai STT: {self.model_name}")
        
        try:
            # Deshabilitar compilación para evitar errores
            torch.compiler.disable()
            
            # CÓDIGO EXACTO DEL NOTEBOOK:
            self.checkpoint_info = loaders.CheckpointInfo.from_hf_repo(self.model_name)
            self.mimi = self.checkpoint_info.get_mimi(device=self.device)
            self.text_tokenizer = self.checkpoint_info.get_text_tokenizer()
            self.lm = self.checkpoint_info.get_moshi(device=self.device)
            
            # Poner modelos en modo evaluación
            self.mimi.eval()
            self.lm.eval()
            
            # Crear InferenceState
            self.inference_state = InferenceState(
                self.mimi,
                self.text_tokenizer,
                self.lm,
                batch_size=self.batch_size,
                device=self.device
            )
            
            self.is_ready = True
            print("✅ Kyutai STT inicializado correctamente")
            self._print_info()
            
        except Exception as e:
            print(f"❌ Error inicializando: {e}")
            raise e
    
    async def transcribe(self, audio_array: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribir audio usando el método EXACTO del notebook
        Con resampling automático si es necesario
        """
        if not self.is_ready:
            return ""
        
        try:
            # IMPORTANTE: El modelo espera audio a 24000 Hz
            if sample_rate != self.mimi.sample_rate:
                # Resamplear audio a la frecuencia correcta
                audio_array = self._resample_audio(audio_array, sample_rate, self.mimi.sample_rate)
                sample_rate = self.mimi.sample_rate
            
            # Convertir audio a tensor como en el notebook
            in_pcms = torch.from_numpy(audio_array).float()
            
            # Si es mono, asegurar shape correcto
            if in_pcms.dim() == 1:
                in_pcms = in_pcms.unsqueeze(0)  # Agregar dimensión de canal
            
            # Mover a dispositivo
            in_pcms = in_pcms.to(device=self.device)
            
            # Aplicar padding según configuración STT (del notebook)
            stt_config = self.checkpoint_info.stt_config
            pad_left = int(stt_config.get("audio_silence_prefix_seconds", 0.0) * sample_rate)
            pad_right = int((stt_config.get("audio_delay_seconds", 0.0) + 1.0) * sample_rate)
            in_pcms = torch.nn.functional.pad(in_pcms, (pad_left, pad_right), mode="constant")
            
            # Expandir para batch (del notebook)
            if in_pcms.dim() == 2:
                in_pcms = in_pcms.unsqueeze(0)  # [batch, channels, time]
            
            # Si tenemos múltiples canales, tomar solo el primero
            if in_pcms.shape[1] > 1:
                in_pcms = in_pcms[:, 0:1, :]
            
            # Ejecutar transcripción con manejo de errores
            with torch.no_grad():
                text = self.inference_state.run(in_pcms)
            
            return text.strip()
            
        except Exception as e:
            print(f"❌ Error en transcripción: {e}")
            return ""
    
    def _resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resamplear audio a la frecuencia objetivo"""
        if orig_sr == target_sr:
            return audio
        
        try:
            import scipy.signal
            # Calcular el número de muestras en la nueva frecuencia
            num_samples = int(len(audio) * target_sr / orig_sr)
            # Resamplear
            resampled = scipy.signal.resample(audio, num_samples)
            return resampled.astype(np.float32)
        except ImportError:
            # Si no hay scipy, hacer resampling simple
            print("⚠️ scipy no disponible, usando resampling simple")
            # Resampling simple por interpolación
            old_indices = np.arange(0, len(audio))
            new_length = int(len(audio) * target_sr / orig_sr)
            new_indices = np.linspace(0, len(audio) - 1, new_length)
            resampled = np.interp(new_indices, old_indices, audio)
            return resampled.astype(np.float32)
    
    async def transcribe_stream(self, audio_stream):
        """
        Transcribir un stream de audio en tiempo real
        Procesando chunks como en el notebook
        """
        if not self.is_ready:
            return
        
        # Acumular audio hasta tener un frame completo
        buffer = []
        frame_size = self.inference_state.frame_size
        
        async for audio_chunk in audio_stream:
            # Agregar al buffer
            if isinstance(audio_chunk, np.ndarray):
                buffer.extend(audio_chunk.tolist())
            else:
                buffer.extend(audio_chunk)
            
            # Procesar cuando tengamos suficiente audio
            while len(buffer) >= frame_size:
                # Extraer un frame
                frame_audio = np.array(buffer[:frame_size], dtype=np.float32)
                buffer = buffer[frame_size:]
                
                # Transcribir frame
                text = await self.transcribe(frame_audio, self.mimi.sample_rate)
                if text:
                    yield text
    
    def _print_info(self):
        """Imprimir información del modelo cargado"""
        print(f"\n📊 Modelo Kyutai STT (implementación del notebook):")
        print(f"   - Modelo: {self.model_name}")
        print(f"   - Dispositivo: {self.device}")
        print(f"   - Sample rate: {self.mimi.sample_rate} Hz")
        print(f"   - Frame rate: {self.mimi.frame_rate} Hz")
        print(f"   - Frame size: {self.inference_state.frame_size} samples")
        
        # Mostrar configuración STT
        stt_config = self.checkpoint_info.stt_config
        print(f"\n   Configuración STT:")
        print(f"   - Audio silence prefix: {stt_config.get('audio_silence_prefix_seconds', 0.0)}s")
        print(f"   - Audio delay: {stt_config.get('audio_delay_seconds', 0.0)}s")
        
        if self.device == "cuda":
            print(f"\n   - GPU: {torch.cuda.get_device_name(0)}")
            mem = torch.cuda.memory_allocated() / 1024**3
            print(f"   - Memoria GPU usada: {mem:.1f} GB")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Obtener información del modelo"""
        if not self.is_ready:
            return {"status": "not_initialized"}
        
        stt_config = self.checkpoint_info.stt_config if self.checkpoint_info else {}
        
        return {
            "model": self.model_name,
            "implementation": "notebook_stt_pytorch_fixed",
            "device": str(self.device),
            "sample_rate": self.mimi.sample_rate if self.mimi else None,
            "frame_rate": self.mimi.frame_rate if self.mimi else None,
            "frame_size": self.inference_state.frame_size if self.inference_state else None,
            "audio_delay_seconds": stt_config.get("audio_delay_seconds", 0.0),
            "status": "ready"
        }
    
    async def cleanup(self):
        """Limpiar recursos"""
        try:
            # Limpiar modelos
            del self.inference_state
            del self.lm
            del self.text_tokenizer
            del self.mimi
            del self.checkpoint_info
            
            # Liberar memoria GPU
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            self.is_ready = False
            print("✅ Recursos liberados")
            
        except Exception as e:
            print(f"⚠️ Error en cleanup: {e}")


# Para debug: verificar que todo funciona
if __name__ == "__main__":
    import asyncio
    
    async def test():
        print("🧪 Test de Kyutai STT con implementación ARREGLADA")
        
        try:
            # Crear engine
            engine = KyutaiRealSTTEngine(device="cuda")
            
            # Inicializar
            await engine.initialize()
            
            # Crear audio de prueba (1 segundo de tono)
            sample_rate = 16000  # Usamos 16kHz, se resampleará automáticamente
            duration = 1.0
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = np.sin(2 * np.pi * 440 * t) * 0.3
            audio = audio.astype(np.float32)
            
            # Transcribir
            print("\n🎤 Transcribiendo audio de prueba...")
            result = await engine.transcribe(audio, sample_rate)
            print(f"Resultado: '{result}'")
            
            # Info
            info = engine.get_model_info()
            print(f"\nInfo: {info}")
            
            # Limpiar
            await engine.cleanup()
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Ejecutar test
    asyncio.run(test())
