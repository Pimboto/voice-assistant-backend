# test_kyutai_notebook.py
"""
Test de la implementación EXACTA del notebook stt_pytorch.ipynb
"""
import asyncio
import numpy as np
import sys
import os

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_implementation():
    print("🧪 TEST DE KYUTAI STT - IMPLEMENTACIÓN DEL NOTEBOOK")
    print("="*60)
    
    try:
        # Importar el engine
        from src.core.kyutai_real_engine import KyutaiRealSTTEngine, MOSHI_AVAILABLE
        
        if not MOSHI_AVAILABLE:
            print("❌ Moshi no está disponible")
            print("   Instala con: pip install moshi>=0.2.6")
            return
        
        print("✅ Moshi disponible")
        
        # Crear engine
        print("\n1. Creando engine...")
        engine = KyutaiRealSTTEngine(
            model_name="kyutai/stt-1b-en_fr",
            device="cuda"  # Cambia a "cpu" si no tienes GPU
        )
        
        # Inicializar
        print("\n2. Inicializando (esto descargará el modelo si es necesario)...")
        await engine.initialize()
        
        # Crear audio de prueba
        print("\n3. Creando audio de prueba...")
        
        # Opción 1: Tono simple (probablemente no dará texto significativo)
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Crear un sweep de frecuencia para que sea más interesante
        freq_start = 200
        freq_end = 800
        frequency = np.linspace(freq_start, freq_end, len(t))
        audio = np.sin(2 * np.pi * frequency * t) * 0.3
        
        # Agregar algo de ruido blanco suave
        noise = np.random.normal(0, 0.02, len(audio))
        audio = audio + noise
        audio = audio.astype(np.float32)
        
        print(f"   - Duración: {duration}s")
        print(f"   - Samples: {len(audio)}")
        print(f"   - Sample rate: {sample_rate} Hz")
        
        # Opción 2: Si tienes un archivo de audio real
        # import soundfile as sf
        # audio, sample_rate = sf.read("tu_archivo.wav")
        
        # Transcribir
        print("\n4. Transcribiendo...")
        result = await engine.transcribe(audio, sample_rate)
        
        print(f"\n📝 RESULTADO: '{result}'")
        print("   (Nota: un tono sintético probablemente no dará texto significativo)")
        
        # Mostrar información del modelo
        print("\n5. Información del modelo:")
        info = engine.get_model_info()
        for key, value in info.items():
            print(f"   - {key}: {value}")
        
        # Test de streaming (simulado)
        print("\n6. Test de streaming...")
        chunks_processed = 0
        async def audio_generator():
            # Simular chunks de audio
            chunk_size = int(sample_rate * 0.1)  # 100ms chunks
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i+chunk_size]
                if len(chunk) == chunk_size:  # Solo chunks completos
                    yield chunk
        
        async for text in engine.transcribe_stream(audio_generator()):
            if text:
                print(f"   Chunk {chunks_processed}: '{text}'")
            chunks_processed += 1
        
        print(f"   Total chunks procesados: {chunks_processed}")
        
        # Limpiar
        print("\n7. Limpiando recursos...")
        await engine.cleanup()
        
        print("\n✅ TEST COMPLETADO EXITOSAMENTE")
        
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n💡 Sugerencias:")
        print("1. Verifica que Moshi esté instalado: pip install moshi>=0.2.6")
        print("2. Si es la primera vez, el modelo se descargará (~4GB)")
        print("3. Si no tienes GPU, cambia device='cpu' en el código")
        print("4. Para audio real, usa un archivo .wav o .mp3")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("TEST DE KYUTAI STT - IMPLEMENTACIÓN OFICIAL")
    print("="*60)
    print("Basado en: https://colab.research.google.com/github/kyutai-labs/")
    print("           delayed-streams-modeling/blob/main/stt_pytorch.ipynb")
    print("="*60 + "\n")
    
    # Ejecutar test
    asyncio.run(test_implementation())
