# test_stt.py
import requests
import json

def test_transcribe():
    """Probar transcripción con archivo"""
    # URL del servidor
    url = "http://127.0.0.1:8000/api/stt/transcribe"
    
    # Archivo de prueba (asegúrate de tener un archivo .wav)
    files = {'file': open('test_audio.wav', 'rb')}
    
    # Hacer petición
    response = requests.post(url, files=files)
    
    # Mostrar resultado
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Transcripción: {result['text']}")
    else:
        print(f"❌ Error: {response.text}")

def test_health():
    """Verificar que el servidor está activo"""
    response = requests.get("http://127.0.0.1:8000/health")
    print(f"Health: {response.json()}")

if __name__ == "__main__":
    print("Probando servidor STT...")
    test_health()
    # test_transcribe()  # Descomenta cuando tengas un archivo de audio
