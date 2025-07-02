import sys
print(f"Python: {sys.version}\n")

# Verificar módulos
modules = {
    'torch': 'PyTorch',
    'fastapi': 'FastAPI',
    'transformers': 'Transformers',
    'soundfile': 'SoundFile',
    'pyaudio': 'PyAudio'
}

for module, name in modules.items():
    try:
        __import__(module)
        print(f"✅ {name} instalado")
        if module == 'torch':
            import torch
            print(f"   CUDA: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"   GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print(f"❌ {name} NO instalado")
