# 🎤 Kyutai Voice Assistant

**Sistema de transcripción en tiempo real usando Kyutai STT con GPU NVIDIA RTX 4060**

## ✨ Características

- 🚀 **Kyutai STT oficial** - Modelo `kyutai/stt-1b-en_fr` 
- ⚡ **GPU acelerado** - CUDA con RTX 4060
- 🌍 **Multiidioma** - Inglés y Francés
- ⏱️ **Tiempo real** - Latencia de 500ms
- 🔄 **Fallback automático** - Whisper como respaldo

## 🎯 Uso Rápido

### 1. Iniciar servidor
```bash
python main.py
```

### 2. Abrir interfaz
```bash
# Abrir client_realtime.html en tu navegador
start client_realtime.html
```

### 3. ¡Hablar!
1. Clic en **"🎤 Iniciar"**
2. Habla en inglés o francés
3. Ve la transcripción en tiempo real
4. Clic en **"⏹️ Detener"** para parar

## 📂 Archivos Principales

```
voice-assistant-backend/
├── main.py                 # 🚀 Servidor principal
├── client_realtime.html    # 🌐 Interfaz web simple
├── src/
│   ├── core/
│   │   └── kyutai_real_engine.py  # 🧠 Motor Kyutai STT
│   ├── config/
│   │   └── settings.py            # ⚙️ Configuración
│   └── api/
│       └── stt_routes.py          # 📡 API WebSocket
└── requirements.txt        # 📦 Dependencias
```

## ⚙️ Configuración GPU

El sistema está **optimizado para NVIDIA RTX 4060** con:
- CUDA habilitado por defecto
- Float16 para mejor rendimiento  
- PyTorch optimizations deshabilitadas (evita problemas con Triton)

## 🛠️ Características Técnicas

- **Modelo**: `kyutai/stt-1b-en_fr` (1.2B parámetros)
- **Audio**: 16kHz, mono, con cancelación de eco
- **Streaming**: Chunks de 500ms
- **Conexión**: WebSocket en puerto 8000
- **GPU**: CUDA con precision float16

## 📊 Estado del Sistema

Cuando el servidor inicia correctamente verás:
```
✅ CUDA verificado y funcional
✅ Modelo STT de Kyutai verificado correctamente  
✅ Motor Kyutai STT inicializado correctamente
INFO: Uvicorn running on http://0.0.0.0:8000
```

## 🎮 Atajos de Teclado

- **Ctrl + Espacio**: Iniciar/Detener grabación

## 📝 Logs

El sistema muestra en tiempo real:
- Estado de conexión
- Palabras transcritas  
- Precisión del modelo
- Sesiones completadas

---

**¡Listo para transcribir!** 🎉 
