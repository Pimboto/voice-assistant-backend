# check_real_moshi_api.py
"""
Script para encontrar la API REAL de Moshi según la documentación
NO más adivinanzas, vamos a ver qué hay realmente
"""

import sys
import inspect
import importlib.util

print("🔍 BUSCANDO LA API REAL DE MOSHI")
print("="*60)

# 1. Verificar qué versión de moshi tenemos
try:
    import moshi
    print(f"✅ Moshi instalado")
    print(f"   Versión: {getattr(moshi, '__version__', 'no especificada')}")
    print(f"   Ubicación: {moshi.__file__}")
    print(f"   Paquete: {moshi.__package__}")
except ImportError as e:
    print(f"❌ Moshi no instalado: {e}")
    sys.exit(1)

# 2. Buscar específicamente las funciones mencionadas en el Colab
print("\n📓 Buscando funciones del notebook stt_pytorch.ipynb...")

# Según el patrón típico de Colab notebooks, deberían existir:
expected_imports = [
    "from moshi import load_model",
    "from moshi.models import something",  
    "from moshi.inference import run",
    "from moshi import STTModel",
]

# Verificar cada posible import
for imp in expected_imports:
    try:
        # Simular el import
        parts = imp.replace("from ", "").replace(" import ", ".").split(".")
        module_path = parts[0]
        
        if len(parts) > 1:
            # Es un import específico
            module = __import__(module_path, fromlist=[parts[-1]])
            if hasattr(module, parts[-1]):
                print(f"✅ {imp} - EXISTE")
            else:
                print(f"❌ {imp} - NO existe")
        else:
            print(f"✅ {module_path} importable")
    except Exception as e:
        print(f"❌ {imp} - Error: {type(e).__name__}")

# 3. Explorar todo lo que hay en moshi
print("\n📦 Contenido COMPLETO de moshi:")

def explore_recursive(obj, name, level=0, max_level=2):
    """Explorar recursivamente un módulo"""
    if level > max_level:
        return
    
    indent = "  " * level
    
    # Solo explorar atributos públicos
    attrs = [a for a in dir(obj) if not a.startswith('_')]
    
    for attr in attrs:
        try:
            value = getattr(obj, attr)
            
            # Clasificar el tipo
            if inspect.ismodule(value):
                print(f"{indent}{attr}/ (módulo)")
                # Explorar submódulos de moshi solamente
                if 'moshi' in str(value):
                    explore_recursive(value, f"{name}.{attr}", level + 1)
                    
            elif inspect.isclass(value):
                print(f"{indent}{attr} [clase]")
                # Mostrar métodos importantes
                methods = [m for m in dir(value) if not m.startswith('_') and m in ['from_pretrained', 'load', 'forward', 'transcribe', 'generate']]
                if methods:
                    print(f"{indent}  → métodos: {', '.join(methods)}")
                    
            elif inspect.isfunction(value):
                print(f"{indent}{attr}() [función]")
                # Mostrar signatura
                try:
                    sig = inspect.signature(value)
                    params = list(sig.parameters.keys())
                    if params:
                        print(f"{indent}  → params: {', '.join(params[:3])}{'...' if len(params) > 3 else ''}")
                except:
                    pass
                    
        except Exception as e:
            print(f"{indent}{attr} (error: {type(e).__name__})")

print("\nmoshi/")
explore_recursive(moshi, "moshi")

# 4. Buscar específicamente patrones de STT
print("\n🎤 Buscando específicamente componentes STT:")

stt_patterns = ['stt', 'speech', 'transcribe', 'asr', 'STT', 'Speech']

for pattern in stt_patterns:
    print(f"\nPatrón '{pattern}':")
    found = False
    
    # Buscar en todo moshi
    for attr in dir(moshi):
        if pattern.lower() in attr.lower():
            print(f"  ✅ moshi.{attr}")
            found = True
    
    # Buscar en submódulos
    for submodule in ['models', 'inference', 'utils']:
        if hasattr(moshi, submodule):
            sub = getattr(moshi, submodule)
            for attr in dir(sub):
                if pattern.lower() in attr.lower():
                    print(f"  ✅ moshi.{submodule}.{attr}")
                    found = True
    
    if not found:
        print(f"  ❌ No se encontró nada con '{pattern}'")

# 5. Verificar si hay ejemplos en el paquete
print("\n📄 Buscando archivos de ejemplo:")

import os
moshi_path = os.path.dirname(moshi.__file__)

# Buscar archivos relevantes
for root, dirs, files in os.walk(moshi_path):
    # No profundizar demasiado
    depth = root.replace(moshi_path, '').count(os.sep)
    if depth > 2:
        continue
        
    for file in files:
        if any(word in file.lower() for word in ['example', 'demo', 'test', 'stt', 'inference']):
            rel_path = os.path.relpath(os.path.join(root, file), moshi_path)
            print(f"  - {rel_path}")

# 6. Intentar el patrón más común
print("\n🧪 Probando patrones comunes de uso:")

# Patrón 1: HuggingFace style
try:
    print("\n1. Patrón HuggingFace (from_pretrained):")
    
    # Buscar clases con from_pretrained
    for attr in dir(moshi):
        obj = getattr(moshi, attr)
        if inspect.isclass(obj) and hasattr(obj, 'from_pretrained'):
            print(f"  ✅ {attr}.from_pretrained() disponible")
            
            # Intentar usarlo
            try:
                print(f"     Intentando cargar kyutai/stt-1b-en_fr...")
                model = obj.from_pretrained("kyutai/stt-1b-en_fr")
                print(f"     ✅ ¡ÉXITO! Modelo cargado: {type(model)}")
                
                # Ver métodos del modelo
                methods = [m for m in dir(model) if not m.startswith('_') and callable(getattr(model, m))]
                print(f"     Métodos disponibles: {', '.join(methods[:10])}...")
                
            except Exception as e:
                print(f"     ❌ Error: {type(e).__name__}: {str(e)[:100]}")
                
except Exception as e:
    print(f"  ❌ Error general: {e}")

# Patrón 2: load_model style
try:
    print("\n2. Patrón load_model:")
    
    # Buscar funciones load
    for attr in dir(moshi):
        if 'load' in attr and callable(getattr(moshi, attr)):
            func = getattr(moshi, attr)
            print(f"  ✅ moshi.{attr}() encontrado")
            
            # Ver signatura
            try:
                sig = inspect.signature(func)
                print(f"     Signatura: {sig}")
            except:
                pass
                
except Exception as e:
    print(f"  ❌ Error: {e}")

# 7. Resumen final
print("\n" + "="*60)
print("📊 RESUMEN:")
print("="*60)

# Contar lo que encontramos
module_count = sum(1 for a in dir(moshi) if inspect.ismodule(getattr(moshi, a)))
class_count = sum(1 for a in dir(moshi) if inspect.isclass(getattr(moshi, a)))
func_count = sum(1 for a in dir(moshi) if inspect.isfunction(getattr(moshi, a)))

print(f"Módulos: {module_count}")
print(f"Clases: {class_count}")
print(f"Funciones: {func_count}")

print("\n💡 SIGUIENTE PASO:")
print("Basado en lo encontrado, actualiza kyutai_real_engine.py")
print("para usar las funciones/clases REALES que existen.")
