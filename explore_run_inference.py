# explore_run_inference.py
"""
Vamos a ver QUÉ HAY REALMENTE en run_inference.py
que es el ÚNICO archivo relacionado con inferencia que existe
"""

import sys
import inspect

print("🔍 EXPLORANDO run_inference.py - EL ARCHIVO REAL")
print("="*60)

try:
    # Importar run_inference
    from moshi import run_inference
    
    print("✅ run_inference importado\n")
    
    # Ver TODO lo que contiene
    print("📋 Contenido COMPLETO de run_inference:")
    print("-"*40)
    
    all_items = []
    
    for name in dir(run_inference):
        if not name.startswith('__'):
            obj = getattr(run_inference, name)
            obj_type = type(obj).__name__
            
            # Clasificar
            if inspect.isfunction(obj):
                category = "FUNCIÓN"
            elif inspect.isclass(obj):
                category = "CLASE"
            elif inspect.ismodule(obj):
                category = "MÓDULO"
            else:
                category = "VARIABLE"
            
            all_items.append((category, name, obj, obj_type))
    
    # Mostrar por categorías
    for category in ["FUNCIÓN", "CLASE", "MÓDULO", "VARIABLE"]:
        items = [item for item in all_items if item[0] == category]
        if items:
            print(f"\n{category}ES:")
            for _, name, obj, obj_type in items:
                print(f"  - {name} ({obj_type})")
                
                # Para funciones, mostrar signatura
                if category == "FUNCIÓN":
                    try:
                        sig = inspect.signature(obj)
                        print(f"    Signatura: {sig}")
                        
                        # Ver docstring
                        if obj.__doc__:
                            first_line = obj.__doc__.strip().split('\n')[0]
                            print(f"    Doc: {first_line[:60]}...")
                    except:
                        pass
                
                # Para clases, mostrar métodos principales
                elif category == "CLASE":
                    methods = [m for m in dir(obj) if not m.startswith('_')]
                    if methods:
                        print(f"    Métodos: {', '.join(methods[:5])}")
    
    # Buscar la función main
    print("\n" + "="*60)
    print("🎯 BUSCANDO FUNCIÓN PRINCIPAL:")
    
    if hasattr(run_inference, 'main'):
        print("✅ Función main() encontrada!")
        
        # Ver signatura
        try:
            sig = inspect.signature(run_inference.main)
            print(f"   Signatura: {sig}")
        except:
            pass
        
        # Ver código si es posible
        try:
            source = inspect.getsource(run_inference.main)
            print("\n📄 Primeras líneas del código de main():")
            lines = source.split('\n')[:20]
            for i, line in enumerate(lines):
                print(f"   {i+1:2d}: {line}")
        except:
            print("   No se puede ver el código fuente")
    
    # Ver si es ejecutable
    print("\n" + "="*60)
    print("🏃 FORMA DE EJECUTAR:")
    
    if hasattr(run_inference, '__file__'):
        print(f"Archivo: {run_inference.__file__}")
        print(f"\nEjecución por CLI:")
        print(f"  python -m moshi.run_inference --help")
    
    # Buscar patrones de uso
    print("\n" + "="*60)
    print("🔎 PATRONES DE USO:")
    
    # Buscar funciones relacionadas con cargar modelos
    load_functions = [name for name in dir(run_inference) 
                     if 'load' in name.lower() or 'model' in name.lower()]
    
    if load_functions:
        print("\nFunciones relacionadas con cargar modelos:")
        for func_name in load_functions:
            print(f"  - {func_name}")
    
    # Buscar funciones de transcripción
    transcribe_functions = [name for name in dir(run_inference)
                           if any(word in name.lower() for word in ['transcribe', 'inference', 'process', 'run'])]
    
    if transcribe_functions:
        print("\nFunciones relacionadas con transcripción:")
        for func_name in transcribe_functions:
            print(f"  - {func_name}")
            
except ImportError as e:
    print(f"❌ Error importando run_inference: {e}")
    
    # Intentar importarlo de otra forma
    print("\nIntentando importación alternativa...")
    
    try:
        import moshi.run_inference
        print("✅ Importado como moshi.run_inference")
        
        # Repetir exploración básica
        print("\nContenido:")
        for attr in dir(moshi.run_inference):
            if not attr.startswith('_'):
                print(f"  - {attr}")
                
    except Exception as e2:
        print(f"❌ También falló: {e2}")

print("\n" + "="*60)
print("💡 CONCLUSIÓN:")
print("run_inference.py es el punto de entrada para usar Kyutai STT")
print("Debemos usar las funciones que encontramos aquí")
