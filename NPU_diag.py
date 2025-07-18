#!/usr/bin/env python3
import tflite_runtime.interpreter as tflite
import os
import glob

print("🔍 DIAGNÓSTICO NPU")
print("=" * 50)

# 1. Verificar delegados disponibles
delegate_paths = [
    '/usr/lib/libvx_delegate.so',
    '/usr/lib/aarch64-linux-gnu/libvx_delegate.so',
    '/opt/imx-gpu-viv/libvx_delegate.so',
    '/usr/local/lib/libvx_delegate.so'
]

print("📁 Buscando delegados NPU:")
for path in delegate_paths:
    exists = os.path.exists(path)
    print(f"   {path}: {'✅' if exists else '❌'}")

# 2. Buscar archivos vx_delegate
print("\n🔍 Buscando archivos *vx_delegate*:")
for pattern in ['/usr/lib/*vx_delegate*', '/opt/*vx_delegate*', '/lib/*vx_delegate*']:
    files = glob.glob(pattern)
    for f in files:
        print(f"   ✅ Encontrado: {f}")

# 3. Probar carga de delegado
print("\n🧪 Probando carga de delegado:")
for path in delegate_paths:
    if os.path.exists(path):
        try:
            delegate = tflite.load_delegate(path)
            print(f"   ✅ {path}: FUNCIONA")
            break
        except Exception as e:
            print(f"   ❌ {path}: ERROR - {e}")
else:
    print("   ❌ Ningún delegado NPU funciona")

# 4. Verificar modelos
print("\n📦 Verificando modelos:")
models = ['best_float16.tflite', 'best_float32.tflite']
for model in models:
    if os.path.exists(model):
        try:
            # Probar con CPU
            interp_cpu = tflite.Interpreter(model_path=model)
            interp_cpu.allocate_tensors()
            print(f"   ✅ {model}: CPU OK")
            
            # Probar con NPU (si está disponible)
            for path in delegate_paths:
                if os.path.exists(path):
                    try:
                        delegate = tflite.load_delegate(path)
                        interp_npu = tflite.Interpreter(
                            model_path=model,
                            experimental_delegates=[delegate]
                        )
                        interp_npu.allocate_tensors()
                        print(f"   🚀 {model}: NPU OK con {path}")
                        break
                    except Exception as e:
                        print(f"   ⚠️ {model}: NPU falla con {path} - {e}")
            
        except Exception as e:
            print(f"   ❌ {model}: ERROR - {e}")
    else:
        print(f"   ❌ {model}: No encontrado")

print("\n" + "=" * 50)
print("💡 Si no hay delegados NPU disponibles:")
print("   - Tu modelo funcionará solo en CPU")
print("   - Esto es normal en algunas configuraciones")
print("   - El código seguirá funcionando, pero más lento")