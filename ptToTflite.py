import torch
from ultralytics import YOLO
import tensorflow as tf
import os

# Forzar torch.load sin weights_only
torch.serialization.weights_only = False

# Cargar modelo
model = YOLO('best.pt')

# Exportar a ONNX (más compatible)
model.export(format='onnx', imgsz=640)

# Usar el método directo de Ultralytics para TFLite
print("🔄 Intentando conversión directa a TFLite...")
try:
    model.export(format='tflite', imgsz=640, int8=False)
    print("✅ Conversión directa exitosa")
except Exception as e:
    print(f"❌ Conversión directa falló: {e}")
    
    # Método alternativo: usar tf2onnx
    print("🔄 Probando método alternativo...")
    try:
        import subprocess
        
        # Instalar tf2onnx si no está
        subprocess.run(["pip", "install", "tf2onnx"], capture_output=True)
        
        # Convertir ONNX a TensorFlow
        subprocess.run([
            "python", "-m", "tf2onnx.convert", 
            "--onnx", "best.onnx",
            "--output", "best_tf.pb"
        ], capture_output=True)
        
        print("⚠️ Método alternativo complejo - usar ONNX directamente")
        
    except Exception as e2:
        print(f"❌ Método alternativo falló: {e2}")

# Verificar archivos generados
print("\n📁 Archivos generados:")
files = ['best.onnx', 'best.tflite']
for file in files:
    if os.path.exists(file):
        size = os.path.getsize(file) / 1024 / 1024
        print(f"   ✅ {file}: {size:.1f} MB")
    else:
        print(f"   ❌ {file}: No generado")

print("\n💡 Recomendación: Si TFLite no se genera, usa best.onnx")
print("   ONNX funciona excelente en placas embedded con onnxruntime")

# Información adicional
if os.path.exists('best.onnx'):
    print(f"\n🎯 Para usar ONNX en tu placa 8MPLUS-BB:")
    print(f"   1. Instala onnxruntime en la placa")
    print(f"   2. Usa best.onnx como modelo")
    print(f"   3. ONNX suele ser más rápido que TFLite")