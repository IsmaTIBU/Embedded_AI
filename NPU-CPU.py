#!/usr/bin/env python3
import cv2
import time
import numpy as np
import tflite_runtime.interpreter as tflite

def test_performance():
    """Comparar rendimiento NPU vs CPU"""
    
    print("🧪 TEST COMPARATIVO NPU vs CPU")
    print("=" * 50)
    
    # Cargar imagen de prueba
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test 1: CPU puro
    print("🧠 Probando CPU...")
    try:
        interpreter_cpu = tflite.Interpreter(
            model_path="best_float16.tflite",
            num_threads=4
        )
        interpreter_cpu.allocate_tensors()
        
        cpu_times = []
        for i in range(10):
            start = time.time()
            
            # Preprocesar
            input_details = interpreter_cpu.get_input_details()[0]
            h, w = input_details['shape'][1:3]
            resized = cv2.resize(dummy_frame, (w, h))
            input_data = np.expand_dims(resized.astype(np.float32) / 255.0, 0)
            
            # Inferencia
            interpreter_cpu.set_tensor(input_details['index'], input_data)
            interpreter_cpu.invoke()
            
            cpu_times.append((time.time() - start) * 1000)
        
        cpu_avg = np.mean(cpu_times[2:])  # Ignorar primeras 2 para warmup
        print(f"   ✅ CPU promedio: {cpu_avg:.1f}ms")
        
    except Exception as e:
        print(f"   ❌ CPU error: {e}")
        cpu_avg = 0
    
    # Test 2: NPU
    print("🚀 Probando NPU...")
    try:
        delegates = [tflite.load_delegate('/usr/lib/libvx_delegate.so')]
        interpreter_npu = tflite.Interpreter(
            model_path="best_float16.tflite",
            experimental_delegates=delegates
        )
        interpreter_npu.allocate_tensors()
        
        npu_times = []
        for i in range(10):
            start = time.time()
            
            # Preprocesar
            input_details = interpreter_npu.get_input_details()[0]
            h, w = input_details['shape'][1:3]
            resized = cv2.resize(dummy_frame, (w, h))
            input_data = np.expand_dims(resized.astype(np.float32) / 255.0, 0)
            
            # Inferencia
            interpreter_npu.set_tensor(input_details['index'], input_data)
            interpreter_npu.invoke()
            
            npu_times.append((time.time() - start) * 1000)
        
        npu_avg = np.mean(npu_times[2:])  # Ignorar primeras 2 para warmup
        print(f"   🚀 NPU promedio: {npu_avg:.1f}ms")
        
    except Exception as e:
        print(f"   ❌ NPU error: {e}")
        npu_avg = 0
    
    # Comparación
    print("\n📊 RESULTADOS:")
    print("=" * 50)
    if cpu_avg > 0 and npu_avg > 0:
        speedup = cpu_avg / npu_avg
        if speedup > 1.1:
            print(f"🚀 NPU es {speedup:.1f}x MÁS RÁPIDA que CPU")
            print(f"   CPU: {cpu_avg:.1f}ms → NPU: {npu_avg:.1f}ms")
            print("   ✅ La NPU SÍ está funcionando")
        elif speedup < 0.9:
            print(f"🧠 CPU es {1/speedup:.1f}x más rápida que NPU")
            print(f"   NPU: {npu_avg:.1f}ms → CPU: {cpu_avg:.1f}ms")
            print("   ⚠️ La NPU no está optimizada para este modelo")
        else:
            print(f"🤝 Rendimiento similar: CPU {cpu_avg:.1f}ms vs NPU {npu_avg:.1f}ms")
            print("   ⚠️ Diferencia mínima - NPU no está acelerando significativamente")
    else:
        print("❌ No se pudo completar la comparación")
    
    print("\n💡 Recomendación:")
    if cpu_avg > 0 and npu_avg > 0 and (cpu_avg / npu_avg) > 1.1:
        print("   Usa el código con NPU - hay mejora de rendimiento")
    else:
        print("   Usa CPU optimizado - NPU no mejora este modelo YOLO")

if __name__ == "__main__":
    test_performance()