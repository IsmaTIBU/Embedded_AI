#!/usr/bin/env python3
import cv2
import time
import numpy as np
import tflite_runtime.interpreter as tflite

def test_performance():
    """Comparar rendimiento NPU vs CPU"""
    
    print("NPU vs CPU test")
    print("=" * 50)
    
    # Cargar imagen de prueba
    dummy_frame = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    
    # Test 1: CPU puro
    print("Testing CPU")
    try:
        interpreter_cpu = tflite.Interpreter(
            model_path="modelo_convertido.tflite",
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
        print(f"   CPU average: {cpu_avg:.1f}ms")
        
    except Exception as e:
        print(f"   CPU error: {e}")
        cpu_avg = 0
    
    # Test 2: NPU
    print("Testing NPU")
    try:
        delegates = [tflite.load_delegate('/usr/lib/libvx_delegate.so')]
        interpreter_npu = tflite.Interpreter(
            model_path="modelo_convertido.tflite",
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
        print(f"   NPU average: {npu_avg:.1f}ms")
        
    except Exception as e:
        print(f"   NPU error: {e}")
        npu_avg = 0
    
    # Comparación
    print("\nRESULTS:")
    print("=" * 50)
    if cpu_avg > 0 and npu_avg > 0:
        speedup = cpu_avg / npu_avg
        if speedup > 1.1:
            print(f"NPU is {speedup:.1f}x faster than CPU")
            print(f"   CPU: {cpu_avg:.1f}ms → NPU: {npu_avg:.1f}ms")
            print("   NPU running efficiently")
        elif speedup < 0.9:
            print(f"CPU is {1/speedup:.1f}x faster than NPU")
            print(f"   NPU: {npu_avg:.1f}ms → CPU: {cpu_avg:.1f}ms")
            print("   NPU not optimized for this model")
        else:
            print(f"Similar performance: CPU {cpu_avg:.1f}ms vs NPU {npu_avg:.1f}ms")
            print("   NPU not optimized for this model")
    else:
        print("Couldn' complete the tests")
    
    print("\nConclusion:")
    if cpu_avg > 0 and npu_avg > 0 and (cpu_avg / npu_avg) > 1.1:
        print("   Use the code based on NPU for better performance")
    else:
        print("   Use the code based on CPU for better performance")

if __name__ == "__main__":
    test_performance()