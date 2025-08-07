#!/usr/bin/env python3
import numpy as np
import tflite_runtime.interpreter as tflite

def test_basic_inference():
    model_path = "detection-balanced-npu.tflite"
    
    try:
        # SIN delegado para depurar
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Crear entrada dummy
        input_shape = input_details[0]['shape']
        print(f"🔄 Creando entrada dummy: {input_shape}")
        
        if input_details[0]['dtype'] == np.uint8:
            dummy_input = np.random.randint(0, 255, input_shape, dtype=np.uint8)
        else:
            dummy_input = np.random.random(input_shape).astype(np.float32)
        
        # Inferencia
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        
        # Obtener TODAS las salidas
        print(f"🚀 Inferencia exitosa!")
        for i, output in enumerate(output_details):
            result = interpreter.get_tensor(output['index'])
            print(f"   Salida[{i}]: {result.shape} - Rango: [{result.min():.3f}, {result.max():.3f}]")
            
            # Mostrar algunos valores
            if result.size < 20:
                print(f"   Valores: {result.flatten()[:10]}")
        
    except Exception as e:
        print(f"❌ Error en inferencia: {e}")

if __name__ == "__main__":
    test_basic_inference()