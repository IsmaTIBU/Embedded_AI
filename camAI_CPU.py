#!/usr/bin/env python3
"""
Clasificación de imágenes en tiempo real para i.MX 8M Plus con NPU
Basado en tu modelo de 3 clases: Background, Circle, Square
"""
import cv2
import threading
import time
import gi
import numpy as np
import tflite_runtime.interpreter as tflite
import os

gi.require_version('Gtk', '3.0')
gi.require_version('GdkPixbuf', '2.0')
from gi.repository import Gtk, GdkPixbuf, GLib

class LEDClassificationNPU:
    def __init__(self):
        self.cap = None
        self.running = False
        self.interpreter = None
        
        # Configuración del modelo de clasificación
        self.model_config = {
            'model_path': "modelo_convertido.tflite",  # Tu modelo TFLite
            'delegate_path': "/usr/lib/libvx_delegate.so",
            'confidence_threshold': 0.60,  # Umbral de confianza
            'classes': {
                0: 'Background',
                1: 'Circle', 
                2: 'Square'
            }
        }
        
        self.setup_model()
        self.setup_ui()

    def setup_model(self):
        """Configurar modelo SOLO CPU (NPU deshabilitado)"""
        try:
            self.load_cpu()
            self.processing_method = "CPU"
                
        except Exception as e:
            print(f"Error configurando modelo CPU: {e}")
            self.interpreter = None
            self.processing_method = "ERROR"

    def try_load_npu(self):
        """Intentar cargar con delegado NPU"""
        try:
            delegate_path = self.model_config['delegate_path']
            
            if not os.path.exists(delegate_path):
                print(f"NPU delegate not found: {delegate_path}")
                return False
            
            # Cargar delegado NPU
            delegate = tflite.load_delegate(delegate_path)
            
            # Crear intérprete con NPU
            self.interpreter = tflite.Interpreter(
                model_path=self.model_config['model_path'],
                experimental_delegates=[delegate]
            )
            self.interpreter.allocate_tensors()
            
            # Obtener detalles del modelo
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print(f"Model successfully loaded")
            print(f"   Input shape: {self.input_details[0]['shape']}")
            print(f"   Output shape: {self.output_details[0]['shape']}")
            
            # Test de velocidad NPU
            self.test_inference_speed("NPU")
            return True
            
        except Exception as e:
            print(f"Error loading NPU: {e}")
            return False

    def load_cpu(self):
        """Cargar modelo solo CPU"""
        try:
            print("Loading model on CPU...")
            
            self.interpreter = tflite.Interpreter(
                model_path=self.model_config['model_path']
            )
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print(f"Model successfully loaded")
            print(f"   Input shape: {self.input_details[0]['shape']}")
            print(f"   Output shape: {self.output_details[0]['shape']}")
            
            # Test de velocidad CPU
            self.test_inference_speed("CPU")
            
        except Exception as e:
            print(f"Error loading on CPU: {e}")
            self.interpreter = None

    def test_inference_speed(self, method):
        """Probar velocidad de inferencia"""
        if not self.interpreter:
            return
            
        input_shape = self.input_details[0]['shape']
        input_dtype = self.input_details[0]['dtype']
        
        # Crear datos dummy
        if input_dtype == np.uint8:
            dummy_input = np.random.randint(0, 255, input_shape, dtype=np.uint8)
        else:
            dummy_input = np.random.random(input_shape).astype(np.float32)
        
        # Medir tiempo
        times = []
        for _ in range(10):
            start = time.time()
            self.interpreter.set_tensor(self.input_details[0]['index'], dummy_input)
            self.interpreter.invoke()
            times.append((time.time() - start) * 1000)
        
        avg_time = np.mean(times)
        print(f"{method} - Average time: {avg_time:.1f}ms ({1000/avg_time:.1f} FPS)")

    def setup_ui(self):
        """Configurar interfaz gráfica"""
        self.window = Gtk.Window(title="CPU based classifier")
        self.window.set_default_size(800, 600)
        self.window.connect("destroy", self.on_destroy)
        
        # Layout principal
        vbox = Gtk.VBox(spacing=10)
        
        # Widget de imagen
        self.image_widget = Gtk.Image()
        
        # Labels de información
        self.status_label = Gtk.Label(label="Iniciando clasificador...")
        self.classification_label = Gtk.Label()
        self.classification_label.set_markup("<big><b>Clasificación: Esperando...</b></big>")
        
        # Información de confianza
        self.confidence_label = Gtk.Label(label="Confianza: ---%")
        self.performance_label = Gtk.Label(label="Rendimiento: --- ms")
        
        # Agregar widgets
        vbox.pack_start(self.image_widget, True, True, 0)
        vbox.pack_start(self.classification_label, False, False, 5)
        vbox.pack_start(self.confidence_label, False, False, 0)
        vbox.pack_start(self.performance_label, False, False, 0)
        vbox.pack_start(self.status_label, False, False, 5)
        
        self.window.add(vbox)

    def classify_image(self, frame):
        """Clasificar imagen usando el modelo"""
        if not self.interpreter:
            return None
            
        try:
            # Preprocesar imagen según el modelo
            input_shape = self.input_details[0]['shape']
            input_dtype = self.input_details[0]['dtype']
            
            # Redimensionar imagen (height, width según tu modelo: 640x480)
            h, w = input_shape[1], input_shape[2]
            resized_frame = cv2.resize(frame, (w, h))
            
            # Preparar input según el tipo de dato
            if input_dtype == np.uint8:
                # Modelo cuantizado INT8
                input_data = np.expand_dims(resized_frame, axis=0).astype(np.uint8)
            else:
                # Modelo FLOAT32
                input_data = np.expand_dims(resized_frame, axis=0).astype(np.float32) / 255.0
            
            # Realizar inferencia
            start_time = time.time()
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            inference_time = (time.time() - start_time) * 1000
            
            # Obtener resultados
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            
            # Aplicar softmax si es necesario (para logits)
            if np.max(output_data) > 1.0:
                probabilities = self.softmax(output_data)
            else:
                probabilities = output_data
            
            # Encontrar clase con mayor probabilidad
            predicted_class_id = np.argmax(probabilities)
            confidence = probabilities[predicted_class_id]
            
            return {
                'class_id': predicted_class_id,
                'class_name': self.model_config['classes'].get(predicted_class_id, 'Unknown'),
                'confidence': confidence,
                'probabilities': probabilities,
                'inference_time': inference_time
            }
            
        except Exception as e:
            print(f"Classifying error: {e}")
            return None

    def softmax(self, x):
        """Aplicar softmax para convertir logits en probabilidades"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def start_camera(self):
        """Inicializar y comenzar captura de cámara"""        
        # Probar diferentes dispositivos de video
        for device in ['/dev/video3', '/dev/video0', 0, 1, 2]:
            try:
                self.cap = cv2.VideoCapture(device)
                if self.cap.isOpened():
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        print(f"Camera found: {device}")
                        break
                self.cap.release()
            except:
                continue
        
        if not self.cap or not self.cap.isOpened():
            self.status_label.set_text("No camera found")
            return
        
        # Configurar cámara
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera frame shape: {actual_w}x{actual_h}")
        
        self.running = True
        threading.Thread(target=self.capture_loop, daemon=True).start()
        
        # Actualizar status
        method_text = f"{self.processing_method}"  # Siempre CPU ahora
        self.status_label.set_text(f"{method_text}")

    def capture_loop(self):
        """Loop principal de captura y clasificación"""
        frame_count = 0
        fps_start_time = time.time()
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Error leyendo frame")
                break
            
            # Clasificar imagen
            result = self.classify_image(frame)
            
            if result:
                # Dibujar información en el frame
                self.draw_classification_info(frame, result)
                
                # Log cada 30 frames
                if frame_count % 30 == 0:
                    elapsed = time.time() - fps_start_time
                    fps = 30 / elapsed if elapsed > 0 else 0
                    
                    print(f"Frame {frame_count}: {result['class_name']} "
                          f"({result['confidence']:.1%}) - {result['inference_time']:.1f}ms - {fps:.1f}FPS")
                    
                    fps_start_time = time.time()
                
                # Actualizar UI en hilo principal
                GLib.idle_add(self.update_ui, frame, result)
            
            frame_count += 1
            time.sleep(0.01)  # Control de FPS

    def draw_classification_info(self, frame, result):
        """Dibujar información de clasificación en el frame"""
        # Color según la clase
        colors = {
            0: (128, 128, 128),  # Background - Gris
            1: (0, 255, 0),      # Circle - Verde  
            2: (0, 0, 255)       # Square - Rojo
        }
        
        color = colors.get(result['class_id'], (255, 255, 255))
        
        # Texto principal
        main_text = f"{result['class_name']}"
        confidence_text = f"{result['confidence']:.1%}"
        
        # Dibujar fondo para el texto
        cv2.rectangle(frame, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 120), color, 2)
        
        # Texto principal
        cv2.putText(frame, main_text, (20, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        
        # Confianza
        cv2.putText(frame, confidence_text, (20, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Método de procesamiento
        method_text = f"{self.processing_method} - {result['inference_time']:.0f}ms"
        cv2.putText(frame, method_text, (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Barra de confianza
        bar_width = 200
        bar_height = 10
        bar_x, bar_y = 20, 110
        
        # Fondo de barra
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (50, 50, 50), -1)
        
        # Barra de confianza
        conf_width = int(bar_width * result['confidence'])
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + conf_width, bar_y + bar_height), 
                     color, -1)

    def update_ui(self, frame, result):
        """Actualizar interfaz gráfica"""
        try:
            # Convertir frame para GTK
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, c = frame_rgb.shape
            pixbuf = GdkPixbuf.Pixbuf.new_from_data(
                frame_rgb.tobytes(), GdkPixbuf.Colorspace.RGB, 
                False, 8, w, h, w * c
            )
            
            # Actualizar imagen
            self.image_widget.set_from_pixbuf(pixbuf)
            
            # Actualizar labels
            class_text = f"<big><b>Classification: {result['class_name']}</b></big>"
            self.classification_label.set_markup(class_text)
            
            self.confidence_label.set_text(f"Confidence: {result['confidence']:.1%}")
            self.performance_label.set_text(f"Perfomance: {result['inference_time']:.1f}ms ({self.processing_method})")
            
            # Mostrar probabilidades de todas las clases
            prob_text = " | ".join([
                f"{self.model_config['classes'][i]}: {prob:.1%}" 
                for i, prob in enumerate(result['probabilities'])
            ])
            self.status_label.set_text(f"Probabilities: {prob_text}")
            
        except Exception as e:
            print(f"Error actualizando UI: {e}")

    def on_destroy(self, widget):
        """Limpiar recursos al cerrar"""
        print("Closing program")
        self.running = False
        if self.cap:
            self.cap.release()
        Gtk.main_quit()

    def run(self):
        """Ejecutar aplicación"""
        self.window.show_all()
        GLib.idle_add(self.start_camera)
        print("Running program...")
        Gtk.main()

if __name__ == "__main__":
    print("=" * 50)
    print("CPU CLASSIFIER - i.MX 8M Plus")
    print("   Background/Circle/Square Detection")
    print("=" * 50)
    
    try:
        app = LEDClassificationNPU()
        app.run()
    except KeyboardInterrupt:
        print("\nKeyboard interruption")
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        print("Closing program")