#!/usr/bin/env python3
import cv2
import threading
import time
import gi
import numpy as np
import tflite_runtime.interpreter as tflite

gi.require_version('Gtk', '3.0')
gi.require_version('GdkPixbuf', '2.0')
from gi.repository import Gtk, GdkPixbuf, GLib

class LEDDetectionSSD:
    def __init__(self):
        self.cap = None
        self.running = False
        self.interpreter = None
        
        # Configuración para SSD MobileNet
        self.model_config = {
            'model_path': "modelo_cuantizado_estandar.tflite",  # ← NUESTRO MODELO
            'delegate_path': "/usr/lib/libvx_delegate.so",
            'score_threshold': 0.3,
            'nms_threshold': 0.5,
            # COCO classes - las que detecta nuestro modelo
            'classes': {
                1: 'square', 2: 'circle'
            }
        }
        
        self.setup_model()
        self.setup_ui()
    
    def setup_model(self):
        """Configurar modelo SSD MobileNet con NPU"""
        try:
            print("🔄 Inicializando modelo SSD MobileNet con NPU...")
            
            # Cargar delegado NPU
            delegates = []
            try:
                delegates = [tflite.load_delegate(self.model_config['delegate_path'])]
                print("✅ NPU delegado cargado")
            except Exception as e:
                print(f"⚠️ NPU no disponible, usando CPU: {e}")
            
            # Crear intérprete
            self.interpreter = tflite.Interpreter(
                model_path=self.model_config['model_path'],
                experimental_delegates=delegates
            )
            self.interpreter.allocate_tensors()
            
            # Obtener detalles de entrada y salida
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print(f"🚀 Modelo SSD inicializado:")
            print(f"   📥 Entrada: {self.input_details[0]['shape']} ({self.input_details[0]['dtype']})")
            print(f"   📤 Salidas: {len(self.output_details)} outputs")
            
            # Mostrar detalles de las salidas
            for i, output in enumerate(self.output_details):
                print(f"   📤 Output {i}: {output['shape']} - {output['name']}")
            
        except Exception as e:
            print(f"❌ Error inicializando modelo: {e}")
            self.interpreter = None
    
    def setup_ui(self):
        """UI simple"""
        self.window = Gtk.Window(title="Object Detection - SSD MobileNet NPU")
        self.window.set_default_size(800, 600)
        self.window.connect("destroy", self.on_destroy)
        
        vbox = Gtk.VBox(spacing=10)
        self.image_widget = Gtk.Image()
        self.status_label = Gtk.Label(label="Iniciando SSD MobileNet...")
        
        vbox.pack_start(self.image_widget, True, True, 0)
        vbox.pack_start(self.status_label, False, False, 5)
        self.window.add(vbox)
    
    def detect_objects(self, frame):
        """Detectar objetos con SSD MobileNet"""
        if not self.interpreter:
            return []
        
        # Preprocesar imagen
        input_shape = self.input_details[0]['shape']
        h, w = input_shape[1], input_shape[2]
        
        resized = cv2.resize(frame, (w, h))
        
        # Preparar entrada según tipo
        if self.input_details[0]['dtype'] == np.uint8:
            input_data = np.expand_dims(resized, axis=0).astype(np.uint8)
        else:
            input_data = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)
        
        # Inferencia
        start_time = time.time()
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        inference_time = (time.time() - start_time) * 1000
        
        # Obtener resultados SSD
        # Típicamente SSD tiene 4 salidas: boxes, classes, scores, num_detections
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        num_detections = int(self.interpreter.get_tensor(self.output_details[3]['index'])[0])
        
        # Procesar detecciones
        detections = []
        frame_h, frame_w = frame.shape[:2]
        
        for i in range(min(num_detections, len(boxes))):
            score = scores[i]
            if score > self.model_config['score_threshold']:
                # SSD boxes están en formato [y1, x1, y2, x2] normalizado
                y1, x1, y2, x2 = boxes[i]
                
                # Convertir a coordenadas de píxeles
                x1 = int(x1 * frame_w)
                y1 = int(y1 * frame_h)
                x2 = int(x2 * frame_w)
                y2 = int(y2 * frame_h)
                
                # Clamp
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame_w, x2), min(frame_h, y2)
                
                # Obtener nombre de clase
                class_id = int(classes[i])
                class_name = self.model_config['classes'].get(class_id, f'class_{class_id}')
                
                detections.append({
                    'bbox': (x1, y1, x2-x1, y2-y1),
                    'class': class_name,
                    'class_id': class_id,
                    'conf': score,
                    'inference_time': inference_time
                })
        
        return detections
    
    def start_camera(self):
        """Iniciar cámara"""
        for device in ['/dev/video3', '/dev/video0', 0, 1]:
            self.cap = cv2.VideoCapture(device)
            if self.cap.isOpened() and self.cap.read()[0]:
                break
            if self.cap:
                self.cap.release()
        
        if not self.cap or not self.cap.isOpened():
            self.status_label.set_text("❌ No hay cámara")
            return
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.running = True
        threading.Thread(target=self.capture_loop, daemon=True).start()
        
        method = "🚀 NPU" if len(self.interpreter.get_signature_list()) > 0 else "💻 CPU"
        self.status_label.set_text(f"{method} SSD MobileNet - Detectando...")
    
    def capture_loop(self):
        """Loop principal"""
        frame_count = 0
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Detectar objetos
            detections = self.detect_objects(frame)
            inference_time = detections[0]['inference_time'] if detections else 0
            
            # Dibujar detecciones
            for det in detections:
                x, y, w, h = det['bbox']
                
                # Colores según clase
                if 'person' in det['class']:
                    color = (0, 255, 0)  # Verde para personas
                elif 'car' in det['class'] or 'truck' in det['class']:
                    color = (0, 0, 255)  # Rojo para vehículos
                elif 'bottle' in det['class'] or 'cup' in det['class']:
                    color = (255, 0, 0)  # Azul para objetos pequeños
                else:
                    color = (255, 255, 0)  # Amarillo para otros
                
                # Dibujar bbox
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Etiqueta
                label = f"{det['class']} {det['conf']:.2f}"
                cv2.putText(frame, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Log performance cada 30 frames
            if frame_count % 30 == 0:
                fps = 1000 / inference_time if inference_time > 0 else 0
                print(f"🚀 Frame {frame_count}: {len(detections)} objetos, {inference_time:.1f}ms, {fps:.1f}FPS")
                
                # Mostrar objetos detectados
                if detections:
                    objects = [d['class'] for d in detections]
                    print(f"   📦 Detectados: {', '.join(set(objects))}")
            
            # Actualizar UI
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, c = frame_rgb.shape
            pixbuf = GdkPixbuf.Pixbuf.new_from_data(
                frame_rgb.tobytes(), GdkPixbuf.Colorspace.RGB, False, 8, w, h, w*c)
            
            # Status con objetos detectados
            unique_objects = list(set(d['class'] for d in detections))
            status = f"📦 {len(detections)} objetos | 🚀 {inference_time:.1f}ms"
            if unique_objects:
                status += f" | {', '.join(unique_objects[:3])}"
            
            GLib.idle_add(lambda: (
                self.image_widget.set_from_pixbuf(pixbuf),
                self.status_label.set_text(status)
            ))
            
            frame_count += 1
            time.sleep(0.033)  # ~30 FPS
    
    def on_destroy(self, widget):
        self.running = False
        if self.cap:
            self.cap.release()
        Gtk.main_quit()
    
    def run(self):
        self.window.show_all()
        GLib.idle_add(self.start_camera)
        Gtk.main()

if __name__ == "__main__":
    LEDDetectionSSD().run()