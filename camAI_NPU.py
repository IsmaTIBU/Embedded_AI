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

class LEDDetectionNPU:
    def __init__(self):
        self.cap = None
        self.running = False
        self.interpreter = None
        
        # Configuración corregida según diagnóstico
        self.model_config = {
            'model_path': "best_float16.tflite",  # ← CONFIRMADO que funciona
            'delegate_path': "/usr/lib/libvx_delegate.so",  # ← CONFIRMADO que existe
            'score_threshold': 0.3,
            'nms_threshold': 0.5,
            'classes': ['Circle', 'Square']
        }
        
        self.setup_model()
        self.setup_ui()
    
    def setup_model(self):
        """Configurar modelo con NPU - Usando rutas confirmadas"""
        try:
            print("🔄 Inicializando modelo LED con NPU...")
            
            # Cargar delegado NPU con ruta confirmada
            delegates = []
            try:
                delegates = [tflite.load_delegate(self.model_config['delegate_path'])]
                print("✅ NPU delegado cargado desde ruta confirmada")
            except Exception as e:
                print(f"❌ Error cargando NPU: {e}")
                return
            
            # Crear intérprete con NPU
            self.interpreter = tflite.Interpreter(
                model_path=self.model_config['model_path'],
                experimental_delegates=delegates
            )
            self.interpreter.allocate_tensors()
            
            # Obtener detalles
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print(f"🚀 Modelo NPU inicializado: {self.input_details[0]['shape']}")
            print(f"🎯 Entrada: {self.input_details[0]['dtype']}")
            print(f"🎯 Salida: {self.output_details[0]['shape']}")
            
        except Exception as e:
            print(f"❌ Error inicializando modelo NPU: {e}")
            self.interpreter = None
    
    def setup_ui(self):
        """UI simple"""
        self.window = Gtk.Window(title="LED Detection - NPU Confirmed")
        self.window.set_default_size(800, 600)
        self.window.connect("destroy", self.on_destroy)
        
        vbox = Gtk.VBox(spacing=10)
        self.image_widget = Gtk.Image()
        self.status_label = Gtk.Label(label="Iniciando NPU...")
        
        vbox.pack_start(self.image_widget, True, True, 0)
        vbox.pack_start(self.status_label, False, False, 5)
        self.window.add(vbox)
    
    def detect_leds(self, frame):
        """Detectar LEDs con NPU confirmada"""
        if not self.interpreter:
            return []
        
        # Preprocesar para float16
        input_shape = self.input_details[0]['shape']
        h, w = input_shape[1], input_shape[2]
        
        resized = cv2.resize(frame, (w, h))
        
        # Preparar entrada según tipo detectado
        if self.input_details[0]['dtype'] == np.uint8:
            input_data = np.expand_dims(resized, axis=0).astype(np.uint8)
        else:
            input_data = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)
        
        # Inferencia NPU
        start_time = time.time()
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        inference_time = (time.time() - start_time) * 1000
        
        # Obtener resultados
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        # Decodificar YOLO
        detections = []
        frame_h, frame_w = frame.shape[:2]
        
        for detection in output_data:
            if len(detection) >= 6:
                x, y, w, h, confidence = detection[:5]
                class_scores = detection[5:7]  # Solo 2 clases
                
                if confidence > self.model_config['score_threshold']:
                    class_id = np.argmax(class_scores)
                    class_conf = class_scores[class_id]
                    
                    if class_conf > self.model_config['score_threshold']:
                        # Convertir coordenadas
                        x1 = int((x - w/2) * frame_w)
                        y1 = int((y - h/2) * frame_h)
                        x2 = int((x + w/2) * frame_w)
                        y2 = int((y + h/2) * frame_h)
                        
                        # Clamp
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame_w, x2), min(frame_h, y2)
                        
                        detections.append({
                            'bbox': (x1, y1, x2-x1, y2-y1),
                            'class': self.model_config['classes'][class_id],
                            'conf': confidence * class_conf,
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
        
        method = "🚀 NPU CONFIRMADA" if self.interpreter else "❌ CPU Fallback"
        self.status_label.set_text(f"{method} - Detectando...")
    
    def capture_loop(self):
        """Loop principal"""
        frame_count = 0
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Detectar con NPU
            detections = self.detect_leds(frame)
            inference_time = detections[0]['inference_time'] if detections else 0
            
            # Dibujar
            for det in detections:
                x, y, w, h = det['bbox']
                color = (0, 255, 0) if det['class'] == 'Circle' else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                cv2.putText(frame, f"{det['class']} {det['conf']:.2f}", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Log performance
            if frame_count % 30 == 0:
                fps = 1000 / inference_time if inference_time > 0 else 0
                print(f"🚀 NPU Frame {frame_count}: {len(detections)} LEDs, {inference_time:.1f}ms, {fps:.1f}FPS")
            
            # UI
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, c = frame_rgb.shape
            pixbuf = GdkPixbuf.Pixbuf.new_from_data(
                frame_rgb.tobytes(), GdkPixbuf.Colorspace.RGB, False, 8, w, h, w*c)
            
            circles = sum(1 for d in detections if d['class'] == 'Circle')
            squares = sum(1 for d in detections if d['class'] == 'Square')
            
            status = f"🔴 {circles} | 🔶 {squares} | 🚀 NPU: {inference_time:.1f}ms"
            
            GLib.idle_add(lambda: (
                self.image_widget.set_from_pixbuf(pixbuf),
                self.status_label.set_text(status)
            ))
            
            frame_count += 1
            time.sleep(0.033)
    
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
    LEDDetectionNPU().run()