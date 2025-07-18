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

class LEDDetectionCPU:
    def __init__(self):
        self.cap = None
        self.running = False
        self.interpreter = None
        self.classes = ['Circle', 'Square']
        self.setup_model()
        self.setup_ui()
    
    def setup_model(self):
        """Modelo CPU optimizado - SIN NPU"""
        try:
            print("🔄 Inicializando modelo LED con CPU optimizado...")
            
            # SOLO CPU - Sin delegados NPU
            self.interpreter = tflite.Interpreter(
                model_path="best_float16.tflite",
                num_threads=4  # Usar múltiples threads CPU
            )
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print(f"✅ Modelo CPU inicializado: {self.input_details[0]['shape']}")
            print(f"🧠 Usando 4 threads CPU para optimización")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            self.interpreter = None
    
    def setup_ui(self):
        self.window = Gtk.Window(title="LED Detection - CPU Optimizado")
        self.window.set_default_size(800, 600)
        self.window.connect("destroy", self.on_destroy)
        
        vbox = Gtk.VBox(spacing=10)
        self.image_widget = Gtk.Image()
        self.status_label = Gtk.Label(label="Iniciando...")
        
        vbox.pack_start(self.image_widget, True, True, 0)
        vbox.pack_start(self.status_label, False, False, 5)
        self.window.add(vbox)
    
    def detect_leds(self, frame):
        if not self.interpreter:
            return []
        
        # Preparar entrada
        input_shape = self.input_details[0]['shape']
        h, w = input_shape[1], input_shape[2]
        resized = cv2.resize(frame, (w, h))
        input_data = np.expand_dims(resized.astype(np.float32) / 255.0, 0)
        
        # Inferencia CPU
        start_time = time.time()
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        inference_time = (time.time() - start_time) * 1000
        
        # Procesar resultados
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        detections = []
        frame_h, frame_w = frame.shape[:2]
        
        for det in output:
            if len(det) >= 6 and det[4] > 0.3:
                x, y, w, h, conf = det[:5]
                class_scores = det[5:7]
                class_id = np.argmax(class_scores)
                class_conf = class_scores[class_id]
                
                if class_conf > 0.3:
                    x1 = int((x - w/2) * frame_w)
                    y1 = int((y - h/2) * frame_h)
                    w1 = int(w * frame_w)
                    h1 = int(h * frame_h)
                    
                    detections.append({
                        'bbox': (max(0, x1), max(0, y1), w1, h1),
                        'class': self.classes[class_id],
                        'conf': conf * class_conf,
                        'inference_time': inference_time
                    })
        
        return detections
    
    def start_camera(self):
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
        self.status_label.set_text("🧠 CPU Optimizado - Detectando...")
    
    def capture_loop(self):
        frame_count = 0
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Detectar
            detections = self.detect_leds(frame)
            inference_time = detections[0]['inference_time'] if detections else 0
            
            # Dibujar
            for det in detections:
                x, y, w, h = det['bbox']
                color = (0, 255, 0) if det['class'] == 'Circle' else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{det['class']} {det['conf']:.2f}", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Log cada 30 frames
            if frame_count % 30 == 0:
                fps = 1000 / inference_time if inference_time > 0 else 0
                print(f"🧠 CPU Frame {frame_count}: {len(detections)} LEDs, {inference_time:.1f}ms, {fps:.1f}FPS")
            
            # UI
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, c = frame_rgb.shape
            pixbuf = GdkPixbuf.Pixbuf.new_from_data(
                frame_rgb.tobytes(), GdkPixbuf.Colorspace.RGB, False, 8, w, h, w*c)
            
            circles = sum(1 for d in detections if d['class'] == 'Circle')
            squares = sum(1 for d in detections if d['class'] == 'Square')
            
            status = f"🔴 {circles} | 🔶 {squares} | 🧠 CPU: {inference_time:.1f}ms"
            
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
    LEDDetectionCPU().run()