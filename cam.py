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

class CameraDetection:
    def __init__(self):
        self.cap = None
        self.running = False
        
        # Inicializar TensorFlow Lite + NPU
        print("🔄 Inicializando NPU...")
        self.setup_npu()
        
        # Crear ventana
        self.window = Gtk.Window()
        self.window.set_title("Detección de Caras - NPU")
        self.window.set_default_size(800, 600)
        self.window.connect("destroy", self.on_destroy)
        
        # Layout simple
        vbox = Gtk.VBox(spacing=10)
        self.window.add(vbox)
        
        # Widget de imagen
        self.image_widget = Gtk.Image()
        vbox.pack_start(self.image_widget, True, True, 0)
        
        # Status
        self.status_label = Gtk.Label()
        self.status_label.set_text("Iniciando...")
        vbox.pack_start(self.status_label, False, False, 5)
    
    def setup_npu(self):
        """Configurar TensorFlow Lite con NPU"""
        model_path = "/opt/gopoint-apps/downloads/face_detection_ptq.tflite"
        
        try:
            print(f"📁 Cargando modelo: {model_path}")
            
            # Crear delegado VX para NPU
            try:
                vx_delegate = tflite.load_delegate('/usr/lib/libvx_delegate.so')
                print("✅ Delegado VX (NPU) cargado")
                delegates = [vx_delegate]
            except Exception as e:
                print(f"⚠️  NPU no disponible, usando CPU: {e}")
                delegates = []
            
            # Crear intérprete
            self.interpreter = tflite.Interpreter(
                model_path=model_path,
                experimental_delegates=delegates
            )
            
            self.interpreter.allocate_tensors()
            print("✅ Modelo TensorFlow Lite inicializado")
            
            # Obtener detalles de entrada y salida
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print(f"📋 Input shape: {self.input_details[0]['shape']}")
            print(f"📋 Output details: {len(self.output_details)} salidas")
            
        except Exception as e:
            print(f"❌ Error inicializando NPU: {e}")
            print("🔄 Fallback a Haar Cascades...")
            self.interpreter = None
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def detect_faces_npu(self, frame):
        """Detectar caras usando NPU"""
        if self.interpreter is None:
            # Fallback a OpenCV
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            return [(x, y, w, h) for (x, y, w, h) in faces]
        
        # Preparar entrada para el modelo
        input_shape = self.input_details[0]['shape']
        height, width = input_shape[1], input_shape[2]
        
        # Redimensionar frame
        resized = cv2.resize(frame, (width, height))
        input_data = np.expand_dims(resized, axis=0).astype(np.uint8)
        
        # Inferencia
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Obtener resultados (esto depende del modelo específico)
        # Para face_detection_ptq, necesitamos verificar el formato de salida
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])
        scores = self.interpreter.get_tensor(self.output_details[1]['index'])
        
        # Convertir a formato OpenCV (x, y, w, h)
        frame_h, frame_w = frame.shape[:2]
        faces = []
        
        for i in range(len(boxes[0])):
            if scores[0][i] > 0.5:  # Umbral de confianza
                box = boxes[0][i]
                x = int(box[1] * frame_w)
                y = int(box[0] * frame_h)
                w = int((box[3] - box[1]) * frame_w)
                h = int((box[2] - box[0]) * frame_h)
                faces.append((x, y, w, h))
        
        return faces
    
    def start_camera(self):
        """Iniciar cámara automáticamente"""
        print("🎥 Buscando cámara...")
        
        # Buscar cámara disponible
        for device in ['/dev/video3', '/dev/video0', '/dev/video1', '/dev/video2', 0, 1, 2, 3]:
            self.cap = cv2.VideoCapture(device)
            if self.cap.isOpened():
                ret, _ = self.cap.read()
                if ret:
                    print(f"✅ Cámara encontrada: {device}")
                    break
                self.cap.release()
        
        if not self.cap or not self.cap.isOpened():
            print("❌ No se encontró cámara")
            self.status_label.set_text("No se encontró cámara")
            return
        
        # Configurar cámara
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Iniciar captura
        self.running = True
        threading.Thread(target=self.capture_loop, daemon=True).start()
        method = "NPU" if self.interpreter else "CPU (Haar)"
        self.status_label.set_text(f"🚀 Detectando caras con {method}...")
        print(f"🚀 Iniciando detección con {method}")
    
    def capture_loop(self):
        """Loop principal de captura y detección"""
        frame_count = 0
        inference_times = []
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Medir tiempo de inferencia
            start_time = time.time()
            faces = self.detect_faces_npu(frame)
            inference_time = (time.time() - start_time) * 1000  # ms
            inference_times.append(inference_time)
            
            # Debug cada 30 frames
            if frame_count % 30 == 0:
                avg_time = np.mean(inference_times[-30:]) if inference_times else 0
                fps = 1000 / avg_time if avg_time > 0 else 0
                print(f"🔍 Frame {frame_count}: {len(faces)} caras, {avg_time:.1f}ms, {fps:.1f}FPS")
            
            # Dibujar rectángulos
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, 'CARA', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Convertir BGR a RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Mostrar en GTK
            height, width, channels = frame_rgb.shape
            pixbuf = GdkPixbuf.Pixbuf.new_from_data(
                frame_rgb.tobytes(), GdkPixbuf.Colorspace.RGB,
                False, 8, width, height, width * channels
            )
            
            # Actualizar UI
            GLib.idle_add(self.update_ui, pixbuf, len(faces), frame_count, inference_time)
            
            frame_count += 1
            time.sleep(0.01667)  # ~60 FPS
    
    def update_ui(self, pixbuf, face_count, frame_count, inference_time):
        """Actualizar interfaz"""
        self.image_widget.set_from_pixbuf(pixbuf)
        method = "NPU" if self.interpreter else "CPU"
        self.status_label.set_text(f"Caras: {face_count} | {method} | {inference_time:.1f}ms")
        return False
    
    def on_destroy(self, widget):
        """Limpiar al cerrar"""
        print("🛑 Cerrando aplicación...")
        self.running = False
        if self.cap:
            self.cap.release()
        Gtk.main_quit()
    
    def run(self):
        """Ejecutar aplicación"""
        self.window.show_all()
        GLib.idle_add(self.start_camera)
        Gtk.main()

if __name__ == "__main__":
    app = CameraDetection()
    app.run()