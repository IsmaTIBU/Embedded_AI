#!/usr/bin/env python3
import cv2
import threading
import time
import gi
import os

gi.require_version('Gtk', '3.0')
gi.require_version('GdkPixbuf', '2.0')
from gi.repository import Gtk, GdkPixbuf, GLib

class CameraDetection:
    def __init__(self):
        self.cap = None
        self.running = False
        
        # Detector de caras con ruta alternativa
        self.face_cascade = self.load_cascade()
        
        # Crear ventana
        self.window = Gtk.Window()
        self.window.set_title("Detección de Caras")
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
    
    def load_cascade(self):
        """Cargar cascade con múltiples rutas"""
        cascade_paths = [
            # Ruta estándar (si cv2.data existe)
            getattr(cv2.data, 'haarcascades', '') + 'haarcascade_frontalface_default.xml' if hasattr(cv2, 'data') else '',
            # Rutas alternativas comunes
            '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
        ]
        
        for path in cascade_paths:
            if path and os.path.exists(path):
                cascade = cv2.CascadeClassifier(path)
                if not cascade.empty():
                    return cascade
        
        # Si no encuentra, crear uno vacío
        return cv2.CascadeClassifier()
    
    def start_camera(self):
        """Iniciar cámara automáticamente"""
        # Buscar cámara disponible
        for device in ['/dev/video3', '/dev/video0', '/dev/video1', '/dev/video2', 0, 1, 2, 3]:
            self.cap = cv2.VideoCapture(device)
            if self.cap.isOpened():
                ret, _ = self.cap.read()
                if ret:
                    break
                self.cap.release()
        
        if not self.cap or not self.cap.isOpened():
            self.status_label.set_text("No se encontró cámara")
            return
        
        # Configurar cámara
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Iniciar captura
        self.running = True
        threading.Thread(target=self.capture_loop, daemon=True).start()
        status = "🎥 Detectando caras..." if not self.face_cascade.empty() else "🎥 Cámara activa (sin detección)"
        self.status_label.set_text(status)
    
    def capture_loop(self):
        """Loop principal de captura y detección"""
        frame_count = 0
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            faces = []
            # Detectar caras solo si el cascade está disponible
            if not self.face_cascade.empty():
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                
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
            GLib.idle_add(self.update_ui, pixbuf, len(faces), frame_count)
            
            frame_count += 1
            time.sleep(0.01667)  # ~60 FPS
    
    def update_ui(self, pixbuf, face_count, frame_count):
        """Actualizar interfaz"""
        self.image_widget.set_from_pixbuf(pixbuf)
        self.status_label.set_text(f"Caras: {face_count}")
        return False
    
    def on_destroy(self, widget):
        """Limpiar al cerrar"""
        self.running = False
        if self.cap:
            self.cap.release()
        Gtk.main_quit()
    
    def run(self):
        """Ejecutar aplicación"""
        self.window.show_all()
        GLib.idle_add(self.start_camera)  # Iniciar automáticamente
        Gtk.main()

if __name__ == "__main__":
    app = CameraDetection()
    app.run()