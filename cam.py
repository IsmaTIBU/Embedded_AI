#!/usr/bin/env python3
import cv2
import threading
import time
import gi
import numpy as np
import tflite_runtime.interpreter as tflite
from pathlib import Path

gi.require_version('Gtk', '3.0')
gi.require_version('GdkPixbuf', '2.0')
from gi.repository import Gtk, GdkPixbuf, GLib

class FaceDetectionNXP:
    def __init__(self):
        self.cap = None
        self.running = False
        self.interpreter = None
        self.anchors = None
        
        # Configuración del modelo
        self.model_config = {
            'model_path': "/opt/gopoint-apps/downloads/face_detection_ptq.tflite",
            'anchors_path': "/opt/gopoint-apps/downloads/box_priors.txt",
            'center_variance': 0.1,
            'size_variance': 0.2,
            'score_threshold': 0.5,
            'nms_threshold': 0.5
        }
        
        self.setup_model()
        self.setup_ui()
    
    def setup_model(self):
        """Configurar modelo TFLite con delegado NPU"""
        try:
            print("🔄 Inicializando modelo...")
            
            # Cargar anchors
            self.anchors = self._load_anchors()
            if self.anchors is None:
                raise Exception("No se pudieron cargar box priors")
            
            # Configurar delegado NPU
            delegates = []
            try:
                delegates = [tflite.load_delegate('/usr/lib/libvx_delegate.so')]
                print("✅ Delegado NPU cargado")
            except Exception as e:
                print(f"⚠️ NPU no disponible, usando CPU: {e}")
            
            # Crear intérprete
            self.interpreter = tflite.Interpreter(
                model_path=self.model_config['model_path'],
                experimental_delegates=delegates
            )
            self.interpreter.allocate_tensors()
            
            # Obtener detalles I/O
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print(f"✅ Modelo inicializado - Input: {self.input_details[0]['shape']}")
            
        except Exception as e:
            print(f"❌ Error inicializando modelo: {e}")
            self.interpreter = None
    
    def _load_anchors(self):
        """Cargar box priors desde archivo"""
        try:
            with open(self.model_config['anchors_path'], 'r') as f:
                numbers = [float(x) for x in f.read().strip().split()]
            
            if len(numbers) >= 3584:  # 896 * 4
                anchors = np.array(numbers[:3584]).reshape(896, 4)
                print(f"✅ {len(anchors)} anchors cargados")
                return anchors
            else:
                print(f"❌ Datos insuficientes para anchors: {len(numbers)}")
                return None
                
        except Exception as e:
            print(f"❌ Error cargando anchors: {e}")
            return None
    
    def setup_ui(self):
        """Configurar interfaz gráfica"""
        self.window = Gtk.Window(title="Face Detection - NXP Style")
        self.window.set_default_size(800, 600)
        self.window.connect("destroy", self.on_destroy)
        
        vbox = Gtk.VBox(spacing=10)
        self.window.add(vbox)
        
        self.image_widget = Gtk.Image()
        vbox.pack_start(self.image_widget, True, True, 0)
        
        self.status_label = Gtk.Label(label="Iniciando...")
        vbox.pack_start(self.status_label, False, False, 5)
    
    def detect_faces(self, frame):
        """Detectar caras usando el modelo NPU"""
        if not self.interpreter or self.anchors is None:
            return []
        
        # Preprocesar
        input_shape = self.input_details[0]['shape']
        h, w = input_shape[1], input_shape[2]
        
        resized = cv2.resize(frame, (w, h))
        input_data = np.expand_dims(resized, axis=0).astype(
            np.uint8 if self.input_details[0]['dtype'] == np.uint8 else np.float32
        )
        
        if self.input_details[0]['dtype'] != np.uint8:
            input_data = input_data / 255.0
        
        # Inferencia
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Obtener salidas
        locations = self.interpreter.get_tensor(self.output_details[1]['index'])
        scores = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Decodificar y filtrar
        detections = self._decode_detections(locations, scores)
        final_detections = self._apply_nms(detections)
        
        # Convertir a formato OpenCV
        frame_h, frame_w = frame.shape[:2]
        return [
            (int(det['box'][0] * frame_w), int(det['box'][1] * frame_h),
             int((det['box'][2] - det['box'][0]) * frame_w), 
             int((det['box'][3] - det['box'][1]) * frame_h))
            for det in final_detections
        ]
    
    def _decode_detections(self, locations, scores):
        """Decodificar detecciones SSD MobileNet"""
        detections = []
        cfg = self.model_config
        
        for i, anchor in enumerate(self.anchors):
            # Calcular score con sigmoid
            raw_score = scores[0][i][0] if len(scores.shape) > 2 else scores[0][i]
            score = 1.0 / (1.0 + np.exp(-raw_score))
            
            if score > cfg['score_threshold']:
                # Decodificar coordenadas
                anchor_cx, anchor_cy, anchor_w, anchor_h = anchor
                dx, dy, dw, dh = locations[0][i][:4]
                
                # Aplicar transformaciones SSD
                decoded_cx = dx * cfg['center_variance'] * anchor_w + anchor_cx
                decoded_cy = dy * cfg['center_variance'] * anchor_h + anchor_cy
                decoded_w = anchor_w * np.exp(dw * cfg['size_variance'])
                decoded_h = anchor_h * np.exp(dh * cfg['size_variance'])
                
                # Convertir a coordenadas de esquina y clip
                x_min = np.clip(decoded_cx - decoded_w / 2.0, 0, 1)
                y_min = np.clip(decoded_cy - decoded_h / 2.0, 0, 1)
                x_max = np.clip(decoded_cx + decoded_w / 2.0, 0, 1)
                y_max = np.clip(decoded_cy + decoded_h / 2.0, 0, 1)
                
                if x_max > x_min and y_max > y_min:
                    detections.append({
                        'box': [x_min, y_min, x_max, y_max],
                        'score': score
                    })
        
        return detections
    
    def _apply_nms(self, detections):
        """Aplicar Non-Maximum Suppression"""
        if not detections:
            return []
        
        # Ordenar por score
        detections.sort(key=lambda x: x['score'], reverse=True)
        
        keep = []
        while detections:
            current = detections.pop(0)
            keep.append(current)
            
            # Filtrar por IoU
            detections = [
                det for det in detections 
                if self._calculate_iou(current['box'], det['box']) < self.model_config['nms_threshold']
            ]
        
        return keep
    
    def _calculate_iou(self, box1, box2):
        """Calcular Intersection over Union"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Intersección
        inter_x_min, inter_y_min = max(x1_min, x2_min), max(y1_min, y2_min)
        inter_x_max, inter_y_max = min(x1_max, x2_max), min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        
        return inter_area / (area1 + area2 - inter_area)
    
    def start_camera(self):
        """Iniciar cámara"""
        print("🎥 Iniciando cámara...")
        
        # Buscar dispositivo de cámara
        for device in ['/dev/video3', '/dev/video0', '/dev/video1', '/dev/video2', 0, 1, 2, 3]:
            self.cap = cv2.VideoCapture(device)
            if self.cap.isOpened() and self.cap.read()[0]:
                print(f"✅ Cámara encontrada: {device}")
                break
            if self.cap:
                self.cap.release()
        
        if not self.cap or not self.cap.isOpened():
            print("❌ No se encontró cámara")
            self.status_label.set_text("No se encontró cámara")
            return
        
        # Configurar resolución
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.running = True
        threading.Thread(target=self.capture_loop, daemon=True).start()
        
        method = "NPU" if self.interpreter else "Fallback"
        self.status_label.set_text(f"🚀 Detectando caras con {method}...")
    
    def capture_loop(self):
        """Loop principal de captura y detección"""
        frame_count = 0
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Detección
            start_time = time.time()
            faces = self.detect_faces(frame)
            inference_time = (time.time() - start_time) * 1000
            
            # Dibujar detecciones
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)
                cv2.putText(frame, 'FACE', (x+550, y+450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Log cada 30 frames
            if frame_count % 30 == 0:
                fps = 1000 / inference_time if inference_time > 0 else 0
                print(f"🔍 Frame {frame_count}: {len(faces)} caras, {inference_time:.1f}ms, {fps:.1f}FPS")
            
            # Actualizar UI
            self._update_display(frame, len(faces), inference_time)
            
            frame_count += 1
            time.sleep(0.033)  # ~30 FPS
    
    def _update_display(self, frame, face_count, inference_time):
        """Actualizar imagen en pantalla"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = frame_rgb.shape
        
        pixbuf = GdkPixbuf.Pixbuf.new_from_data(
            frame_rgb.tobytes(), GdkPixbuf.Colorspace.RGB,
            False, 8, w, h, w * c
        )
        
        GLib.idle_add(self._update_ui, pixbuf, face_count, inference_time)
    
    def _update_ui(self, pixbuf, face_count, inference_time):
        """Actualizar interfaz (thread-safe)"""
        self.image_widget.set_from_pixbuf(pixbuf)
        self.status_label.set_text(f"Caras: {face_count} | NPU | {inference_time:.1f}ms")
        return False
    
    def on_destroy(self, widget):
        """Limpiar recursos al cerrar"""
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
    FaceDetectionNXP().run()