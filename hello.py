#!/usr/bin/env python3
import cv2

# Abrir cámara
cap = cv2.VideoCapture('/dev/video3')

# Loop principal
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow('Camera', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpiar
cap.release()
cv2.destroyAllWindows()