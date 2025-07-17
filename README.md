# Placa i.MX 8M Plus Power EVK
### ¿Qué es?
Es una placa de desarrollo profesional de NXP basada en el procesador i.MX 8M Plus, diseñada específicamente para aplicaciones de IA y machine learning en edge computing.  

[Link to datasheet](https://www.nxp.com/products/IMX8MPLUS)

#### Procesador principal:
CPU: ARM Cortex-A53 quad-core hasta 1.8 GHz  
GPU: Vivante GC7000UL (gráficos 3D y aceleración)  
NPU: Neural Processing Unit de 2.3 TOPS para IA  
VPU: Video Processing Unit para codificación/decodificación H.264/H.265  

### Capacidades clave:
IA/ML: NPU dedicado que acelera TensorFlow Lite, ONNX, PyTorch  
Visión: Múltiples cámaras, procesamiento de imagen en tiempo real  
Audio: DSP dedicado para procesamiento de audio avanzado  
Conectividad: Ethernet, WiFi, Bluetooth, múltiples USB  
Pantallas: Soporte para múltiples displays simultáneos  

### Transmision de ficheros

IP de la placa: 192.168.59.125  
Mi IP: 192.168.59.38

#### Para enviar desde el ordenador
Desde el ordenador ejecutar: 
```
cd PlacaKepar
python -m http.server 8000
```  
Desde la placa ejecutar:
```
cd PlacaKepar
wget http://192.168.59.38:8000/[Archivo]
```

#### Para enviar desde la placa
Desde el ordenador ejecutar: 
```
cd PlacaKepar
Invoke-WebRequest -Uri "http://192.168.59.125:8000/[Archivo]" -OutFile "[Archivo]"
```  
Desde la placa ejecutar:
```
cd PlacaKepar
python3 -m http.server 8000
```

### Ejecucion de ficheros
Activar primero el entorno virtual: ```source Kepar/bin/activate```
Ejecutar ficheros python como se quiera con ```python3 [archivo]```
