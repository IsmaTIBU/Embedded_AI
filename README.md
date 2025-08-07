# Placa i.MX 8M Plus Power
> A tener en cuenta: Este proyecto fue desarrollado para una aplicación industrial específica que involucra el control de calidad de un producto en especifico pero se focalizo desde una primera instancia en la misma deteccion de LEDs que en [LedType-detection](https://github.com/IsmaTIBU/LedType-detection/tree/main) para una vision mas clara de sus capacidades y limitacines al tratarse de un modelo de IA integrado en una placa. Sin embargo los resultados obtenidos basandonos en la deteccion de LEDs es perfectamente extrapolable a un control de calidad relacionado a cualquier otro elemento de la placa.

### ¿Qué es la i.MX 8M Plus Power?
Es una placa de desarrollo profesional de NXP basada en el procesador i.MX 8M Plus, diseñada específicamente para aplicaciones de IA y machine learning en edge computing.  

[Link to datasheet](https://www.nxp.com/products/IMX8MPLUS)

![images/Placa](Placa.webp)

#### Procesador principal:
CPU: ARM Cortex-A53 quad-core hasta 1.8 GHz  
GPU: Vivante GC7000UL (gráficos 3D y aceleración)  
NPU: Neural Processing Unit de 2.3 TOPS para IA  
VPU: Video Processing Unit para codificación/decodificación H.264/H.265  

### Capacidades clave:
IA/ML: NPU dedicado que acelera modelos en formato .tflite(TensorFlow Lite)  
Visión: Múltiples cámaras, procesamiento de imagen en tiempo real  
Audio: DSP dedicado para procesamiento de audio avanzado  
Conectividad: Ethernet, WiFi, Bluetooth, múltiples USB  
Pantallas: Soporte para múltiples displays simultáneos  

### Ejemplo de transmision de ficheros

IP de la placa: 194.178.59.125  
Mi IP: 194.178.59.38

#### Para enviar desde el ordenador
Desde el ordenador ejecutar: 
```
cd PlacaKepar
python -m http.server 8000
```  
Desde la placa ejecutar:
```
cd PlacaKepar
wget http://194.178.59.38:8000/[Archivo]
```

#### Para enviar desde la placa
Desde el ordenador ejecutar: 
```
cd PlacaKepar
Invoke-WebRequest -Uri "http://194.178.59.125:8000/[Archivo]" -OutFile "[Archivo]"
```  
Desde la placa ejecutar:
```
cd PlacaKepar
python3 -m http.server 8000
```

## Funcionamiento de los programas actuales 

### Ejecucion de ficheros
Ejecutar archivos python con ```python3 [archivo]```


