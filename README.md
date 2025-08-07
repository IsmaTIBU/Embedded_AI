# Placa i.MX 8M Plus Power
> A tener en cuenta: Este proyecto es un prototipo desarrollado para una aplicación industrial específica que involucra el control de calidad de un producto en especifico pero se focalizo desde una primera instancia en la misma deteccion de LEDs que en [LedType-detection](https://github.com/IsmaTIBU/LedType-detection/tree/main) para una vision mas clara de sus capacidades y limitacines al tratarse de un modelo de IA integrado en una placa. Sin embargo los resultados obtenidos basandonos en la deteccion de LEDs es perfectamente extrapolable a un control de calidad relacionado a cualquier otro elemento de la placa.

### ¿Qué es la i.MX 8M Plus Power?
Es una placa de desarrollo profesional de NXP basada en el procesador i.MX 8M Plus, diseñada específicamente para aplicaciones de IA y machine learning en edge computing.  

[Link to datasheet](https://www.nxp.com/products/IMX8MPLUS)

![Placa](images/Placa.webp)

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

## Funcionamiento de los programas actuales 
### Primero de todo cargar [modelo_convertido.tflite](https://github.com/IsmaTIBU/Embedded_AI/releases/tag/Tflite_model) en el mismo directorio que los programas.
NPU-CPU.py: Analiza el rendimiento del modelo cargado en el codigo para confirmar si esta bien optimizado para ser utilizado con la NPU o la CPU del sistema.     
camAI_CPU.py: Realiza una clasificacion de imagenes utilizando la CPU. Muestra la latencia del modelo cargado y la exactitud de sus predicciones en tiempo real.  
camAI_NPU.py: Realiza una clasificacion de imagenes utilizando la NPU. Muestra la latencia del modelo cargado y la exactitud de sus predicciones en tiempo real.  

### Ejecucion de ficheros
Ejecutar archivos python con ```python3 [archivo]```

## Ejemplo de transmision de ficheros

IP de la placa: 194.178.59.125  
Mi IP: 194.178.59.38

#### Para enviar desde el ordenador (Windows)
Desde el ordenador ejecutar: 
```
cd Embedded_AI
python -m http.server 8000
```  
Desde la placa ejecutar:
```
cd PlacaKepar
wget http://194.178.59.38:8000/[Archivo]
```

#### Para enviar desde la placa (Linux - yocto)
Desde el ordenador ejecutar: 
```
cd Embedded_AI
Invoke-WebRequest -Uri "http://194.178.59.125:8000/[Archivo]" -OutFile "[Archivo]"
```  
Desde la placa ejecutar:
```
cd PlacaKepar
python3 -m http.server 8000
```
## Entrenamiento del modelo
El modelo fue enteramente desarrollado con Tensorflow y Keras. El objetivo principal era desarrollar un modelo eficiente para la tarea que tampoco pesase demasiado, complicando lo minimo su arquitectura pero obteniendo resultados correctos. El modelo actual cuenta con poco mas de 900k parametros entrenados durante su fine-tunning.

<table>
<tr>
<td><img src="images/Model1.png" width="200"/></td>
<td><img src="images/Model2.png" width="200"/></td>
</tr>
<tr>
<td colspan="2" align="center">
  <em>Cambio de la arquitectura del modelo<br>para suavizar las curvas de entrenamiento y validación</em>
</td>
</tr>
</table>

<table>
<tr>
<td><img src="images/simple_training_history1.png" width="600"/></td>
</tr>
<tr>
<td><img src="images/simple_training_history3.png" width="600"/></td>
</tr>
<tr>
<td colspan="2" align="center"><em>Resultados del entrenamiento del modelo inicial y del modelo final</em></td>
</tr>
</table>

<table>
<tr>
<td><img src="images/KERAS_predictions.png" width="400"/></td>
<td><img src="images/TFLITE_predictions.png" width="400"/></td>
</tr>
<tr>
<td colspan="2" align="center"><em>Resultado de deteccion entre un modelo keras y tflite</em></td>
</tr>
</table>

|          |modelo .keras|modelo .tflite|
|----------|-----------|------------|
|Peso|11.3 MB|1.05 MB|
|Precision|100 %|98.59 %|
|Confianza promedio|97.5 %|94.2 %|
|Confianza maxima|100 %|99.9 %|
|Confianza minima|77.3 %|50.2 %|

Estos resultados podrian mejorar, especialmente para el modelo en formato .tflite, aumentando el dataset considerablemente (actualmente contamos con unicamente 590 imagenes de entrenamiento).

## Resultados con modelo cargado en ordenador 
### Camara utilizada: [ELP 5MP HD USB Camera](https://www.elpcctv.com/elp-5mp-hd-usb-camera-board-free-driver-usb-camera-module-with-ov5640-sensor-elpusb500w02ml21-p-51.html)

## Resultados con modelo integrado sobre la placa
### Camara utilizada: [4K MIPI CMOS Camera](https://www.nxp.com/design/design-center/development-boards-and-designs/4K-MIPI-CMOS-CAMERA-MODULE)





