# Embedded_AI - Intégration d'un model d'IA sur une plaque electronique ayant comme but de controler la qualité de produits

> [!Note]
> Ce projet est un prototype développé pour une application industrielle impliquant le contrôle qualité d'un produit spécifique, mais il s'est initialement focalisé sur la même détection de LED que dans [LedType-detection](https://github.com/IsmaTIBU/LedType-detection/tree/main) pour une vision plus claire de ses capacités et limitations lors du traitement d'un modèle d'IA intégré dans une carte. Cependant, le modèle obtenu basé sur la détection de LED est parfaitement extrapolable au contrôle qualité lié à tout autre élément de la carte.

# Index
### - [Le Hardware](#quest-ce-que-la-8mplus-bb-)
### - [Le Software](#fonctionnement-des-programmes-actuels)
### - [Entraînement du model](#entraînement-du-modèle)
### - [Résultats du model](#résultats-avec-modèle-chargé-sur-ordinateur)
### - [Possibles Améliorations](#deux-solutions-possibles)

### Qu'est-ce que la 8MPLUS-BB ?
C'est une carte de développement professionnelle de NXP basée sur le processeur i.MX 8M Plus, conçue spécifiquement pour les applications d'IA et d'apprentissage automatique en edge computing.

[Link to datasheet](https://www.nxp.com/products/i.MX8MPLUS)

![Placa](images/Placa.webp)

#### Processeur principal :
CPU : ARM Cortex-A53 quad-core jusqu'à 1,8 GHz  
GPU : Vivante GC7000UL (graphiques 3D et accélération)  
NPU : Neural Processing Unit de 2,3 TOPS pour l'IA  
VPU : Video Processing Unit pour l'encodage/décodage H.264/H.265  

### Capacités clés :
IA/ML : NPU dédié qui accélère les modèles au format .tflite (TensorFlow Lite)  
Vision : Multiples caméras, traitement d'image en temps réel  
Audio : DSP dédié pour le traitement audio avancé  
Connectivité : Ethernet, WiFi, Bluetooth, multiples USB  
Écrans : Support pour multiples écrans simultanés  

## Fonctionnement des programmes actuels 
### Tout d'abord, charger [modelo_convertido.tflite](https://github.com/IsmaTIBU/Embedded_AI/releases/tag/Tflite_model) dans le même répertoire que les programmes.
NPU-CPU.py : Analyse les performances du modèle chargé dans le code pour confirmer s'il est bien optimisé pour être utilisé avec le NPU ou le CPU du système.     
camAI_CPU.py : Effectue une classification d'images en utilisant le CPU. Montre la latence du modèle chargé et la précision de ses prédictions en temps réel.  
camAI_NPU.py : Effectue une classification d'images en utilisant le NPU. Montre la latence du modèle chargé et la précision de ses prédictions en temps réel.  

### Exécution des fichiers
Exécuter les fichiers python avec ```python3 [fichier]```

## Exemple de transfert de fichiers

IP de la carte : 194.178.59.125  
Mon IP : 194.178.59.38

Architecture du répertoire sur l'ordinateur :
```
Embedded_AI
    |-------NPU-CPU.py
    |-------camAI_CPU.py
    |-------camAI_NPU.py
    |-------README.md
```

Architecture actuelle du répertoire sur la carte :
```
PlacaKepar
    |-------NPU-CPU.py
    |-------camAI_CPU.py
    |-------camAI_NPU.py
    |-------modelo_convertido.tflite
```

#### Pour envoyer depuis l'ordinateur (Windows)
Depuis l'ordinateur exécuter :
```
cd Embedded_AI
python -m http.server 8000
```

Depuis la carte exécuter :
```
cd PlacaKepar
wget http://194.178.59.38:8000/[Fichier]
```

#### Pour envoyer depuis la carte (Linux - yocto)
Depuis l'ordinateur exécuter :
```
cd Embedded_AI
Invoke-WebRequest -Uri "http://194.178.59.125:8000/[Fichier]" -OutFile "[Fichier]"
```

Depuis la carte exécuter :
```
cd PlacaKepar
python3 -m http.server 8000
```

## Entraînement du modèle
Le modèle a été entièrement développé avec Tensorflow et Keras. L'objectif principal était de développer un modèle efficace pour la tâche qui ne pèse pas trop, en compliquant le moins possible son architecture mais en obtenant des résultats corrects. Le modèle actuel compte avec un peu plus de 900k paramètres entraînés lors de son fine-tuning.

<table>
<tr>
<td><img src="images/Model1.png" width="200"/></td>
<td><img src="images/Model2.png" width="200"/></td>
</tr>
<tr>
<td colspan="2" align="center">
  <em>Changement de l'architecture du modèle<br>pour lisser les courbes d'entraînement et de validation</em>
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
<td colspan="2" align="center"><em>Résultats d'entraînement du modèle initial et du modèle final</em></td>
</tr>
</table>

<table>
<tr>
<td><img src="images/KERAS_predictions.png" width="400"/></td>
<td><img src="images/TFLITE_predictions.png" width="400"/></td>
</tr>
<tr>
<td colspan="2" align="center"><em>Résultats de détection entre un modèle .keras et .tflite</em></td>
</tr>
</table>

|          |modèle .keras|modèle .tflite|
|----------|-----------|------------|
|Poids|11.3 MB|1.05 MB|
|Précision|100 %|98.59 %|
|Confiance moyenne|97.5 %|94.2 %|
|Confiance maximale|100 %|99.9 %|
|Confiance minimale|77.3 %|50.2 %|

Ces résultats pourraient s'améliorer, surtout pour le modèle au format .tflite, en augmentant considérablement le dataset (actuellement nous n'avons que 590 images d'entraînement).

## Résultats avec modèle chargé sur ordinateur 
### Caméra utilisée : [ELP 5MP HD USB Camera](https://www.elpcctv.com/elp-5mp-hd-usb-camera-board-free-driver-usb-camera-module-with-ov5640-sensor-elpusb500w02ml21-p-51.html)
<img src="images/camTests.PNG" width="100"/>

<table>
<tr>
<td><img src="images/Comp_testKeras.gif" width="400"/></td>
<td><img src="images/Comp_testTflite.gif" width="400"/></td>
</tr>
<tr>
<td colspan="2" align="center"><em>Résultats de détection en temps réel d'un modèle .keras et .tflite </em></td>
</tr>
</table>

## Résultats avec modèle intégré sur la carte
### Caméra utilisée : [4K MIPI CMOS Camera](https://www.nxp.com/design/design-center/development-boards-and-designs/4K-MIPI-CMOS-CAMERA-MODULE) 
<img src="images/camPlaca.PNG" width="150"/>

<table>
<tr>
<td><img src="images/b_cpu.jpg" width="400"/></td>
<td><img src="images/b_npu.jpg" width="400"/></td>
</tr>
<tr>
<td><img src="images/cg_cpu.jpg" width="400"/></td>
<td><img src="images/cg_npu.jpg" width="400"/></td>
</tr>
<tr>
<td><img src="images/cug_cpu.jpg" width="400"/></td>
<td><img src="images/cug_npu.jpg" width="400"/></td>
</tr>
<tr>
<td colspan="2" align="center">
  <em>Détection du fond, LED circulaires et carrées sur la carte NXP en utilisant le CPU ou le NPU</em>
</td>
</tr>
</table>

Comme je l'ai déjà dit, pour le moment ce n'est rien de plus qu'un prototype pour se familiariser le plus rapidement possible avec ces concepts et utiliser la carte, mais on voit clairement une différence de performance entre les détections par CPU et par NPU, cette dernière étant beaucoup plus lente (~ x2.07) alors qu'elle ne devrait pas l'être. Il faudra donc débugger les erreurs d'optimisation, même si heureusement pour notre cas le modèle actuel est si léger qu'il s'exécute suffisamment rapidement par CPU.
En plus de cela, les LED carrées n'arrivent jamais à être détectées, même s'il s'agit du même modèle qui, comme nous l'avons vu auparavant, avait une précision très élevée. Ceci très probablement à cause d'un dataset pas bien "spécialisé" pour cette caméra. Le mieux serait d'utiliser la propre caméra qui serait utilisée pour la détection pour prendre les photos utilisées pour construire le dataset. L'emplacement est aussi crucial puisqu'il se reflétera dans les changements de lumière et les reflets captés par la caméra.

### Deux solutions possibles :
1. Créer un dataset beaucoup plus étendu (~2000-3000 images d'entraînement) dans lequel on implémenterait des photos avec beaucoup de différence d'intensité lumineuse.  
   **- Avantages :** Il s'agirait très probablement d'un dataset qui fonctionnerait avec presque n'importe quel type de caméra sous presque n'importe quel type de conditions.  
   **- Inconvénients :** Construire un dataset de cette envergure prendrait assez de temps (1 semaine environ, sans compter les possibles erreurs d'étiquetage pour aller vite).
2. Créer un dataset spécialisé (~500-600 images d'entraînement) pour être utilisé uniquement avec cette caméra. Les photos devraient être prises avec cette caméra sous les conditions et
   l'emplacement où on la disposerait.  
   **- Avantages :** Construire ce dataset prendrait beaucoup moins de temps (1 jour approximativement). De plus, s'agissant d'un dataset "spécialisé", la précision des prédictions serait toujours
   très élevée.  
   **- Inconvénients :** Dataset "spécialisé" donc on ne pourrait pas l'utiliser pour le reste des emplacements ou caméras. Soit on verrait des baisses de performance soit il faudrait implémenter
   plus d'images au dataset si on désire l'utiliser pour d'autres cas.
