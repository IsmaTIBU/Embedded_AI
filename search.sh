#!/bin/bash

echo "🔍 Búsqueda exhaustiva de componentes NPU/ML en el sistema"
echo "========================================================="

echo -e "\n📁 Buscando archivos NPU/VIP/Galcore..."
find / -name "*npu*" -o -name "*galcore*" -o -name "*vip*" -o -name "*neural*" 2>/dev/null | head -20

echo -e "\n📁 Buscando TensorFlow Lite..."
find / -name "*tflite*" -o -name "*tensorflow*" 2>/dev/null | head -20

echo -e "\n📁 Buscando librerías de inferencia..."
find / -name "*inference*" -o -name "*runtime*" -o -name "*delegate*" 2>/dev/null | head -20

echo -e "\n🔧 Buscando módulos del kernel..."
find /lib/modules/$(uname -r) -name "*.ko" | xargs grep -l "galcore\|npu\|vip" 2>/dev/null

echo -e "\n📚 Buscando librerías dinámicas..."
ldconfig -p | grep -i "npu\|tflite\|galcore\|vip\|neural"

echo -e "\n🐍 Buscando paquetes Python..."
find /usr/lib*/python* -name "*tflite*" -o -name "*tensorflow*" -o -name "*onnx*" 2>/dev/null

echo -e "\n📦 Buscando paquetes instalados..."
if command -v opkg &> /dev/null; then
    opkg list-installed | grep -i "npu\|tflite\|neural\|ml"
elif command -v rpm &> /dev/null; then
    rpm -qa | grep -i "npu\|tflite\|neural\|ml"
elif command -v dpkg &> /dev/null; then
    dpkg -l | grep -i "npu\|tflite\|neural\|ml"
fi

echo -e "\n🔍 Buscando en archivos de configuración..."
grep -r "npu\|galcore\|tflite" /etc/ 2>/dev/null | head -10

echo -e "\n💻 Buscando ejecutables..."
find /usr/bin /usr/sbin /bin /sbin -name "*tflite*" -o -name "*npu*" -o -name "*neural*" 2>/dev/null

echo -e "\n📋 Buscando headers de desarrollo..."
find /usr/include -name "*npu*" -o -name "*tflite*" -o -name "*vip*" 2>/dev/null

echo -e "\n🔧 Verificando dispositivos del sistema..."
ls -la /dev/ | grep -i "video\|npu\|vip\|galcore"

echo -e "\n🖥️ Verificando información del kernel..."
dmesg | grep -i "npu\|vip\|galcore\|neural" | tail -10

echo -e "\n✅ Búsqueda completada!"
echo "Si no aparece nada relacionado con NPU, es probable que necesites:"
echo "1. Una imagen Yocto con soporte NPU habilitado"
echo "2. Instalar los paquetes NPU manualmente"
echo "3. Recompilar con las opciones NPU activadas"