# 🎧 Filtro de Ruido de Audio (FFT y Números Complejos)

> Una herramienta de procesamiento digital de señales basada en Python que transforma señales de audio desde el "dominio del tiempo" al "dominio de la frecuencia" utilizando la Transformada Rápida de Fourier (FFT) y la manipulación de números complejos para aislar y eliminar el ruido de fondo.

---

## 🚀 Acerca del Proyecto

Las grabaciones de audio del mundo real y los datos de sensores IoT suelen estar contaminados por ruido de fondo e interferencia eléctrica. Manipular las ondas brutas directamente en el dominio del tiempo suele ser ineficiente o impracticable.

Este proyecto implementa un flujo completo de **Procesamiento Digital de Señales (DSP)** que:
1. Lee señales de audio crudas.
2. Proyecta la señal en el dominio de la frecuencia mediante la **Transformada Rápida de Fourier (FFT)**, utilizando números complejos para representar tanto la amplitud como la fase.
3. Aplica una máscara de filtrado espectral personalizada para apuntar y suprimir componentes de frecuencia no deseados.
4. Reconstruye la señal limpia de regreso al dominio del tiempo mediante la **Transformada Rápida de Fourier Inversa (IFFT)**.

---

## 🛠️ Stack Tecnológico y Dependencias

* **Python 3.x**
* **NumPy:** Para operaciones con matrices de alto rendimiento y matemáticas con números complejos.
* **SciPy:** Para leer/escribir archivos de audio WAV y realizar cálculos científicos.
* **Matplotlib:** Para generar visualizaciones comparativas del espectro y del dominio del tiempo.

---

## ⚙️ Cómo instalar el proyecto

Ejecuta los siguientes comandos en tu terminal para clonar el repositorio y configurar todo el entorno:

```bash
# 1. Clona el repositorio
git clone [https://github.com/tu-usuario/audio-noise-filter.git](https://github.com/tu-usuario/audio-noise-filter.git)
cd audio-noise-filter

# 2. Crea y activa el entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows usa: venv\Scripts\activate

# 3. Instala las dependencias necesarias
pip install -r requirements.txt