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
git clone https://github.com/YandreyC/audio-noise-flitrer.git
cd audio-noise-flitrer

# 2. Crea y activa el entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows usa: venv\Scripts\activate

# 3. Instala las dependencias necesarias
pip install -r requirements.txt
```

## Uso

Coloca un archivo WAV en `data/input_audio.wav` y ejecuta:

```powershell
python src/filter.py
```

Por defecto se elimina el rango de 3000 a 4000 Hz. Puedes cambiarlo y elegir otros archivos:

```powershell
python src/filter.py --input data/input_audio.wav --output outputs/clean_audio.wav --min-hz 3000 --max-hz 4000 --plot plots/comparison.png
```

El programa conserva mono o estéreo, escribe el audio filtrado en `outputs/clean_audio.wav` y genera una comparación temporal y espectral en `plots/comparison.png`.

### Reducción automática de ruido

Si los primeros segundos del audio contienen principalmente ruido de fondo, puedes estimar su perfil automáticamente:

```powershell
python src/filter.py --auto-noise --noise-duration 1.5 --strength 1.5
```

`--noise-duration` indica cuántos segundos iniciales se usan para medir el ruido. `--strength` aumenta o reduce la agresividad: valores mayores eliminan más ruido, pero pueden afectar sonidos débiles. Este modo funciona mejor con ruido constante, como ventiladores o zumbidos.