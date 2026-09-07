import numpy as np
from scipy.io import wavfile
import os

def process_audio_filter():
    # Rutas de archivos
    input_path = "data/input_audio.wav"
    output_path = "outputs/clean_audio.wav"
    
    if not os.path.exists(input_path):
        print(f"❌ Error: No se encontró el archivo '{input_path}'. Coloca un archivo WAV de prueba ahí.")
        return

    print("📂 Leyendo archivo de audio real...")
    sample_rate, audio_data = wavfile.read(input_path)
    
    # Si el audio es estéreo (2 canales), lo convertimos a mono (1 canal) promediando ambos
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    
    # Convertir a tipo float para evitar desbordamientos durante la matemática compleja
    audio_float = audio_data.astype(np.float64)

    print("🔄 Aplicando Transformada Rápida de Fourier (FFT)...")
    # 1. FFT: Pasa del dominio del tiempo al dominio de la frecuencia (Arreglo de Números Complejos)
    fft_spectrum = np.fft.fft(audio_float)
    frequencies = np.fft.fftfreq(len(audio_float), 1/sample_rate)

    print("🛡️ Aplicando filtro espectral (Eliminando frecuencias indeseadas)...")
    # 2. DISEÑO DEL FILTRO: 
    # Aquí puedes aislar una frecuencia específica (ej. un zumbido eléctrico molesto).
    # Por ejemplo, vamos a atenuar frecuencias específicas de ruido (puedes ajustar este rango según tu audio).
    # Supongamos que queremos filtrar un ruido punzante o zumbido en un rango determinado (ej. entre 3000 Hz y 4000 Hz)
    # O un notch filter básico: apagamos un rango de frecuencias indeseadas multiplicando por 0.
    
    noise_min_hz = 3000  # Límite inferior del ruido a eliminar
    noise_max_hz = 4000  # Límite superior del ruido a eliminar
    
    # Creamos una copia o modificamos el espectro complejo directamente
    # Apagamos tanto las frecuencias positivas como sus simétricas negativas
    mask = (np.abs(frequencies) >= noise_min_hz) & (np.abs(frequencies) <= noise_max_hz)
    
    # Aplicamos la máscara multiplicando el espectro complejo por 0 en esas frecuencias
    fft_spectrum[mask] = 0.0

    print("🔙 Aplicando Transformada Inversa (IFFT) para reconstruir la señal...")
    # 3. IFFT: Regresa del dominio de la frecuencia al dominio del tiempo utilizando la parte real
    cleaned_float = np.ifft(fft_spectrum)
    cleaned_signal = np.real(cleaned_float)

    # Normalizar los valores de nuevo al rango de audio de 16 bits (-32768 a 32767) para evitar distorsión
    max_val = np.max(np.abs(cleaned_signal))
    if max_val > 0:
        cleaned_signal = cleaned_signal / max_val * 32767
    
    cleaned_int16 = cleaned_signal.astype(np.int16)

    # Asegurar que la carpeta 'outputs' exista
    os.makedirs("outputs", exist_ok=True)
    
    # Guardar el resultado limpio
    wavfile.write(output_path, sample_rate, cleaned_int16)
    print(f"✨ ¡Audio procesado y limpio guardado con éxito en: {output_path}!")

if _name_ == "_main_":
    process_audio_filter()