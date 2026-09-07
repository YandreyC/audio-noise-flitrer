"""Herramienta para eliminar ruido de archivos de audio WAV.

El módulo ofrece dos modos de trabajo:

* Filtro manual: elimina un rango de frecuencias indicado por el usuario.
* Filtro automático: estima el ruido a partir del inicio del audio y reduce
    las frecuencias que se parecen a ese perfil de ruido.

También puede guardar una comparación visual de la señal y su espectro.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import istft, stft


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# PROJECT_ROOT apunta a la carpeta del proyecto, dos niveles por encima de
# este archivo. Se usa para construir rutas predeterminadas portables.


def _to_float(audio_data):
    """Convierte las muestras del audio a `float64` para procesarlas.

    El procesamiento espectral necesita operar con valores decimales. La
    conversión evita modificar directamente el arreglo original leído por
    SciPy y permite trabajar con audio entero o de punto flotante.
    """
    return audio_data.astype(np.float64, copy=False)


def _restore_dtype(audio_data, original_data):
    """Devuelve el audio al tipo y rango del archivo original.

    Los formatos enteros se redondean y limitan a sus valores permitidos.
    Los formatos flotantes se limitan al rango normalizado [-1, 1]. Esto evita
    escribir muestras inválidas o producir saturación inesperada en el WAV.
    """
    if np.issubdtype(original_data.dtype, np.integer):
        info = np.iinfo(original_data.dtype)
        audio_data = np.clip(np.rint(audio_data), info.min, info.max)
    elif np.issubdtype(original_data.dtype, np.floating):
        audio_data = np.clip(audio_data, -1.0, 1.0)
    return audio_data.astype(original_data.dtype)


def _filter_channel(channel, sample_rate, noise_min_hz, noise_max_hz):
    """Elimina un rango de frecuencias de un canal mediante FFT/IFFT.

    La FFT transforma la señal del dominio temporal al frecuencial. Después
    se ponen a cero las frecuencias del rango solicitado, incluyendo sus
    equivalentes negativas, y la IFFT reconstruye la señal temporal.
    """
    spectrum = np.fft.fft(channel)
    frequencies = np.fft.fftfreq(channel.size, d=1 / sample_rate)
    mask = (np.abs(frequencies) >= noise_min_hz) & (np.abs(frequencies) <= noise_max_hz)
    spectrum[mask] = 0
    return np.real(np.fft.ifft(spectrum))


def _auto_filter_channel(channel, sample_rate, noise_duration, strength):
    """Reduce automáticamente el ruido de un canal usando STFT.

    Se calcula una representación tiempo-frecuencia del audio. La mediana
    de los primeros `noise_duration` segundos se toma como perfil de ruido;
    cada componente se atenúa según cuánto supera ese perfil.

    `strength` controla la sensibilidad: valores mayores exigen una señal
    más fuerte para conservarla. El inicio del audio debe contener ruido
    principalmente para que la estimación sea útil.
    """
    noise_samples = int(noise_duration * sample_rate)
    if noise_samples < 1 or noise_samples >= channel.size:
        raise ValueError("noise_duration debe ser mayor que 0 y menor que la duración del audio")
    if strength <= 0:
        raise ValueError("strength debe ser mayor que 0")

    # STFT divide la señal en ventanas solapadas para analizar cómo cambia
    # su contenido frecuencial a lo largo del tiempo.
    segment_length = min(2048, channel.size)
    _, times, spectrum = stft(
        channel,
        fs=sample_rate,
        nperseg=segment_length,
        noverlap=segment_length // 2,
        boundary="zeros",
    )
    noise_frames = np.searchsorted(times, noise_duration, side="right")
    noise_frames = max(1, min(noise_frames, spectrum.shape[1]))
    # La mediana es menos sensible a picos aislados que la media y produce un
    # perfil más estable del ruido de fondo.
    noise_profile = np.median(np.abs(spectrum[:, :noise_frames]), axis=1, keepdims=True)
    magnitude = np.abs(spectrum)
    threshold = noise_profile * strength
    ratio = np.divide(magnitude, threshold, out=np.zeros_like(magnitude), where=threshold > 0)
    # La ganancia queda entre 0.1 y 1 para reducir el ruido sin crear cortes
    # completos que puedan generar artefactos audibles.
    gain = np.clip((ratio - 1) / strength, 0.1, 1)
    gain = np.where(threshold <= np.finfo(np.float64).eps, 1, gain)
    filtered_spectrum = spectrum * gain
    _, cleaned = istft(
        filtered_spectrum,
        fs=sample_rate,
        nperseg=segment_length,
        noverlap=segment_length // 2,
        input_onesided=True,
        boundary=True,
    )
    return cleaned[:channel.size]


def _save_plots(original, cleaned, sample_rate, noise_min_hz, noise_max_hz, plot_path, auto_noise):
    """Guarda una gráfica comparando audio original y audio filtrado.

    La figura contiene la señal en el tiempo y la magnitud de su espectro.
    En modo manual se marca el rango eliminado; en modo automático se cambia
    el título para indicar que se aplicó reducción basada en el perfil de
    ruido.
    """
    original_mono = original.mean(axis=1) if original.ndim > 1 else original
    cleaned_mono = cleaned.mean(axis=1) if cleaned.ndim > 1 else cleaned
    frequencies = np.fft.rfftfreq(original_mono.size, 1 / sample_rate)
    original_spectrum = np.abs(np.fft.rfft(original_mono))
    cleaned_spectrum = np.abs(np.fft.rfft(cleaned_mono))
    time = np.arange(original_mono.size) / sample_rate

    figure, axes = plt.subplots(2, 1, figsize=(12, 7))
    axes[0].plot(time, original_mono, label="Original", alpha=0.7)
    axes[0].plot(time, cleaned_mono, label="Filtrada", alpha=0.7)
    axes[0].set(xlabel="Tiempo (s)", ylabel="Amplitud", title="Señal de audio")
    axes[0].legend()
    axes[1].plot(frequencies, original_spectrum, label="Original")
    axes[1].plot(frequencies, cleaned_spectrum, label="Filtrada")
    if auto_noise:
        axes[1].set_title("Espectro con reducción automática de ruido")
    else:
        axes[1].axvspan(noise_min_hz, noise_max_hz, color="red", alpha=0.15, label="Rango eliminado")
    axes[1].set(xlabel="Frecuencia (Hz)", ylabel="Magnitud", title="Espectro")
    axes[1].set_xlim(0, sample_rate / 2)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)


def process_audio_filter(
    input_path,
    output_path,
    noise_min_hz=3000,
    noise_max_hz=4000,
    plot_path=None,
    auto_noise=False,
    noise_duration=1.0,
    strength=1.5,
):
    """Procesa un WAV y guarda una versión filtrada.

    Args:
        input_path: Archivo WAV de entrada.
        output_path: Ruta donde se guardará el audio limpio.
        noise_min_hz: Frecuencia inicial del rango manual a eliminar.
        noise_max_hz: Frecuencia final del rango manual a eliminar.
        plot_path: Ruta de la gráfica; `None` desactiva su generación.
        auto_noise: Si es `True`, usa el filtro automático basado en STFT.
        noise_duration: Duración inicial usada para estimar el ruido.
        strength: Sensibilidad del filtro automático.

    Returns:
        La ruta del archivo WAV generado.

    Raises:
        FileNotFoundError: Si el archivo de entrada no existe.
        ValueError: Si el audio, el rango o los parámetros no son válidos.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo WAV: {input_path}")
    if not auto_noise and (noise_min_hz < 0 or noise_max_hz <= noise_min_hz):
        raise ValueError("El rango de frecuencias no es válido")

    sample_rate, original_data = wavfile.read(input_path)
    if original_data.size == 0:
        raise ValueError("El archivo WAV está vacío")
    if not auto_noise and noise_max_hz >= sample_rate / 2:
        raise ValueError(f"noise_max_hz debe ser menor que Nyquist ({sample_rate / 2:g} Hz)")

    audio_float = _to_float(original_data)
    # Cada canal se procesa por separado para conservar la estructura mono o
    # estéreo del archivo original.
    filter_channel = (
        lambda channel: _auto_filter_channel(channel, sample_rate, noise_duration, strength)
        if auto_noise
        else _filter_channel(channel, sample_rate, noise_min_hz, noise_max_hz)
    )
    if audio_float.ndim == 1:
        cleaned_float = filter_channel(audio_float)
    else:
        cleaned_float = np.column_stack([
            filter_channel(audio_float[:, channel])
            for channel in range(audio_float.shape[1])
        ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(output_path, sample_rate, _restore_dtype(cleaned_float, original_data))
    if plot_path is not None:
        plot_path = Path(plot_path)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        _save_plots(audio_float, cleaned_float, sample_rate, noise_min_hz, noise_max_hz, plot_path, auto_noise)
    return output_path


def main():
    """Define la interfaz de consola y ejecuta el procesamiento solicitado."""
    parser = argparse.ArgumentParser(description="Reduce ruido de fondo o elimina un rango de frecuencias de un WAV")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "input_audio.wav")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "clean_audio.wav")
    parser.add_argument("--min-hz", type=float, default=3000)
    parser.add_argument("--max-hz", type=float, default=4000)
    parser.add_argument("--plot", type=Path, default=PROJECT_ROOT / "plots" / "comparison.png")
    parser.add_argument("--auto-noise", action="store_true", help="estima el ruido usando el inicio del audio")
    parser.add_argument("--noise-duration", type=float, default=1.0, help="segundos iniciales usados como perfil de ruido")
    parser.add_argument("--strength", type=float, default=1.5, help="sensibilidad de la puerta espectral automática")
    # argparse convierte automáticamente las rutas y valores numéricos al
    # tipo declarado en cada argumento.
    args = parser.parse_args()
    try:
        output = process_audio_filter(
            args.input,
            args.output,
            args.min_hz,
            args.max_hz,
            args.plot,
            args.auto_noise,
            args.noise_duration,
            args.strength,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(f"Audio filtrado guardado en: {output}")


if __name__ == "__main__":
    main()