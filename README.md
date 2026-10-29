# EchoWriter AI: Transcripción Universal, Diarización & Traducción Local

Esta es una aplicación local para Windows diseñada para **transcribir audios de cualquier formato a texto limpio** (incluyendo notas de voz de WhatsApp, notas de audio, grabaciones de voz) e **identificar diferentes hablantes (diarización)** en reuniones y entrevistas. 

Está impulsada por **WhisperX**, **Pyannote Diarization (community-1 / 3.1)** y el motor de traducción neuronal **Meta NLLB-200**, con una interfaz web moderna en **Gradio** optimizada para tarjetas gráficas **NVIDIA (RTX 5050 / 8GB VRAM o superiores)**.

---

## ✨ Modos Principales y Novedades

### ⚡ 1. Extractor Rápido de Texto (WhatsApp / Audios / Notas de Voz)
* **Formatos soportados:** `.opus` (notas de WhatsApp), `.ogg`, `.mp3`, `.m4a`, `.wav`, `.aac`, `.flac`, `.wma`, `.webm` o grabación directa con micrófono.
* **Sin necesidad de Hugging Face Token:** Funciona al instante sin credenciales externas.
* **Caché Inteligente de Modelos en VRAM:** Tras la primera carga, los audios subsiguientes se transcriben de forma instantánea (< 1 segundo) sin recargar el modelo desde el disco.
* **Modelos Disponibles:** Selector de modelos Whisper: `turbo` (ultrarrápido), `large-v3` (máxima fidelidad acústica), `medium` y `small`.
* **Selector de Precisión GPU & Batch Size:** Alterna entre `float16` y `int8`, y ajusta el tamaño de lote (`batch_size` 1 a 32) para exprimir el paralelismo de la GPU.
* **Monitor de Rendimiento en Tiempo Real:** Métricas exactas en UI y consola de tiempo de proceso, velocidad (tiempo real `X x`), pico de VRAM GPU y consumo de RAM.
* **Traducción Neuronal Offline (Meta NLLB-200):** Traducción directa desde más de 200 idiomas al español sin descargas en runtime ni pérdida de calidad.
* **Botón "🧹 Liberar VRAM":** Desaloja el modelo de la tarjeta gráfica con 1 clic para dejar la GPU 100% libre para juegos u otras tareas.

### 👥 2. Transcripción Avanzada & Diarización (Multi-Hablante)
* **Identificación de Personas:** Separa quién dijo qué (`[SPEAKER_00]`, `[SPEAKER_01]`, etc.) con marcas de tiempo en milisegundos.
* **Modelos Pyannote SOTA:** Compatible con el nuevo `pyannote/speaker-diarization-community-1` y la versión `3.1`.
* **Alineación Fonética de Precisión:** Con modelos Wav2Vec2 para sincronización palabra por palabra.
* **Aislamiento de VRAM:** Desaloja automáticamente modelos en reposo para evitar cualquier error de memoria (OOM).

### 🎬 3. Auditoría Multimedia y Renombrado Rápido
* **Reproductor Multimedia Sincronizado:** Video y audio vinculados a la transcripción.
* **Tabla Interactiva:** Edición de texto y tiempos celda por celda.
* **🏷️ Renombrado Rápido de Hablantes:** Reemplaza etiquetas genéricas (ej. `SPEAKER_00` por "Carlos") en todo el documento y exportaciones con 1 clic.
* **Exportación Profesional y Nombres Dinámicos:** Subtítulos `.srt`, guion formateado en `.docx` y texto limpio en `.txt`, adoptando automáticamente el nombre del video o audio procesado (ej. `MiVideo_transcripcion.docx`, `MiVideo.srt`).

### 📦 4. Procesamiento por Lotes (Multi-Archivos, Cola y Carpeta Local)
* **Doble Modo de Entrada:**
  * Arrastra múltiples archivos de video o audio directamente a la interfaz web.
  * O escribe la ruta de una carpeta local en Windows (ej. `C:\Videos\MisGrabaciones`) para escanear y procesar todos sus archivos automáticamente.
* **🧹 Modo Subtítulos Limpios:** Casilla pre-configurable para exportar `.srt` y `.txt` sin etiquetas de hablantes (`[SPEAKER_XX]`), dejándolos 100% listos para importar directamente en Premiere Pro, DaVinci Resolve, CapCut o YouTube.
* **Cola Secuencial Inteligente:** Procesa archivo por archivo manteniendo el modelo en VRAM para máxima velocidad de inferencia sin sobrecalentamientos ni recargas lentas.
* **Traducción Automática al Español:** Traduce en lote con Meta NLLB-200 cualquier video en otro idioma.
* **Guardado Automático en Carpeta Origen:** Guarda los `.srt`, `.txt` y `.docx` al lado del video original en Windows.
* **📦 Descarga en Archivo ZIP:** Empaqueta todos los subtítulos y transcripciones generados en un archivo `.zip` descargable con un solo clic.

---

## 🚀 Requisitos Previos

1. **Python (3.10 a 3.12):** Instalado y marcado en el PATH de Windows.
2. **Git:** Necesario para descargar WhisperX.
3. **FFmpeg:** Indispensable para decodificar audio y video. Si no lo tienes:
   ```bash
   winget install ffmpeg
   ```
4. **Drivers NVIDIA y CUDA:** Tarjeta gráfica NVIDIA (serie RTX recomendada).

---

## 🔧 Instalación

El proceso de instalación está automatizado:

1. Haz doble clic en `install.bat`.
2. El script detectará la versión de CUDA instalada en tu sistema.
3. Creará el entorno virtual (`venv`) e instalará **PyTorch con aceleración CUDA**, **WhisperX**, **CTranslate2**, **Transformers** y dependencias de UI.

---

## 🔑 Configuración de Hugging Face (Solo para Diarización)

> **Nota:** Si solo deseas transcribir audios a texto limpio en la **Pestaña 1**, **NO necesitas token**.

Para la separación de hablantes (Pestaña 2):
1. Inicia sesión en [Hugging Face](https://huggingface.co/).
2. Acepta los términos de uso en:
   * 📄 [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) *(Recomendado)*
   * 📄 [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   * 📄 [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3. En [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), crea un token de lectura (**Read**) e ingrésalo en la pestaña 2 de la app (se guardará automáticamente).

---

## 💻 Uso de la Aplicación

1. Haz doble clic en `start_app.bat` para iniciar el servidor local.
2. Abre tu navegador web en `http://127.0.0.1:7860`.
3. Disfruta de transcripciones y diarizaciones a máxima velocidad en tu GPU local.
