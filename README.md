# EchoWriter AI: Transcripción & Diarización Local

Esta es una aplicación local para Windows diseñada para transcribir audios/videos e identificar los diferentes hablantes (diarización), impulsada por **WhisperX** y **Pyannote Diarization**. La aplicación cuenta con una interfaz web moderna en **Gradio** y está optimizada para ejecutarse en computadoras con tarjetas gráficas **NVIDIA (como la RTX 5050 de 8GB VRAM)** sin agotar la memoria de video.

---

## 🚀 Requisitos Previos

Antes de ejecutar el instalador, asegúrate de tener instalado en tu sistema:
1. **Python (versión 3.10 o posterior):** Debe estar en el PATH de Windows.
2. **Git:** Necesario para descargar WhisperX.
3. **FFmpeg:** Indispensable para decodificar archivos de audio/video. Si no lo tienes, puedes instalarlo desde una terminal de Windows ejecutando:
   ```bash
   winget install ffmpeg
   ```
4. **Drivers de NVIDIA y CUDA:** Asegúrate de tener tu tarjeta gráfica configurada con soporte para CUDA.

---

## 🔧 Instalación

El proceso de instalación está totalmente automatizado:

1. Haz doble clic en el archivo `install.bat`.
2. El script validará que cumplas con los pre-requisitos (Python, Git, FFmpeg).
3. Detectará de forma inteligente la versión de CUDA instalada en tu sistema host.
4. Creará un entorno virtual local (`venv`) e instalará la versión correspondiente de **PyTorch con CUDA**.
5. Instalará **WhisperX** y las dependencias adicionales (Gradio).

*Nota: La instalación puede tardar varios minutos dependiendo de tu velocidad de conexión a internet, ya que descarga PyTorch con soporte para GPU (~2.5 GB).*

---

## 🔑 Configuración Obligatoria de Hugging Face

Para utilizar la funcionalidad de separación de hablantes (diarización), el pipeline hace uso de los modelos pre-entrenados de Pyannote hospedados en **Hugging Face**. Sigue estos pasos para obtener el acceso:

1. Crea o inicia sesión en tu cuenta de [Hugging Face](https://huggingface.co/).
2. Acepta de manera individual los términos y condiciones de los siguientes dos modelos:
   * 📄 [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   * 📄 [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3. Ve a la sección de configuración de tu perfil en Hugging Face: **Settings -> Access Tokens** (o ingresa a [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)).
4. Crea un nuevo token de tipo **Read** (lectura) y cópialo. Lo necesitarás para ingresarlo en el campo correspondiente en la interfaz de la aplicación.

---

## 💻 Uso de la Aplicación

1. Haz doble clic en `start_app.bat` para iniciar el servidor local.
2. Se abrirá una terminal y se iniciará Gradio. En unos segundos, verás que la aplicación está lista e indicará una dirección local similar a: `http://127.0.0.1:7860`.
3. Abre esa dirección en tu navegador web.
4. **Instrucciones en la Interfaz:**
   * **Carga tu archivo** de audio o video.
   * **Introduce tu Token de Hugging Face** en el campo enmascarado.
   * **Selecciona el modelo de Whisper** (`turbo` es la opción por defecto y recomendada por su excelente relación velocidad/precisión). El modelo `large` se encuentra deshabilitado para evitar errores de falta de memoria (Out of Memory) en GPUs de 8GB de VRAM.
   * *(Opcional)* Especifica el número mínimo o máximo de hablantes si los conoces con certeza.
   * Presiona **Comenzar Transcripción**.
5. Al finalizar el procesamiento, verás el texto estructurado con etiquetas como `[SPEAKER_01] (0.00s - 4.50s): Hola mundo`.
6. Utiliza los botones de descarga de la derecha para exportar los resultados a formato **TXT** o a formato de subtítulos **SRT** alineados temporalmente.

---

## 🛠️ Optimización y Gestión de Memoria

Esta aplicación está diseñada para no sobrecargar la GPU. Utiliza un pipeline por etapas:
* Primero transcribe el audio y libera completamente la memoria gráfica.
* Luego alinea los textos fonéticamente y vuelve a limpiar la VRAM.
* Finalmente, realiza la diarización con Pyannote y libera la memoria.

Esto permite que tarjetas de **8GB de VRAM** ejecuten tareas complejas sin interrupciones ni bloqueos de pantalla.
