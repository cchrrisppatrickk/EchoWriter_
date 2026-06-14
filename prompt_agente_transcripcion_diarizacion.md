# Especificación Técnica y Blueprint: App Local de Transcripción y Diarización (WhisperX + Gradio)

Este documento es un **Prompt de Sistema de Alta Precisión** para el agente de IA. Contiene las reglas estrictas, la arquitectura y los scripts de instalación necesarios para construir una aplicación de escritorio local para transcripción y separación de hablantes (diarización).

---

## 1. OBJETIVO DEL PROYECTO
Desarrollar una aplicación web local (interfaz Gradio) impulsada por **WhisperX**. La aplicación se centrará EXCLUSIVAMENTE en: **Transcribir audio/video e Identificar a los hablantes (Diarización)**. 
Debe estar empaquetada con scripts `.bat` para una instalación aislada y automatizada en Windows, optimizada para una **NVIDIA RTX 5050 (8GB VRAM)**.

---

## 2. ARQUITECTURA DE ARCHIVOS REQUERIDA
El agente debe generar la siguiente estructura de proyecto:

```text
whisperx-diarization-app/
│
├── install.bat            # Script inteligente de instalación (crea venv, instala PyTorch CUDA y dependencias)
├── start_app.bat          # Script para lanzar la interfaz de usuario
├── app.py                 # Lógica principal y UI en Gradio
├── requirements.txt       # Dependencias adicionales (Gradio, etc.)
└── README.md              # Instrucciones de uso
```

---

## 3. LÓGICA DEL SCRIPT DE INSTALACIÓN (install.bat)
El agente debe escribir un archivo `install.bat` robusto que ejecute secuencialmente lo siguiente:
1. **Validación de Sistema:** Comprobar si `python` y `git` están instalados y en el PATH. Comprobar si `ffmpeg` está en el PATH (si no está, pausar y advertir al usuario que debe instalarlo vía `winget install ffmpeg`).
2. **Aislamiento (venv):** Crear un entorno virtual (`python -m venv venv`) y activarlo.
3. **Detección Inteligente de GPU:** Ejecutar `nvidia-smi` para detectar la versión de CUDA instalada en el sistema host.
4. **Instalación de PyTorch:** Basado en la versión de CUDA detectada, instalar la rueda (wheel) específica de PyTorch (ej. `cu118`, `cu121`, `cu124`). Si no hay NVIDIA, detenerse informando que este proyecto requiere GPU.
5. **Instalación de WhisperX:** Ejecutar estrictamente `pip install git+https://github.com/m-bain/WhisperX.git`
6. **Dependencias Extra:** Instalar `gradio` leyendo el `requirements.txt`.

*El script `start_app.bat` simplemente debe activar el venv (`call venv\Scripts\activate`) y ejecutar `python app.py`.*

---

## 4. REQUISITOS DE LA INTERFAZ DE USUARIO (app.py con Gradio)
La interfaz debe ser sencilla y enfocada en el flujo de trabajo de diarización:

**Inputs (Entradas):**
1. **Carga de Archivo:** Componente de arrastrar y soltar compatible con video y audio (`.mp4`, `.mp3`, `.wav`, etc.).
2. **Token de Hugging Face:** Un campo de texto enmascarado (`type="password"`) OBLIGATORIO para descargar los modelos de Pyannote. (Debe incluir un texto de ayuda recordando al usuario que debe aceptar los términos de pyannote en HF).
3. **Selección de Modelo Whisper:** Dropdown con `small`, `medium`, `turbo`. (Valor por defecto: `turbo`). *El agente debe bloquear u ocultar el modelo `large`, ya que la suma de Whisper large + Pyannote Diarization excederá los 8GB de VRAM.*
4. **Número de Hablantes (Opcional):** Dos campos numéricos (Min Speakers / Max Speakers).

**Outputs (Salidas):**
1. Un `gradio.Textbox` grande para mostrar el texto resultante con los hablantes. (Ej: `[SPEAKER_01]: Hola Mundo`).
2. Un botón de descarga para exportar el resultado como archivo `.txt` y `.srt` (con etiquetas de hablantes en los tiempos correspondientes).

---

## 5. LÓGICA DEL MOTOR (WhisperX Pipeline)
El código en `app.py` debe implementar el pipeline de WhisperX gestionando la memoria con cuidado:

1. **Paso 1 - Transcripción:** Cargar modelo WhisperX con `compute_type="float16"` y forzar `batch_size=4`. Realizar transcripción de los segmentos.
2. **Paso 2 - Limpieza de Memoria:** *CRÍTICO.* Antes de pasar a los siguientes modelos, el agente debe programar la limpieza de la memoria VRAM usando `gc.collect()` y `torch.cuda.empty_cache()` para evitar el error Out of Memory en la RTX 5050.
3. **Paso 3 - Alineación:** Cargar modelo wav2vec2, alinear segmentos y borrar el modelo de la memoria (repetir limpieza).
4. **Paso 4 - Diarización:** Inyectar el Token de Hugging Face provisto por el usuario en el componente de UI a la función `DiarizationPipeline(use_auth_token=TOKEN)`.
5. **Paso 5 - Fusión:** Ejecutar `assign_word_speakers` y formatear el resultado.

---

## 6. INSTRUCCIÓN FINAL PARA EL AGENTE
Agente, tu tarea es generar el código completo de los archivos listados en la Arquitectura (Sección 2). No asumas que el usuario configurará el entorno manualmente. El script `install.bat` es el pilar de este proyecto. Asegúrate de que el manejo de errores en Python capture fallos de autenticación (Token HF inválido) o de falta de FFmpeg, mostrándolos limpiamente en la interfaz de Gradio.