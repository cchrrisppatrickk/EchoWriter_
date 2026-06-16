# 🚀 Mejoras Futuras para EchoWriter AI

¡Felicidades por tener tu aplicación de IA de transcripción y diarización funcionando al 100% de manera local! Aquí tienes una lista de ideas y funcionalidades que podrías añadir en el futuro para llevar la aplicación de un prototipo funcional a una herramienta de nivel profesional y comercial.

## 🌟 1. Mejoras de la Interfaz de Usuario (UI) y Experiencia

- [x] **Guardar el Token de Hugging Face automáticamente**: Modificar `app.py` para leer un archivo `.env` invisible que guarde el token.
- [ ] **Procesamiento por Lotes (Batch Processing):** Permitir subir múltiples audios/videos a la vez, procesarlos secuencialmente y generar un ZIP final.
- [x] **Reproductor de Audio/Video Integrado:** Mostrar el archivo multimedia subido junto a la caja de transcripción.
- [ ] **Colores para Hablantes:** Usar HTML/Markdown enriquecido en la caja de salida para darle colores distintos a cada "SPEAKER_XX".

## ⚡ 2. Optimizaciones de Rendimiento

- [x] **Selección Manual de Idioma:** Agregar un menú desplegable para elegir el idioma y ahorrar el tiempo de autodetección, ganando también precisión.
- [ ] **Selector de Precisión (Hardware):** Selector UI para escoger entre `float16` y `int8` dependiendo de los recursos del PC o tamaño del audio.
- [ ] **Control de Alucinaciones (Glosario):** Casilla de texto opcional para pasar un "Glosario" inicial a la IA con términos especializados.

## 🛠️ 3. Nuevas Funcionalidades del Motor

- [x] **Traducción Automática Integrada:** Habilitar el task de traducción ("translate") de Whisper para que transcripciones en otros idiomas salgan automáticamente en inglés.
- [ ] **Nuevos Formatos de Exportación:** Añadir descargas en formato `.vtt`, `.docx` o `PDF` con formato pulcro y distinguido por colores.
- [ ] **Ajustes Avanzados de Silencio (VAD):** Exponer en la UI los controles de sensibilidad de Silero VAD para evitar que corte audio durante ruidos fuertes.
- [ ] **Edición en Caliente:** Permitir editar la transcripción final en la interfaz web de Gradio antes de descargar el archivo.
