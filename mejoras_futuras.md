# 🚀 Mejoras Futuras para EchoWriter AI

¡Felicidades por tener tu aplicación de IA de transcripción y diarización funcionando al 100% de manera local! Aquí tienes una lista de ideas y funcionalidades que podrías añadir en el futuro para llevar la aplicación de un prototipo funcional a una herramienta de nivel profesional y comercial.

## 🌟 1. Mejoras de la Interfaz de Usuario (UI) y Experiencia

- [x] **Guardar el Token de Hugging Face automáticamente**: Modificar `app.py` para leer un archivo `.env` invisible que guarde el token.
- [ ] **Procesamiento por Lotes (Batch Processing):** Permitir subir múltiples audios/videos a la vez, procesarlos secuencialmente y generar un ZIP final.
- [x] **Reproductor de Audio/Video Integrado:** Mostrar el archivo multimedia subido junto a la caja de transcripción.


## ⚡ 2. Optimizaciones de Rendimiento

- [x] **Selección Manual de Idioma:** Agregar un menú desplegable para elegir el idioma y ahorrar el tiempo de autodetección, ganando también precisión.
- [ ] **Selector de Precisión (Hardware):** Selector UI para escoger entre `float16` y `int8` dependiendo de los recursos del PC o tamaño del audio.
- [ ] **Control de Alucinaciones (Glosario):** Casilla de texto opcional para pasar un "Glosario" inicial a la IA con términos especializados.

## 🛠️ 3. Nuevas Funcionalidades del Motor

- [x] **Traducción Automática Integrada:** Habilitar el task de traducción ("translate") de Whisper para que transcripciones en otros idiomas salgan automáticamente en inglés.
- [ ] **Nuevos Formatos de Exportación:** Añadir descargas en formato `.vtt`, `.docx` o `PDF` con formato pulcro y distinguido por colores.
- [ ] **Ajustes Avanzados de Silencio (VAD):** Exponer en la UI los controles de sensibilidad de Silero VAD para evitar que corte audio durante ruidos fuertes.
- [ ] **Edición en Caliente:** Permitir editar la transcripción final en la interfaz web de Gradio antes de descargar el archivo.

## 🎨 4. Mejoras Avanzadas de Formato y Transcripción

- [ ] **Colores Dinámicos para Hablantes:** Usar HTML/Markdown enriquecido en la caja de salida para darle un color distintivo y consistente a cada `SPEAKER_XX` en toda la transcripción.
- [ ] **Renombrado Interactivo de Hablantes:** Una interfaz donde si la app detecta "SPEAKER_00" y "SPEAKER_01", puedas escribir sus nombres reales ("Juan", "Entrevistador") y reemplazarlos en todo el documento con un clic antes de exportarlo.
- [ ] **Fusión Inteligente de Diálogos (Smart Merge):** A veces la IA corta la frase de un hablante en dos líneas si este respira profundo. Un filtro que una intervenciones consecutivas de la misma persona haría la lectura mucho más fluida.
- [ ] **Exportación Profesional a Guion:** Permitir descargar la transcripción en formato `.pdf` o `.docx` con un formato de guion de entrevista profesional (nombres en negrita, colores, márgenes correctos), listo para entregar a clientes.
- [ ] **Generación Automática de Resumen (LLM):** Integrar una opción que, al terminar la transcripción, la envíe a un modelo de lenguaje ligero para que extraiga los "puntos clave" y un "resumen ejecutivo" de la reunión.
