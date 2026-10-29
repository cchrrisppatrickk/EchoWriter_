# 🚀 Mejoras Futuras para EchoWriter AI

¡EchoWriter AI cuenta ahora con un motor de transcripción SOTA, caché inteligente en VRAM y traducción de alto rendimiento con Meta NLLB-200! Aquí tienes el seguimiento del progreso y las próximas ideas de evolución.

## 🌟 1. Mejoras de la Interfaz de Usuario (UI) y Experiencia

- [x] **Guardar el Token de Hugging Face automáticamente**: Lee y persiste el token en `.env`.
- [x] **Reproductor de Audio/Video Integrado:** Vista multimedia sincronizada con la tabla de edición.
- [x] **Control de Memoria VRAM en Tiempo Real:** Barra visual en la cabecera con botón de 1 clic para liberar la memoria de la tarjeta gráfica (`🧹 Liberar VRAM`).
- [x] **Procesamiento por Lotes (Batch Processing):** Procesamiento de múltiples videos/audios mediante cola interactiva o escaneo de carpetas en Windows, modo de subtítulos limpios, nombrado dinámico y descarga de todo en un único archivo ZIP.

## ⚡ 2. Optimizaciones de Rendimiento y Motor de IA

- [x] **Selección Manual de Idioma:** Ahorra tiempo de autodetección y previene saltos de idioma.
- [x] **Selector de Precisión de Hardware (`compute_type`):** Selector para alternar entre `float16` (máxima velocidad en GPU RTX) e `int8` (mínimo consumo de VRAM).
- [x] **Desbloqueo de Whisper Large V3:** Disponible junto a `turbo`, `medium` y `small`.
- [x] **Memoria Caché del Modelo (Singleton):** El modelo se mantiene caliente en VRAM tras la primera transcripción; audios consecutivos se transcriben en menos de 1 segundo.
- [x] **Traducción Neuronal Superior (Meta NLLB-200 vía CTranslate2):** Reemplazo de Argos por NLLB-200 INT8 con 200 idiomas directos a español sin descargas en runtime.
- [x] **Compatibilidad con Pyannote Diarization SOTA:** Selector entre `community-1` y `3.1`.
- [ ] **Ajustes Avanzados de Silencio (VAD):** Exponer en la UI los controles de sensibilidad de Silero VAD para evitar que corte audio durante ruidos fuertes.
- [ ] **Control de Alucinaciones (Glosario / Prompt Inicial):** Casilla de texto opcional para pasar un "Prompt" inicial a Whisper con términos especializados.

## 🎨 3. Formato, Edición y Salida Profesional

- [x] **Edición en Caliente:** Tabla interactiva en vivo antes de descargar archivos.
- [x] **Renombrado Interactivo de Hablantes:** Herramienta para renombrar masivamente etiquetas (`SPEAKER_00` -> `Carlos`) en todo el guion y descargas con 1 clic.
- [x] **Exportación a Guion Profesional (`.docx`):** Distinción visual de hablantes con colores y marcas de tiempo.
- [ ] **Fusión Inteligente de Diálogos (Smart Merge):** Filtro que una intervenciones consecutivas de la misma persona si la pausa es menor a 1.5s.
- [ ] **Generación Automática de Resumen (LLM):** Integración con modelo de lenguaje ligero (vía Ollama o GGUF local) para extraer "puntos clave" y "resumen ejecutivo".
