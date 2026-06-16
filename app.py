import os
import gc
import tempfile
import torch
import gradio as gr
import whisperx
from whisperx.diarize import DiarizationPipeline
from dotenv import load_dotenv, set_key
import argostranslate.package
import argostranslate.translate

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(env_path)

def format_timestamp(seconds: float) -> str:
    """Convierte segundos a formato de tiempo SRT: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int(round((seconds - int(seconds)) * 1000))
    # Asegurar límites de milisegundos
    milliseconds = min(999, max(0, milliseconds))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

def transcribe_and_diarize(file_path, hf_token, whisper_model, source_language_ui, min_speakers, max_speakers, traducir_a_es, progress=gr.Progress(track_tqdm=True)):
    # Validaciones iniciales
    if not file_path:
        return "Error: Por favor, carga un archivo de audio o video.", None, None, gr.update()
    if not hf_token or not hf_token.strip():
        return "Error: El token de Hugging Face es obligatorio para descargar los modelos de Diarización (Pyannote).", None, None, gr.update()
        
    # Auto-guardado del token en .env para la próxima vez
    current_token = os.getenv("HF_TOKEN", "")
    if hf_token.strip() != current_token:
        try:
            if not os.path.exists(env_path):
                open(env_path, 'w').close()
            set_key(env_path, "HF_TOKEN", hf_token.strip())
            os.environ["HF_TOKEN"] = hf_token.strip()
            print("[INFO] Token de Hugging Face guardado localmente en .env.")
        except Exception as e:
            print(f"[WARNING] No se pudo guardar el token en .env: {e}")
            
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return "Error: No se detectó una GPU compatible con CUDA. Esta aplicación requiere aceleración por GPU (NVIDIA) para funcionar.", None, None, gr.update()
        
    compute_type = "int8"  # int8 evita errores de kernel ('no kernel image') en GPUs más antiguas y optimiza la VRAM.
    batch_size = 4
    
    temp_txt_path = None
    temp_srt_path = None
    
    try:
        # --- PASO 1: Transcripción con WhisperX ---
        progress(0.1, desc="Cargando modelo WhisperX en VRAM...")
        print(f"[PROCESS] Cargando modelo WhisperX: {whisper_model}...")
        model = whisperx.load_model(whisper_model, device, compute_type=compute_type)
        
        progress(0.2, desc="Procesando audio de entrada...")
        print("[PROCESS] Cargando audio...")
        audio = whisperx.load_audio(file_path)
        
        progress(0.3, desc="Ejecutando transcripción de audio...")
        print("[PROCESS] Iniciando transcripción...")
        
        # Mapeo de idioma de la UI a código Whisper
        lang_map = {
            "Español": "es", "Inglés": "en", "Francés": "fr", 
            "Alemán": "de", "Italiano": "it", "Portugués": "pt"
        }
        language_code = lang_map.get(source_language_ui, None)
        
        if language_code:
            raw_result = model.transcribe(audio, batch_size=batch_size, language=language_code)
        else:
            raw_result = model.transcribe(audio, batch_size=batch_size)
        
        # Guardar idioma detectado para cargar el modelo de alineación correspondiente
        detected_language = raw_result.get("language", "es")
        print(f"[PROCESS] Idioma detectado: {detected_language}")
        
        # Limpieza de memoria (VRAM) inmediata
        progress(0.4, desc="Liberando VRAM de transcripción...")
        print("[PROCESS] Liberando memoria de transcripción...")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        # --- PASO 2: Alineación fina (Wav2Vec2) ---
        progress(0.5, desc="Cargando modelo de alineación fonética...")
        print("[PROCESS] Cargando modelo de alineación...")
        align_model, metadata = whisperx.load_align_model(language_code=detected_language, device=device)
        
        progress(0.6, desc="Alineando palabras con precisión de milisegundos...")
        print("[PROCESS] Ejecutando alineación...")
        aligned_result = whisperx.align(raw_result["segments"], align_model, metadata, audio, device, return_char_alignments=False)
        
        # Limpieza de memoria (VRAM) inmediata
        progress(0.7, desc="Liberando VRAM de alineación...")
        print("[PROCESS] Liberando memoria de alineación...")
        del align_model
        gc.collect()
        torch.cuda.empty_cache()
        
        # --- PASO 3: Diarización (Pyannote) ---
        progress(0.8, desc="Iniciando Diarización de hablantes (Pyannote)...")
        print("[PROCESS] Inicializando pipeline de Diarización...")
        
        # Configurar los parámetros opcionales de cantidad de hablantes
        diarize_kwargs = {}
        if min_speakers is not None and min_speakers > 0:
            diarize_kwargs["min_speakers"] = int(min_speakers)
        if max_speakers is not None and max_speakers > 0:
            diarize_kwargs["max_speakers"] = int(max_speakers)
            
        diarize_model = DiarizationPipeline(token=hf_token.strip(), device=device)
        
        progress(0.85, desc="Separando pistas de voces y hablantes...")
        print("[PROCESS] Ejecutando diarización...")
        diarize_segments = diarize_model(audio, **diarize_kwargs)
        
        # Limpieza de memoria (VRAM) inmediata
        progress(0.9, desc="Liberando VRAM de diarización...")
        print("[PROCESS] Liberando memoria de diarización...")
        del diarize_model
        gc.collect()
        torch.cuda.empty_cache()
        
        # --- PASO 4: Fusión final de palabras y hablantes ---
        progress(0.95, desc="Fusionando texto con etiquetas de hablantes...")
        print("[PROCESS] Asignando hablantes a palabras...")
        final_result = whisperx.assign_word_speakers(diarize_segments, aligned_result)
        
        # --- PASO 4.5: Traducción (Opcional) ---
        if traducir_a_es and detected_language != "es":
            progress(0.96, desc="Inicializando motor de Traducción (Argos)...")
            print("[PROCESS] Preparando traducción a Español...")
            try:
                argostranslate.package.update_package_index()
                available_packages = argostranslate.package.get_available_packages()
                
                # Buscar el paquete adecuado (Origen -> Español)
                package_to_install = next(
                    filter(
                        lambda x: x.from_code == detected_language and x.to_code == "es", available_packages
                    ), None
                )
                
                if package_to_install is not None:
                    progress(0.97, desc=f"Descargando paquete de idioma ({detected_language} -> es) si es necesario...")
                    package_to_install.download()
                    package_to_install.install()
                    
                    installed_languages = argostranslate.translate.get_installed_languages()
                    from_lang = list(filter(lambda x: x.code == detected_language, installed_languages))[0]
                    to_lang = list(filter(lambda x: x.code == "es", installed_languages))[0]
                    translation_obj = from_lang.get_translation(to_lang)
                    
                    progress(0.98, desc="Traduciendo transcripción al Español...")
                    print("[PROCESS] Traducción en progreso...")
                    for segment in final_result["segments"]:
                        original_text = segment.get("text", "").strip()
                        if original_text:
                            segment["text"] = translation_obj.translate(original_text)
                else:
                    print(f"[WARNING] No se encontró paquete de traducción directo de '{detected_language}' a 'es'.")
            except Exception as tr_e:
                print(f"[ERROR] Falló la traducción: {tr_e}")

        # --- PASO 5: Generar y formatear salidas ---
        progress(0.99, desc="Formateando subtítulos finales...")
        text_lines = []
        srt_lines = []
        srt_counter = 1
        
        for segment in final_result["segments"]:
            speaker = segment.get("speaker", "SPEAKER_UNKNOWN")
            text = segment.get("text", "").strip()
            start = segment.get("start", 0.0)
            end = segment.get("end", 0.0)
            
            # Formato de visualización de texto en la caja de la UI
            display_line = f"[{speaker}] ({start:.2f}s - {end:.2f}s): {text}"
            text_lines.append(display_line)
            
            # Formato de archivo SRT
            srt_lines.append(f"{srt_counter}")
            srt_lines.append(f"{format_timestamp(start)} --> {format_timestamp(end)}")
            srt_lines.append(f"[{speaker}]: {text}\n")
            srt_counter += 1
            
        full_text = "\n".join(text_lines)
        full_srt = "\n".join(srt_lines)
        
        if not full_text:
            full_text = "Proceso terminado, pero no se detectó contenido de voz en el archivo."
            
        # Guardar en archivos físicos temporales para descarga
        temp_dir = tempfile.gettempdir()
        temp_txt_path = os.path.join(temp_dir, "transcripcion_diarizada.txt")
        temp_srt_path = os.path.join(temp_dir, "transcripcion_diarizada.srt")
        
        with open(temp_txt_path, "w", encoding="utf-8") as f:
            f.write(full_text)
            
        with open(temp_srt_path, "w", encoding="utf-8") as f:
            f.write(full_srt)
            
        progress(1.0, desc="¡Proceso finalizado!")
        print("[SUCCESS] Proceso de diarización y transcripción completado.")
        return full_text, temp_txt_path, temp_srt_path, gr.Tabs(selected=2)
        
    except Exception as e:
        error_msg = f"ERROR EN EL PROCESAMIENTO:\n\n{str(e)}\n\n"
        error_msg += "Sugerencia: Asegúrate de que el token de Hugging Face sea válido y de haber aceptado las licencias de Pyannote en su web."
        print(f"[ERROR] {error_msg}")
        
        # Liberar memoria VRAM en caso de error crítico
        try:
            gc.collect()
            torch.cuda.empty_cache()
        except:
            pass
            
        return error_msg, None, None, gr.update()

# --- CSS Personalizado para una Estética Premium Oscura ---
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

body, .gradio-container {
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    background: radial-gradient(circle at top, #1e1b4b 0%, #030712 100%) !important;
}

/* Panel principal con Glassmorphism */
.glass-panel {
    background: rgba(17, 24, 39, 0.7) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 20px !important;
    box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.5) !important;
    padding: 25px !important;
}

/* Header estilizado */
.app-header {
    text-align: center;
    padding: 2rem 1rem;
    margin-bottom: 2rem;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(168, 85, 247, 0.1) 100%);
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: 16px;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
}

.app-title {
    font-size: 2.6rem;
    font-weight: 700;
    margin: 0;
    background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.03em;
}

.app-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    margin-top: 0.5rem;
    font-weight: 400;
}

/* Botones de acción principal */
.action-btn {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
}

.action-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.5) !important;
}

/* Modificaciones de cajas de Gradio */
input, textarea, select {
    transition: border-color 0.2s, box-shadow 0.2s !important;
}

input:focus, textarea:focus, select:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 3px rgba(129, 140, 248, 0.25) !important;
}
"""

# --- CONSTRUCCIÓN DE LA INTERFAZ DE USUARIO ---
with gr.Blocks(theme=gr.themes.Default(primary_hue="indigo", secondary_hue="slate", neutral_hue="zinc"), css=custom_css) as demo:
    
    # Encabezado principal
    gr.HTML(
        """
        <div class="app-header">
            <h1 class="app-title">EchoWriter AI</h1>
            <p class="app-subtitle">Transcripción Local Inteligente & Separación de Hablantes (Diarización)</p>
        </div>
        """
    )
    
    with gr.Tabs() as main_tabs:
        with gr.TabItem("🎬 1. Configuración y Video", id=1):
            with gr.Column(elem_classes="glass-panel"):
                gr.Markdown("### 🛠️ Archivo Multimedia y Configuración")
                
                audio_video_input = gr.Video(
                    label="Reproductor Integrado (Sube o arrastra tu video/audio aquí)",
                    sources=["upload"],
                    interactive=True
                )
                
                with gr.Accordion("⚙️ Parámetros del Motor de IA", open=True):
                    hf_token_input = gr.Textbox(
                        label="Token de Hugging Face (pyannote)",
                        placeholder="hf_...",
                        type="password",
                        value=os.getenv("HF_TOKEN", ""),
                        info="Requerido para la diarización. Asegúrate de haber aceptado la licencia de Pyannote 3.1 en Hugging Face."
                    )
                    
                    with gr.Row():
                        whisper_model_input = gr.Dropdown(
                            label="Modelo de Transcripción Whisper",
                            choices=["small", "medium", "turbo"],
                            value="turbo",
                            info="El modelo 'large' está deshabilitado para evitar errores (OOM)."
                        )
                        
                        source_language_input = gr.Dropdown(
                            label="Idioma del Audio (Opcional)",
                            choices=["Automático", "Español", "Inglés", "Francés", "Alemán", "Italiano", "Portugués"],
                            value="Automático",
                            info="Seleccionar el idioma manualmente salta el paso de auto-detección."
                        )
                    
                    with gr.Row():
                        min_speakers_input = gr.Number(
                            label="Min Hablantes (Opcional)", 
                            precision=0, 
                            minimum=0,
                            value=0
                        )
                        max_speakers_input = gr.Number(
                            label="Max Hablantes (Opcional)", 
                            precision=0, 
                            minimum=0,
                            value=0
                        )
                        
                    traducir_es_input = gr.Checkbox(
                        label="Traducir transcripción al Español (Offline)",
                        value=False,
                        info="Si el audio está en otro idioma, la IA lo traducirá automáticamente al español."
                    )
                    
                submit_btn = gr.Button("Comenzar Transcripción", elem_classes="action-btn")
        
        with gr.TabItem("📝 2. Resultados", id=2):
            with gr.Column(elem_classes="glass-panel"):
                gr.Markdown("### 📝 Resultados de la Transcripción y Diarización")
                
                output_textbox = gr.Textbox(
                    label="Transcripción por Segmentos",
                    placeholder="Los resultados se mostrarán aquí...",
                    lines=15,
                    max_lines=30,
                    show_copy_button=True
                )
                
                with gr.Row():
                    txt_download_btn = gr.File(
                        label="Descargar Transcripción (.txt)",
                        interactive=False
                    )
                    srt_download_btn = gr.File(
                        label="Descargar Subtítulos (.srt)",
                        interactive=False
                    )

    # Vinculación de eventos
    submit_btn.click(
        fn=transcribe_and_diarize,
        inputs=[
            audio_video_input,
            hf_token_input,
            whisper_model_input,
            source_language_input,
            min_speakers_input,
            max_speakers_input,
            traducir_es_input
        ],
        outputs=[
            output_textbox,
            txt_download_btn,
            srt_download_btn,
            main_tabs
        ]
    )

# Lanzar aplicación localmente
if __name__ == "__main__":
    # La aplicación corre localmente en el puerto 7860
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
