import os
import gc
import re
import time
import tempfile
import zipfile
import shutil
import psutil
import torch
import gradio as gr
import whisperx
from whisperx.diarize import DiarizationPipeline
from dotenv import load_dotenv, set_key
import pandas as pd
from docx import Document
from docx.shared import RGBColor, Pt
import transformers
import ctranslate2

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(env_path)

# ==========================================
# 🌐 MOTOR DE TRADUCCIÓN: META NLLB-200 (CTranslate2)
# ==========================================
WHISPER_TO_NLLB = {
    "es": "spa_Latn", "en": "eng_Latn", "fr": "fra_Latn",
    "de": "deu_Latn", "it": "ita_Latn", "pt": "por_Latn",
    "ca": "cat_Latn", "gl": "glg_Latn", "eu": "eus_Latn",
    "nl": "nld_Latn", "ru": "rus_Cyrl", "uk": "ukr_Cyrl",
    "pl": "pol_Latn", "ro": "ron_Latn", "zh": "zho_Hans",
    "ja": "jpn_Jpan", "ko": "kor_Hang", "ar": "ara_Arab",
    "hi": "hin_Deva", "tr": "tur_Latn", "sv": "swe_Latn",
    "el": "ell_Grek", "he": "heb_Hebr", "cs": "ces_Latn",
    "da": "dan_Latn", "fi": "fin_Latn", "hu": "hun_Latn",
    "no": "nob_Latn",
}

class NLLBTranslator:
    def __init__(self, model_rel_path="models/nllb-200-ct2-int8"):
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_rel_path)
        self.tokenizer = None
        self.translator = None

    def _ensure_loaded(self):
        if self.translator is None:
            if not os.path.exists(self.model_path) or not os.path.exists(os.path.join(self.model_path, "model.bin")):
                print("[NLLB] Descargando modelo NLLB-200 INT8 localmente (sin enlaces simbólicos)...")
                from huggingface_hub import snapshot_download
                snapshot_download("JustFrederik/nllb-200-distilled-600M-ct2-int8", local_dir=self.model_path)

            print("[NLLB] Inicializando motor de traducción NLLB-200...")
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_path)
            # Inferencia en CPU para preservar el 100% de la VRAM para Whisper y Pyannote
            self.translator = ctranslate2.Translator(self.model_path, device="cpu", inter_threads=2, intra_threads=4)

    def translate_texts(self, texts: list, source_whisper_lang: str, target_lang: str = "spa_Latn") -> list:
        if not texts:
            return []
        self._ensure_loaded()
        src_lang = WHISPER_TO_NLLB.get(source_whisper_lang, f"{source_whisper_lang}_Latn")
        self.tokenizer.src_lang = src_lang

        valid_indices = [i for i, t in enumerate(texts) if t and str(t).strip()]
        if not valid_indices:
            return texts

        tokenized_batch = [
            self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(str(texts[i]).strip()))
            for i in valid_indices
        ]

        results = self.translator.translate_batch(
            tokenized_batch,
            target_prefix=[[target_lang]] * len(tokenized_batch),
            beam_size=4,
            max_decoding_length=256
        )

        translated_results = list(texts)
        for idx, res in zip(valid_indices, results):
            hyp = res.hypotheses[0]
            clean_tokens = [t for t in hyp if t != target_lang]
            trans_str = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(clean_tokens))
            translated_results[idx] = trans_str.strip()

        return translated_results

nllb_translator = NLLBTranslator()

def resolve_whisper_model_path(model_name: str) -> str:
    """
    Resuelve o descarga el modelo Whisper localmente sin usar enlaces simbólicos (symlinks).
    Evita por completo el error [WinError 1314] en Windows.
    """
    if os.path.isdir(model_name):
        return model_name
        
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    target_dir = os.path.join(models_dir, f"whisper-{model_name}")
    
    # Si ya está descargado y completo
    if os.path.isdir(target_dir) and os.path.isfile(os.path.join(target_dir, "model.bin")):
        return target_dir
        
    import faster_whisper
    print(f"[DOWNLOAD] Descargando modelo Whisper '{model_name}' de forma segura (sin enlaces simbólicos) en: {target_dir}...")
    try:
        model_path = faster_whisper.download_model(model_name, output_dir=target_dir)
        return model_path
    except Exception as e:
        print(f"[WARNING] Falló descarga directa a carpeta ({e}), usando identificador estándar...")
        return model_name

# ==========================================
# 🧠 GESTOR DE CACHÉ DE MODELOS WHISPER (SINGLETON)
# ==========================================
class WhisperModelManager:
    def __init__(self):
        self.model = None
        self.current_model_name = None
        self.current_compute_type = None
        self.current_device = None

    def get_model(self, model_name: str, compute_type: str, device: str = "cuda"):
        if (
            self.model is not None
            and self.current_model_name == model_name
            and self.current_compute_type == compute_type
            and self.current_device == device
        ):
            print(f"[CACHE] Reutilizando modelo Whisper '{model_name}' ({compute_type}) en VRAM. ¡Cero tiempo de recarga!")
            return self.model, False  # was_reloaded = False

        self.unload()
        resolved_path = resolve_whisper_model_path(model_name)
        print(f"[CACHE] Cargando modelo Whisper '{model_name}' ({compute_type}) en {device}...")
        self.model = whisperx.load_model(resolved_path, device, compute_type=compute_type)
        self.current_model_name = model_name
        self.current_compute_type = compute_type
        self.current_device = device
        return self.model, True  # was_reloaded = True

    def unload(self):
        if self.model is not None:
            print(f"[CACHE] Desalojando modelo '{self.current_model_name}' de la VRAM...")
            del self.model
            self.model = None
            self.current_model_name = None
            self.current_compute_type = None
            self.current_device = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return "✅ Memoria VRAM liberada exitosamente."
        return "ℹ️ La VRAM ya estaba libre (sin modelo en caché)."

    def get_status_text(self):
        if self.model is not None:
            return f"🟢 **Modelo en VRAM (Listo para usar):** `{self.current_model_name}` | Precisión: `{self.current_compute_type}`"
        return "⚪ **Estado VRAM:** Libre (sin modelo cargado en memoria)"

model_manager = WhisperModelManager()

# ==========================================
# 🛠️ FUNCIONES DE UTILIDAD Y EXPORTACIÓN
# ==========================================
def get_resource_metrics(start_time: float, audio_duration: float):
    """Calcula el tiempo de procesamiento, factor de velocidad, uso de VRAM y RAM"""
    elapsed_time = max(0.001, time.perf_counter() - start_time)
    speed_factor = (audio_duration / elapsed_time) if elapsed_time > 0 and audio_duration > 0 else 0.0
    
    peak_vram_gb = 0.0
    total_vram_gb = 0.0
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        
    process_ram_gb = 0.0
    sys_ram_percent = 0.0
    try:
        process = psutil.Process()
        process_ram_gb = process.memory_info().rss / (1024 ** 3)
        sys_ram_percent = psutil.virtual_memory().percent
    except:
        pass
        
    return {
        "elapsed_time": elapsed_time,
        "speed_factor": speed_factor,
        "peak_vram_gb": peak_vram_gb,
        "total_vram_gb": total_vram_gb,
        "process_ram_gb": process_ram_gb,
        "sys_ram_percent": sys_ram_percent
    }

def format_timestamp(seconds: float) -> str:
    """Convierte segundos a formato de tiempo SRT: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int(round((seconds - int(seconds)) * 1000))
    milliseconds = min(999, max(0, milliseconds))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

def format_simple_timestamp(seconds: float) -> str:
    """Convierte segundos a formato MM:SS o HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"

def get_browser_safe_video(file_path: str, srt_path: str):
    """
    Evita que Gradio se quede colgado al 99% recodificando videos pesados o audios en CPU.
    Solo envía el archivo al reproductor si ya es nativo y reproducible en el navegador.
    """
    if not file_path:
        return None
    ext = os.path.splitext(file_path)[1].lower()
    AUDIO_EXTS = {'.mp3', '.wav', '.ogg', '.opus', '.m4a', '.flac', '.aac', '.wma', '.alac'}
    if ext in AUDIO_EXTS:
        return None
        
    try:
        import gradio.processing_utils as gpu
        if gpu.video_is_playable(file_path):
            return (file_path, srt_path)
    except Exception:
        pass
    
    print("[INFO] Vista previa de video omitida para evitar la recodificación lenta de FFmpeg en CPU. Los resultados y descargas se entregan de inmediato.")
    return None

def extract_base_filename(file_path: str, default_prefix: str = "transcripcion") -> str:
    """Extrae el nombre base del archivo multimedia sin extensión y limpio de caracteres conflictivos"""
    if not file_path:
        return default_prefix
    base = os.path.basename(file_path)
    raw_name = os.path.splitext(base)[0]
    # Limpiar caracteres prohibidos en nombres de archivo de Windows (\ / * ? : " < > |)
    clean_name = re.sub(r'[\\/*?:"<>|]', "", raw_name).strip()
    return clean_name if clean_name else default_prefix

def generate_clean_downloads(text_content: str, doc_title: str = "Transcripción de Audio", base_name: str = "transcripcion"):
    """Genera archivos .txt y .docx a partir de un texto limpio adaptando el nombre del archivo original"""
    temp_dir = tempfile.gettempdir()
    temp_txt_path = os.path.join(temp_dir, f"{base_name}_transcripcion.txt")
    temp_docx_path = os.path.join(temp_dir, f"{base_name}_transcripcion.docx")
    
    with open(temp_txt_path, "w", encoding="utf-8") as f:
        f.write(text_content)
        
    doc = Document()
    doc.add_heading(doc_title, 0)
    for paragraph in text_content.split("\n\n"):
        p_strip = paragraph.strip()
        if p_strip:
            doc.add_paragraph(p_strip)
    doc.save(temp_docx_path)
    
    return temp_txt_path, temp_docx_path

# ==========================================
# ⚡ MODO 1: EXTRACCIÓN RÁPIDA DE TEXTO (WHATSAPP / NOTAS / CUALQUIER AUDIO O VIDEO)
# ==========================================
def quick_transcribe_audio(file_path, mic_path, whisper_model, compute_type, batch_size, source_language_ui, format_mode, traducir_a_es, progress=gr.Progress(track_tqdm=True)):
    actual_path = file_path if file_path else mic_path
    if not actual_path:
        return "⚠️ Error: Por favor sube un archivo de audio o video, o graba un mensaje con el micrófono.", "### 📊 Métricas\n*No hay archivo cargado.*", None, None, model_manager.get_status_text()
    file_path = actual_path
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return "⚠️ Error: No se detectó una GPU compatible con CUDA. Esta aplicación requiere aceleración por GPU (NVIDIA).", "### 📊 Métricas\n*Error de GPU.*", None, None, model_manager.get_status_text()
        
    batch_size = int(batch_size) if batch_size and int(batch_size) > 0 else 16
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()
    
    try:
        progress(0.15, desc=f"Obteniendo modelo WhisperX ({whisper_model} - {compute_type})...")
        model, was_reloaded = model_manager.get_model(whisper_model, compute_type, device)
        status_note = "Modelo cargado desde disco" if was_reloaded else "⚡ Reutilizado desde VRAM (Ultrarrápido)"
        
        progress(0.35, desc="Cargando y decodificando audio...")
        print(f"[QUICK] Cargando audio: {file_path} ({status_note})...")
        audio = whisperx.load_audio(file_path)
        audio_duration = len(audio) / 16000.0
        
        progress(0.55, desc="Extrayendo texto del audio...")
        print("[QUICK] Transcribiendo...")
        
        lang_map = {
            "Español": "es", "Inglés": "en", "Francés": "fr", 
            "Alemán": "de", "Italiano": "it", "Portugués": "pt",
            "Catalán": "ca", "Ruso": "ru", "Chino": "zh", "Japonés": "ja"
        }
        language_code = lang_map.get(source_language_ui, None)
        
        if language_code:
            result = model.transcribe(audio, batch_size=batch_size, language=language_code)
        else:
            result = model.transcribe(audio, batch_size=batch_size)
            
        detected_lang = result.get("language", "es")
        print(f"[QUICK] Idioma detectado: {detected_lang}")
        
        segments = result.get("segments", [])
        
        # Traducción opcional a Español con Meta NLLB-200
        if traducir_a_es and detected_lang != "es":
            progress(0.85, desc=f"Traduciendo de {detected_lang.upper()} a Español con Meta NLLB-200...")
            try:
                raw_texts = [seg.get("text", "").strip() for seg in segments]
                translated_texts = nllb_translator.translate_texts(raw_texts, detected_lang, target_lang="spa_Latn")
                for seg, tr_txt in zip(segments, translated_texts):
                    seg["text"] = tr_txt
            except Exception as tr_err:
                print(f"[QUICK ERROR] Error en traducción NLLB-200: {tr_err}")
                
        # Formateo de texto según preferencia
        progress(0.95, desc="Formateando salida de texto limpio...")
        if format_mode == "Texto Continuo / Párrafo":
            raw_texts = [seg.get("text", "").strip() for seg in segments if seg.get("text", "").strip()]
            final_text = " ".join(raw_texts)
        elif format_mode == "Con Marcas de Tiempo":
            lines = []
            for seg in segments:
                start = seg.get("start", 0.0)
                end = seg.get("end", 0.0)
                txt = seg.get("text", "").strip()
                lines.append(f"[{format_simple_timestamp(start)} - {format_simple_timestamp(end)}] {txt}")
            final_text = "\n".join(lines)
        else:  # "Párrafos por Pausas / Frases"
            lines = [seg.get("text", "").strip() for seg in segments if seg.get("text", "").strip()]
            final_text = "\n\n".join(lines)
            
        if not final_text.strip():
            final_text = "No se detectó contenido de voz en el archivo."
            
        word_count = len(final_text.split())
        char_count = len(final_text)
        mins = int(audio_duration // 60)
        secs = int(audio_duration % 60)
        
        m = get_resource_metrics(start_time, audio_duration)
        print(f"📊 [BENCHMARK] Tiempo: {m['elapsed_time']:.2f}s | Velocidad: {m['speed_factor']:.1f}x tiempo real | Pico VRAM: {m['peak_vram_gb']:.2f} GB / {m['total_vram_gb']:.1f} GB | RAM Proceso: {m['process_ram_gb']:.2f} GB ({m['sys_ram_percent']:.0f}% sys) | Batch Size: {batch_size}")
        
        stats_md = f"""
### 📊 Rendimiento y Recursos del Sistema
| ⏱️ Duración Audio | ⚡ Tiempo Proceso | 🚀 Velocidad | 🎮 Pico VRAM GPU | 💾 RAM Proceso | 📦 Batch Size |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **{mins:02d}:{secs:02d} min** | **{m['elapsed_time']:.2f} s** | **{m['speed_factor']:.1f}x tiempo real** | **{m['peak_vram_gb']:.2f} GB / {m['total_vram_gb']:.1f} GB** | **{m['process_ram_gb']:.2f} GB ({m['sys_ram_percent']:.0f}%)** | **{batch_size}** |

| 🌐 Idioma | 📝 Palabras | 🔤 Caracteres | ⚙️ Configuración | ⚡ Estado Motor |
| :---: | :---: | :---: | :---: | :---: |
| **{detected_lang.upper()}** | **{word_count} palabras** | **{char_count} caracteres** | **{whisper_model} ({compute_type})** | **{status_note}** |
"""
        base_name = extract_base_filename(actual_path, default_prefix="audio")
        temp_txt, temp_docx = generate_clean_downloads(final_text, doc_title=f"Transcripción: {base_name}", base_name=base_name)
        
        progress(1.0, desc="¡Texto extraído con éxito!")
        print("[SUCCESS] Transcripción rápida completada.")
        return final_text, stats_md, temp_txt, temp_docx, model_manager.get_status_text()
        
    except Exception as e:
        err_msg = f"ERROR EN LA TRANSCRIPCIÓN RÁPIDA:\n\n{str(e)}"
        print(f"[QUICK ERROR] {err_msg}")
        return err_msg, "### ⚠️ Error durante el procesamiento", None, None, model_manager.get_status_text()

# ==========================================
# 👥 MODO 2: TRANSCRIPCIÓN Y DIARIZACIÓN AVANZADA
# ==========================================
def transcribe_and_diarize(file_path, hf_token, whisper_model, compute_type, diarize_batch_size, diarize_model_name, source_language_ui, min_speakers, max_speakers, traducir_a_es, progress=gr.Progress(track_tqdm=True)):
    if not file_path:
        return pd.DataFrame([{"Hablante": "ERROR", "Inicio (s)": 0.0, "Fin (s)": 0.0, "Texto": "Por favor, carga un archivo de audio o video."}]), None, None, None, gr.update(), None, model_manager.get_status_text(), "### 📊 Métricas\n*No hay archivo cargado.*", "transcripcion"
    if not hf_token or not hf_token.strip():
        return pd.DataFrame([{"Hablante": "ERROR", "Inicio (s)": 0.0, "Fin (s)": 0.0, "Texto": "El token de Hugging Face es obligatorio para Diarización (Pyannote)."}]), None, None, None, gr.update(), None, model_manager.get_status_text(), "### 📊 Métricas\n*Token requerido.*", "transcripcion"
        
    # Auto-guardado del token en .env
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
        return pd.DataFrame([{"Hablante": "ERROR", "Inicio (s)": 0.0, "Fin (s)": 0.0, "Texto": "No se detectó una GPU compatible con CUDA."}]), None, None, None, gr.update(), None, model_manager.get_status_text(), "### 📊 Métricas\n*Error de GPU.*", "transcripcion"
        
    # Para Diarización desalojamos cualquier modelo en reposo para maximizar VRAM
    model_manager.unload()
    batch_size = int(diarize_batch_size) if diarize_batch_size and int(diarize_batch_size) > 0 else 8
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()
    
    try:
        # --- PASO 1: Transcripción con WhisperX ---
        resolved_path = resolve_whisper_model_path(whisper_model)
        progress(0.1, desc=f"Cargando modelo WhisperX ({whisper_model} - {compute_type}) en VRAM...")
        print(f"[PROCESS] Cargando modelo WhisperX: {whisper_model} ({compute_type})...")
        model = whisperx.load_model(resolved_path, device, compute_type=compute_type)
        
        progress(0.2, desc="Procesando audio de entrada...")
        print("[PROCESS] Cargando audio...")
        audio = whisperx.load_audio(file_path)
        
        progress(0.3, desc="Ejecutando transcripción de audio...")
        print("[PROCESS] Iniciando transcripción...")
        
        lang_map = {
            "Español": "es", "Inglés": "en", "Francés": "fr", 
            "Alemán": "de", "Italiano": "it", "Portugués": "pt",
            "Catalán": "ca", "Ruso": "ru", "Chino": "zh", "Japonés": "ja"
        }
        language_code = lang_map.get(source_language_ui, None)
        
        if language_code:
            raw_result = model.transcribe(audio, batch_size=batch_size, language=language_code)
        else:
            raw_result = model.transcribe(audio, batch_size=batch_size)
        
        detected_language = raw_result.get("language", "es")
        print(f"[PROCESS] Idioma detectado: {detected_language}")
        
        # Limpieza de memoria (VRAM)
        progress(0.4, desc="Liberando VRAM de transcripción...")
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
        
        # Limpieza de memoria (VRAM)
        progress(0.7, desc="Liberando VRAM de alineación...")
        del align_model
        gc.collect()
        torch.cuda.empty_cache()
        
        # --- PASO 3: Diarización (Pyannote) ---
        progress(0.8, desc=f"Iniciando Diarización de hablantes ({diarize_model_name})...")
        print(f"[PROCESS] Inicializando pipeline de Diarización con {diarize_model_name}...")
        
        diarize_kwargs = {}
        if min_speakers is not None and min_speakers > 0:
            diarize_kwargs["min_speakers"] = int(min_speakers)
        if max_speakers is not None and max_speakers > 0:
            diarize_kwargs["max_speakers"] = int(max_speakers)
            
        diarize_model = DiarizationPipeline(model_name=diarize_model_name, token=hf_token.strip(), device=device)
        
        progress(0.85, desc="Separando pistas de voces y hablantes...")
        print("[PROCESS] Ejecutando diarización...")
        diarize_segments = diarize_model(audio, **diarize_kwargs)
        
        # Limpieza de memoria (VRAM)
        progress(0.9, desc="Liberando VRAM de diarización...")
        del diarize_model
        gc.collect()
        torch.cuda.empty_cache()
        
        # --- PASO 4: Fusión final de palabras y hablantes ---
        progress(0.95, desc="Fusionando texto con etiquetas de hablantes...")
        print("[PROCESS] Asignando hablantes a palabras...")
        final_result = whisperx.assign_word_speakers(diarize_segments, aligned_result)
        
        # --- PASO 4.5: Traducción (Opcional con Meta NLLB-200) ---
        if traducir_a_es and detected_language != "es":
            progress(0.97, desc="Traduciendo transcripción al Español con Meta NLLB-200...")
            print("[PROCESS] Preparando traducción a Español con NLLB-200...")
            try:
                segments = final_result.get("segments", [])
                raw_texts = [seg.get("text", "").strip() for seg in segments]
                translated_texts = nllb_translator.translate_texts(raw_texts, detected_language, target_lang="spa_Latn")
                for seg, tr_txt in zip(segments, translated_texts):
                    seg["text"] = tr_txt
            except Exception as tr_e:
                print(f"[ERROR] Falló la traducción NLLB-200: {tr_e}")

        # --- PASO 5: Generar y formatear salidas ---
        progress(0.99, desc="Formateando salidas finales...")
        df_rows = []
        
        for segment in final_result["segments"]:
            speaker = segment.get("speaker", "SPEAKER_UNKNOWN")
            text = segment.get("text", "").strip()
            start = segment.get("start", 0.0)
            end = segment.get("end", 0.0)
            df_rows.append({"Hablante": speaker, "Inicio (s)": round(start, 2), "Fin (s)": round(end, 2), "Texto": text})
            
        df_data = pd.DataFrame(df_rows)
        
        if df_data.empty:
            df_data = pd.DataFrame([{"Hablante": "Sistema", "Inicio (s)": 0.0, "Fin (s)": 0.0, "Texto": "No se detectó contenido de voz en el archivo."}])
            
        base_name = extract_base_filename(file_path, default_prefix="reunion")
        temp_txt_path, temp_srt_path, temp_docx_path = update_downloads_from_df(df_data, clean_export=False, base_name=base_name)
        
        audio_duration = len(audio) / 16000.0 if 'audio' in locals() and audio is not None else 0.0
        mins = int(audio_duration // 60)
        secs = int(audio_duration % 60)
        m = get_resource_metrics(start_time, audio_duration)
        print(f"📊 [BENCHMARK DIARIZACIÓN] Tiempo Total: {m['elapsed_time']:.2f}s | Velocidad: {m['speed_factor']:.1f}x tiempo real | Pico VRAM: {m['peak_vram_gb']:.2f} GB / {m['total_vram_gb']:.1f} GB | RAM Proceso: {m['process_ram_gb']:.2f} GB ({m['sys_ram_percent']:.0f}% sys) | Batch Size: {batch_size}")
        
        diarize_stats_md = f"""
### 📊 Rendimiento de Diarización Multi-Hablante
| ⏱️ Duración Audio | ⚡ Tiempo Total | 🚀 Velocidad | 🎮 Pico VRAM GPU | 💾 RAM Proceso | 📦 Batch Size |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **{mins:02d}:{secs:02d} min** | **{m['elapsed_time']:.2f} s** | **{m['speed_factor']:.1f}x tiempo real** | **{m['peak_vram_gb']:.2f} GB / {m['total_vram_gb']:.1f} GB** | **{m['process_ram_gb']:.2f} GB ({m['sys_ram_percent']:.0f}%)** | **{batch_size}** |
"""
            
        progress(1.0, desc="¡Proceso finalizado!")
        print("[SUCCESS] Proceso de diarización y transcripción completado.")
        # Cambia automáticamente a la pestaña de Auditoría (Tab id=3)
        video_preview = get_browser_safe_video(file_path, temp_srt_path)
        return df_data, temp_txt_path, temp_srt_path, temp_docx_path, gr.Tabs(selected=3), video_preview, model_manager.get_status_text(), diarize_stats_md, base_name
        
    except Exception as e:
        error_msg = f"ERROR EN EL PROCESAMIENTO:\n\n{str(e)}\n\n"
        error_msg += "Sugerencia: Asegúrate de que el token de Hugging Face sea válido y de haber aceptado las licencias del modelo de Pyannote seleccionado en Hugging Face."
        print(f"[ERROR] {error_msg}")
        
        try:
            gc.collect()
            torch.cuda.empty_cache()
        except:
            pass
            
        error_df = pd.DataFrame([{"Hablante": "ERROR", "Inicio (s)": 0.0, "Fin (s)": 0.0, "Texto": error_msg}])
        return error_df, None, None, None, gr.update(), None, model_manager.get_status_text(), "### ⚠️ Error en Diarización", "transcripcion"

def get_speaker_color(speaker_name):
    hash_val = sum(ord(c) for c in speaker_name)
    colors = [
        RGBColor(0, 114, 178),  # Blue
        RGBColor(213, 94, 0),   # Vermillion
        RGBColor(0, 158, 115),  # Bluish Green
        RGBColor(204, 121, 167),  # Reddish Purple
        RGBColor(230, 159, 0),  # Orange
        RGBColor(86, 180, 233),  # Sky Blue
    ]
    return colors[hash_val % len(colors)]

def update_downloads_from_df(df, clean_export=False, base_name="transcripcion"):
    if df is None or df.empty:
        return None, None, None
    
    clean_base = extract_base_filename(base_name, default_prefix="reunion")
    srt_lines = []
    txt_lines = []
    srt_counter = 1
    
    doc = Document()
    doc.add_heading(f'Guion de Transcripción: {clean_base}', 0)
    
    for _, row in df.iterrows():
        speaker = str(row.get("Hablante", "SPEAKER_UNKNOWN"))
        try:
            start = float(row.get("Inicio (s)", 0.0))
            end = float(row.get("Fin (s)", 0.0))
        except:
            start, end = 0.0, 0.0
        text = str(row.get("Texto", "")).strip()
        
        # DOCX Logic
        p = doc.add_paragraph()
        speaker_run = p.add_run(f"[{speaker}]")
        speaker_run.bold = True
        speaker_run.font.color.rgb = get_speaker_color(speaker)
        p.add_run(f"\n({start:.2f}s - {end:.2f}s) {text}")
        
        if clean_export:
            txt_lines.append(f"({start:.2f}s - {end:.2f}s): {text}")
            srt_lines.append(f"{srt_counter}")
            srt_lines.append(f"{format_timestamp(start)} --> {format_timestamp(end)}")
            srt_lines.append(f"{text}\n")
        else:
            txt_lines.append(f"[{speaker}] ({start:.2f}s - {end:.2f}s): {text}")
            srt_lines.append(f"{srt_counter}")
            srt_lines.append(f"{format_timestamp(start)} --> {format_timestamp(end)}")
            srt_lines.append(f"[{speaker}]: {text}\n")
            
        srt_counter += 1
        
    full_text = "\n".join(txt_lines)
    full_srt = "\n".join(srt_lines)
    
    temp_dir = tempfile.gettempdir()
    temp_txt_path = os.path.join(temp_dir, f"{clean_base}_diarizada.txt")
    temp_srt_path = os.path.join(temp_dir, f"{clean_base}.srt")
    temp_docx_path = os.path.join(temp_dir, f"{clean_base}_guion.docx")
    
    with open(temp_txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)
        
    with open(temp_srt_path, "w", encoding="utf-8") as f:
        f.write(full_srt)
        
    doc.save(temp_docx_path)
        
    return temp_txt_path, temp_srt_path, temp_docx_path

def rename_speaker_in_df(df, old_name, new_name, clean_export, base_name="transcripcion"):
    """Reemplaza masivamente un identificador de hablante en todo el DataFrame y actualiza descargas con el nombre del video"""
    if df is None or df.empty or not old_name or not new_name:
        return df, None, None, None
    df_copy = df.copy()
    old_val = str(old_name).strip()
    new_val = str(new_name).strip()
    df_copy["Hablante"] = df_copy["Hablante"].replace(old_val, new_val)
    txt_path, srt_path, docx_path = update_downloads_from_df(df_copy, clean_export=clean_export, base_name=base_name)
    return df_copy, txt_path, srt_path, docx_path

# ==========================================
# 📦 MODO 4: PROCESAMIENTO POR LOTES (MULTI-ARCHIVOS / CARPETA)
# ==========================================
MEDIA_EXTENSIONS = {
    # Video
    '.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv', '.m4v', '.ts', '.3gp', '.mpeg', '.mpg',
    # Audio
    '.mp3', '.wav', '.m4a', '.opus', '.ogg', '.flac', '.aac', '.wma', '.alac', '.aiff', '.amr'
}

def generate_files_for_segments(segments: list, clean_export: bool = False, base_name: str = "transcripcion", output_dir: str = None):
    """
    Genera archivos .srt, .txt y .docx a partir de una lista de segmentos de transcripción.
    Respeta el modo de subtítulos limpios (sin etiquetas de hablante).
    """
    if output_dir is None:
        output_dir = tempfile.gettempdir()
    os.makedirs(output_dir, exist_ok=True)
    
    clean_base = extract_base_filename(base_name, default_prefix="audio")
    srt_path = os.path.join(output_dir, f"{clean_base}.srt")
    txt_path = os.path.join(output_dir, f"{clean_base}.txt")
    docx_path = os.path.join(output_dir, f"{clean_base}_transcripcion.docx")
    
    srt_lines = []
    txt_lines = []
    
    doc = Document()
    doc.add_heading(f"Transcripción: {clean_base}", 0)
    
    srt_counter = 1
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
            
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        speaker = seg.get("speaker", None)
        
        # Word DOCX
        p = doc.add_paragraph()
        if speaker and not clean_export:
            speaker_run = p.add_run(f"[{speaker}] ")
            speaker_run.bold = True
            speaker_run.font.color.rgb = get_speaker_color(speaker)
        time_run = p.add_run(f"({format_simple_timestamp(start)} - {format_simple_timestamp(end)}) ")
        time_run.font.color.rgb = RGBColor(128, 128, 128)
        p.add_run(text)
        
        # Subtítulos SRT y TXT
        if clean_export or not speaker:
            txt_lines.append(f"({format_simple_timestamp(start)} - {format_simple_timestamp(end)}): {text}")
            srt_lines.append(f"{srt_counter}")
            srt_lines.append(f"{format_timestamp(start)} --> {format_timestamp(end)}")
            srt_lines.append(f"{text}\n")
        else:
            txt_lines.append(f"[{speaker}] ({format_simple_timestamp(start)} - {format_simple_timestamp(end)}): {text}")
            srt_lines.append(f"{srt_counter}")
            srt_lines.append(f"{format_timestamp(start)} --> {format_timestamp(end)}")
            srt_lines.append(f"[{speaker}]: {text}\n")
            
        srt_counter += 1
        
    if not srt_lines:
        srt_lines.append("1\n00:00:00,000 --> 00:00:01,000\n[Sin contenido de voz detectable]\n")
        txt_lines.append("[Sin contenido de voz detectable]")
        doc.add_paragraph("[Sin contenido de voz detectable]")
        
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))
        
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))
        
    doc.save(docx_path)
    
    return srt_path, txt_path, docx_path

def get_batch_file_list(uploaded_files, folder_path):
    """Obtiene la lista deduplicada de archivos de audio/video desde subidas directas o carpeta local"""
    candidate_paths = []
    
    # 1. Archivos subidos en Gradio
    if uploaded_files:
        if isinstance(uploaded_files, list):
            for item in uploaded_files:
                p = item if isinstance(item, str) else getattr(item, "name", str(item))
                if p and os.path.isfile(p):
                    candidate_paths.append(p)
        elif isinstance(uploaded_files, str) and os.path.isfile(uploaded_files):
            candidate_paths.append(uploaded_files)
            
    # 2. Archivos escaneados de carpeta local en Windows
    if folder_path and str(folder_path).strip():
        clean_dir = str(folder_path).strip().strip('"').strip("'")
        if os.path.isdir(clean_dir):
            try:
                for entry in os.scandir(clean_dir):
                    if entry.is_file():
                        ext = os.path.splitext(entry.name)[1].lower()
                        if ext in MEDIA_EXTENSIONS:
                            candidate_paths.append(entry.path)
            except Exception as e:
                print(f"[BATCH] Error al escanear carpeta '{clean_dir}': {e}")
                
    # Deduplicar preservando el orden original
    seen = set()
    final_list = []
    for path in candidate_paths:
        canonical = os.path.realpath(path)
        if canonical not in seen and os.path.isfile(path):
            seen.add(canonical)
            final_list.append(path)
            
    return final_list

def scan_batch_queue(uploaded_files, folder_path):
    """Inspecciona y previsualiza los archivos en cola antes de procesar"""
    files = get_batch_file_list(uploaded_files, folder_path)
    if not files:
        stats_md = "### ⚠️ No se detectaron archivos multimedia\n*Arrastra archivos arriba o escribe la ruta de una carpeta válida en Windows.*"
        empty_df = pd.DataFrame([{"#": 1, "Archivo": "Ningún archivo compatible detectado", "Duración": "--", "Tiempo": "--", "Velocidad": "--", "Estado": "⚠️ Vacío"}])
        return stats_md, empty_df, None
        
    rows = []
    for idx, f in enumerate(files):
        rows.append({
            "#": idx + 1,
            "Archivo": os.path.basename(f),
            "Duración": "--",
            "Tiempo": "--",
            "Velocidad": "--",
            "Estado": "⏳ En cola"
        })
    stats_md = f"### 📋 Cola Lista: **{len(files)}** archivo(s) detectado(s)\n*Haz clic en **🚀 Iniciar Procesamiento por Lotes** para comenzar.*"
    return stats_md, pd.DataFrame(rows), None

def batch_process_media(
    uploaded_files,
    folder_path,
    save_in_source,
    clean_subtitles,
    whisper_model,
    compute_type,
    batch_size,
    source_language,
    traducir_a_es,
    enable_diarization,
    hf_token,
    progress=gr.Progress(track_tqdm=True)
):
    files = get_batch_file_list(uploaded_files, folder_path)
    if not files:
        err_msg = "### ⚠️ No se encontraron archivos multimedia válidos.\nPor favor sube archivos o indica una carpeta que contenga audios o videos (.mp4, .mp3, .mkv, .wav, .m4a, etc.)."
        err_df = pd.DataFrame([{"#": 1, "Archivo": "Ningún archivo encontrado", "Duración": "--", "Tiempo": "--", "Velocidad": "--", "Estado": "⚠️ Sin archivos"}])
        return err_msg, err_df, None, model_manager.get_status_text()
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        err_msg = "### ⚠️ Se requiere GPU NVIDIA (CUDA)\nEsta aplicación requiere aceleración por GPU para procesar por lotes."
        err_df = pd.DataFrame([{"#": 1, "Archivo": "Error", "Duración": "--", "Tiempo": "--", "Velocidad": "--", "Estado": "❌ Sin GPU CUDA"}])
        return err_msg, err_df, None, model_manager.get_status_text()
        
    if enable_diarization and (not hf_token or not str(hf_token).strip()):
        err_msg = "### ⚠️ Token de Hugging Face requerido para Diarización.\nPor favor ingresa tu token o desmarca la opción de Diarización para procesar en modo rápido."
        err_df = pd.DataFrame([{"#": 1, "Archivo": "Error", "Duración": "--", "Tiempo": "--", "Velocidad": "--", "Estado": "❌ Falta Token HF"}])
        return err_msg, err_df, None, model_manager.get_status_text()
        
    # Guardar token en .env si se suministró
    if hf_token and str(hf_token).strip():
        current_token = os.getenv("HF_TOKEN", "")
        if str(hf_token).strip() != current_token:
            try:
                set_key(env_path, "HF_TOKEN", str(hf_token).strip())
                os.environ["HF_TOKEN"] = str(hf_token).strip()
            except Exception:
                pass
                
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    batch_temp_dir = os.path.join(tempfile.gettempdir(), f"echowriter_batch_{timestamp_str}")
    os.makedirs(batch_temp_dir, exist_ok=True)
    
    lang_map = {
        "Español": "es", "Inglés": "en", "Francés": "fr", 
        "Alemán": "de", "Italiano": "it", "Portugués": "pt",
        "Catalán": "ca", "Ruso": "ru", "Chino": "zh", "Japonés": "ja"
    }
    language_code = lang_map.get(source_language, None)
    
    total_files = len(files)
    batch_size = int(batch_size) if batch_size and int(batch_size) > 0 else 16
    total_audio_seconds = 0.0
    batch_start_time = time.perf_counter()
    table_rows = []
    generated_count = 0
    
    print(f"[BATCH] Iniciando lote de {total_files} archivo(s)...")
    
    for idx, file_path in enumerate(files):
        file_name = os.path.basename(file_path)
        base_name = extract_base_filename(file_path)
        file_start = time.perf_counter()
        prog_start = idx / total_files
        prog_step = 1.0 / total_files
        
        progress(prog_start, desc=f"[{idx+1}/{total_files}] Iniciando: {file_name}")
        
        try:
            # 1. Cargar audio
            progress(prog_start + 0.15 * prog_step, desc=f"[{idx+1}/{total_files}] Decodificando audio: {file_name}...")
            audio = whisperx.load_audio(file_path)
            audio_duration = len(audio) / 16000.0
            total_audio_seconds += audio_duration
            
            if enable_diarization:
                model_manager.unload()
                resolved_path = resolve_whisper_model_path(whisper_model)
                
                progress(prog_start + 0.30 * prog_step, desc=f"[{idx+1}/{total_files}] Transcribiendo WhisperX: {file_name}...")
                w_model = whisperx.load_model(resolved_path, device, compute_type=compute_type)
                if language_code:
                    raw_res = w_model.transcribe(audio, batch_size=batch_size, language=language_code)
                else:
                    raw_res = w_model.transcribe(audio, batch_size=batch_size)
                det_lang = raw_res.get("language", "es")
                del w_model; gc.collect(); torch.cuda.empty_cache()
                
                progress(prog_start + 0.50 * prog_step, desc=f"[{idx+1}/{total_files}] Alineando fonemas: {file_name}...")
                align_m, meta = whisperx.load_align_model(language_code=det_lang, device=device)
                aligned_res = whisperx.align(raw_res["segments"], align_m, meta, audio, device, return_char_alignments=False)
                del align_m; gc.collect(); torch.cuda.empty_cache()
                
                progress(prog_start + 0.70 * prog_step, desc=f"[{idx+1}/{total_files}] Diarizando hablantes (Pyannote): {file_name}...")
                diar_pipe = DiarizationPipeline(model_name="pyannote/speaker-diarization-community-1", token=str(hf_token).strip(), device=device)
                diar_segs = diar_pipe(audio)
                del diar_pipe; gc.collect(); torch.cuda.empty_cache()
                
                progress(prog_start + 0.85 * prog_step, desc=f"[{idx+1}/{total_files}] Asignando hablantes: {file_name}...")
                final_res = whisperx.assign_word_speakers(diar_segs, aligned_res)
                segments = final_res.get("segments", [])
                detected_language = det_lang
            else:
                progress(prog_start + 0.40 * prog_step, desc=f"[{idx+1}/{total_files}] Transcribiendo Whisper ({whisper_model}): {file_name}...")
                model, _ = model_manager.get_model(whisper_model, compute_type, device)
                if language_code:
                    result = model.transcribe(audio, batch_size=batch_size, language=language_code)
                else:
                    result = model.transcribe(audio, batch_size=batch_size)
                detected_language = result.get("language", "es")
                segments = result.get("segments", [])
                
            # Traducción opcional a Español con Meta NLLB-200
            if traducir_a_es and detected_language != "es":
                progress(prog_start + 0.88 * prog_step, desc=f"[{idx+1}/{total_files}] Traduciendo de {detected_language.upper()} a Español: {file_name}...")
                try:
                    raw_texts = [seg.get("text", "").strip() for seg in segments]
                    translated_texts = nllb_translator.translate_texts(raw_texts, detected_language, target_lang="spa_Latn")
                    for seg, tr_txt in zip(segments, translated_texts):
                        seg["text"] = tr_txt
                except Exception as tr_err:
                    print(f"[BATCH TRADUCCIÓN ERROR] {file_name}: {tr_err}")
                    
            # Generar archivos .srt, .txt y .docx en el directorio del lote
            progress(prog_start + 0.95 * prog_step, desc=f"[{idx+1}/{total_files}] Guardando archivos: {file_name}...")
            srt_path, txt_path, docx_path = generate_files_for_segments(
                segments=segments,
                clean_export=clean_subtitles,
                base_name=base_name,
                output_dir=batch_temp_dir
            )
            generated_count += 3
            
            # Guardar también en la carpeta origen si save_in_source está activo
            if save_in_source:
                try:
                    file_dir = os.path.dirname(os.path.realpath(file_path))
                    if file_dir.lower() != tempfile.gettempdir().lower() and os.path.isdir(file_dir) and os.access(file_dir, os.W_OK):
                        shutil.copy2(srt_path, os.path.join(file_dir, os.path.basename(srt_path)))
                        shutil.copy2(txt_path, os.path.join(file_dir, os.path.basename(txt_path)))
                        shutil.copy2(docx_path, os.path.join(file_dir, os.path.basename(docx_path)))
                        print(f"[BATCH] Archivos exportados a carpeta origen: {file_dir}")
                except Exception as copy_err:
                    print(f"[BATCH WARNING] No se pudo copiar a carpeta origen: {copy_err}")
                    
            f_elapsed = max(0.01, time.perf_counter() - file_start)
            speed_factor = (audio_duration / f_elapsed) if f_elapsed > 0 else 0.0
            mins = int(audio_duration // 60)
            secs = int(audio_duration % 60)
            
            table_rows.append({
                "#": idx + 1,
                "Archivo": file_name,
                "Duración": f"{mins:02d}:{secs:02d}",
                "Tiempo": f"{f_elapsed:.1f}s",
                "Velocidad": f"{speed_factor:.1f}x",
                "Estado": "✅ Completado"
            })
            
            del audio
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as err:
            err_str = str(err)
            print(f"[BATCH ERROR] Error en {file_name}: {err_str}")
            table_rows.append({
                "#": idx + 1,
                "Archivo": file_name,
                "Duración": "--",
                "Tiempo": "--",
                "Velocidad": "--",
                "Estado": f"❌ Error: {err_str[:40]}..."
            })
            
    # Empaquetar en ZIP
    progress(0.99, desc="Empaquetando todos los archivos generados en ZIP...")
    zip_filename = os.path.join(tempfile.gettempdir(), f"EchoWriter_Lote_{timestamp_str}.zip")
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for root, _, batch_files in os.walk(batch_temp_dir):
            for bf in batch_files:
                full_bf_path = os.path.join(root, bf)
                zip_file.write(full_bf_path, arcname=bf)
                
    batch_elapsed = max(0.01, time.perf_counter() - batch_start_time)
    avg_speed = (total_audio_seconds / batch_elapsed) if batch_elapsed > 0 else 0.0
    tot_mins = int(total_audio_seconds // 60)
    tot_secs = int(total_audio_seconds % 60)
    success_count = sum(1 for r in table_rows if "✅" in r["Estado"])
    
    summary_md = f"""
### 📊 Resumen Final del Lote
| 📁 Procesados | ⏱️ Duración Total Audio | ⚡ Tiempo de Lote | 🚀 Velocidad Media | 📦 Archivos Generados | 💾 Estado |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **{success_count} / {total_files} exitosos** | **{tot_mins:02d}:{tot_secs:02d} min** | **{batch_elapsed:.1f} s** | **{avg_speed:.1f}x tiempo real** | **{generated_count} archivos** | **✅ Finalizado** |
"""
    
    df_result = pd.DataFrame(table_rows)
    progress(1.0, desc="¡Lote finalizado con éxito!")
    print(f"[BATCH SUCCESS] Finalizado lote de {total_files} archivos en {batch_elapsed:.1f}s.")
    return summary_md, df_result, zip_filename, model_manager.get_status_text()

# ==========================================
# 🎨 ESTILOS CSS PERSONALIZADOS (DARK THEME)
# ==========================================
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

body, .gradio-container {
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    background: radial-gradient(circle at top, #1e1b4b 0%, #030712 100%) !important;
}

.glass-panel {
    background: rgba(17, 24, 39, 0.7) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 20px !important;
    box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.5) !important;
    padding: 22px !important;
    margin-bottom: 16px !important;
}

.app-header {
    text-align: center;
    padding: 1.8rem 1rem;
    margin-bottom: 1.2rem;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(168, 85, 247, 0.12) 100%);
    border: 1px solid rgba(99, 102, 241, 0.2);
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

.action-btn {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    font-size: 1.05rem !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
}

.action-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.5) !important;
}

.vram-bar {
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    border-radius: 12px !important;
    padding: 10px 16px !important;
    display: flex !important;
    align-items: center !important;
    margin-bottom: 15px !important;
}

.stat-box {
    background: rgba(30, 41, 59, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    padding: 15px !important;
    margin-bottom: 15px !important;
}
"""

# ==========================================
# 🖥️ CONSTRUCCIÓN DE LA INTERFAZ DE USUARIO
# ==========================================
with gr.Blocks(theme=gr.themes.Default(primary_hue="indigo", secondary_hue="slate", neutral_hue="zinc"), css=custom_css, title="EchoWriter AI") as demo:
    
    # Encabezado principal
    gr.HTML(
        """
        <div class="app-header">
            <h1 class="app-title">EchoWriter AI</h1>
            <p class="app-subtitle">Transcripción Universal SOTA & Diarización Inteligente con IA Local</p>
        </div>
        """
    )
    
    # Barra de Control de Memoria VRAM (GPU)
    with gr.Row(elem_classes="vram-bar"):
        with gr.Column(scale=4):
            vram_status_display = gr.Markdown(model_manager.get_status_text())
        with gr.Column(scale=1):
            free_vram_btn = gr.Button("🧹 Liberar VRAM de GPU", size="sm")

    # Estado global para persistir el nombre base del archivo cargado
    current_video_basename = gr.State(value="reunion")

    with gr.Tabs() as main_tabs:
        
        # =========================================================================
        # PESTAÑA 1: EXTRACTOR RÁPIDO DE TEXTO LIMPIO (AUDIOS, WHATSAPP, NOTAS)
        # =========================================================================
        with gr.TabItem("⚡ 1. Extractor Rápido de Texto (WhatsApp / Audios)", id=1):
            with gr.Column(elem_classes="glass-panel"):
                gr.Markdown("### 🎙️ Extracción Rápida de Texto Limpio\n*Ideal para notas de voz de WhatsApp (`.opus`, `.ogg`), audios (`.mp3`, `.wav`, `.m4a`) o micrófono directo.*")
                
                with gr.Row():
                    # Columna Izquierda: Entrada y Controles
                    with gr.Column(scale=1):
                        with gr.Tabs():
                            with gr.TabItem("📁 Subir Archivo (Audio / Video)"):
                                quick_file_input = gr.File(
                                    label="Arrastra cualquier archivo de Audio o Video (.mp3, .mp4, .m4a, .opus, .wav, .mkv, .webm, etc.)",
                                    file_types=["audio", "video"],
                                    type="filepath"
                                )
                            with gr.TabItem("🎙️ Grabar con Micrófono"):
                                quick_mic_input = gr.Audio(
                                    label="Grabar audio directamente desde tu micrófono",
                                    sources=["microphone"],
                                    type="filepath"
                                )
                        
                        with gr.Accordion("⚙️ Opciones de Transcripción y Hardware", open=True):
                            with gr.Row():
                                quick_model_input = gr.Dropdown(
                                    label="Modelo Whisper",
                                    choices=["turbo", "large-v3", "medium", "small"],
                                    value="turbo",
                                    info="Turbo: Ultrarrápido y balanceado. Large-v3: Máxima fidelidad en audios complejos."
                                )
                                quick_compute_input = gr.Dropdown(
                                    label="Precisión GPU (Hardware)",
                                    choices=["float16", "int8"],
                                    value="float16",
                                    info="float16: Máxima velocidad en RTX 5050. int8: Menor uso de VRAM."
                                )
                                quick_batch_size = gr.Slider(
                                    label="📦 Tamaño de Lote (Batch Size)",
                                    minimum=1,
                                    maximum=32,
                                    step=1,
                                    value=16,
                                    info="Segmentos en paralelo en GPU (16 es óptimo para RTX 5050 8GB)."
                                )
                                
                            with gr.Row():
                                quick_lang_input = gr.Dropdown(
                                    label="Idioma del Audio",
                                    choices=["Automático", "Español", "Inglés", "Francés", "Alemán", "Italiano", "Portugués", "Catalán", "Ruso", "Chino", "Japonés"],
                                    value="Automático",
                                    info="Auto-detecta el idioma o elígelo manualmente."
                                )
                                quick_traducir_es = gr.Checkbox(
                                    label="Traducir a Español (Meta NLLB-200)",
                                    value=False,
                                    info="Si el audio está en otro idioma, se traducirá con Meta NLLB-200."
                                )
                                
                            quick_format_mode = gr.Radio(
                                label="Formato del Texto Limpio",
                                choices=["Texto Continuo / Párrafo", "Párrafos por Pausas / Frases", "Con Marcas de Tiempo"],
                                value="Texto Continuo / Párrafo",
                                info="Elige cómo estructurar el texto extraído."
                            )
                        
                        quick_submit_btn = gr.Button("⚡ Extraer Texto Limpio", elem_classes="action-btn")
                        
                    # Columna Derecha: Salida de Texto y Descargas
                    with gr.Column(scale=1):
                        quick_stats_output = gr.Markdown(
                            """
| ⏱️ Duración | 🌐 Idioma | 📝 Palabras | 🔤 Caracteres | ⚡ Estado del Motor |
| :---: | :---: | :---: | :---: | :---: |
| *--:--* | *--* | *0 palabras* | *0 caracteres* | *Listo* |
                            """,
                            elem_classes="stat-box"
                        )
                        
                        quick_text_output = gr.Textbox(
                            label="📄 Texto Limpio Extraído (Usa el botón superior para copiar con 1 clic)",
                            lines=12,
                            show_copy_button=True,
                            interactive=True,
                            placeholder="El texto extraído de tu audio aparecerá aquí listo para leer, copiar o exportar..."
                        )
                        
                        with gr.Row():
                            quick_txt_download = gr.File(
                                label="Descargar Texto (.txt)",
                                interactive=False
                            )
                            quick_docx_download = gr.File(
                                label="Descargar Documento (.docx)",
                                interactive=False
                            )

        # =========================================================================
        # PESTAÑA 2: CONFIGURACIÓN DE DIARIZACIÓN AVANZADA (MULTI-HABLANTE)
        # =========================================================================
        with gr.TabItem("👥 2. Transcripción & Diarización (Multi-Hablante)", id=2):
            with gr.Column(elem_classes="glass-panel"):
                gr.Markdown("### 🛠️ Transcripción con Identificación de Hablantes\n*Para entrevistas, reuniones y videos donde se requiere identificar quién dijo qué y exportar subtítulos sincronizados.*")
                
                audio_video_input = gr.File(
                    label="Subir o arrastrar Archivo (Audio/Video)",
                    file_types=["audio", "video"],
                    type="filepath"
                )
                
                with gr.Accordion("⚙️ Parámetros del Motor de IA y Diarización", open=True):
                    hf_token_input = gr.Textbox(
                        label="Token de Hugging Face (Pyannote)",
                        placeholder="hf_...",
                        type="password",
                        value=os.getenv("HF_TOKEN", ""),
                        info="Requerido. Asegúrate de haber aceptado la licencia de pyannote/speaker-diarization-community-1 o pyannote/speaker-diarization-3.1 en Hugging Face."
                    )
                    
                    with gr.Row():
                        whisper_model_input = gr.Dropdown(
                            label="Modelo Whisper",
                            choices=["turbo", "large-v3", "medium", "small"],
                            value="turbo",
                            info="Turbo es el más veloz. Large-v3 ofrece precisión suprema."
                        )
                        compute_type_input = gr.Dropdown(
                            label="Precisión GPU (Hardware)",
                            choices=["float16", "int8"],
                            value="float16",
                            info="float16 para máxima velocidad. int8 para ahorrar VRAM."
                        )
                        diarize_batch_size = gr.Slider(
                            label="📦 Tamaño de Lote Whisper (Batch Size)",
                            minimum=1,
                            maximum=24,
                            step=1,
                            value=8,
                            info="Segmentos en paralelo para Whisper en etapa 1 (8 recomendado para Diarización)."
                        )
                        diarize_model_input = gr.Dropdown(
                            label="Modelo de Diarización (Pyannote)",
                            choices=["pyannote/speaker-diarization-community-1", "pyannote/speaker-diarization-3.1"],
                            value="pyannote/speaker-diarization-community-1",
                            info="community-1 es el modelo SOTA de última generación."
                        )
                    
                    with gr.Row():
                        source_language_input = gr.Dropdown(
                            label="Idioma del Audio (Opcional)",
                            choices=["Automático", "Español", "Inglés", "Francés", "Alemán", "Italiano", "Portugués", "Catalán", "Ruso", "Chino", "Japonés"],
                            value="Automático",
                            info="Seleccionar el idioma manualmente salta el paso de auto-detección."
                        )
                        min_speakers_input = gr.Number(
                            label="Mín. Hablantes (Opcional)", 
                            precision=0, 
                            minimum=0,
                            value=0
                        )
                        max_speakers_input = gr.Number(
                            label="Máx. Hablantes (Opcional)", 
                            precision=0, 
                            minimum=0,
                            value=0
                        )
                        
                    traducir_es_input = gr.Checkbox(
                        label="Traducir transcripción al Español (Meta NLLB-200)",
                        value=False,
                        info="Si el audio está en otro idioma, se traducirá automáticamente al español."
                    )
                    
                diarize_submit_btn = gr.Button("🚀 Comenzar Transcripción & Diarización", elem_classes="action-btn")
        
        # =========================================================================
        # PESTAÑA 3: AUDITORÍA Y EDICIÓN DE DIARIZACIÓN
        # =========================================================================
        with gr.TabItem("🎬 3. Auditoría y Edición (Diarización)", id=3):
            with gr.Column(elem_classes="glass-panel"):
                gr.Markdown("### 📝 Auditoría de Transcripción (Sincronización)")
                
                with gr.Row():
                    # Columna Izquierda: Reproductor
                    with gr.Column(scale=1):
                        playback_video = gr.Video(
                            label="Reproductor de Auditoría",
                            interactive=False
                        )
                        
                    # Columna Derecha: Tabla y Exportación
                    with gr.Column(scale=1):
                        diarize_stats_output = gr.Markdown(
                            """
| ⏱️ Duración | ⚡ Tiempo Total | 🚀 Velocidad | 🎮 Pico VRAM | 💾 RAM Proceso |
| :---: | :---: | :---: | :---: | :---: |
| *--:--* | *-- s* | *--* | *-- GB* | *-- GB* |
                            """,
                            elem_classes="stat-box"
                        )
                        
                        output_dataframe = gr.Dataframe(
                            headers=["Hablante", "Inicio (s)", "Fin (s)", "Texto"],
                            datatype=["str", "number", "number", "str"],
                            label="Transcripción Editable (Doble clic en una celda para editar)",
                            interactive=True,
                            wrap=True
                        )
                        
                        # Herramienta de Renombrado Masivo de Hablantes
                        with gr.Accordion("🏷️ Renombrado Rápido de Hablantes", open=False):
                            with gr.Row():
                                old_speaker_name = gr.Textbox(
                                    label="Etiqueta Actual",
                                    placeholder="Ej: SPEAKER_00",
                                    scale=2
                                )
                                new_speaker_name = gr.Textbox(
                                    label="Nombre Real",
                                    placeholder="Ej: Juan Pérez / Entrevistador",
                                    scale=2
                                )
                                rename_btn = gr.Button("⚡ Reemplazar en Guion", scale=1)
                        
                        clean_export_input = gr.Checkbox(
                            label="🧹 Modo Subtítulos Limpios (Ocultar nombres de hablantes en descarga)",
                            value=False,
                            info="Útil para exportar a Premiere, YouTube o DaVinci Resolve sin etiquetas de hablantes."
                        )
                        
                        apply_edits_btn = gr.Button("💾 Aplicar Ediciones y Actualizar Descargas", elem_classes="action-btn")
                        
                        with gr.Row():
                            txt_download_btn = gr.File(
                                label="Descargar Transcripción (.txt)",
                                interactive=False
                            )
                            srt_download_btn = gr.File(
                                label="Descargar Subtítulos (.srt)",
                                interactive=False
                            )
                            docx_download_btn = gr.File(
                                label="📄 Descargar Guion Profesional (.docx)",
                                interactive=False
                            )

        # =========================================================================
        # PESTAÑA 4: PROCESAMIENTO POR LOTES (COLA / CARPETA)
        # =========================================================================
        with gr.TabItem("📦 4. Procesamiento por Lotes (Cola / Carpeta)", id=4):
            with gr.Column(elem_classes="glass-panel"):
                gr.Markdown("### 📦 Procesamiento por Lotes y Cola de Archivos\n*Procesa múltiples videos o audios en serie: transcribe, traduce y genera subtítulos (`.srt`), transcripciones (`.txt`) y guiones (`.docx`) nombrados automáticamente según cada video, con descarga directa individual o empaquetada en un archivo `.zip`.*")
                
                with gr.Row():
                    # Columna Izquierda: Entrada de Archivos y Configuración
                    with gr.Column(scale=1):
                        batch_files_input = gr.File(
                            label="1. Subir o arrastrar múltiples archivos de Audio/Video",
                            file_count="multiple",
                            file_types=["audio", "video"],
                            type="filepath"
                        )
                        
                        batch_folder_input = gr.Textbox(
                            label="2. O escribir la ruta de una carpeta local en Windows",
                            placeholder=r"Ej: C:\Users\krieg\Videos\MisGrabaciones",
                            info="Escaneará automáticamente todos los videos y audios compatibles dentro de la carpeta."
                        )
                        
                        with gr.Row():
                            scan_queue_btn = gr.Button("🔍 Inspeccionar Archivos en Cola", size="sm")
                            batch_save_in_source = gr.Checkbox(
                                label="💾 Guardar también en la carpeta origen del video",
                                value=True,
                                info="Crea los .srt, .txt y .docx directamente al lado de cada video en Windows."
                            )
                        
                        with gr.Accordion("⚙️ Opciones de Lote, Subtítulos y Traducción", open=True):
                            batch_clean_subtitles = gr.Checkbox(
                                label="🧹 Modo Subtítulos Limpios (Ocultar nombres de hablantes en descarga)",
                                value=True,
                                info="Genera los subtítulos .srt y transcripciones listos para DaVinci / Premiere / YouTube sin etiquetas [SPEAKER_XX]."
                            )
                            
                            with gr.Row():
                                batch_model_input = gr.Dropdown(
                                    label="Modelo Whisper",
                                    choices=["turbo", "large-v3", "medium", "small"],
                                    value="turbo",
                                    info="Turbo es ultrarrápido para procesar muchos videos."
                                )
                                batch_compute_input = gr.Dropdown(
                                    label="Precisión GPU (Hardware)",
                                    choices=["float16", "int8"],
                                    value="float16",
                                    info="float16 para máxima velocidad. int8 para ahorrar VRAM."
                                )
                                batch_size_slider = gr.Slider(
                                    label="📦 Tamaño de Lote (Batch Size)",
                                    minimum=1,
                                    maximum=32,
                                    step=1,
                                    value=16,
                                    info="Paralelismo en GPU (16 es ideal en RTX 5050)."
                                )
                                
                            with gr.Row():
                                batch_lang_input = gr.Dropdown(
                                    label="Idioma del Audio",
                                    choices=["Automático", "Español", "Inglés", "Francés", "Alemán", "Italiano", "Portugués", "Catalán", "Ruso", "Chino", "Japonés"],
                                    value="Automático",
                                    info="Auto-detecta el idioma o fuerza uno específico."
                                )
                                batch_traducir_es = gr.Checkbox(
                                    label="Traducir a Español (Meta NLLB-200)",
                                    value=False,
                                    info="Traduce cualquier video en otro idioma al español con Meta NLLB-200."
                                )
                                
                            batch_enable_diarization = gr.Checkbox(
                                label="👥 Activar Diarización Multi-Hablante (Separar voces)",
                                value=False,
                                info="Desactivado (por defecto): Subtítulos con marcas de tiempo a máxima velocidad. Activado: Identifica quién habló con Pyannote."
                            )
                            
                            batch_hf_token = gr.Textbox(
                                label="Token de Hugging Face (Requerido solo si se activa Diarización)",
                                placeholder="hf_...",
                                type="password",
                                value=os.getenv("HF_TOKEN", "")
                            )
                            
                        batch_start_btn = gr.Button("🚀 Iniciar Procesamiento por Lotes", elem_classes="action-btn")
                        
                    # Columna Derecha: Resumen, Descarga ZIP y Tabla de Seguimiento
                    with gr.Column(scale=1):
                        batch_stats_summary = gr.Markdown(
                            """
### 📊 Resumen del Lote
| 📁 Archivos | ⏱️ Tiempo Total | ⚡ Velocidad Media | 💾 Estado |
| :---: | :---: | :---: | :---: |
| *0 archivos* | *-- s* | *--* | *En espera* |
                            """,
                            elem_classes="stat-box"
                        )
                        
                        batch_zip_download = gr.File(
                            label="📦 Descargar Todos los Archivos Generados (.zip)",
                            interactive=False
                        )
                        
                        batch_results_table = gr.Dataframe(
                            headers=["#", "Archivo", "Duración", "Tiempo", "Velocidad", "Estado"],
                            datatype=["number", "str", "str", "str", "str", "str"],
                            label="📋 Registro de Procesamiento por Archivo (Cola)",
                            interactive=False,
                            wrap=True
                        )

    # =========================================================================
    # EVENTOS Y VINCULACIONES
    # =========================================================================
    
    # Evento Liberar VRAM
    def handle_free_vram():
        model_manager.unload()
        return model_manager.get_status_text()

    free_vram_btn.click(
        fn=handle_free_vram,
        outputs=[vram_status_display]
    )

    # Evento de Transcripción Rápida (Pestaña 1)
    quick_submit_btn.click(
        fn=quick_transcribe_audio,
        inputs=[
            quick_file_input,
            quick_mic_input,
            quick_model_input,
            quick_compute_input,
            quick_batch_size,
            quick_lang_input,
            quick_format_mode,
            quick_traducir_es
        ],
        outputs=[
            quick_text_output,
            quick_stats_output,
            quick_txt_download,
            quick_docx_download,
            vram_status_display
        ]
    )
    
    # Evento de Diarización Avanzada (Pestaña 2)
    diarize_submit_btn.click(
        fn=transcribe_and_diarize,
        inputs=[
            audio_video_input,
            hf_token_input,
            whisper_model_input,
            compute_type_input,
            diarize_batch_size,
            diarize_model_input,
            source_language_input,
            min_speakers_input,
            max_speakers_input,
            traducir_es_input
        ],
        outputs=[
            output_dataframe,
            txt_download_btn,
            srt_download_btn,
            docx_download_btn,
            main_tabs,
            playback_video,
            vram_status_display,
            diarize_stats_output,
            current_video_basename
        ]
    )
    
    # Evento de Renombrado Masivo de Hablantes (Pestaña 3)
    rename_btn.click(
        fn=rename_speaker_in_df,
        inputs=[output_dataframe, old_speaker_name, new_speaker_name, clean_export_input, current_video_basename],
        outputs=[output_dataframe, txt_download_btn, srt_download_btn, docx_download_btn]
    )

    # Evento de Aplicar Ediciones en Auditoría (Pestaña 3)
    apply_edits_btn.click(
        fn=update_downloads_from_df,
        inputs=[output_dataframe, clean_export_input, current_video_basename],
        outputs=[txt_download_btn, srt_download_btn, docx_download_btn]
    )

    # Evento de Inspeccionar Cola de Lotes (Pestaña 4)
    scan_queue_btn.click(
        fn=scan_batch_queue,
        inputs=[batch_files_input, batch_folder_input],
        outputs=[batch_stats_summary, batch_results_table, batch_zip_download]
    )

    # Evento de Iniciar Procesamiento por Lotes (Pestaña 4)
    batch_start_btn.click(
        fn=batch_process_media,
        inputs=[
            batch_files_input,
            batch_folder_input,
            batch_save_in_source,
            batch_clean_subtitles,
            batch_model_input,
            batch_compute_input,
            batch_size_slider,
            batch_lang_input,
            batch_traducir_es,
            batch_enable_diarization,
            batch_hf_token
        ],
        outputs=[
            batch_stats_summary,
            batch_results_table,
            batch_zip_download,
            vram_status_display
        ]
    )

# Lanzar aplicación localmente
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
