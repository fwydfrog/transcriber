import sys
import os
import glob
import gc
import torch
import pickle
import subprocess
from dotenv import load_dotenv

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

venv_path = sys.prefix
nvidia_base = os.path.join(venv_path, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages", "nvidia")
if not os.path.exists(nvidia_base):
    nvidia_base = os.path.join(venv_path, "lib64", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages", "nvidia")

extra_paths = [
    os.path.join(nvidia_base, "cublas", "lib"),
    os.path.join(nvidia_base, "cudnn", "lib"),
]

existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = ":".join(extra_paths + [existing_ld]).strip(":")

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

def get_diarization_tracks(diar_result):
    """Универсальная распаковка любого формата возврата PyAnnote."""
    if hasattr(diar_result, "speaker_diarization"):
        return diar_result.speaker_diarization.itertracks(yield_label=True)
    if hasattr(diar_result, "speaker_timeline"):
        return diar_result.speaker_timeline.itertracks(yield_label=True)
    if hasattr(diar_result, "annotation"):
        return diar_result.annotation.itertracks(yield_label=True)
    if hasattr(diar_result, "to_annotation"):
        return diar_result.to_annotation().itertracks(yield_label=True)
    if hasattr(diar_result, "itertracks"):
        return diar_result.itertracks(yield_label=True)
    
    raise AttributeError(f"Не удалось извлечь таймлайны из объекта типа {type(diar_result)}")

def convert_to_wav(input_path):
    """Конвертирует медиафайл в временный WAV-файл (16kHz, Mono)."""
    wav_path = input_path + ".tmp.wav"
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        wav_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return wav_path

def transcribe_with_diarization(file_path, output_dir, cache_dir):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    use_cuda = torch.cuda.is_available()
    cache_file = os.path.join(cache_dir, f"{base_name}.diarization.pkl")
    output_txt = os.path.join(output_dir, f"{base_name}_diarized.txt")

    print(f"\n[{base_name}] 0/2. Подготовка аудиофайла (конвертация в WAV 16kHz)...")
    wav_file = convert_to_wav(file_path)

    try:
        if os.path.exists(cache_file):
            print(f"[{base_name}] 1/2. Загружаем готовую диаризацию из кэша ({cache_file})...")
            with open(cache_file, "rb") as f:
                diar_result = pickle.load(f)
        else:
            print(f"[{base_name}] 1/2. Определение спикеров (PyAnnote на GPU)...")
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=HF_TOKEN
            )
            if use_cuda:
                pipeline.to(torch.device("cuda"))
            else:
                pipeline.to(torch.device("cpu"))

            with ProgressHook() as hook:
                diar_result = pipeline(wav_file, hook=hook)

            # Сохраняем кэш в папку cache
            with open(cache_file, "wb") as f:
                pickle.dump(diar_result, f)

            del pipeline
            gc.collect()
            if use_cuda:
                torch.cuda.empty_cache()

        tracks = list(get_diarization_tracks(diar_result))

        # 2. РАСПОЗНАВАНИЕ РЕЧИ (Whisper)
        print(f"[{base_name}] 2/2. Распознавание речи (Faster-Whisper)...")
        try:
            whisper_model = WhisperModel("medium", device="cuda" if use_cuda else "cpu", compute_type="int8_float16" if use_cuda else "int8")
        except Exception as e:
            print(f"GPU недоступен для Whisper ({e}), переключаемся на CPU...")
            whisper_model = WhisperModel("medium", device="cpu", compute_type="int8")

        segments, info = whisper_model.transcribe(wav_file, beam_size=5, language="en")
        print(f"[{base_name}] Язык: {info.language} (вероятность: {info.language_probability:.2f})\n")
        print("\nФормирование текста...\n")
        with open(output_txt, "w", encoding="utf-8") as f:
            current_speaker = None
            
            for segment in segments:
                # 1. Определяем спикера для текущего сегмента
                speaker_durations = {}
                for turn, _, speaker in tracks:
                    overlap_start = max(segment.start, turn.start)
                    overlap_end = min(segment.end, turn.end)
                    overlap = max(0, overlap_end - overlap_start)
                    if overlap > 0:
                        speaker_durations[speaker] = speaker_durations.get(speaker, 0) + overlap
                
                speaker = max(speaker_durations, key=speaker_durations.get) if speaker_durations else "UNKNOWN"
                text = segment.text.strip()

                # 2. Логика объединения текста
                if speaker == current_speaker:
                    # Тот же спикер продолжает говорить - добавляем пробел и текст
                    print(f" {text}", end="", flush=True)
                    f.write(f" {text}")
                else:
                    # Заговорил новый спикер - начинаем с новой строки (если это не первая строчка вообще)
                    prefix = "\n" if current_speaker is not None else ""
                    line_start = f"{prefix}[{speaker}]: {text}"
                    print(line_start, end="", flush=True)
                    f.write(line_start)
                    current_speaker = speaker
                    
            # Добавляем финальный перенос строки в конце файла
            print()
            f.write("\n")
        print(f"\nГотово! Результат сохранен в {output_txt}")

    finally:
        if os.path.exists(wav_file):
            os.remove(wav_file)

if __name__ == "__main__":
    current_dir = os.getcwd()
    
    videos_dir = os.path.join(current_dir, "Videos")
    script_dir = os.path.join(current_dir, "script")
    cache_dir = os.path.join(current_dir, "cache")

    if not os.path.exists(videos_dir):
        print(f"Папка 'Videos' не найдена по пути: {videos_dir}")
        print("Пожалуйста, создайте её и поместите туда файлы .mp3")
        sys.exit(1)

    mp3_files = glob.glob(os.path.join(videos_dir, "*.mp3"))
    if not mp3_files:
        print(f"В папке {videos_dir} не найдено файлов формата .mp3")
        sys.exit(0)

    os.makedirs(script_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Найдено {len(mp3_files)} файлов для обработки. Начинаем...")
    print(f"Тексты будут сохранены в: {script_dir}")
    print(f"Кэш диаризации в: {cache_dir}")
    print("-" * 50)

    for i, mp3_file in enumerate(mp3_files, 1):
        print(f"\n=== Обработка файла {i}/{len(mp3_files)}: {os.path.basename(mp3_file)} ===")
        transcribe_with_diarization(mp3_file, script_dir, cache_dir)
        
    print("\n✅ Все файлы успешно обработаны!")