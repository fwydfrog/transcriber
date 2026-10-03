import sys
import os

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

import gc
import torch
import pickle
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

from dotenv import load_dotenv
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

import subprocess

def convert_to_wav(input_path):
    """Конвертирует медиафайл в временный WAV-файл (16kHz, Mono)."""
    wav_path = input_path + ".tmp.wav"
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        wav_path
    ]
    # Запуск FFmpeg без вывода лишнего спама в консоль
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return wav_path

def transcribe_with_diarization(file_path):
    use_cuda = torch.cuda.is_available()
    cache_file = file_path + ".diarization.pkl"

    # 0. Конвертируем исходный MP4 в чистый WAV (16kHz, mono)
    print("0/2. Подготовка аудиофайла (конвертация в WAV 16kHz)...")
    wav_file = convert_to_wav(file_path)

    try:
        # 1. ДИАРИЗАЦИЯ (С КЭШИРОВАНИЕМ)
        if os.path.exists(cache_file):
            print(f"1/2. Загружаем готовую диаризацию из кэша ({cache_file})...")
            with open(cache_file, "rb") as f:
                diar_result = pickle.load(f)
        else:
            print("1/2. Определение спикеров (PyAnnote на GPU)...")
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=HF_TOKEN
            )
            if use_cuda:
                pipeline.to(torch.device("cuda"))
            else:
                pipeline.to(torch.device("cpu"))

            with ProgressHook() as hook:
                # ВАЖНО: передаем подготовленный wav_file
                diar_result = pipeline(wav_file, hook=hook)

            # Сохраняем кэш
            with open(cache_file, "wb") as f:
                pickle.dump(diar_result, f)

            del pipeline
            gc.collect()
            if use_cuda:
                torch.cuda.empty_cache()

        # Извлекаем треки спикеров
        tracks = list(get_diarization_tracks(diar_result))

        # 2. РАСПОЗНАВАНИЕ РЕЧИ (Whisper)
        print("\n2/2. Распознавание речи (Faster-Whisper)...")
        try:
            whisper_model = WhisperModel("medium", device="cuda" if use_cuda else "cpu", compute_type="int8_float16" if use_cuda else "int8")
        except Exception as e:
            print(f"GPU недоступен для Whisper ({e}), переключаемся на CPU...")
            whisper_model = WhisperModel("medium", device="cpu", compute_type="int8")

        # Whisper лучше распознает из обработанного WAV
        segments, info = whisper_model.transcribe(wav_file, beam_size=5)
        print(f"Язык: {info.language} (вероятность: {info.language_probability:.2f})\n")

        # 3. СКЛЕЙКА И СОХРАНЕНИЕ
        output_txt = file_path + "_diarized.txt"
        with open(output_txt, "w", encoding="utf-8") as f:
            for segment in segments:
                speaker_durations = {}
                for turn, _, speaker in tracks:
                    overlap_start = max(segment.start, turn.start)
                    overlap_end = min(segment.end, turn.end)
                    overlap = max(0, overlap_end - overlap_start)
                    if overlap > 0:
                        speaker_durations[speaker] = speaker_durations.get(speaker, 0) + overlap
                
                speaker = max(speaker_durations, key=speaker_durations.get) if speaker_durations else "UNKNOWN"

                t_start = int(segment.start)
                t_end = int(segment.end)
                line = f"[{t_start//60:02d}:{t_start%60:02d} -> {t_end//60:02d}:{t_end%60:02d}] [{speaker}]: {segment.text.strip()}\n"
                print(line, end="")
                f.write(line)

        print(f"\nГотово! Результат сохранен в {output_txt}")

    finally:
        # Удаляем временный WAV файл после завершения
        if os.path.exists(wav_file):
            os.remove(wav_file)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path_file = sys.argv[1]
    else:
        path_file = input("Введите путь к файлу: ").strip('"')
        
    transcribe_with_diarization(path_file)