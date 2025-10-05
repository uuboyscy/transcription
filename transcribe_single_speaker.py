import multiprocessing
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel

INPUT_DIR = Path("resource")
OUTPUT_DIR = Path("output")
MODEL_SIZE = "medium"               # tiny, base, small, medium, large
PREFERRED_LANGUAGE: Optional[str] = None
MAX_CONCURRENT_CONVERSIONS = 3


def _cpu_workers() -> int:
    cores = multiprocessing.cpu_count()
    return max(1, min(cores, cores // 2 or 1))


def load_model() -> WhisperModel:
    import torch
    has_mps = torch.backends.mps.is_available()
    threads = multiprocessing.cpu_count()
    worker_count = _cpu_workers()
    # attempts = [
    #     # ("metal", "int8_float16"),
    #     ("mps", "int8_float16"),  # use MPS backend for Apple Silicon
    #     ("cpu", "int8"),
    # ]
    attempts = []
    if has_mps:
        attempts.append(("mps", "int8_float16"))
    attempts.append(("cpu", "int8"))

    last_error: Optional[Exception] = None
    for device, compute_type in attempts:
        try:
            print(f"→ Loading model on {device} ({compute_type})")
            model = WhisperModel(
                MODEL_SIZE,
                device=device,
                compute_type=compute_type,
                cpu_threads=threads,
                num_workers=worker_count,
            )
            print(f"✅ Model loaded on {device} ({compute_type})")
            return model
        except Exception as exc:  # RuntimeError for unsupported configs
            last_error = exc
            print(f"⚠️  Falling back from {device} ({compute_type}): {exc}")
    print("⚠️  All attempts to load the model failed.")
    raise RuntimeError("Unable to load WhisperModel with the configured options") from last_error


def needs_update(source: Path, target: Path) -> bool:
    if not source.exists():
        return False
    if not target.exists():
        return True
    return target.stat().st_mtime < source.stat().st_mtime


def convert_m4a_to_wav(m4a_path: Path, wav_path: Path):
    if not needs_update(m4a_path, wav_path):
        return wav_path

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(m4a_path),
        "-ar",
        "16000",
        "-ac",
        "1",
        str(wav_path),
    ]
    subprocess.run(command, check=True)
    print(f"✔ Converted: {m4a_path.name} → {wav_path.name}")
    return wav_path


def transcribe_audio(wav_path: Path, txt_path: Path, model: WhisperModel):
    if not needs_update(wav_path, txt_path):
        return

    try:
        segments, info = model.transcribe(
            str(wav_path),
            language=PREFERRED_LANGUAGE,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
        )
    except Exception as e:
        print(f"❌ Error transcribing {wav_path.name}: {e}")
        return

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"[Language: {info.language}]\n\n")
        for segment in segments:
            f.write(f"[{segment.start:.2f}s → {segment.end:.2f}s] {segment.text}\n")
    print(f"📝 Transcribed: {wav_path.name} → {txt_path.name}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    m4a_files = sorted(INPUT_DIR.glob("*.m4a"))
    if not m4a_files:
        print("No .m4a files found in resource/")
        return

    model = load_model()

    conversions = []
    max_workers = min(MAX_CONCURRENT_CONVERSIONS, len(m4a_files))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for wav_path in pool.map(_prepare_wav, m4a_files):
            conversions.append(wav_path)

    print(f"🔊 Starting transcription for {len(conversions)} files...")
    for wav_path in conversions:
        txt_path = wav_path.with_suffix(".txt")
        transcribe_audio(wav_path, txt_path, model)

    print(f"🔊 Ending transcription for {len(conversions)} files...")


def _prepare_wav(m4a_path: Path) -> Path:
    wav_path = OUTPUT_DIR / f"{m4a_path.stem}.wav"
    convert_m4a_to_wav(m4a_path, wav_path)
    return wav_path

if __name__ == "__main__":
    main()
