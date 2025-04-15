import subprocess
from pathlib import Path

from faster_whisper import WhisperModel

INPUT_DIR = Path("resource")
OUTPUT_DIR = Path("output")
MODEL_SIZE = "medium"         # Can be: tiny, base, small, medium, large
COMPUTE_TYPE = "int8"        # Use "float16" or "int8_float16" for GPU if available

def convert_m4a_to_wav(m4a_path: Path, wav_path: Path):
    command = [
        "ffmpeg", "-y",
        "-i", str(m4a_path),
        "-ar", "16000", "-ac", "1",
        str(wav_path)
    ]
    subprocess.run(command, check=True)
    print(f"✔ Converted: {m4a_path.name} → {wav_path.name}")

def transcribe_audio(wav_path: Path, txt_path: Path, model):
    segments, info = model.transcribe(str(wav_path))
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"[Language: {info.language}]\n\n")
        for segment in segments:
            f.write(f"[{segment.start:.2f}s → {segment.end:.2f}s] {segment.text}\n")
    print(f"📝 Transcribed: {wav_path.name} → {txt_path.name}")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = WhisperModel(MODEL_SIZE, compute_type=COMPUTE_TYPE)

    for m4a_file in INPUT_DIR.glob("*.m4a"):
        base_name = m4a_file.stem
        wav_file = OUTPUT_DIR / f"{base_name}.wav"
        txt_file = OUTPUT_DIR / f"{base_name}.txt"

        convert_m4a_to_wav(m4a_file, wav_file)
        transcribe_audio(wav_file, txt_file, model)

if __name__ == "__main__":
    main()
