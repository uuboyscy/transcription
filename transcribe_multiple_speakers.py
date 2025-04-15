import os
import subprocess
from pathlib import Path

import torchaudio
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

# === Configuration ===
INPUT_DIR = Path("resource")
OUTPUT_DIR = Path("output")
MODEL_SIZE = "medium"
COMPUTE_TYPE = "int8"
HF_TOKEN = os.getenv("HF_TOKEN")  # Set via env var for security

# === Load Diarization Pipeline ===
print("📦 Loading speaker diarization model...")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=HF_TOKEN
)

# === Helper Functions ===
def convert_m4a_to_wav(m4a_path: Path, wav_path: Path):
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(m4a_path),
        "-ar", "16000", "-ac", "1",
        str(wav_path)
    ], check=True)
    print(f"✔ Converted: {m4a_path.name} → {wav_path.name}")

def diarize_audio(wav_path: Path):
    print(f"👥 Running speaker diarization on {wav_path.name}...")
    return pipeline(str(wav_path))

def transcribe_segments(wav_path: Path, diarization, model, txt_path: Path):
    waveform, sample_rate = torchaudio.load(str(wav_path))

    with open(txt_path, "w", encoding="utf-8") as f:
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start = turn.start
            end = turn.end
            segment_waveform = waveform[:, int(start * sample_rate):int(end * sample_rate)]

            # Save temp file
            segment_path = wav_path.parent / f"temp_{start:.2f}_{end:.2f}.wav"
            torchaudio.save(str(segment_path), segment_waveform, sample_rate)

            segments, _ = model.transcribe(str(segment_path))

            for s in segments:
                f.write(f"[{speaker} | {s.start + start:.2f}s → {s.end + start:.2f}s] {s.text.strip()}\n")

            segment_path.unlink()

    print(f"📝 Transcribed with speakers: {wav_path.name} → {txt_path.name}")

# === Main Workflow ===
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = WhisperModel(MODEL_SIZE, compute_type=COMPUTE_TYPE)

    for m4a_file in INPUT_DIR.glob("*.m4a"):
        base_name = m4a_file.stem
        wav_file = OUTPUT_DIR / f"{base_name}.wav"
        txt_file = OUTPUT_DIR / f"{base_name}_speaker.txt"

        convert_m4a_to_wav(m4a_file, wav_file)
        diarization = diarize_audio(wav_file)
        transcribe_segments(wav_file, diarization, model, txt_file)

if __name__ == "__main__":
    main()
