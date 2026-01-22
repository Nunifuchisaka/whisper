import sys
import os
import subprocess
import torch
from faster_whisper import WhisperModel

MODEL_NAME = "large-v3"
LANGUAGE = "ja"
SKIP_ERRORS = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_TRACK = 1   # ← 2番目の音声トラック（0始まりなので1）

def extract_track(input_file, track_index):
    wav_path = os.path.splitext(input_file)[0] + f"_track{track_index}.wav"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_file,
        "-map", f"0:a:{track_index}",
        "-ac", "1",
        "-ar", "16000",
        wav_path
    ]

    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return wav_path


if len(sys.argv) < 2:
    print("Usage: python Whisper.py <file_or_folder>")
    sys.exit(1)

input_path = sys.argv[1]

video_exts = [".mp4", ".mkv", ".avi", ".mov"]
files = []

if os.path.isfile(input_path):
    if os.path.splitext(input_path)[1].lower() in video_exts:
        files.append(input_path)
elif os.path.isdir(input_path):
    for f in os.listdir(input_path):
        if os.path.splitext(f)[1].lower() in video_exts:
            files.append(os.path.join(input_path, f))
else:
    print(f"Path not found: {input_path}")
    sys.exit(1)

if not files:
    print("No video files found.")
    sys.exit(0)

print(f"Loading model {MODEL_NAME} on {DEVICE}...")
model = WhisperModel(MODEL_NAME, device=DEVICE)

for file in files:
    print(f"\nProcessing: {file}")
    output_txt = os.path.splitext(file)[0] + ".txt"

    try:
        # 2番目トラック抽出
        wav = extract_track(file, TARGET_TRACK)

        segments, info = model.transcribe(
            wav,
            language=LANGUAGE,
            vad_filter=True
        )

        with open(output_txt, "w", encoding="utf-8") as f:
            for segment in segments:
                print(f"[{segment.start:.2f} - {segment.end:.2f}] {segment.text}")
                f.write(segment.text + "\n")

        print(f"Saved transcription: {output_txt}")

        # 一時ファイル削除
        if os.path.exists(wav):
            os.remove(wav)

    except Exception as e:
        print(f"Error processing {file}: {e}")
        if not SKIP_ERRORS:
            raise
        print("Skipping to next file...")
