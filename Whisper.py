import sys
import os
from faster_whisper import WhisperModel

# モデル設定
MODEL_NAME = "large-v3"
DEVICE = "cuda"  # GPU
SKIP_ERRORS = True
LANGUAGE = "ja"

# 入力取得
if len(sys.argv) < 2:
    print("Usage: python transcribe_folder.py <file_or_folder>")
    sys.exit(1)

input_path = sys.argv[1]

# 対象ファイルリスト作成
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

# モデルロード
print(f"Loading model {MODEL_NAME} on {DEVICE}...")
model = WhisperModel(MODEL_NAME, device=DEVICE)

for file in files:
    print(f"\nProcessing: {file}")
    output_txt = os.path.splitext(file)[0] + ".txt"
    try:
        segments, info = model.transcribe(file, language=LANGUAGE)

        with open(output_txt, "w", encoding="utf-8") as f:
            for segment in segments:
                # コンソールに表示
                print(f"[{segment.start:.2f} - {segment.end:.2f}] {segment.text}")
                # txt に書き込み
                f.write(segment.text + "\n")

        print(f"Saved transcription: {output_txt}")
    except Exception as e:
        print(f"Error processing {file}: {e}")
        if not SKIP_ERRORS:
            raise
        else:
            print("Skipping to next file...")
