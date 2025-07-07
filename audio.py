import os
import math
import subprocess
import whisper

# === CONFIGURATION ===
input_audio = r"E:\Pycharm All Project\PythonProject\audio\R01_IMP_RES_104.wav"
output_dir = os.path.dirname(input_audio)
split_prefix = os.path.join(output_dir, "part_")
model_size = "medium"  # or "large", "large-v3" for higher accuracy
language = "bn"
segments = 10  # how many parts to split into

# === STEP 1: Get audio duration (in seconds) using ffprobe ===
def get_duration(file_path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", file_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    return float(result.stdout)

duration = get_duration(input_audio)
segment_time = math.ceil(duration / segments)

print(f"Audio Duration: {duration:.2f} seconds")
print(f"Splitting into {segments} parts of ~{segment_time} seconds each...")

# === STEP 2: Split the audio using ffmpeg ===
split_command = [
    "ffmpeg",
    "-i", input_audio,
    "-f", "segment",
    "-segment_time", str(segment_time),
    "-c", "copy",
    f"{split_prefix}%02d.wav"
]

subprocess.run(split_command, check=True)
print("✅ Audio split complete.")

# === STEP 3: Translate each part using Whisper ===
model = whisper.load_model(model_size)
translated_text = ""

files = sorted(f for f in os.listdir(output_dir) if f.startswith("part_") and f.endswith(".wav"))

for f in files:
    full_path = os.path.join(output_dir, f)
    print(f"🔄 Translating {f} ...")
    result = model.transcribe(full_path, language=language, task="translate", fp16=False)
    translated_text += result["text"].strip() + "\n\n"

# === STEP 4: Save the full translated text ===
output_txt = os.path.join(output_dir, "translated_output.txt")
with open(output_txt, "w", encoding="utf-8") as f:
    f.write(translated_text)

print(f"✅ Translation complete. Output saved to:\n{output_txt}")
