# audio_capture.py
import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import requests
import time
from faster_whisper import WhisperModel

API_URL = "http://127.0.0.1:8000/ingest_text"
MODEL_SIZE = "small"   # tiny / base / small / medium (trade-off speed/accuracy)
DEVICE = "cpu"         # "cpu" or "cuda"

CHUNK_SECONDS = 5      # size of audio chunks to transcribe
SAMPLE_RATE = 16000
LANG = None            # None = auto language detect; or "vi" for Vietnamese

print("Loading Whisper model (this may take a while)...")
whisper_model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type="int8")

def record_chunk(duration=CHUNK_SECONDS, sr=SAMPLE_RATE):
    print(f"Recording {duration}s...")
    data = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='int16')
    sd.wait()
    return data.flatten()

def save_wav(arr: np.ndarray, sr=SAMPLE_RATE):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, arr.astype('int16'), sr)
    return tmp.name

def transcribe_file(path):
    segments, info = whisper_model.transcribe(path, beam_size=5, language=LANG)
    text = " ".join([seg.text for seg in segments]).strip()
    return text

def post_text(text, speaker=None):
    try:
        payload = {"text": text}
        if speaker:
            payload["speaker"] = speaker
        r = requests.post(API_URL, json=payload, timeout=10)
        if r.ok:
            print("Posted:", text[:80], "->", r.json().get("entry", {}).get("label"))
        else:
            print("API error:", r.status_code, r.text)
    except Exception as e:
        print("Post error:", e)

def main_loop():
    print("Starting audio capture loop. Press Ctrl+C to stop.")
    while True:
        try:
            arr = record_chunk()
            wav = save_wav(arr)
            text = transcribe_file(wav)
            if text:
                print("Transcribed:", text)
                post_text(text)
            else:
                print("No speech detected in chunk.")
            time.sleep(0.1)  # small pause
        except KeyboardInterrupt:
            print("Stopping.")
            break
        except Exception as e:
            print("Error in loop:", e)
            time.sleep(1)

if __name__ == "__main__":
    main_loop()
