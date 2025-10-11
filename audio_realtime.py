import sounddevice as sd
import numpy as np
import queue
import threading
import requests
from faster_whisper import WhisperModel

# Cấu hình mô hình Whisper
model = WhisperModel("base", device="cpu")
API_URL = "http://127.0.0.1:8000/predict"

q = queue.Queue()
recording = True

def audio_callback(indata, frames, time, status):
    q.put(indata.copy())

def transcribe_audio():
    global recording
    samplerate = 16000
    blocksize = 4000

    with sd.InputStream(samplerate=samplerate, channels=1, dtype='float32', callback=audio_callback):
        print("🎤 Listening... Press Ctrl+C to stop.")
        buffer = np.array([], dtype=np.float32)

        while recording:
            block = q.get()
            buffer = np.concatenate((buffer, block[:, 0]))

            if len(buffer) > samplerate * 5:  # mỗi 5 giây xử lý 1 lần
                audio_chunk = buffer[:samplerate * 5]
                buffer = buffer[samplerate * 5:]

                segments, _ = model.transcribe(audio_chunk, beam_size=5)
                text = " ".join([seg.text for seg in segments]).strip()

                if text:
                    print("🗣️ Text:", text)
                    try:
                        r = requests.post(API_URL, json={"text": text})
                        print("✅ Sent to API:", r.json())
                    except Exception as e:
                        print("❌ API error:", e)

try:
    transcribe_audio()
except KeyboardInterrupt:
    print("⏹️ Stopped.")
    recording = False
