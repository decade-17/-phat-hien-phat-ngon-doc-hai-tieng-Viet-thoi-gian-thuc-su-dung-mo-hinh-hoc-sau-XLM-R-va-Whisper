from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import tempfile
from faster_whisper import WhisperModel
import os

app = FastAPI(title="Toxicity Realtime VN API")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ Model phát hiện bình luận độc hại (đa ngôn ngữ, có thể hiểu tiếng Việt)
CLASSIFIER_NAME = "textdetox/xlmr-base-toxicity-classifier"
WHISPER_SIZE = "small"

print("🔹 Đang tải model phân loại (classifier)...")
tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_NAME)
model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_NAME)
model.to(DEVICE)
model.eval()
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if DEVICE == "cuda" else -1)
print("✅ Classifier loaded.")

print("🔹 Đang tải Whisper model...")
whisper = WhisperModel(WHISPER_SIZE, device=DEVICE, compute_type="int8")
print("✅ Whisper loaded.")


class TextIn(BaseModel):
    text: str


def classify_text(text: str):
    if not text.strip():
        return {"label": "empty", "score": 0.0}

    res = pipe(text)[0]
    label = res["label"].lower()
    score = float(res["score"])

    # 🌐 Chuyển nhãn sang tiếng Việt thân thiện
    if "toxic" in label:
        label = "Độc hại"
    elif "neutral" in label or "non-toxic" in label:
        label = "Bình thường"
    else:
        label = "Không xác định"

    return {"label": label, "score": round(score, 4)}


@app.post("/predict")
async def predict(payload: TextIn):
    res = classify_text(payload.text)
    return {"text": payload.text, "label": res["label"], "score": res["score"]}


@app.post("/speech")
async def predict_speech(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    segments, info = whisper.transcribe(tmp_path, beam_size=5, language="vi")
    text = " ".join([seg.text for seg in segments]).strip()
    os.remove(tmp_path)

    if not text:
        return {"text": "", "label": "empty", "score": 0.0}

    res = classify_text(text)
    return {"text": text, "label": res["label"], "score": res["score"]}
