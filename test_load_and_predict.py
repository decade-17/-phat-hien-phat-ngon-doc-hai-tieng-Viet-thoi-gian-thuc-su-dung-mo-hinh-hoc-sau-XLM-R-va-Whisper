# test_load_and_predict.py
from pathlib import Path
import pickle

BASE_DIR = Path(__file__).resolve().parent
TFIDF_PATH = BASE_DIR / "tf_idf.pkt"
MODEL_PATH = BASE_DIR / "toxicity_model.pkt"

print("TFIDF path:", TFIDF_PATH)
print("MODEL path:", MODEL_PATH)

try:
    tfidf = pickle.load(open(TFIDF_PATH, "rb"))
    model = pickle.load(open(MODEL_PATH, "rb"))
except Exception as e:
    print("ERROR loading pickles:", e)
    raise

print("Loaded types:", type(tfidf), type(model))

tests = ["I hate you moron", "Hôm nay thật tuyệt, tôi rất vui."]
for t in tests:
    X = tfidf.transform([t])
    pred = model.predict(X)
    print(f"INPUT: {t}")
    print("  predict array:", pred, "=>", int(pred[0]))
    if hasattr(model, "predict_proba"):
        print("  proba:", model.predict_proba(X)[0])
