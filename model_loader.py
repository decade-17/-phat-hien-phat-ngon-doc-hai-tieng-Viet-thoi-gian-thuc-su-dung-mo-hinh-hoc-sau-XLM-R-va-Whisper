# model_loader.py
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

CLASSIFIER_MODEL = "unitary/multilingual-toxic-xlm-roberta"  # multilingual toxic model

_classifier = None

def get_classifier(device: str = "cpu"):
    global _classifier
    if _classifier is None:
        # device: "cpu" or "cuda:0"
        _classifier = pipeline(
            "text-classification",
            model=CLASSIFIER_MODEL,
            device=0 if (device != "cpu" and torch.cuda.is_available()) else -1,
            truncation=True,
            top_k=None
        )
    return _classifier

def predict_text(text: str, threshold: float = 0.5):
    clf = get_classifier()
    if not text or not text.strip():
        return {"label": "empty", "score": 0.0}
    res = clf(text[:1000])  # limit length
    # res is a list of dicts or a single dict depending on model; normalize:
    if isinstance(res, list) and len(res) > 0:
        # some multilingual toxic models return multiple labels; we look for 'toxic' related labels
        # find highest scoring toxic-related label
        best = max(res, key=lambda x: x.get("score", 0.0))
        label = best.get("label", "")
        score = float(best.get("score", 0.0))
    elif isinstance(res, dict):
        label = res.get("label", "")
        score = float(res.get("score", 0.0))
    else:
        label = "unknown"
        score = 0.0

    label_lower = label.lower()
    # heuristics: treat labels containing 'toxic'/'abuse'/'hate' as toxic
    toxic_words = ["toxic", "abusive", "abuse", "hate", "offensive", "insult"]
    is_toxic = any(w in label_lower for w in toxic_words) or (score >= threshold)
    return {"label": "toxic" if is_toxic else "non-toxic", "orig_label": label, "score": score}
