from functools import lru_cache
from pathlib import Path

import joblib


MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "spam"


@lru_cache(maxsize=1)
def _load_spam_model():
    classifier = joblib.load(MODEL_DIR / "spam_classifier.pkl")
    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
    return classifier, vectorizer


def run_spam(cleaned_text: str) -> dict:
    try:
        classifier, vectorizer = _load_spam_model()
        prediction = classifier.predict(vectorizer.transform([cleaned_text]))[0]
        label = str(prediction).lower()
        is_spam = label == "spam"
        return {"is_spam": is_spam, "label": "spam" if is_spam else "ham"}
    except Exception as exc:
        return {"is_spam": None, "label": None, "error": str(exc)}
