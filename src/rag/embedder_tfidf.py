from __future__ import annotations
import numpy as np
from pathlib import Path
import joblib

try:
    # sklearn >=1.0
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception as e:
    raise RuntimeError("scikit-learn is required. Try: pip install scikit-learn") from e

class TfidfEmbedder:
    """
    Lightweight TF-IDF encoder for local RAG (no native deps).
    Saves/loads the vectorizer to artifacts/tfidf_vectorizer.pkl by default.
    """
    def __init__(self, model_path: str | Path = "artifacts/tfidf_vectorizer.pkl"):
        self.path = Path(model_path)
        self.vectorizer: TfidfVectorizer | None = None
        self.dim: int | None = None

    def fit(self, texts: list[str]):
        self.vectorizer = TfidfVectorizer(min_df=3, max_df=0.9, ngram_range=(1,2))
        self.vectorizer.fit(texts)
        self.dim = len(self.vectorizer.vocabulary_)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, self.path)

    def load(self):
        self.vectorizer = joblib.load(self.path)
        self.dim = len(self.vectorizer.vocabulary_)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.vectorizer is None:
            self.load()
        mat = self.vectorizer.transform(texts)  # scipy CSR matrix
        return mat.toarray().astype("float32")