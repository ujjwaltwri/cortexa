from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.neighbors import NearestNeighbors

class SklearnStore:
    """
    Cosine-similarity NearestNeighbors index (no FAISS).
    Persists index (pkl) and metadata (json).
    """
    def __init__(self, index_path: str | Path = "artifacts/tfidf_nbrs.pkl",
                 meta_path: str | Path = "artifacts/cortexa.meta.json"):
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        self.nbrs: NearestNeighbors | None = None
        self.meta: list[dict] = []

    def build(self, vecs: np.ndarray, metas: list[dict], n_neighbors: int = 50):
        self.nbrs = NearestNeighbors(metric="cosine", n_neighbors=n_neighbors)
        self.nbrs.fit(vecs)
        self.meta = metas
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.nbrs, self.index_path)
        self.meta_path.write_text(json.dumps(self.meta))

    def load(self):
        self.nbrs = joblib.load(self.index_path)
        self.meta = json.loads(self.meta_path.read_text())
        return self

    def search(self, qvec: np.ndarray, k: int = 5):
        if self.nbrs is None:
            self.load()
        dist, idx = self.nbrs.kneighbors(qvec.reshape(1, -1), n_neighbors=k, return_distance=True)
        sims = 1.0 - dist[0]  # cosine similarity -> similarity
        out = []
        for s, i in zip(sims.tolist(), idx[0].tolist()):
            if 0 <= i < len(self.meta):
                out.append((float(s), self.meta[i]))
        return out